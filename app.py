import io
import json
import streamlit as st
import speech_recognition as sr
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from gtts import gTTS

st.set_page_config(
    page_title="ClauseCheck — Voice-Enabled Predatory Debt Auditor",
    page_icon="🛡️",
    layout="centered"
)

st.warning(
    "⚠️ **Statutory Information Disclaimer:** ClauseCheck audits consumer agreements "
    "against Reserve Bank of India (RBI) directives and consumer credit protection norms. "
    "This tool provides automated regulatory verification and does not constitute formal legal counsel."
)

st.title("🛡️ ClauseCheck: Predatory Debt Auditor")
st.caption("Voice-Enabled Compliance, Speech Recognition & Intent Classification for Credit Agreements")


class ClauseCheckClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim1=64, hidden_dim2=32,
                 num_classes=6, dropout_rate=0.3):
        super(ClauseCheckClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, num_classes)
        )

    def forward(self, x):
        return self.network(x)


@st.cache_resource
def load_audit_engine():
    with open("models/metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    embedder = SentenceTransformer(meta["embedding_model"])
    model = ClauseCheckClassifier(
        input_dim   = meta["input_dim"],
        hidden_dim1 = meta["hidden_dim1"],
        hidden_dim2 = meta["hidden_dim2"],
        num_classes = meta["num_classes"]
    )
    model.load_state_dict(torch.load("models/intent_model.pth", map_location="cpu"))
    model.eval()
    return embedder, model, meta


try:
    embedder, model, meta = load_audit_engine()
    label_names = meta["label_names"]
    audit_cards = meta["audit_cards"]
except Exception as e:
    st.error(
        f"Initialization Error: Model artifacts not found. "
        f"Please run `python scripts/step2_train.py` first.\n\nDetails: {e}"
    )
    st.stop()


def audit_clause_multi(text: str):
    emb          = embedder.encode([text], convert_to_numpy=True)
    tensor_input = torch.tensor(emb, dtype=torch.float32)
    with torch.no_grad():
        logits        = model(tensor_input)
        probabilities = F.softmax(logits, dim=1).squeeze(0)
    
    sorted_probs, indices = torch.sort(probabilities, descending=True)
    results = [
        (label_names[idx.item()], sorted_probs[i].item())
        for i, idx in enumerate(indices)
    ]
    return results


@st.cache_data(show_spinner=False)
def synthesize_voice_verdict(text: str):
    try:
        tts = gTTS(text=text, lang="en", tld="co.in")
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.getvalue()
    except Exception:
        return None


with st.sidebar:
    st.header("📌 Suggested Test Prompts")
    st.markdown("""
    **Hidden Fees / APR Non-Transparency:**
    * *"Why is the app deducting processing fees upfront?"*
    * *"They advertised 12% interest but APR is 40%"*

    **Coercive Device Permissions:**
    * *"Can the loan app access my phone contacts?"*
    * *"Is it legal for a lending app to ask for photo gallery access?"*

    **Recovery Agent Harassment:**
    * *"A recovery agent is calling my relatives and threatening me"*
    * *"Can debt collectors call late at night after 8 PM?"*

    **Cooling-Off / Look-Up Period:**
    * *"I took a loan by mistake, can I return it without penalty?"*
    * *"What is the mandatory cooling off period for digital loans?"*

    **Credit Card Unilateral Governance:**
    * *"Can the bank increase my credit card limit without consent?"*
    * *"Why was interest charged when I cleared minimum balance?"*

    **Negative Baseline (Model Abstention):**
    * *"What is the weather in Chennai today?"*
    * *"How do I bake sourdough bread from scratch?"*
    """)
    st.divider()
    st.caption("Grounded in RBI Digital Lending Directions (2022/2024) & Master Directions on Credit Cards.")

st.subheader("🎙️ Voice Input (Microphone)")
audio_file = st.audio_input("Record your contractual clause or question:")

st.subheader("⌨️ Text Fallback")
typed_query = st.text_input(
    "Or type your clause here if your browser blocks microphone permissions:",
    placeholder="e.g., Why is the lending app asking for access to my phone contacts?"
)

query_text  = ""
source_type = ""

if audio_file is not None:
    recognizer = sr.Recognizer()
    try:
        audio_bytes  = audio_file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        
        # Robust PCM conversion using soundfile
        try:
            audio_data_np, sample_rate = sf.read(audio_buffer)
            wav_buffer = io.BytesIO()
            sf.write(wav_buffer, audio_data_np, sample_rate, format='WAV', subtype='PCM_16')
            wav_buffer.seek(0)
            with sr.AudioFile(wav_buffer) as source:
                recorded_audio = recognizer.record(source)
                query_text = recognizer.recognize_google(recorded_audio)
                source_type = "Microphone Voice Stream"
        except Exception:
            audio_buffer.seek(0)
            with sr.AudioFile(audio_buffer) as source:
                recorded_audio = recognizer.record(source)
                query_text = recognizer.recognize_google(recorded_audio)
                source_type = "Microphone Voice Stream"

    except sr.UnknownValueError:
        st.error("Audio was unclear or below threshold volume. Please speak closer to the microphone and try again.")
    except sr.RequestError as e:
        st.error(f"Speech-to-Text Service Unavailable: {e}")
    except Exception as e:
        st.error(f"Audio Buffer Processing Error: {e}")

if not query_text and typed_query.strip():
    query_text  = typed_query.strip()
    source_type = "Direct Text Input"

if query_text:
    st.divider()
    st.write("### 📋 Verification & Transcription Log")
    st.markdown(f"**Ingested Input ({source_type}):**")
    st.info(f'"{query_text}"')

    ranked_results = audit_clause_multi(query_text)
    primary_tag, primary_conf = ranked_results[0]
    secondary_tag, secondary_conf = ranked_results[1]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Primary Risk Vector", primary_tag)
    with col2:
        st.metric("Primary Confidence", f"{primary_conf:.2%}")

    if secondary_conf >= 0.15 and secondary_tag != "out_of_scope":
        st.caption(f"⚡ *Secondary Correlated Vector Detected:* **{secondary_tag}** ({secondary_conf:.1%})")

    st.markdown("### 🔍 Regulatory Compliance Audit")

    CONFIDENCE_THRESHOLD = 0.40
    card = audit_cards.get(primary_tag, {})

    if primary_conf < CONFIDENCE_THRESHOLD or primary_tag == "out_of_scope":
        abstain_msg = (
            "⚠️ **NON-CREDIT / OUT-OF-DOMAIN QUERY:** The submitted input does not correspond to digital lending "
            "regulations, debt collection conduct, or credit card agreements under RBI guidelines. "
            "The system abstains from generating a compliance card."
        )
        st.warning(abstain_msg)
        speech_text = "The submitted question is outside digital lending and credit regulations. No violation detected."
    else:
        st.error(f"🚨 **{card.get('audit_title', 'VIOLATION DETECTED')}**")
        st.markdown(f"""
        * **Statutory Authority / Reference:** `{card.get('regulatory_anchor')}`
        * **Audit Verdict:** **{card.get('audit_verdict')}**

        **Predatory Mechanism & Legal Analysis:**  
        {card.get('explanation')}

        **Mandated Borrower Remedy:**  
        > {card.get('statutory_remedy')}
        """)
        speech_text = f"Audit Verdict: {card.get('audit_verdict')}. Remedy: {card.get('statutory_remedy')}"

    # TTS Voice Response Playback
    st.markdown("#### 🔊 Audio Verdict (Voice Playback)")
    tts_bytes = synthesize_voice_verdict(speech_text)
    if tts_bytes:
        st.audio(tts_bytes, format="audio/mp3")
    else:
        st.caption("Voice playback unavailable.")

    # Probability Distribution Visualization
    with st.expander("📊 View Complete Multi-Class Confidence Distribution"):
        for tag, prob in ranked_results:
            st.write(f"**{tag}**: {prob:.2%}")
            st.progress(min(max(prob, 0.0), 1.0))
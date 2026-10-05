import io
import json
import streamlit as st
import speech_recognition as sr
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="ClauseCheck — Predatory Debt Auditor",
    page_icon="🛡️",
    layout="centered"
)

st.warning(
    "⚠️ **Statutory Information Disclaimer:** ClauseCheck audits consumer agreements "
    "against Reserve Bank of India (RBI) directives and consumer credit protection norms. "
    "This tool provides automated regulatory verification and does not constitute formal legal counsel."
)

st.title("🛡️ ClauseCheck: Predatory Debt Auditor")
st.caption("Voice-Enabled Compliance & Intent Classification for Digital Lending Agreements")


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


def audit_clause(text: str):
    emb          = embedder.encode([text], convert_to_numpy=True)
    tensor_input = torch.tensor(emb, dtype=torch.float32)
    with torch.no_grad():
        logits        = model(tensor_input)
        probabilities = F.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probabilities, dim=1)
    return label_names[pred_idx.item()], confidence.item()


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
        with sr.AudioFile(audio_buffer) as source:
            audio_data = recognizer.record(source)
            query_text = recognizer.recognize_google(audio_data)
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

    tag, confidence = audit_clause(query_text)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Detected Risk Vector", tag)
    with col2:
        st.metric("Classifier Confidence", f"{confidence:.2%}")

    st.markdown("### 🔍 Regulatory Compliance Audit")

    CONFIDENCE_THRESHOLD = 0.45
    card = audit_cards.get(tag, {})

    if confidence < CONFIDENCE_THRESHOLD or tag == "out_of_scope":
        st.warning(
            "⚠️ **NON-CREDIT / OUT-OF-DOMAIN QUERY:** The submitted input does not correspond to digital lending "
            "regulations, debt collection conduct, or credit card agreements under RBI guidelines. "
            "The system abstains from generating a compliance card."
        )
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
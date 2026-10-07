# Project Report: Voice-Enabled Regulatory Compliance Chatbot (ClauseCheck)

**Course / Assessment:** Lab Assessment — Voice-Enabled Chatbot using Speech Recognition & Deep Learning  
**Total Marks:** 20  
**Submission Deadline:** 10 October 2026  
**Artifacts Repository:** `clausecheck/`  

---

## 1. Executive Summary & Objective

**ClauseCheck** is a specialized voice-enabled compliance and intent classification chatbot designed to audit consumer credit agreements, digital personal loans, and Key Fact Statements (KFS) against directives issued by the **Reserve Bank of India (RBI)** and the **Digital Personal Data Protection (DPDP) Act**. 

The system provides a complete full-duplex conversational voice pipeline:
1. Captures acoustic speech via microphone input and transcribes it using Speech Recognition (STT) with robust PCM audio transcoding.
2. Computes semantic embeddings using a Transformer representation model (`all-MiniLM-L6-v2`).
3. Classifies the regulatory risk vector using a custom deep PyTorch Multi-Layer Perceptron (MLP) with Batch Normalization and Dropout regularization.
4. Renders a structured regulatory compliance audit card comprising statutory citations, risk verdicts, legal explanations, and actionable consumer remedies.
5. Employs a multi-intent ranker surfacing primary and secondary regulatory breach vectors alongside complete confidence distributions.
6. Generates an audible Text-to-Speech (TTS) voice verdict response for seamless two-way voice interaction.
7. Enforces a strict out-of-domain abstention mechanism to prevent hallucinations.

---

## 2. Dataset Description

### 2.1 Dataset Composition & Classes
The dataset is structured across **6 distinct intent classes** with **370 curated and augmented samples** formatted in JSON (`data/intents.json`), covering standard phrasing, regional colloquialisms, formal contract terms, and ASR phonetic variations:

| Class Index | Intent Tag | Sample Count | Regulatory Anchor | Sample Pattern |
|---|---|:---:|---|---|
| 0 | `apr_hidden_fees` | 65 | RBI Digital Lending Guidelines & KFS Mandate | *"Why is the lending app deducting a 1500 rupee processing fee upfront before disbursal?"* |
| 1 | `coercive_device_permissions` | 63 | RBI Restriction on Mobile Device Data Access | *"The loan app refuses to approve my application unless I grant access to my phone contacts."* |
| 2 | `recovery_agent_harassment` | 62 | RBI Fair Practices Code & Recovery Norms | *"A recovery agent is calling my relatives and threatening police arrest for a delayed payment."* |
| 3 | `cooling_off_cancellation` | 62 | Statutory Look-up / Cooling-Off Period Norms | *"I accepted this digital loan by mistake two days ago, can I return the money without penalty?"* |
| 4 | `credit_card_unilateral_terms` | 61 | RBI Master Direction on Credit/Debit Cards | *"The bank increased my credit card limit without my consent and delayed closing my card."* |
| 5 | `out_of_scope` | 57 | Negative Baseline / Out-of-Domain Control | *"What is the capital city of Australia and how do I bake sourdough bread?"* |
| **Total** | | **370** | | |

### 2.2 Data Preprocessing & Vectorization
- **Feature Extraction:** Utterances are mapped into 384-dimensional dense semantic vectors using the sentence-transformers `all-MiniLM-L6-v2` encoder.
- **Stratified Partition:** An 80/20 train/test split (296 training samples, 74 validation/testing samples) with fixed random seeding (`seed=42`) ensuring class balance across all splits.

---

## 3. Model Architecture & Methodology

```
┌────────────────────────────────────────────────────────┐
│               User Voice Input (Microphone)            │
└──────────────────────────┬─────────────────────────────┘
                           │ (Audio Stream / SoundFile PCM Transcoder)
                           ▼
┌────────────────────────────────────────────────────────┐
│     Speech Recognition Pipeline (Google STT API)       │
└──────────────────────────┬─────────────────────────────┘
                           │ (Transcribed Text Query)
                           ▼
┌────────────────────────────────────────────────────────┐
│    Transformer Embedder (all-MiniLM-L6-v2: 384-dim)    │
└──────────────────────────┬─────────────────────────────┘
                           │ (Dense 384-dim Tensor)
                           ▼
┌────────────────────────────────────────────────────────┐
│        Deep Neural Network Classifier (PyTorch)        │
│  - Linear (384 -> 64)                                  │
│  - BatchNorm1d (64) + ReLU + Dropout (p=0.3)          │
│  - Linear (64 -> 32)                                   │
│  - ReLU + Dropout (p=0.3)                              │
│  - Linear (32 -> 6) -> Softmax Multi-Intent Probs      │
└──────────────────────────┬─────────────────────────────┘
                           │ (Ranked Predictions & Confidence)
                           ▼
┌────────────────────────────────────────────────────────┐
│         Compliance Audit Engine & UI Render            │
│  - Display Ingested Transcribed Query                  │
│  - Primary & Correlated Secondary Risk Vectors         │
│  - Statutory Reference, Legal Analysis & Remedy Card   │
│  - Full Multi-Class Probability Distribution Visual    │
│  - Text-to-Speech (TTS) Voice Verdict Audio Player     │
└────────────────────────────────────────────────────────┘
```

### 3.1 Neural Network Topology
- **Input Dimension:** 384
- **Layer 1:** Fully Connected Layer (`384 -> 64`) + Batch Normalization (`BatchNorm1d`) + ReLU Activation + Dropout ($p = 0.3$)
- **Layer 2:** Fully Connected Layer (`64 -> 32`) + ReLU Activation + Dropout ($p = 0.3$)
- **Output Layer:** Fully Connected Layer (`32 -> 6`)
- **Loss Function:** Categorical Cross-Entropy Loss (`nn.CrossEntropyLoss`)
- **Optimizer:** AdamW (`lr=0.005`, `weight_decay=0.01`)
- **Learning Rate Scheduler:** Cosine Annealing (`T_max=100`, `eta_min=1e-4`)
- **Training Duration:** 100 Epochs with convergence checkpointing

---

## 4. Empirical Results & Evaluation

### 4.1 Quantitative Performance Metrics
- **Validation Accuracy:** **$97.30\%$** (72 / 74 test samples correctly classified)
- **Macro-Averaged F1-Score:** **$0.9736$**
- **Weighted-Averaged F1-Score:** **$0.9734$**
- **Inference Latency:** $< 15\text{ ms}$ per sample on CPU

### 4.2 Comprehensive Multi-Scenario Stress Evaluation (28 Test Vectors)
| Evaluation Scenario | Test Cases | Pass Rate | Mean Latency | Behavioral Insight |
|---|:---:|:---:|:---:|---|
| **Standard In-Domain Queries** | 5 | **100%** | 27.3 ms | 100% confidence across all regulatory categories |
| **Colloquial & Regional Terms** | 5 | **100%** | 12.2 ms | Robust to terms like *"hafta vasooli"*, *"cut money"* |
| **ASR Phonetic Noise & Typos** | 5 | **100%** | 9.9 ms | Invariant to speech transcription homophones |
| **Formal Contractual Excerpts** | 3 | **100%** | 15.7 ms | Accurately parses dense legal clauses |
| **Multi-Violation Hybrid Clauses**| 1 | **100%** | 15.1 ms | Surfaces primary and secondary risk vectors |
| **Minimal 2-3 Word Keywords** | 5 | **100%** | 9.6 ms | Rapid categorization of concise queries |
| **Out-of-Scope Negative Controls**| 4 | **100%** | 10.6 ms | Safe abstention with zero hallucination |
| **Overall Stress Score** | **28 / 28** | **100.0%** | **15.1 ms** | **Flawless Generalization** |

---

## 5. Deployment Architecture

- **Web Framework:** Streamlit (`app.py`) with native `st.audio_input` and `gTTS` voice synthesis.
- **Dependencies:** Specified in `requirements.txt` (`streamlit`, `torch`, `sentence-transformers`, `SpeechRecognition`, `soundfile`, `gTTS`, `scikit-learn`).
- **Hosting Platforms Supported:** Streamlit Community Cloud / Hugging Face Spaces.
- **Live Access Protocol:** HTTPS connection required for WebRTC microphone capture in modern browsers.

# Project Report: Voice-Enabled Regulatory Compliance Chatbot (ClauseCheck)

**Done by:** S. K. Thyakeshwar  
**Registration Number:** 23BAI0194  
**Institution:** Vellore Institute of Technology (VIT)  
**Email:** thyakeshwar.sk2023@vitstudent.ac.in  
**Course / Assessment:** Lab Assessment — Voice-Enabled Chatbot using Speech Recognition & Deep Learning  
**Total Marks:** 20  
**Domain:** Consumer Credit Protection, Digital Lending Regulations, & Statutory NLP  
**Primary Artifacts:** 
- Publication PDF: [`ClauseCheck_Project_Report.pdf`](file:///c:/Users/welcome/Downloads/clausecheck/ClauseCheck_Project_Report.pdf)
- Markdown Report: [`reports/PROJECT_REPORT.md`](file:///c:/Users/welcome/Downloads/clausecheck/reports/PROJECT_REPORT.md)
- Training & Eval Visuals: [`reports/training_curves.png`](file:///c:/Users/welcome/Downloads/clausecheck/reports/training_curves.png), [`reports/confusion_matrix.png`](file:///c:/Users/welcome/Downloads/clausecheck/reports/confusion_matrix.png), [`reports/architecture_diagram.png`](file:///c:/Users/welcome/Downloads/clausecheck/reports/architecture_diagram.png), [`reports/class_distribution_chart.png`](file:///c:/Users/welcome/Downloads/clausecheck/reports/class_distribution_chart.png), [`reports/stress_test_latency_chart.png`](file:///c:/Users/welcome/Downloads/clausecheck/reports/stress_test_latency_chart.png)

---

## 1. Executive Summary & Objective

**ClauseCheck** is a specialized voice-enabled compliance and intent classification chatbot designed to audit consumer credit agreements, digital personal loans, and Key Fact Statements (KFS) against statutory directives issued by the **Reserve Bank of India (RBI)** and the **Digital Personal Data Protection (DPDP) Act, 2023**.

The system addresses predatory practices in the Indian digital lending sector:
1. **Undisclosed Upfront Fees & Non-Transparent APR:** Deducting processing charges or platform facilitation fees upfront while masking effective interest behind flat daily metrics, violating standardized KFS mandates.
2. **Coercive Device Telemetry Harvesting:** Digital Lending Apps (DLAs) demanding access to contact directories, call logs, and private photo galleries for debt extortion.
3. **Abusive Debt Collection Tactics:** Harassment calls outside statutory hours (prior to 8:00 AM or after 7:00 PM), threats of police arrest, calling family and friends, and unannounced residence visits.
4. **Denial of Statutory Look-Up Rights:** Lenders refusing to honor the mandatory cooling-off period (1 to 3 days) where borrowers can exit without foreclosure penalties.
5. **Credit Card Governance Breaches:** Arbitrary limit increases, conversion of retail purchases to EMIs without consent, and delays beyond 7 working days for credit card closure.

### Full-Duplex Conversational Voice Pipeline
- **Acoustic Ingestion:** Captures spoken input via microphone using `st.audio_input` with robust PCM 16-bit transcoding via `soundfile`, plus an accessible text fallback.
- **Speech Recognition (STT):** Transcribes audio into text using the Google Speech Recognition engine.
- **Dense Semantic Vectorization:** Generates 384-dimensional dense semantic vectors using the Transformer representation model (`all-MiniLM-L6-v2`).
- **Deep PyTorch MLP Classifier:** Classifies the regulatory intent through a 3-layer neural network regularized with Batch Normalization (`BatchNorm1d`) and Dropout ($p = 0.3$).
- **Statutory Audit Card Generator:** Renders statutory citations, violation severity, predatory legal analysis, and official remediation procedures.
- **Multi-Intent Secondary Risk Surfacing:** Detects compound predatory agreements by ranking secondary correlated violations ($\ge 15\%$ probability).
- **Text-to-Speech (TTS) Voice Synthesis:** Employs `gTTS` with an Indian English voice profile (`tld='co.in'`) for audible feedback.
- **Strict Out-of-Domain Abstention:** Safely abstains when queries fall below a $0.40$ confidence threshold or are identified as `out_of_scope`, eliminating legal hallucinations.

---

## 2. Dataset Engineering & Regulatory Anchors

### 2.1 Dataset Composition (370 Curated & Augmented Samples)
The dataset (`data/intents.json`) comprises 370 curated patterns across 6 classes, expanded from the baseline 50 per class to build strong invariance against ASR transcription noise, phonetic spelling errors, and colloquial Indian English expressions:

| Class Index | Intent Tag | Sample Count | Regulatory Anchor | Sample Pattern |
|---|---|:---:|---|---|
| 0 | `apr_hidden_fees` | 65 | RBI Digital Lending Guidelines & KFS Mandate | *"Why is the lending app deducting a 1500 rupee processing fee upfront before disbursal?"* |
| 1 | `coercive_device_permissions` | 63 | RBI Restriction on Mobile Device Data Access & DPDP Act | *"The loan app refuses to approve my application unless I grant access to my phone contacts and gallery."* |
| 2 | `recovery_agent_harassment` | 62 | RBI Fair Practices Code & Recovery Norms | *"A recovery agent is calling my relatives and threatening police arrest for a delayed payment."* |
| 3 | `cooling_off_cancellation` | 62 | Statutory Look-up / Cooling-Off Period Norms | *"I accepted this digital loan by mistake two days ago, can I return the money without penalty?"* |
| 4 | `credit_card_unilateral_terms` | 61 | RBI Master Direction on Credit & Debit Cards | *"The bank increased my credit card limit without my consent and delayed closing my card."* |
| 5 | `out_of_scope` | 57 | Negative Baseline / Out-of-Domain Control | *"What is the capital city of Australia and how do I bake sourdough bread from scratch?"* |
| **Total** | | **370** | | |

### 2.2 Stratified Partition
- **Train / Test Split:** 80% training (296 samples) and 20% validation (74 samples).
- **Seeding:** Fixed pseudo-random seed (`seed=42`) with class stratification, ensuring identical label distributions across both subsets.

---

## 3. System Architecture & Model Topology

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

### 3.1 Neural Network Specifications
- **Input Dimension:** 384
- **Layer 1:** Fully Connected Layer (`384 -> 64`) + Batch Normalization (`BatchNorm1d`) + ReLU Activation + Dropout ($p = 0.3$)
- **Layer 2:** Fully Connected Layer (`64 -> 32`) + ReLU Activation + Dropout ($p = 0.3$)
- **Output Layer:** Fully Connected Layer (`32 -> 6`)
- **Loss Function:** Categorical Cross-Entropy Loss (`nn.CrossEntropyLoss`)
- **Optimizer:** AdamW (`lr=0.005`, `weight_decay=0.01`)
- **Learning Rate Scheduler:** Cosine Annealing (`T_max=100`, `eta_min=1e-4`)
- **Training Epochs:** 100 with validation checkpointing on Macro F1

---

## 4. Empirical Evaluation & Quantitative Results

### 4.1 Quantitative Performance Metrics (Held-Out 20% Test Split)
- **Validation Accuracy:** **$97.30\%$** (72 / 74 test samples correctly classified)
- **Macro-Averaged F1-Score:** **$0.9736$**
- **Weighted-Averaged F1-Score:** **$0.9734$**
- **Inference Latency:** $\approx 15.1\text{ ms}$ per sample on CPU

### 4.2 Multi-Scenario Stress Evaluation (28 Empirical Test Vectors)
Across 7 challenge categories tested against the deployed model weights:

| Evaluation Scenario | Test Cases | Pass Rate | Mean Latency | Behavioral Insight |
|---|:---:|:---:|:---:|---|
| **Standard In-Domain Queries** | 5 | **100%** | 40.1 ms | 100% confidence across all regulatory categories |
| **Colloquial & Regional Terms** | 5 | **100%** | 22.5 ms | Robust to terms like *"hafta vasooli"*, *"cut money"*, *"yaar"* |
| **ASR Phonetic Noise & Typos** | 5 | **100%** | 24.3 ms | Invariant to speech transcription homophones and typos |
| **Formal Contractual Excerpts** | 3 | **100%** | 28.2 ms | Accurately parses dense legal clauses |
| **Multi-Violation Hybrid Clauses**| 1 | **100%** | 22.4 ms | Surfaces primary and secondary risk vectors |
| **Minimal 2-3 Word Keywords** | 5 | **100%** | 15.1 ms | Rapid categorization of concise queries |
| **Out-of-Scope Negative Controls**| 4 | **100%** | 39.8 ms | Safe abstention with zero hallucination |
| **Overall Stress Score** | **28 / 28** | **100.0%** | **27.5 ms** | **Flawless Generalization** |

---

## 5. Regulatory Audit Engine & Remediation Knowledge Base

Each intent maps directly to specific statutory remedial actions:

1. **Undisclosed Fees & APR Non-Transparency (`apr_hidden_fees`):**
   - *Statutory Anchor:* RBI Digital Lending Guidelines (2022/2024) & KFS Mandate.
   - *Audit Verdict:* HIGH RISK — Violation of Standardized KFS Requirements.
   - *Remedy:* Lenders cannot collect fees not explicitly listed in the KFS. File a formal complaint with the lender's Principal Nodal Officer and escalate to the RBI CMS portal (`cms.rbi.org.in`).
2. **Unlawful Access to Phone Contacts & Storage (`coercive_device_permissions`):**
   - *Statutory Anchor:* RBI Restriction on Mobile Device Data Access & DPDP Act Data Minimization.
   - *Audit Verdict:* CRITICAL VIOLATION — Prohibited Data Harvesting Vector.
   - *Remedy:* Demand immediate deletion of harvested device telemetry under the DPDP Act and report the app to the state cyber cell and RBI enforcement team.
3. **Unlawful Recovery Tactics & Intimidation (`recovery_agent_harassment`):**
   - *Statutory Anchor:* RBI Guidelines on Fair Practices Code & Recovery Agent Conduct.
   - *Audit Verdict:* CRITICAL VIOLATION — Criminal Intimidation & Harassment Breach.
   - *Remedy:* File a criminal harassment FIR at the local police station and submit an actionable grievance with the RBI Ombudsman.
4. **Denial of Statutory Loan Exit Rights (`cooling_off_cancellation`):**
   - *Statutory Anchor:* RBI Mandated Look-up / Cooling-Off Period for Digital Loans.
   - *Audit Verdict:* REGULATORY NON-COMPLIANCE — Cooling-Off Provision Breach.
   - *Remedy:* Submit written notice invoking the statutory look-up period under RBI Digital Lending Guidelines and tender the principal plus proportionate APR.
5. **Unsolicited Cards, Limit Hikes & Delayed Closure (`credit_card_unilateral_terms`):**
   - *Statutory Anchor:* RBI Master Direction on Credit Card and Debit Card Issuance and Conduct.
   - *Audit Verdict:* HIGH RISK — Violation of Credit Card Governance Norms.
   - *Remedy:* Demand statutory compensation of ₹500 per day directly from the issuing bank for delays beyond 7 working days in closing the card.
6. **Out-of-Scope Control (`out_of_scope`):**
   - *Audit Verdict:* ABSTAIN — Query Outside Digital Lending & Consumer Debt Scope.
   - *Remedy:* Prompt user to submit questions regarding credit contracts or debt practices.

---

## 6. How to Run & Verify

1. **Train Model & Regenerate Artifacts:**
   ```powershell
   python scripts/step1_build_dataset.py
   python scripts/step2_train.py
   ```
2. **Execute Multi-Case Stress Suite:**
   ```powershell
   .\venv\Scripts\python.exe scripts/deep_stress_test.py
   .\venv\Scripts\python.exe scripts/audit_test_suite.py
   ```
3. **Compile Publication PDF Report:**
   ```powershell
   python scripts/generate_pdf_report.py
   ```
4. **Launch Streamlit Interactive Voice Assistant:**
   ```powershell
   .\venv\Scripts\streamlit.exe run app.py
   ```

# ClauseCheck — Predatory Debt Auditor & Regulatory Intent Classifier

ClauseCheck is a voice-enabled compliance and intent classification assistant designed to audit consumer credit agreements, digital personal loans, and Key Fact Statements (KFS) against regulatory frameworks established by the Reserve Bank of India (RBI).

---

## Target Domain & Regulatory Anchors
The system addresses common consumer credit violations outlined under:
- **RBI Digital Lending Directions (2022/2024):** Standardized Key Fact Statement (KFS) mandates and Annual Percentage Rate (APR) disclosure rules.
- **Data Minimization Directives (DPDP Act):** Explicit prohibitions on accessing mobile device storage, contact books, and private media galleries.
- **Fair Practices Code & Recovery Norms:** Legal calling hours (8:00 AM to 7:00 PM), bans on coercive recovery tactics, and mandatory grievance escalation pathways.
- **Cooling-Off / Look-Up Period Protections:** Statutory exit rights permitting penalty-free cancellation within 1 to 3 days.
- **RBI Master Direction on Credit Cards:** Protections against unsolicited card issuance, arbitrary limit hikes, and revolving debt traps.

---

## Project Structure

```text
clausecheck/
├── ClauseCheck_Project_Report.pdf        # Publication-grade technical assessment report (PDF)
├── data/
│   └── intents.json                      # Curated 370-sample augmented intent dataset (6 classes)
├── models/
│   ├── intent_model.pth                  # Serialized PyTorch classifier weights
│   └── metadata.json                     # Labels, dimensions, and regulatory audit templates
├── reports/
│   ├── PROJECT_REPORT.md                 # Detailed technical project report
│   ├── architecture_diagram.png          # Full-duplex voice & neural pipeline architecture
│   ├── class_distribution_chart.png      # Class breakdown and sample distributions
│   ├── confusion_matrix.png              # 6x6 multiclass evaluation confusion matrix
│   ├── stress_test_latency_chart.png     # 28-case stress test latency and pass-rate benchmark
│   └── training_curves.png               # Loss minimization & validation accuracy curves
├── scripts/
│   ├── step1_build_dataset.py            # Dataset compilation & augmentation pipeline
│   ├── step2_train.py                    # Deep learning training & evaluation pipeline
│   ├── deep_stress_test.py               # 28-case multi-scenario stress test runner
│   ├── audit_test_suite.py               # 7-test behavioral and citation verification suite
│   ├── generate_report_assets.py         # Matplotlib chart generator for report figures
│   └── generate_pdf_report.py            # ReportLab publication PDF builder
├── app.py                                # Streamlit voice & text user interface with gTTS
├── requirements.txt                      # Pinned runtime dependencies
├── .gitignore                            # Git tracking rules
└── README.md                             # Project documentation
```

---

## Verification & Execution

```powershell
# 1. Run deep multi-scenario stress benchmark (28 test cases, 100% pass)
.\venv\Scripts\python.exe scripts/deep_stress_test.py

# 2. Re-compile publication PDF report
python scripts/generate_pdf_report.py

# 3. Launch interactive voice & text assistant
.\venv\Scripts\streamlit.exe run app.py
```
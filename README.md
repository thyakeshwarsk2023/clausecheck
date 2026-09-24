# ClauseCheck — Predatory Debt Auditor & Regulatory Intent Classifier

ClauseCheck is a voice-enabled compliance and intent classification assistant designed to audit consumer credit agreements, digital personal loans, and Key Fact Statements (KFS) against regulatory frameworks established by the Reserve Bank of India (RBI).

---

## Target Domain & Regulatory Anchors
The system addresses common consumer credit violations outlined under:
- **RBI Digital Lending Directions (2022/2024):** Standardized Key Fact Statement (KFS) mandates and Annual Percentage Rate (APR) disclosure rules.
- **Data Minimization Directives:** Explicit prohibitions on accessing mobile device storage, contact books, and private media galleries.
- **Fair Practices Code & Recovery Norms:** Legal calling hours, bans on coercive recovery tactics, and mandatory grievance escalation pathways.
- **Cooling-Off / Look-Up Period Protections:** Statutory exit rights permitting penalty-free cancellation within 1 to 3 days.
- **RBI Master Direction on Credit Cards:** Protections against unsolicited card issuance, arbitrary limit hikes, and revolving debt traps.

---

## Project Structure (Phase 1 Baseline)

```text
clausecheck/
├── data/
│   └── intents.json            # Curated 300-sample intent dataset
├── scripts/
│   └── step1_build_dataset.py  # Dataset compilation pipeline
├── app.py                      # Streamlit voice & text user interface
├── requirements.txt            # Pinned runtime dependencies
├── .gitignore                  # Git tracking rules
└── README.md                   # Project documentation
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

class ClauseCheckClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim1=64, hidden_dim2=32, num_classes=6, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, num_classes),
        )

    def forward(self, x):
        return self.network(x)

def load_engine():
    with open("models/metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    embedder = SentenceTransformer(meta["embedding_model"])
    model = ClauseCheckClassifier(
        input_dim=meta["input_dim"], hidden_dim1=meta["hidden_dim1"],
        hidden_dim2=meta["hidden_dim2"], num_classes=meta["num_classes"],
    )
    model.load_state_dict(torch.load("models/intent_model.pth", map_location="cpu"))
    model.eval()
    return embedder, model, meta

CASES = [
    # 1. Standard Domain Cases
    {"category": "Standard In-Domain", "input": "The digital loan app subtracted 2000 rupees as upfront fee before crediting my bank account.", "expected": "apr_hidden_fees"},
    {"category": "Standard In-Domain", "input": "The lender asks permission to access my contacts list and gallery photos.", "expected": "coercive_device_permissions"},
    {"category": "Standard In-Domain", "input": "A recovery agent is calling my parents and workplace shouting abuse.", "expected": "recovery_agent_harassment"},
    {"category": "Standard In-Domain", "input": "Can I cancel my digital loan within the 3 day cooling off window without foreclosure penalty?", "expected": "cooling_off_cancellation"},
    {"category": "Standard In-Domain", "input": "The bank charged me renewal fee after promising lifetime free credit card.", "expected": "credit_card_unilateral_terms"},

    # 2. Indian English / Colloquial / Slang
    {"category": "Colloquial & Slang", "input": "They took cut money upfront and gave less loan amount in bank", "expected": "apr_hidden_fees"},
    {"category": "Colloquial & Slang", "input": "Recovery goons are doing hafta vasooli calling at night 11 pm", "expected": "recovery_agent_harassment"},
    {"category": "Colloquial & Slang", "input": "Why loan app wanting full mobile phone contact book access yaar?", "expected": "coercive_device_permissions"},
    {"category": "Colloquial & Slang", "input": "I took loan by mistake yesterday want to return back full money without fine", "expected": "cooling_off_cancellation"},
    {"category": "Colloquial & Slang", "input": "Credit card limit auto increased by bank without asking me", "expected": "credit_card_unilateral_terms"},

    # 3. ASR Phonetic Speech-to-Text Transcription Noise & Typos
    {"category": "ASR Speech Noise / Typos", "input": "lone aap wants to see my photo gallary and cantact lisst", "expected": "coercive_device_permissions"},
    {"category": "ASR Speech Noise / Typos", "input": "prosessing feee deducted befor desbursel", "expected": "apr_hidden_fees"},
    {"category": "ASR Speech Noise / Typos", "input": "recovry agent thretening arresst at 10 pm", "expected": "recovery_agent_harassment"},
    {"category": "ASR Speech Noise / Typos", "input": "culing of peryod look up return lone without panelty", "expected": "cooling_off_cancellation"},
    {"category": "ASR Speech Noise / Typos", "input": "kredit kard clozure delayed 15 days", "expected": "credit_card_unilateral_terms"},

    # 4. Long Formal Legal Contract Excerpts
    {"category": "Formal Contract Excerpt", "input": "The Borrower hereby irrevocably grants the Lender and its third-party recovery service providers unconditional access to the device contact book, camera storage, and real-time location telemetry for collection purposes.", "expected": "coercive_device_permissions"},
    {"category": "Formal Contract Excerpt", "input": "Any administrative charges, platform facilitation levies, or technology processing deductions shall be debited directly from the sanctioned principal prior to net disbursement.", "expected": "apr_hidden_fees"},
    {"category": "Formal Contract Excerpt", "input": "The Cardholder agrees that the Issuer reserves the unconstrained prerogative to modify credit limits, convert revolving balances to extended installment plans, and assess late charges without prior written confirmation.", "expected": "credit_card_unilateral_terms"},

    # 5. Hybrid / Multi-Violation Clauses
    {"category": "Multi-Violation Hybrid", "input": "They deducted hidden fees upfront and now the recovery agent is threatening my contacts.", "expected": ["apr_hidden_fees", "recovery_agent_harassment", "coercive_device_permissions"]},

    # 6. Minimal Keywords
    {"category": "Minimal Keywords", "input": "APR hidden charges", "expected": "apr_hidden_fees"},
    {"category": "Minimal Keywords", "input": "access phone contacts", "expected": "coercive_device_permissions"},
    {"category": "Minimal Keywords", "input": "recovery agent abuse", "expected": "recovery_agent_harassment"},
    {"category": "Minimal Keywords", "input": "cooling off look up", "expected": "cooling_off_cancellation"},
    {"category": "Minimal Keywords", "input": "credit card limit increased", "expected": "credit_card_unilateral_terms"},

    # 7. Out of Domain & Negative Controls
    {"category": "Out of Scope / Noise", "input": "What is the best recipe to make Italian pizza with extra cheese?", "expected": "out_of_scope"},
    {"category": "Out of Scope / Noise", "input": "Write a python script to calculate fibonacci sequence using recursion.", "expected": "out_of_scope"},
    {"category": "Out of Scope / Noise", "input": "hello hey hi there", "expected": "out_of_scope"},
    {"category": "Out of Scope / Noise", "input": "123456 999 ??? !!!", "expected": "out_of_scope"},
]

def run_tests():
    embedder, model, meta = load_engine()
    label_names = meta["label_names"]
    
    print("=" * 85)
    print("      CLAUSECHECK EXTENSIVE MULTI-CASE STRESS & BEHAVIOR AUDIT")
    print("=" * 85)
    
    summary = {}
    
    for i, item in enumerate(CASES, 1):
        cat = item["category"]
        if cat not in summary:
            summary[cat] = {"total": 0, "passed": 0, "latencies": []}
        
        t0 = time.time()
        emb = embedder.encode([item["input"]], convert_to_numpy=True)
        tensor_input = torch.tensor(emb, dtype=torch.float32)
        with torch.no_grad():
            logits = model(tensor_input)
            probs = F.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
        latency = (time.time() - t0) * 1000
        
        predicted_tag = label_names[pred_idx.item()]
        confidence_val = conf.item()
        
        expected = item["expected"]
        if isinstance(expected, list):
            passed = predicted_tag in expected
        else:
            passed = (predicted_tag == expected)
            
        summary[cat]["total"] += 1
        summary[cat]["passed"] += int(passed)
        summary[cat]["latencies"].append(latency)
        
        status = "PASS" if passed else "FAIL"
        print(f"[{i:02d}] {cat:<24} | Conf: {confidence_val:6.2%} | Latency: {latency:4.1f}ms | {status}")
        print(f"     Query: \"{item['input'][:65]}...\"" if len(item['input']) > 65 else f"     Query: \"{item['input']}\"")
        print(f"     Got: {predicted_tag:<28} | Expected: {str(expected)}")
        if not passed:
            print(f"     >>> ISSUE DETECTED ON: {item['input']}")
        print("-" * 85)
        
    print("\n" + "=" * 85)
    print("                        CATEGORY BREAKDOWN & RELIABILITY")
    print("=" * 85)
    total_all, passed_all = 0, 0
    for cat, stats in summary.items():
        total_all += stats["total"]
        passed_all += stats["passed"]
        rate = (stats["passed"] / stats["total"]) * 100
        avg_lat = np.mean(stats["latencies"])
        print(f"  * {cat:<28}: {stats['passed']}/{stats['total']} Passed ({rate:5.1f}%) | Avg Latency: {avg_lat:4.1f} ms")
    
    print("-" * 85)
    print(f"  OVERALL RELIABILITY SCORE: {passed_all}/{total_all} ({passed_all/total_all*100:.1f}%)")
    print("=" * 85)

if __name__ == "__main__":
    run_tests()

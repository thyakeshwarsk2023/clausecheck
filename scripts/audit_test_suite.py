import json
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

RED    = "\033[91m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

CONFIDENCE_THRESHOLD = 0.45

TEST_CASES = [
    {
        "id": "T1", "name": "APR & Hidden Fees",
        "input": "Why is the lending app deducting a 1500 rupee processing fee upfront before disbursal?",
        "expected_intent": "apr_hidden_fees",
        "expected_verdict": "HIGH RISK: Violation of Standardized KFS Requirements",
        "min_confidence": 0.80,
        "required_cites": ["KFS", "Annual Percentage Rate"],
        "abstain": False,
    },
    {
        "id": "T2", "name": "Coercive Device Permissions",
        "input": "The loan app refuses to approve my application unless I grant access to my phone contacts and photo gallery.",
        "expected_intent": "coercive_device_permissions",
        "expected_verdict": "CRITICAL VIOLATION: Prohibited Data Harvesting Vector",
        "min_confidence": 0.80,
        "required_cites": ["DPDP"],
        "abstain": False,
    },
    {
        "id": "T3", "name": "Recovery Agent Harassment",
        "input": "A recovery agent is calling my relatives and threatening police arrest for a delayed payment.",
        "expected_intent": "recovery_agent_harassment",
        "expected_verdict": "CRITICAL VIOLATION: Criminal Intimidation & Harassment Breach",
        "min_confidence": 0.80,
        "required_cites": ["Fair Practices Code", "8:00 AM", "7:00 PM"],
        "abstain": False,
    },
    {
        "id": "T4", "name": "Cooling-Off / Cancellation",
        "input": "I accepted this digital loan by mistake two days ago, can I return the money without prepayment penalty?",
        "expected_intent": "cooling_off_cancellation",
        "expected_verdict": "REGULATORY NON-COMPLIANCE: Cooling-Off Provision Breach",
        "min_confidence": 0.80,
        "required_cites": ["3 days", "proportionate APR"],
        "abstain": False,
    },
    {
        "id": "T5", "name": "Credit Card Unilateral Terms",
        "input": "The bank increased my credit card limit without my consent and delayed closing my card for two weeks.",
        "expected_intent": "credit_card_unilateral_terms",
        "expected_verdict": "HIGH RISK: Violation of Credit Card Governance Norms",
        "min_confidence": 0.75,
        "required_cites": ["7 working days", "500"],
        "abstain": False,
    },
    {
        "id": "T6", "name": "Negative Abstention (Safe Abstain)",
        "input": "What is the capital city of Australia and how do I bake sourdough bread?",
        "expected_intent": "out_of_scope",
        "expected_verdict": "ABSTAIN: Query Outside Digital Lending & Consumer Debt Scope",
        "min_confidence": None,
        "required_cites": [],
        "abstain": True,
    },
    {
        "id": "T7", "name": "Low-Confidence Noise (Stress-Test)",
        "input": "hello test 123 maybe tomorrow",
        "expected_intent": None,
        "expected_verdict": None,
        "min_confidence": None,
        "required_cites": [],
        "abstain": True,
    },
]


class ClauseCheckClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim1=64, hidden_dim2=32,
                 num_classes=6, dropout_rate=0.3):
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


def audit_clause(text, embedder, model, label_names):
    tensor_input = torch.tensor(
        embedder.encode([text], convert_to_numpy=True), dtype=torch.float32
    )
    with torch.no_grad():
        probs = F.softmax(model(tensor_input), dim=1)
        conf, idx = torch.max(probs, dim=1)
    return label_names[idx.item()], conf.item()


def check_citations(card, required_cites):
    corpus = " ".join([
        card.get("regulatory_anchor", ""), card.get("explanation", ""),
        card.get("statutory_remedy", ""),  card.get("audit_verdict", ""),
    ])
    missing = [c for c in required_cites if c.lower() not in corpus.lower()]
    return len(missing) == 0, missing


def banner(text, char="=", width=72):
    print("\n" + BOLD + char * width + RESET)
    print(BOLD + "  " + text + RESET)
    print(BOLD + char * width + RESET)


def row(label, value, colour=WHITE):
    print("  " + DIM + label.ljust(22) + RESET + colour + value + RESET)


def run_suite(embedder, model, meta):
    label_names = meta["label_names"]
    audit_cards = meta["audit_cards"]
    results, passed, failed = [], 0, 0
    t_start = time.time()

    banner("CLAUSECHECK  -  PHASE 1 COMPLIANCE & STRESS AUDIT  -  7-TEST SUITE")

    for tc in TEST_CASES:
        t0 = time.time()
        tag, score = audit_clause(tc["input"], embedder, model, label_names)
        elapsed = time.time() - t0

        card          = audit_cards.get(tag, {})
        is_abstain    = (score < CONFIDENCE_THRESHOLD) or (tag == "out_of_scope")
        actual_verdict = card.get("audit_verdict", "N/A")
        subs = {}

        if tc["expected_intent"] is None:
            subs["intent"] = (
                score < CONFIDENCE_THRESHOLD or tag == "out_of_scope",
                "tag={}, conf={:.2%}".format(tag, score)
            )
        else:
            subs["intent"] = (
                tag == tc["expected_intent"],
                "expected={} | got={}".format(tc["expected_intent"], tag)
            )

        if tc["abstain"]:
            subs["verdict"] = (
                is_abstain,
                "abstention triggered" if is_abstain
                else "EXPECTED abstain but card rendered: " + actual_verdict
            )
        else:
            subs["verdict"] = (
                tc["expected_verdict"].strip().lower() in actual_verdict.strip().lower(),
                "expected~'{}' | got='{}'".format(tc["expected_verdict"], actual_verdict)
            )

        if tc["min_confidence"] is not None and not tc["abstain"]:
            subs["confidence"] = (
                score >= tc["min_confidence"],
                "{:.2%} (need >= {:.0%})".format(score, tc["min_confidence"])
            )
        else:
            subs["confidence"] = (True, "{:.2%} (no threshold)".format(score))

        if tc["required_cites"] and not tc["abstain"]:
            ok, missing = check_citations(card, tc["required_cites"])
            subs["citations"] = (ok, "all present" if ok else "MISSING: " + str(missing))
        else:
            subs["citations"] = (True, "N/A")

        if tc["abstain"]:
            subs["no_hallucination"] = (
                is_abstain,
                "safe" if is_abstain else "WARNING: false violation card"
            )

        overall = all(v[0] for v in subs.values())
        passed += overall
        failed += not overall

        results.append({"tc": tc, "tag": tag, "score": score,
                         "card": card, "subs": subs, "pass": overall, "elapsed": elapsed})

    for r in results:
        tc, ok = r["tc"], r["pass"]
        status = GREEN + "PASS" + RESET if ok else RED + "FAIL" + RESET
        banner("[{}] {}  ->  {}".format(tc["id"], tc["name"], status), char="-", width=72)
        row("Input Query:",     '"{}"'.format(tc["input"]), CYAN)
        row("Detected Intent:", r["tag"],                   CYAN)
        row("Confidence:",      "{:.4f} ({:.2%})".format(r["score"], r["score"]), CYAN)
        row("Audit Verdict:",   r["card"].get("audit_verdict", "-"), CYAN)
        row("Inference Time:",  "{:.1f} ms".format(r["elapsed"] * 1000), DIM)
        print("\n  " + BOLD + "Sub-checks:" + RESET)
        for name, (ok_sub, note) in r["subs"].items():
            icon   = GREEN + "v" + RESET if ok_sub else RED + "x" + RESET
            colour = GREEN if ok_sub else RED
            print("    {}  {:<20} {}{}{}".format(icon, name, colour, note, RESET))

    elapsed_total = time.time() - t_start
    banner("SUITE SUMMARY  --  {}/{} PASSED  |  {} FAILED  |  {:.2f}s".format(
        passed, len(TEST_CASES), failed, elapsed_total))

    print("\n  {:<4} {:<38} {:>28} {:>7}  {}".format("ID", "Test Name", "Intent", "Conf", "Pass"))
    print("  " + "-" * 82)
    for r in results:
        tc   = r["tc"]
        icon = GREEN + "PASS" + RESET if r["pass"] else RED + "FAIL" + RESET
        print("  {:<4} {:<38} {:>28} {:>7}  {}".format(
            tc["id"], tc["name"], r["tag"], "{:.2%}".format(r["score"]), icon))
    print()

    if failed == 0:
        print(GREEN + BOLD + "  ALL {} TESTS PASSED".format(passed) + RESET + "\n")
        return 0
    print(RED + BOLD + "  {} TEST(S) FAILED".format(failed) + RESET + "\n")
    return 1


if __name__ == "__main__":
    print("\n" + BOLD + "Loading ClauseCheck audit engine..." + RESET)
    try:
        embedder, model, meta = load_engine()
        print(GREEN + "Engine loaded ({}, {} classes)".format(
            meta["embedding_model"], meta["num_classes"]) + RESET)
    except Exception as exc:
        print(RED + "Engine load failed: {}".format(exc) + RESET)
        sys.exit(2)
    sys.exit(run_suite(embedder, model, meta))

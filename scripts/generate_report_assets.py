import os
import matplotlib.pyplot as plt
import numpy as np

# Set publication styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.dpi'] = 300

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=300)
    ax.set_facecolor("#F8FAFC")
    ax.axis("off")

    stages = [
        {"title": "1. Voice / Audio Ingestion", "sub": "Microphone Audio Stream\nPCM 16-bit Transcoder\nor Fallback Text", "color": "#1E3A8A", "box": (0.02, 0.45, 0.17, 0.46)},
        {"title": "2. Speech Recognition", "sub": "Speech-to-Text API\nError Handling (Silence,\nNoise, Chunking)", "color": "#0284C7", "box": (0.22, 0.45, 0.17, 0.46)},
        {"title": "3. Transformer Encoder", "sub": "all-MiniLM-L6-v2\n384-dim Dense Semantic\nEmbedding Tensor", "color": "#0D9488", "box": (0.42, 0.45, 0.17, 0.46)},
        {"title": "4. Deep PyTorch MLP", "sub": "384 -> 64 -> 32 -> 6\nBatchNorm1d + Dropout (0.3)\nSoftmax Multi-Intent Probs", "color": "#7C3AED", "box": (0.62, 0.45, 0.17, 0.46)},
        {"title": "5. Compliance & TTS", "sub": "RBI Regulatory Card Engine\nAbstention Guard (<0.40)\ngTTS Voice Audio Verdict", "color": "#DC2626", "box": (0.82, 0.45, 0.17, 0.46)},
    ]

    for stage in stages:
        x, y, w, h = stage["box"]
        # Background box
        rect = plt.Rectangle((x, y), w, h, transform=ax.transAxes,
                             facecolor=stage["color"], alpha=0.08,
                             edgecolor=stage["color"], linewidth=2,
                             linestyle='-', zorder=2)
        ax.add_patch(rect)
        # Header banner
        header_rect = plt.Rectangle((x, y + h - 0.13), w, 0.13, transform=ax.transAxes,
                                    facecolor=stage["color"], alpha=0.92,
                                    edgecolor=stage["color"], linewidth=1, zorder=3)
        ax.add_patch(header_rect)
        # Header text
        ax.text(x + w / 2, y + h - 0.065, stage["title"], transform=ax.transAxes,
                ha='center', va='center', color='white', weight='bold', fontsize=8.5, zorder=4)
        # Body text
        ax.text(x + w / 2, y + (h - 0.13) / 2, stage["sub"], transform=ax.transAxes,
                ha='center', va='center', color='#1E293B', fontsize=8, linespacing=1.35, zorder=4)

    # Draw arrows between stages
    for i in range(len(stages) - 1):
        x1 = stages[i]["box"][0] + stages[i]["box"][2]
        x2 = stages[i + 1]["box"][0]
        y_center = stages[i]["box"][1] + stages[i]["box"][3] / 2
        ax.annotate('', xy=(x2, y_center), xytext=(x1, y_center),
                    xycoords='axes fraction', textcoords='axes fraction',
                    arrowprops=dict(arrowstyle="-|>", color="#475569", lw=2.2, mutation_scale=15),
                    zorder=5)

    # Sub-caption / Regulatory anchor footnote
    ax.text(0.5, 0.22,
            "Full Duplex Voice Pipeline: Acoustic Ingestion → SBERT Semantic Vectorization → Regularized PyTorch Classifier → Statutory RBI Card & Voice Feedback",
            transform=ax.transAxes, ha='center', va='center', color='#334155', fontsize=8.5, weight='bold')

    ax.text(0.5, 0.10,
            "Regulatory Anchors: RBI Digital Lending Directions (2022/2024), Fair Practices Code, Master Directions on Credit Cards, and DPDP Act Data Minimization",
            transform=ax.transAxes, ha='center', va='center', color='#64748B', fontsize=7.5, style='italic')

    plt.tight_layout()
    out_path = os.path.join(REPORTS_DIR, "architecture_diagram.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

def generate_class_distribution_chart():
    classes = [
        "apr_hidden_fees",
        "coercive_device_permissions",
        "recovery_agent_harassment",
        "cooling_off_cancellation",
        "credit_card_unilateral_terms",
        "out_of_scope"
    ]
    labels = [
        "APR & Hidden Fees\n(KFS Directives)",
        "Coercive Device Perms\n(DPDP / Storage Bans)",
        "Recovery Harassment\n(Fair Practices Code)",
        "Cooling-Off Exit\n(Look-Up Window)",
        "Credit Card Violations\n(Master Direction)",
        "Out of Scope\n(Negative Controls)"
    ]
    counts = [65, 63, 62, 62, 61, 57]
    colors_list = ["#1E3A8A", "#0D9488", "#DC2626", "#D97706", "#7C3AED", "#64748B"]

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    bars = ax.bar(labels, counts, color=colors_list, width=0.55, edgecolor="#1E293B", linewidth=1.2, alpha=0.9)

    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 1.2, f"{yval} samples",
                ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1E293B')

    ax.set_ylim(0, 78)
    ax.set_ylabel("Number of Augmented Samples", fontsize=10, fontweight='bold')
    ax.set_title("Dataset Composition Across Intent Classes (Total: 370 Samples)", fontsize=12, pad=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    plt.xticks(fontsize=8.5)
    plt.tight_layout()
    out_path = os.path.join(REPORTS_DIR, "class_distribution_chart.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

def generate_stress_test_chart():
    categories = [
        "Standard In-Domain",
        "Colloquial & Slang",
        "ASR Noise & Typos",
        "Formal Contract Excerpt",
        "Multi-Violation Hybrid",
        "Minimal Keywords",
        "Out of Scope / Noise"
    ]
    latencies = [40.1, 22.5, 24.3, 28.2, 22.4, 15.1, 39.8]
    pass_rates = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]

    x = np.arange(len(categories))
    width = 0.55

    fig, ax1 = plt.subplots(figsize=(10, 4.6), dpi=300)

    bars = ax1.bar(x, latencies, width, color="#0284C7", edgecolor="#0369A1", linewidth=1.2, alpha=0.85)
    ax1.set_ylabel("Average CPU Latency (ms)", color="#0369A1", fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 52)
    ax1.tick_params(axis='y', labelcolor="#0369A1")

    for bar in bars:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1.2, f"{h:.1f} ms",
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold', color="#0369A1")

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=22, ha='right', fontsize=8.5)
    ax1.set_title("Multi-Scenario Stress Test Performance & CPU Latency (28 Cases, 100% Pass)", fontsize=12, pad=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(REPORTS_DIR, "stress_test_latency_chart.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated: {out_path}")

if __name__ == "__main__":
    generate_architecture_diagram()
    generate_class_distribution_chart()
    generate_stress_test_chart()

"""Generate results figures for paper and README.

Produces:
  figures/fig1_coverage.png      — coverage (% answered) per version × query type
  figures/fig2_ragas_overall.png — 4 RAGAS metrics per version (overall)
  figures/fig3_ragas_v4.png      — v4 per-query-type RAGAS breakdown
  figures/fig4_loop_efficiency.png — loop count distribution + router accuracy (v4)
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

VERSIONS   = ["v1", "v2", "v3", "v4"]
V_LABELS   = ["v1\nNaive RAG", "v2\nStatic routing", "v3\nLoop, no rewrite", "v4\nFull system"]
QTYPES     = ["factual", "relational", "thematic"]
METRICS    = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
M_LABELS   = ["Faithfulness", "Answer Relevancy", "Context Precision", "Context Recall"]

COLORS = {
    "v1": "#4C72B0",
    "v2": "#DD8452",
    "v3": "#55A868",
    "v4": "#C44E52",
}
QTYPE_COLORS = {
    "factual":    "#4C72B0",
    "relational": "#DD8452",
    "thematic":   "#55A868",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


def avg(lst): return sum(lst) / len(lst) if lst else 0.0


def load_all() -> dict:
    data = {}
    for v in VERSIONS:
        raw    = [json.loads(l) for l in open(f"data/eval/eval_results_{v}.jsonl", encoding="utf-8")]
        scores = [json.loads(l) for l in open(f"data/eval/ragas_scores_{v}.jsonl", encoding="utf-8")]
        answered = [r for r in raw if not r["refused"]]
        by_type_scores: dict[str, list] = defaultdict(list)
        by_type_raw:    dict[str, list] = defaultdict(list)
        for s in scores:
            by_type_scores[s["query_type"]].append(s)
        for r in raw:
            by_type_raw[r["query_type"]].append(r)
        data[v] = {
            "raw": raw, "scores": scores,
            "answered": answered,
            "by_type_scores": by_type_scores,
            "by_type_raw":    by_type_raw,
        }
    return data


# ── Fig 1: Coverage ────────────────────────────────────────────────────────
def plot_coverage(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: overall coverage bar
    ax = axes[0]
    coverages = [len(data[v]["answered"]) / len(data[v]["raw"]) * 100 for v in VERSIONS]
    bars = ax.bar(V_LABELS, coverages, color=[COLORS[v] for v in VERSIONS], width=0.5)
    for bar, val in zip(bars, coverages):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Queries answered (%)")
    ax.set_title("Overall Coverage", fontweight="bold")
    ax.axhline(y=80, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    # Right: coverage by query type (v4 only)
    ax2 = axes[1]
    v4_data = data["v4"]
    qtype_cov = []
    for qt in QTYPES:
        total   = len(v4_data["by_type_raw"][qt])
        ans     = sum(1 for r in v4_data["by_type_raw"][qt] if not r["refused"])
        qtype_cov.append(ans / total * 100 if total else 0)

    bars2 = ax2.bar(QTYPES, qtype_cov, color=[QTYPE_COLORS[q] for q in QTYPES], width=0.4)
    for bar, val in zip(bars2, qtype_cov):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 110)
    ax2.set_ylabel("Queries answered (%)")
    ax2.set_title("v4 Coverage by Query Type", fontweight="bold")

    fig.suptitle("Query Coverage Across Ablation Versions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIGURES_DIR / "fig1_coverage.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Fig 2: Overall RAGAS across versions ──────────────────────────────────
def plot_ragas_overall(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    x     = np.arange(len(METRICS))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for i, v in enumerate(VERSIONS):
        vals = [avg([s[m] for s in data[v]["scores"]]) for m in METRICS]
        bars = ax.bar(x + offsets[i] * width, vals, width,
                      label=V_LABELS[i].replace("\n", " "), color=COLORS[v])
        for bar, val in zip(bars, vals):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(M_LABELS)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("RAGAS Metrics by Ablation Version\n(answered queries only)", fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    plt.tight_layout()
    out = FIGURES_DIR / "fig2_ragas_overall.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Fig 3: v4 per-query-type RAGAS ────────────────────────────────────────
def plot_ragas_v4_by_type(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    x     = np.arange(len(METRICS))
    width = 0.22
    offsets = [-1, 0, 1]

    for i, qt in enumerate(QTYPES):
        group = data["v4"]["by_type_scores"][qt]
        vals  = [avg([s[m] for s in group]) for m in METRICS]
        n     = len(group)
        bars  = ax.bar(x + offsets[i] * width, vals, width,
                       label=f"{qt} (n={n})", color=QTYPE_COLORS[qt])
        for bar, val in zip(bars, vals):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(M_LABELS)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("v4 Full System — RAGAS Scores by Query Type", fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.axhline(y=0.8, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    plt.tight_layout()
    out = FIGURES_DIR / "fig3_ragas_v4.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── Fig 4: Loop efficiency (v4) ────────────────────────────────────────────
def plot_loop_efficiency(data: dict) -> None:
    loop_data = json.load(open("data/eval/loop_metrics.json", encoding="utf-8"))
    le = loop_data["loop_efficiency"]
    ra = loop_data["router_accuracy"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: loop count distribution per query type
    ax = axes[0]
    max_loops = 3
    bar_width = 0.22
    offsets   = [-1, 0, 1]
    x = np.arange(max_loops + 1)

    for i, qt in enumerate(QTYPES):
        dist = le.get(qt, {}).get("loop_dist", {})
        vals = [dist.get(str(lc), dist.get(lc, 0)) for lc in range(max_loops + 1)]
        total = sum(vals)
        pcts  = [v / total * 100 if total else 0 for v in vals]
        ax.bar(x + offsets[i] * bar_width, pcts, bar_width,
               label=qt, color=QTYPE_COLORS[qt])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{i} loop{'s' if i != 1 else ''}" for i in range(max_loops + 1)])
    ax.set_ylabel("% of answered queries")
    ax.set_title("Loop Count Distribution (v4)", fontweight="bold")
    ax.legend(fontsize=9)

    # Right: router accuracy per query type
    ax2 = axes[1]
    acc_vals = [ra.get(qt, {}).get("accuracy_pct", 0) for qt in QTYPES]
    bars = ax2.bar(QTYPES, acc_vals, color=[QTYPE_COLORS[q] for q in QTYPES], width=0.4)
    for bar, val in zip(bars, acc_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Router Accuracy by Query Type (v4)", fontweight="bold")
    ax2.axhline(y=ra["overall"]["accuracy_pct"], color="gray",
                linestyle="--", linewidth=0.8, alpha=0.7,
                label=f"Overall {ra['overall']['accuracy_pct']:.0f}%")
    ax2.legend(fontsize=9)

    fig.suptitle("Loop Efficiency & Router Accuracy — v4 Full System",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "fig4_loop_efficiency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    data = load_all()
    plot_coverage(data)
    plot_ragas_overall(data)
    plot_ragas_v4_by_type(data)
    plot_loop_efficiency(data)
    print("\nAll figures saved to figures/")

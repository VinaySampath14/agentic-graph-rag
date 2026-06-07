"""Compute loop-efficiency and router accuracy metrics from eval results.

Metrics computed (v4 full system only, since loop metrics only meaningful there):
  Loop efficiency:
    - avg loop_count per query type
    - % first_mode_success per query type
    - % rewrite_triggered per query type
    - mode distribution (which modes were used as final answer mode)

  Router accuracy:
    - % queries where first_mode == expected_mode (per query type)
    - Confusion matrix: expected_mode vs first_mode

Output: data/eval/loop_metrics.json + printed summary
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_FILE  = Path("data/eval/eval_results_v4.jsonl")
QUERIES_FILE  = Path("data/eval/eval_queries_validated.jsonl")
OUTPUT_FILE   = Path("data/eval/loop_metrics.json")


def avg(lst: list) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def pct(num: int, den: int) -> float:
    return round(100 * num / den, 1) if den else 0.0


def main() -> None:
    # Load results
    records = [json.loads(l) for l in open(RESULTS_FILE, encoding="utf-8")]

    # Load expected modes
    expected = {}
    for line in open(QUERIES_FILE, encoding="utf-8"):
        q = json.loads(line)
        expected[q["id"]] = q.get("expected_mode", "")

    # Fix first_mode: mode_history is empty for single-pass successes,
    # so derive first_mode from the first retriever node in agent_trace instead.
    RETRIEVER_NODES = {
        "naive_retriever":       "vector",
        "local_graph_retriever": "graph",
        "global_retriever":      "community",
        "web_retriever":         "web",
    }
    for r in records:
        if not r.get("first_mode"):
            for entry in r.get("agent_trace", []):
                mapped = RETRIEVER_NODES.get(entry.get("node", ""))
                if mapped:
                    r["first_mode"] = mapped
                    break

    answered = [r for r in records if not r["refused"]]
    refused  = [r for r in records if r["refused"]]

    qtypes = ["factual", "relational", "thematic"]

    # ── Loop efficiency ──────────────────────────────────────────────────
    print("=" * 60)
    print("LOOP EFFICIENCY (v4 — answered queries only)")
    print("=" * 60)

    loop_metrics: dict = {}

    for qtype in qtypes:
        group = [r for r in answered if r["query_type"] == qtype]
        if not group:
            continue

        loop_counts       = [r["loop_count"] for r in group]
        first_success     = [r for r in group if r["first_mode_success"]]
        rewrites          = [r for r in group if r["rewrite_triggered"]]
        mode_dist         = Counter(r["final_mode"] for r in group)

        loop_metrics[qtype] = {
            "n_answered":          len(group),
            "avg_loop_count":      round(avg(loop_counts), 3),
            "pct_first_mode_success": pct(len(first_success), len(group)),
            "pct_rewrite_triggered":  pct(len(rewrites), len(group)),
            "loop_dist":           dict(Counter(loop_counts)),
            "final_mode_dist":     dict(mode_dist),
        }

        print(f"\n{qtype.upper()} (n={len(group)})")
        print(f"  avg loops:            {avg(loop_counts):.2f}")
        print(f"  first-mode success:   {pct(len(first_success), len(group))}%")
        print(f"  rewrite triggered:    {pct(len(rewrites), len(group))}%")
        print(f"  loop distribution:    {dict(sorted(Counter(loop_counts).items()))}")
        print(f"  final mode dist:      {dict(mode_dist.most_common())}")

    # Overall
    loop_counts_all   = [r["loop_count"] for r in answered]
    first_success_all = [r for r in answered if r["first_mode_success"]]
    rewrites_all      = [r for r in answered if r["rewrite_triggered"]]

    print(f"\nOVERALL (n={len(answered)})")
    print(f"  avg loops:            {avg(loop_counts_all):.2f}")
    print(f"  first-mode success:   {pct(len(first_success_all), len(answered))}%")
    print(f"  rewrite triggered:    {pct(len(rewrites_all), len(answered))}%")
    print(f"  loop distribution:    {dict(sorted(Counter(loop_counts_all).items()))}")

    loop_metrics["overall"] = {
        "n_answered":             len(answered),
        "n_refused":              len(refused),
        "avg_loop_count":         round(avg(loop_counts_all), 3),
        "pct_first_mode_success": pct(len(first_success_all), len(answered)),
        "pct_rewrite_triggered":  pct(len(rewrites_all), len(answered)),
        "loop_dist":              dict(Counter(loop_counts_all)),
    }

    # ── Router accuracy ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ROUTER ACCURACY (first mode vs expected mode — all 80 queries)")
    print("=" * 60)

    router_metrics: dict = {}
    confusion: dict[str, Counter] = defaultdict(Counter)

    for r in records:
        exp  = expected.get(r["id"], "")
        got  = r.get("first_mode", "")
        if exp and got:
            confusion[exp][got] += 1

    total_with_exp = sum(sum(c.values()) for c in confusion.values())
    total_correct  = sum(confusion[m][m] for m in confusion)

    print(f"\nOverall router accuracy: {total_correct}/{total_with_exp} "
          f"({pct(total_correct, total_with_exp)}%)")

    print("\nConfusion matrix (rows=expected, cols=actual first mode):")
    modes = ["vector", "graph", "community", "web", ""]
    header_modes = [m for m in modes if any(confusion[e][m] for e in confusion)]
    print(f"  {'expected':>12}  " + "  ".join(f"{m:>10}" for m in header_modes))
    for exp_mode in sorted(confusion.keys()):
        row = "  ".join(f"{confusion[exp_mode][m]:>10}" for m in header_modes)
        n   = sum(confusion[exp_mode].values())
        cor = confusion[exp_mode][exp_mode]
        print(f"  {exp_mode:>12}  {row}   ({cor}/{n} correct)")

    # Per query type router accuracy
    print("\nPer query type:")
    for qtype in qtypes:
        group = [r for r in records if r["query_type"] == qtype]
        correct = sum(
            1 for r in group
            if r.get("first_mode", "") == expected.get(r["id"], "X")
        )
        print(f"  {qtype:<12} {correct}/{len(group)} ({pct(correct, len(group))}%)")

        router_metrics[qtype] = {
            "n": len(group),
            "correct": correct,
            "accuracy_pct": pct(correct, len(group)),
        }

    router_metrics["overall"] = {
        "n": total_with_exp,
        "correct": total_correct,
        "accuracy_pct": pct(total_correct, total_with_exp),
    }

    # ── Save ─────────────────────────────────────────────────────────────
    output = {
        "loop_efficiency": loop_metrics,
        "router_accuracy":  router_metrics,
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

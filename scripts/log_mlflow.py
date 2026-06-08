"""Log all eval results to MLflow.

Structure:
  Experiment: agentic-graph-rag-eval
  One parent run per ablation version (v1-v4)
    Tags:    version, data_version, eval_date
    Metrics: overall RAGAS scores, refusal rate, coverage
    Metrics: per query-type RAGAS scores
    Metrics: loop efficiency + router accuracy (v4 only)
    Artifacts: ragas_scores_{v}.jsonl, eval_results_{v}.jsonl

Usage:
    python scripts/log_mlflow.py
"""
import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import mlflow
from dotenv import load_dotenv

load_dotenv()

EXPERIMENT_NAME = "agentic-graph-rag-eval"
DATA_VERSION    = "v2"   # matches pipeline rebuild tag
EVAL_DATE       = str(date.today())

VERSIONS = {
    "v1": "Naive RAG — vector only, no routing, no loop",
    "v2": "Static routing — correct dispatch, single pass, no loop",
    "v3": "Agentic loop — loop + web fallback, no query rewrite",
    "v4": "Full system — loop + query rewrite + web fallback",
}

METRICS  = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
QTYPES   = ["factual", "relational", "thematic"]


def avg(lst: list) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def load_version_data(version: str) -> tuple[list[dict], list[dict]]:
    raw    = [json.loads(l) for l in
              open(f"data/eval/eval_results_{version}.jsonl", encoding="utf-8")]
    scores = [json.loads(l) for l in
              open(f"data/eval/ragas_scores_{version}.jsonl", encoding="utf-8")]
    return raw, scores


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)

    loop_data = json.load(open("data/eval/loop_metrics.json", encoding="utf-8"))

    for version, description in VERSIONS.items():
        print(f"\nLogging {version}: {description}")

        raw, scores = load_version_data(version)
        answered = [r for r in raw if not r["refused"]]
        refused  = [r for r in raw if r["refused"]]

        by_type: dict[str, list[dict]] = defaultdict(list)
        for s in scores:
            by_type[s["query_type"]].append(s)

        with mlflow.start_run(run_name=version) as run:
            # ── Tags ──────────────────────────────────────────────────
            mlflow.set_tags({
                "version":      version,
                "description":  description,
                "data_version": DATA_VERSION,
                "eval_date":    EVAL_DATE,
                "judge_model":  "gpt-4o-mini",
                "agent_model":  "llama-3.1-8b-instant",
            })

            # ── Coverage metrics ──────────────────────────────────────
            mlflow.log_metrics({
                "coverage_total":      round(len(answered) / len(raw), 4),
                "refusal_rate_total":  round(len(refused)  / len(raw), 4),
                "n_answered":          len(answered),
                "n_refused":           len(refused),
            })

            for qtype in QTYPES:
                qt_all      = [r for r in raw      if r["query_type"] == qtype]
                qt_answered = [r for r in answered if r["query_type"] == qtype]
                if qt_all:
                    mlflow.log_metrics({
                        f"coverage_{qtype}":     round(len(qt_answered) / len(qt_all), 4),
                        f"refusal_rate_{qtype}": round((len(qt_all) - len(qt_answered)) / len(qt_all), 4),
                    })

            # ── Overall RAGAS ─────────────────────────────────────────
            for metric in METRICS:
                vals = [s[metric] for s in scores]
                mlflow.log_metric(f"ragas_{metric}", round(avg(vals), 4))

            # ── Per query-type RAGAS ──────────────────────────────────
            for qtype in QTYPES:
                group = by_type.get(qtype, [])
                for metric in METRICS:
                    vals = [s[metric] for s in group]
                    mlflow.log_metric(f"ragas_{metric}_{qtype}", round(avg(vals), 4))

            # ── Loop + router metrics (v4 only) ───────────────────────
            if version == "v4":
                le = loop_data["loop_efficiency"]
                ra = loop_data["router_accuracy"]

                mlflow.log_metrics({
                    "loop_avg_count":          le["overall"]["avg_loop_count"],
                    "loop_first_mode_success": le["overall"]["pct_first_mode_success"],
                    "loop_rewrite_rate":       le["overall"]["pct_rewrite_triggered"],
                    "router_accuracy_overall": ra["overall"]["accuracy_pct"],
                    "router_accuracy_factual":    ra.get("factual",    {}).get("accuracy_pct", 0),
                    "router_accuracy_relational": ra.get("relational", {}).get("accuracy_pct", 0),
                    "router_accuracy_thematic":   ra.get("thematic",   {}).get("accuracy_pct", 0),
                })

                for qtype in QTYPES:
                    if qtype in le:
                        mlflow.log_metrics({
                            f"loop_avg_count_{qtype}":    le[qtype]["avg_loop_count"],
                            f"loop_first_success_{qtype}":le[qtype]["pct_first_mode_success"],
                            f"loop_rewrite_rate_{qtype}": le[qtype]["pct_rewrite_triggered"],
                        })

            # ── Artifacts ─────────────────────────────────────────────
            mlflow.log_artifact(f"data/eval/eval_results_{version}.jsonl",  "eval_results")
            mlflow.log_artifact(f"data/eval/ragas_scores_{version}.jsonl",  "ragas_scores")
            if version == "v4":
                mlflow.log_artifact("data/eval/loop_metrics.json", "loop_metrics")

            print(f"  Run ID: {run.info.run_id}")
            print(f"  Coverage: {len(answered)}/{len(raw)} answered")
            overall_f = avg([s["faithfulness"] for s in scores])
            print(f"  Faithfulness: {overall_f:.3f}")

    print(f"\nDone. View results: mlflow ui")


if __name__ == "__main__":
    main()

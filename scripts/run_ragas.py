"""Score eval results with 4 RAGAS-equivalent metrics using GPT-4o-mini as judge.

Metrics (all 0-1):
  faithfulness      — are all answer claims supported by the retrieved context?
  answer_relevancy  — does the answer address the question?
  context_precision — what fraction of the retrieved context is relevant to the question?
  context_recall    — does the retrieved context cover the key points in the reference answer?

Refused queries are excluded from scoring (Option B).
Results saved to data/eval/ragas_scores.jsonl + summary printed to stdout.

Usage:
    python scripts/run_ragas.py
"""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

RAW_FILE    = Path("data/eval/eval_results_raw.jsonl")
OUTPUT_FILE = Path("data/eval/ragas_scores.jsonl")
MODEL       = "gpt-4o-mini"
SLEEP       = 1.0   # seconds between API calls


# ── Prompts ────────────────────────────────────────────────────────────────

FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI-generated answer is faithful to the retrieved context.

Question: {question}

Retrieved context:
{context}

Answer to evaluate:
{answer}

Task: Identify every factual claim in the answer. For each claim, check whether it is
directly supported by the retrieved context (not by general knowledge).

Return ONLY a JSON object with this exact format:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}

Where score = (number of supported claims) / (total claims).
If the answer is empty or makes no claims, return 0.0.
"""

ANSWER_RELEVANCY_PROMPT = """\
You are evaluating whether an AI-generated answer is relevant to the question asked.

Question: {question}

Answer to evaluate:
{answer}

Task: Does the answer directly address the question? Penalise answers that are vague,
off-topic, or address a different question entirely.

Return ONLY a JSON object with this exact format:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}

Where 1.0 = fully addresses the question, 0.0 = completely irrelevant or empty.
"""

CONTEXT_PRECISION_PROMPT = """\
You are evaluating the precision of retrieved context for answering a question.

Question: {question}

Retrieved context:
{context}

Task: What fraction of the retrieved context is actually useful for answering this question?
Penalise context that is mostly irrelevant filler even if the answer happens to be correct.

Return ONLY a JSON object with this exact format:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}

Where 1.0 = all context is relevant, 0.0 = no context is relevant.
"""

CONTEXT_RECALL_PROMPT = """\
You are evaluating whether retrieved context covers the key information needed to answer a question.

Question: {question}

Reference answer (ground truth):
{reference}

Retrieved context:
{context}

Task: Identify the key claims or facts in the reference answer. For each, check whether
the retrieved context contains enough information to support that claim.

Return ONLY a JSON object with this exact format:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}

Where score = (claims supported by context) / (total claims in reference).
"""


# ── Scorer ─────────────────────────────────────────────────────────────────

def score_one(client: OpenAI, prompt: str) -> tuple[float, str]:
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content)
        score = float(data.get("score", 0.0))
        reason = data.get("reason", "")
        return max(0.0, min(1.0, score)), reason
    except Exception as e:
        return 0.0, f"ERROR: {e}"


def score_record(client: OpenAI, rec: dict) -> dict:
    q       = rec["query"]
    answer  = rec["answer"]
    ref     = rec["reference_answer"]

    context = rec.get("context_text", "") or answer

    faith_score,  faith_reason  = score_one(client, FAITHFULNESS_PROMPT.format(
        question=q, context=context, answer=answer))
    time.sleep(SLEEP)

    rel_score,    rel_reason    = score_one(client, ANSWER_RELEVANCY_PROMPT.format(
        question=q, answer=answer))
    time.sleep(SLEEP)

    prec_score,   prec_reason   = score_one(client, CONTEXT_PRECISION_PROMPT.format(
        question=q, context=context))
    time.sleep(SLEEP)

    recall_score, recall_reason = score_one(client, CONTEXT_RECALL_PROMPT.format(
        question=q, reference=ref, context=context))
    time.sleep(SLEEP)

    return {
        "id":                rec["id"],
        "query":             q,
        "query_type":        rec["query_type"],
        "final_mode":        rec["final_mode"],
        "loop_count":        rec["loop_count"],
        "faithfulness":      round(faith_score,  4),
        "answer_relevancy":  round(rel_score,    4),
        "context_precision": round(prec_score,   4),
        "context_recall":    round(recall_score, 4),
        "reasons": {
            "faithfulness":      faith_reason,
            "answer_relevancy":  rel_reason,
            "context_precision": prec_reason,
            "context_recall":    recall_reason,
        },
    }


# ── Main ───────────────────────────────────────────────────────────────────

def load_records() -> list[dict]:
    records = []
    with open(RAW_FILE, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_scored_ids() -> set[int]:
    if not OUTPUT_FILE.exists():
        return set()
    done = set()
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            done.add(json.loads(line)["id"])
    return done


def print_summary(scores: list[dict]) -> None:
    from collections import defaultdict

    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    def avg(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    print("\n" + "="*60)
    print("RAGAS SCORES SUMMARY (refused queries excluded)")
    print("="*60)

    # Overall
    print("\nOVERALL")
    for m in metrics:
        vals = [s[m] for s in scores]
        print(f"  {m:<22} {avg(vals):.4f}  (n={len(vals)})")

    # Per query type
    by_type: dict[str, list[dict]] = defaultdict(list)
    for s in scores:
        by_type[s["query_type"]].append(s)

    for qtype in ["factual", "relational", "thematic"]:
        group = by_type.get(qtype, [])
        print(f"\n{qtype.upper()} (n={len(group)})")
        for m in metrics:
            vals = [s[m] for s in group]
            print(f"  {m:<22} {avg(vals):.4f}")

    print("="*60)


def main() -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI()

    records    = load_records()
    answered   = [r for r in records if not r["refused"]]
    refused    = [r for r in records if r["refused"]]
    scored_ids = load_scored_ids()
    pending    = [r for r in answered if r["id"] not in scored_ids]

    print(f"Total: {len(records)} | Answered: {len(answered)} | Refused: {len(refused)}")
    print(f"Already scored: {len(scored_ids)} | Remaining: {len(pending)}")

    if not pending:
        print("All answered queries already scored.")
    else:
        for i, rec in enumerate(pending):
            print(f"[{len(scored_ids)+i+1}/{len(answered)}] id={rec['id']} ({rec['query_type']}): {rec['query'][:60]}")
            scored = score_record(client, rec)
            print(f"  F={scored['faithfulness']:.3f} R={scored['answer_relevancy']:.3f} "
                  f"CP={scored['context_precision']:.3f} CR={scored['context_recall']:.3f}")
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(scored, ensure_ascii=False) + "\n")

    # Load all scores and print summary
    all_scores = []
    with open(OUTPUT_FILE, encoding="utf-8") as f:
        for line in f:
            all_scores.append(json.loads(line))

    print_summary(all_scores)

    # Refusal rate report
    print("\nREFUSAL RATE")
    from collections import Counter
    ref_by_type = Counter(r["query_type"] for r in refused)
    total_by_type = Counter(r["query_type"] for r in records)
    for qtype in ["factual", "relational", "thematic"]:
        n_ref = ref_by_type.get(qtype, 0)
        n_tot = total_by_type.get(qtype, 0)
        print(f"  {qtype:<12} {n_ref}/{n_tot} refused ({100*n_ref/n_tot:.0f}%)")
    print(f"  {'total':<12} {len(refused)}/{len(records)} refused ({100*len(refused)/len(records):.0f}%)")


if __name__ == "__main__":
    main()

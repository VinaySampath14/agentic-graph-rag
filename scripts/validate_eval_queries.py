"""Phase 1 — validate the 80 generated eval queries with GPT-4o.

For each query, asks GPT-4o whether it is well-formed, answerable from the
corpus, and correctly typed (factual/relational/thematic). Flags any that fail
so the author can review/replace them before building reference answers."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

INPUT_FILE = Path("data/eval/eval_queries.jsonl")
OUTPUT_FILE = Path("data/eval/eval_queries_validated.jsonl")
MODEL = "gpt-4o"

VALIDATION_PROMPT = """You are reviewing an evaluation question for a RAG system over a corpus of ~2000 arXiv CS papers.

Question: "{query}"
Claimed type: {query_type} (expected to test {expected_mode}-mode retrieval)

Definitions:
- factual: about a SINGLE paper's content (what it proposes/solves/reports) — best served by vector search
- relational: about CONNECTIONS between entities (authors, methods, institutions, citations) — best served by graph traversal
- thematic: about HIGH-LEVEL TRENDS or overviews across many papers — best served by community summaries

Judge the question on three criteria:
1. well_formed: Is it a clear, grammatical, single question?
2. answerable: Could it plausibly be answered using a corpus of CS research paper abstracts/metadata (not requiring info outside the corpus)?
3. correctly_typed: Does it genuinely match its claimed type, i.e. would the claimed retrieval mode actually be the right way to answer it?

Respond with a JSON object: {{"well_formed": true/false, "answerable": true/false, "correctly_typed": true/false, "issue": "<short reason if any check failed, else empty string>"}}
"""


def get_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def validate_query(client: OpenAI, entry: dict) -> dict:
    prompt = VALIDATION_PROMPT.format(
        query=entry["query"],
        query_type=entry["query_type"],
        expected_mode=entry["expected_mode"],
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    verdict = json.loads(response.choices[0].message.content)
    passed = verdict.get("well_formed") and verdict.get("answerable") and verdict.get("correctly_typed")
    return {**entry, "validation": verdict, "validation_passed": bool(passed)}


def main() -> None:
    client = get_client()
    entries = [json.loads(l) for l in INPUT_FILE.read_text(encoding="utf-8").splitlines()]

    print(f"Validating {len(entries)} queries with {MODEL}...")
    results = []
    for i, entry in enumerate(entries):
        result = validate_query(client, entry)
        results.append(result)
        status = "OK" if result["validation_passed"] else f"FLAGGED ({result['validation'].get('issue', '')})"
        print(f"  [{i+1}/{len(entries)}] [{entry['query_type']}] {status}: {entry['query'][:70]}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_passed = sum(1 for r in results if r["validation_passed"])
    n_flagged = len(results) - n_passed
    print(f"\nDone. {n_passed}/{len(results)} passed, {n_flagged} flagged for review.")
    print(f"Saved to {OUTPUT_FILE}")

    if n_flagged:
        print("\nFlagged queries:")
        for r in results:
            if not r["validation_passed"]:
                print(f"  [id {r['id']}, {r['query_type']}] {r['query']}")
                print(f"    -> {r['validation']}")


if __name__ == "__main__":
    main()

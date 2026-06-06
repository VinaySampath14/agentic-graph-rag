"""Phase 2 — generate reference (ground-truth) answers for the 80 eval queries with GPT-4o.

For each query, fetches "gold" context directly from the source data (NOT via the
system's own retrieval, to keep references independent of the system under test):
  - factual    -> fuzzy-matched source paper's abstract
  - relational -> abstracts of papers connected to the named author/method via graph traversal
  - thematic   -> the relevant community theme + summary

Then asks GPT-4o for a concise, grounded 2-3 sentence reference answer."""
import difflib
import glob
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

load_dotenv()

INPUT_FILE = Path("data/eval/eval_queries_validated.jsonl")
OUTPUT_FILE = Path("data/eval/reference_answers.jsonl")
MODEL = "gpt-4o"
MAX_CONTEXT_PAPERS = 3

REFERENCE_PROMPT = """You are creating a reference (ground-truth) answer for evaluating a RAG system over arXiv CS papers.

Question: "{query}"

Gold context (the actual source material the answer must be grounded in):
{context}

Write a concise, factual, 2-3 sentence reference answer using ONLY the information in the gold context above.
Do not speculate or add information not present in the context.
Respond with the answer text only — no preamble, no quotes.
"""


def get_driver():
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )


def load_papers() -> list[dict]:
    papers = []
    for path in sorted(glob.glob("data/raw/papers_batch_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                papers.append(json.loads(line))
    return papers


def find_source_paper(query: str, papers: list[dict]) -> dict | None:
    """Fuzzy-match a factual query back to its source paper by title overlap."""
    titles = [p["title"] for p in papers]
    matches = difflib.get_close_matches(query, titles, n=1, cutoff=0.0)
    if not matches:
        return None
    # score by word overlap to avoid bad difflib matches on short titles
    best, best_score = None, 0.0
    query_words = set(re.findall(r"[A-Za-z0-9\-]+", query.lower()))
    for p in papers:
        title_words = set(re.findall(r"[A-Za-z0-9\-]+", p["title"].lower()))
        if not title_words:
            continue
        overlap = len(query_words & title_words) / len(title_words)
        if overlap > best_score:
            best, best_score = p, overlap
    return best if best_score > 0.3 else None


def fetch_factual_context(entry: dict, papers: list[dict]) -> str | None:
    paper = find_source_paper(entry["query"], papers)
    if not paper:
        return None
    return f"Title: {paper['title']}\nAbstract: {paper['abstract']}"


def extract_entity_names(query: str, candidates: list[str]) -> list[str]:
    found = []
    for name in candidates:
        if name.lower() in query.lower():
            found.append(name)
    return found


def _detect_target(query: str) -> str:
    """What entity type is the question actually asking about?"""
    q = query.lower()
    if "institution" in q:
        return "institution"
    if "author" in q or "who" in q or "researcher" in q:
        return "author"
    return "paper"


# Maps natural-language entity names to the actual node names stored in the graph
METHOD_NAME_ALIASES = {
    "Mixture-of-Experts": "MoE",
    "Tree of Thoughts": "ToT",
    "Chain-of-Thought": "CoT",
}


def _resolve_method_name(name: str, methods: list[str]) -> str:
    return METHOD_NAME_ALIASES.get(name, name) if METHOD_NAME_ALIASES.get(name, name) in methods else name


def fetch_relational_context(session, entry: dict, papers_by_id: dict, authors: list[str], methods: list[str]) -> str | None:
    query = entry["query"]
    matched_authors = extract_entity_names(query, authors)
    raw_matched_methods = extract_entity_names(query, methods + list(METHOD_NAME_ALIASES.keys()))
    matched_methods = [_resolve_method_name(m, methods) for m in raw_matched_methods]
    target = _detect_target(query)
    asks_both = bool(re.search(r"\bboth\b|\band\b.*\b(method|technique)s?\b", query.lower()))

    # "Both X and Y" pattern -> compute the real intersection, not single-entity context
    if asks_both and len(matched_methods) >= 2:
        m1, m2 = matched_methods[0], matched_methods[1]
        if target == "author":
            rows = session.run(
                "MATCH (a:Author)<-[:AUTHORED_BY]-(p1:Paper)-[:USES_METHOD]->(:Method {name: $m1}) "
                "MATCH (a)<-[:AUTHORED_BY]-(p2:Paper)-[:USES_METHOD]->(:Method {name: $m2}) "
                "RETURN DISTINCT a.name AS name LIMIT 8",
                m1=m1, m2=m2,
            ).data()
            names = [r["name"] for r in rows]
            if not names:
                return f"Entities: {m1}, {m2}\nNo authors found who have published using both {m1} and {m2}."
            return f"Entities: {m1}, {m2}\nAuthors who have published using both {m1} and {m2}: {names}"
        # target == paper
        rows = session.run(
            "MATCH (p:Paper)-[:USES_METHOD]->(:Method {name: $m1}) "
            "MATCH (p)-[:USES_METHOD]->(:Method {name: $m2}) "
            "RETURN DISTINCT p.arxiv_id AS arxiv_id, p.title AS title LIMIT $n",
            m1=m1, m2=m2, n=MAX_CONTEXT_PAPERS,
        ).data()
        if not rows:
            return f"Entities: {m1}, {m2}\nNo papers found that use both {m1} and {m2}."
        parts = [f"Entities: {m1}, {m2}"]
        for r in rows:
            paper = papers_by_id.get(r["arxiv_id"])
            parts.append(f"Title: {paper['title']}\nAbstract: {paper['abstract'][:400]}" if paper else f"Title: {r['title']}")
        return "\n\n".join(parts)

    # "Which papers did <author> write that use <method>?" -> author+method intersection
    if matched_authors and matched_methods and target == "paper":
        author, method = matched_authors[0], matched_methods[0]
        rows = session.run(
            "MATCH (a:Author {name: $author})<-[:AUTHORED_BY]-(p:Paper)-[:USES_METHOD]->(:Method {name: $method}) "
            "RETURN p.arxiv_id AS arxiv_id, p.title AS title LIMIT $n",
            author=author, method=method, n=MAX_CONTEXT_PAPERS,
        ).data()
        if not rows:
            return f"Entities: {author}, {method}\nNo papers found where {author} is an author and the paper uses {method}."
        parts = [f"Entities: {author}, {method}"]
        for r in rows:
            paper = papers_by_id.get(r["arxiv_id"])
            parts.append(f"Title: {paper['title']}\nAbstract: {paper['abstract'][:400]}" if paper else f"Title: {r['title']}")
        return "\n\n".join(parts)

    anchor_label, anchor_name, anchor_rel = None, None, None
    if matched_authors:
        anchor_label, anchor_name, anchor_rel = "Author", matched_authors[0], "AUTHORED_BY"
    elif matched_methods:
        anchor_label, anchor_name, anchor_rel = "Method", matched_methods[0], "USES_METHOD"

    if not anchor_name:
        return None

    if target == "institution":
        rows = session.run(
            f"MATCH (n:{anchor_label} {{name: $name}})<-[:{anchor_rel}]-(p:Paper)-[:AUTHORED_BY]->(a:Author)"
            f"-[:FROM_INSTITUTION]->(i:Institution) "
            "RETURN DISTINCT i.name AS name LIMIT 8",
            name=anchor_name,
        ).data()
        names = [r["name"] for r in rows]
        if not names:
            return None
        return f"Entity: {anchor_name}\nAssociated institutions (from co-authorship): {names}"

    if target == "author":
        rows = session.run(
            f"MATCH (n:{anchor_label} {{name: $name}})<-[:{anchor_rel}]-(p:Paper)-[:AUTHORED_BY]->(a:Author) "
            "RETURN DISTINCT a.name AS name LIMIT 8",
            name=anchor_name,
        ).data()
        names = [r["name"] for r in rows]
        if not names:
            return None
        return f"Entity: {anchor_name}\nAssociated authors: {names}"

    # target == "paper"
    rows = session.run(
        f"MATCH (n:{anchor_label} {{name: $name}})<-[:{anchor_rel}]-(p:Paper) "
        "RETURN p.arxiv_id AS arxiv_id, p.title AS title LIMIT $n",
        name=anchor_name, n=MAX_CONTEXT_PAPERS,
    ).data()
    if not rows:
        return None
    parts = [f"Entity: {anchor_name}"]
    for r in rows:
        paper = papers_by_id.get(r["arxiv_id"])
        if paper:
            parts.append(f"Title: {paper['title']}\nAbstract: {paper['abstract'][:500]}")
        else:
            parts.append(f"Title: {r['title']}")
    return "\n\n".join(parts)


def fetch_thematic_context(session) -> dict[int, str]:
    """Pre-fetch all community themes/summaries, keyed by community_id."""
    rows = session.run(
        "MATCH (c:Community) RETURN c.community_id AS id, c.theme AS theme, c.summary AS summary"
    ).data()
    contexts = {}
    for r in rows:
        summary = {}
        try:
            summary = json.loads(r["summary"] or "{}")
        except Exception:
            pass
        contexts[r["id"]] = (
            f"Community theme: {r['theme']}\n"
            f"Dominant methods: {summary.get('dominant_methods', [])}\n"
            f"Key authors: {summary.get('key_authors', [])}"
        )
    return contexts


def call_llm(client: OpenAI, query: str, context: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": REFERENCE_PROMPT.format(query=query, context=context)}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    driver = get_driver()
    entries = [json.loads(l) for l in INPUT_FILE.read_text(encoding="utf-8").splitlines()]

    print("Loading papers and grounding entities...")
    papers = load_papers()
    papers_by_id = {p["arxiv_id"]: p for p in papers}

    with driver.session() as session:
        authors = [
            r["name"] for r in session.run(
                "MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper) WITH a, count(p) AS c "
                "ORDER BY c DESC LIMIT 40 RETURN a.name AS name"
            )
        ]
        methods = [
            r["name"] for r in session.run(
                "MATCH (m:Method)<-[:USES_METHOD]-(p:Paper) WITH m, count(p) AS c "
                "ORDER BY c DESC LIMIT 60 RETURN m.name AS name"
            )
        ]
        community_contexts = fetch_thematic_context(session)

    all_theme_context = "\n\n".join(community_contexts.values())

    results = []
    skipped = []
    print(f"\nGenerating reference answers for {len(entries)} queries with {MODEL}...")
    with driver.session() as session:
        for i, entry in enumerate(entries):
            qtype = entry["query_type"]
            if qtype == "factual":
                context = fetch_factual_context(entry, papers)
            elif qtype == "relational":
                context = fetch_relational_context(session, entry, papers_by_id, authors, methods)
            else:
                context = all_theme_context

            if not context:
                skipped.append(entry)
                print(f"  [{i+1}/{len(entries)}] SKIP (no gold context found): {entry['query'][:70]}")
                continue

            answer = call_llm(client, entry["query"], context)
            results.append({
                "id": entry["id"],
                "query": entry["query"],
                "query_type": qtype,
                "expected_mode": entry["expected_mode"],
                "gold_context": context,
                "reference_answer": answer,
            })
            print(f"  [{i+1}/{len(entries)}] OK [{qtype}]: {entry['query'][:60]}")
            print(f"      -> {answer[:120]}")

    driver.close()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. {len(results)}/{len(entries)} reference answers generated.")
    if skipped:
        print(f"{len(skipped)} skipped (no gold context found):")
        for s in skipped:
            print(f"  [id {s['id']}, {s['query_type']}] {s['query']}")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

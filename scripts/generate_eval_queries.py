"""Generate 80 grounded eval queries (30 factual / 30 relational / 20 thematic) using an LLM,
seeded with real entities pulled from the Neo4j graph so questions are answerable from the corpus."""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase

load_dotenv()

OUTPUT_FILE = Path("data/eval/eval_queries.jsonl")
MODEL = "llama-3.1-8b-instant"

# (query_type, expected_mode, count)
PLAN = [
    ("factual", "vector", 30),
    ("relational", "graph", 30),
    ("thematic", "community", 20),
]

PROMPTS = {
    "factual": """You are creating evaluation questions for a RAG system over a corpus of {n_papers} arXiv CS papers.
Factual questions ask about a SPECIFIC paper's content: what it proposes, what problem it solves, what results it reports.
They should be answerable by reading one paper's abstract — NOT about authors, institutions, or trends.

Here are real paper titles from the corpus:
{titles}

Generate exactly {count} factual questions, each grounded in one of these papers (paraphrase the title into a natural question, e.g. "What does <PaperX> propose for <problem>?" or "How does <method> address <Y>?").
Return a JSON object: {{"queries": ["q1", "q2", ...]}}
""",
    "relational": """You are creating evaluation questions for a RAG system over a knowledge graph of arXiv CS papers, authors, institutions, and methods.
Relational questions ask about CONNECTIONS: who wrote what, who collaborates with whom, which papers use which methods, which institutions are involved.

Here are real entities from the graph:
Authors: {authors}
Methods: {methods}

Generate exactly {count} relational questions that combine these entities naturally, e.g. "Which papers did <author> write?", "Who are the authors working on <method>?", "Which authors have published on both <method1> and <method2>?", "What institutions are associated with <method> research?"
Return a JSON object: {{"queries": ["q1", "q2", ...]}}
""",
    "thematic": """You are creating evaluation questions for a RAG system that summarizes research communities/clusters of papers by theme.
Thematic questions ask about HIGH-LEVEL TRENDS, overviews, or comparisons across a research area — not single papers or specific authors.

Here are the real community themes detected in the corpus:
{themes}

Generate exactly {count} thematic questions such as "What are the main trends in <area>?", "Give an overview of research on <topic>", "How do approaches to <X> compare across recent work?"
Return a JSON object: {{"queries": ["q1", "q2", ...]}}
""",
}


def get_driver():
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )


def fetch_grounding(session) -> dict:
    titles = [r["t"] for r in session.run("MATCH (p:Paper) RETURN p.title AS t LIMIT 60")]
    authors = [
        r["name"]
        for r in session.run(
            "MATCH (a:Author)<-[:AUTHORED_BY]-(p:Paper) WITH a, count(p) AS c "
            "ORDER BY c DESC LIMIT 25 RETURN a.name AS name"
        )
    ]
    methods = [
        r["name"]
        for r in session.run(
            "MATCH (m:Method)<-[:USES_METHOD]-(p:Paper) WITH m, count(p) AS c "
            "ORDER BY c DESC LIMIT 25 RETURN m.name AS name"
        )
    ]
    themes = [r["theme"] for r in session.run("MATCH (c:Community) RETURN c.theme AS theme")]
    n_papers = session.run("MATCH (p:Paper) RETURN count(p) AS n").single()["n"]
    return {"titles": titles, "authors": authors, "methods": methods, "themes": themes, "n_papers": n_papers}


def call_llm(client: Groq, prompt: str) -> list[str]:
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            return [q.strip() for q in data.get("queries", []) if q.strip()]
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                print("  Rate limit — waiting 30s...")
                time.sleep(30)
            else:
                raise
    return []


def build_prompt(query_type: str, count: int, grounding: dict) -> str:
    if query_type == "factual":
        titles = "\n".join(f"- {t}" for t in grounding["titles"][:40])
        return PROMPTS["factual"].format(n_papers=grounding["n_papers"], titles=titles, count=count)
    if query_type == "relational":
        return PROMPTS["relational"].format(
            authors=", ".join(grounding["authors"]),
            methods=", ".join(grounding["methods"]),
            count=count,
        )
    return PROMPTS["thematic"].format(themes="\n".join(f"- {t}" for t in grounding["themes"]), count=count)


def main() -> None:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    driver = get_driver()

    with driver.session() as session:
        grounding = fetch_grounding(session)
    driver.close()

    all_queries = []
    qid = 1
    for query_type, expected_mode, count in PLAN:
        print(f"Generating {count} {query_type} queries...")
        prompt = build_prompt(query_type, count, grounding)
        queries = call_llm(client, prompt)
        print(f"  Got {len(queries)} queries")
        for q in queries[:count]:
            all_queries.append({
                "id": qid,
                "query": q,
                "query_type": query_type,
                "expected_mode": expected_mode,
            })
            qid += 1
        time.sleep(2)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in all_queries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_queries)} queries to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

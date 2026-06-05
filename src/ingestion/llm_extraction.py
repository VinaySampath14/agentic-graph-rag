"""LLM-based relation extraction on top 50 most-connected papers."""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase

load_dotenv()

PROMPT_FILE = Path("prompts/relation_extraction_v1.txt")
OUTPUT_FILE = Path("data/processed/llm_extractions.jsonl")
TOP_N = 50


def get_driver():
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )


def load_prompt() -> str:
    lines = PROMPT_FILE.read_text(encoding="utf-8").splitlines()
    return "\n".join(l for l in lines if not l.startswith("#")).strip()


def get_top_papers(session, n: int) -> list[dict]:
    result = session.run(f"""
        MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
        WITH p, count(a) AS author_count
        ORDER BY author_count DESC
        LIMIT {n}
        RETURN p.arxiv_id AS arxiv_id, p.title AS title
    """)
    return result.data()


def get_abstract(arxiv_id: str) -> str:
    raw_dir = Path("data/raw")
    for path in sorted(raw_dir.glob("papers_batch_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                paper = json.loads(line)
                if paper["arxiv_id"] == arxiv_id:
                    return paper["abstract"]
    return ""


def extract_relations(client: Groq, prompt_template: str, abstract: str) -> dict:
    prompt = prompt_template.replace("{abstract}", abstract[:2000])
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                print(f"  Rate limit — waiting 30s...")
                time.sleep(30)
            else:
                return {"methods": [], "datasets": [], "institutions": [], "cites": []}
    return {"methods": [], "datasets": [], "institutions": [], "cites": []}


def update_neo4j(session, arxiv_id: str, extraction: dict) -> None:
    for method in extraction.get("methods", []):
        if method and len(method) > 1:
            session.run("""
                MERGE (m:Method {name: $name})
                WITH m
                MATCH (p:Paper {arxiv_id: $arxiv_id})
                MERGE (p)-[:USES_METHOD]->(m)
            """, name=method, arxiv_id=arxiv_id)

    for inst in extraction.get("institutions", []):
        if inst and len(inst) > 2:
            session.run("""
                MERGE (i:Institution {name: $name})
                WITH i
                MATCH (p:Paper {arxiv_id: $arxiv_id})-[:AUTHORED_BY]->(a:Author)
                WITH i, a LIMIT 1
                MERGE (a)-[:FROM_INSTITUTION]->(i)
            """, name=inst, arxiv_id=arxiv_id)


def main() -> None:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    driver = get_driver()
    prompt_template = load_prompt()

    with driver.session() as session:
        top_papers = get_top_papers(session, TOP_N)

    print(f"Running LLM extraction on top {len(top_papers)} papers...")
    results = []

    with driver.session() as session:
        for i, paper in enumerate(top_papers):
            arxiv_id = paper["arxiv_id"]
            abstract = get_abstract(arxiv_id)
            if not abstract:
                continue

            extraction = extract_relations(client, prompt_template, abstract)
            update_neo4j(session, arxiv_id, extraction)
            results.append({"arxiv_id": arxiv_id, **extraction})

            print(f"  [{i+1}/{len(top_papers)}] {paper['title'][:60]}")
            print(f"    methods: {extraction.get('methods', [])[:3]}")
            time.sleep(2)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. Results saved to {OUTPUT_FILE}")
    driver.close()


if __name__ == "__main__":
    main()

"""Load papers, authors, institutions, methods and edges into Neo4j."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

PAPERS_DIR = Path("data/raw")
ENTITIES_FILE = Path("data/processed/entities_clean.jsonl")
DATA_VERSION = "v1"


def get_driver():
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )


def load_papers(limit: int | None = None) -> list[dict]:
    papers = []
    for path in sorted(PAPERS_DIR.glob("papers_batch_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                papers.append(json.loads(line))
                if limit and len(papers) >= limit:
                    return papers
    return papers


def load_entities() -> dict[str, dict]:
    entities = {}
    with open(ENTITIES_FILE, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            entities[e["arxiv_id"]] = e
    return entities


def derive_venue(paper: dict) -> str:
    journal_ref = paper.get("journal_ref", "")
    categories = paper.get("categories", [])
    if journal_ref:
        for venue in ["NeurIPS", "ICML", "ICLR", "ACL", "EMNLP", "NAACL",
                      "CVPR", "ICCV", "ECCV", "AAAI", "IJCAI"]:
            if venue.lower() in journal_ref.lower():
                return venue
    for cat in categories:
        if "cs.CL" in cat:
            return "NLP Venue"
        if "cs.CV" in cat:
            return "CV Venue"
    return "arXiv"


def create_indexes(session) -> None:
    session.run("CREATE FULLTEXT INDEX paperTitleIndex IF NOT EXISTS FOR (p:Paper) ON EACH [p.title]")
    session.run("CREATE FULLTEXT INDEX authorNameIndex IF NOT EXISTS FOR (a:Author) ON EACH [a.name]")
    session.run("CREATE FULLTEXT INDEX methodNameIndex IF NOT EXISTS FOR (m:Method) ON EACH [m.name]")
    session.run("CREATE INDEX paperYearIndex IF NOT EXISTS FOR (p:Paper) ON (p.year)")
    session.run("CREATE INDEX paperVenueIndex IF NOT EXISTS FOR (p:Paper) ON (p.venue)")
    print("Indexes created.")


def ingest_papers(papers: list[dict], entities: dict[str, dict], session) -> None:
    for i, paper in enumerate(papers):
        arxiv_id = paper["arxiv_id"]
        venue = derive_venue(paper)

        # Paper node
        session.run("""
            MERGE (p:Paper {arxiv_id: $arxiv_id})
            ON CREATE SET
                p.title = $title,
                p.year = $year,
                p.venue = $venue,
                p.community_id = null,
                p.data_version = $data_version
        """, arxiv_id=arxiv_id, title=paper["title"],
            year=paper.get("year", 0), venue=venue,
            data_version=DATA_VERSION)

        ents = entities.get(arxiv_id, {})

        # Author nodes + AUTHORED_BY edges
        for author in paper.get("authors", []):
            session.run("""
                MERGE (a:Author {name: $name})
                WITH a
                MATCH (p:Paper {arxiv_id: $arxiv_id})
                MERGE (p)-[:AUTHORED_BY]->(a)
            """, name=author, arxiv_id=arxiv_id)

        # Institution nodes + FROM_INSTITUTION edges (via author)
        for org in ents.get("orgs", []):
            session.run("""
                MERGE (i:Institution {name: $name})
                WITH i
                MATCH (p:Paper {arxiv_id: $arxiv_id})-[:AUTHORED_BY]->(a:Author)
                WITH i, a LIMIT 1
                MERGE (a)-[:FROM_INSTITUTION]->(i)
            """, name=org, arxiv_id=arxiv_id)

        # Method nodes + USES_METHOD edges
        for method in ents.get("methods", []):
            session.run("""
                MERGE (m:Method {name: $name})
                WITH m
                MATCH (p:Paper {arxiv_id: $arxiv_id})
                MERGE (p)-[:USES_METHOD]->(m)
            """, name=method, arxiv_id=arxiv_id)

        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1}/{len(papers)} papers")


def run_validation(session) -> None:
    print("\n--- Validation queries ---")

    r = session.run("MATCH (p:Paper) RETURN count(p) AS n").single()
    print(f"Paper nodes:       {r['n']}")

    r = session.run("MATCH (a:Author) RETURN count(a) AS n").single()
    print(f"Author nodes:      {r['n']}")

    r = session.run("MATCH (i:Institution) RETURN count(i) AS n").single()
    print(f"Institution nodes: {r['n']}")

    r = session.run("MATCH (m:Method) RETURN count(m) AS n").single()
    print(f"Method nodes:      {r['n']}")

    r = session.run("MATCH ()-[r:AUTHORED_BY]->() RETURN count(r) AS n").single()
    print(f"AUTHORED_BY edges: {r['n']}")

    r = session.run("MATCH ()-[r:USES_METHOD]->() RETURN count(r) AS n").single()
    print(f"USES_METHOD edges: {r['n']}")

    print("\nSample paper + authors:")
    results = session.run("""
        MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)
        RETURN p.title AS title, collect(a.name) AS authors
        LIMIT 3
    """)
    for row in results:
        print(f"  {row['title'][:60]} → {row['authors'][:2]}")


def main(dry_run: bool = False) -> None:
    limit = 20 if dry_run else None
    mode = "DRY RUN (20 papers)" if dry_run else "FULL INGESTION (2000 papers)"
    print(f"Starting {mode}...")

    papers = load_papers(limit=limit)
    entities = load_entities()

    driver = get_driver()
    with driver.session() as session:
        create_indexes(session)
        ingest_papers(papers, entities, session)
        run_validation(session)

    driver.close()
    print(f"\n{mode} complete.")


if __name__ == "__main__":
    import sys
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)

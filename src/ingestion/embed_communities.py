"""Embed community summaries with BGE-M3 and store on Community nodes in Neo4j."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from FlagEmbedding import FlagModel
from neo4j import GraphDatabase

load_dotenv()

CACHE_FILE = Path("data/processed/community_embeddings_cache.json")


def get_driver():
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )


def main() -> None:
    driver = get_driver()

    print("Loading BGE-M3...")
    model = FlagModel("BAAI/bge-m3", use_fp16=True, normalize_embeddings=True)

    with driver.session() as session:
        rows = session.run("""
            MATCH (c:Community)
            RETURN c.community_id AS cid, c.theme AS theme,
                   c.summary AS summary, c.size AS size
            ORDER BY cid
        """).data()

    print(f"Embedding {len(rows)} community summaries...")

    texts = [f"{r['theme']} {r['summary']}" for r in rows]
    embeddings = model.encode(texts)

    cache_records = []
    with driver.session() as session:
        for i, row in enumerate(rows):
            embedding = embeddings[i].tolist()
            session.run("""
                MATCH (c:Community {community_id: $cid})
                SET c.embedding = $embedding
            """, cid=row["cid"], embedding=embedding)
            cache_records.append({
                "cid": row["cid"],
                "theme": row["theme"],
                "summary": row["summary"],
                "embedding": embedding,
                "size": row["size"],
            })
            print(f"  Community {row['cid']} embedded.")

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps(cache_records, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Community cache updated: {CACHE_FILE} ({len(cache_records)} records)")

    driver.close()
    print("\nDone. All community embeddings stored in Neo4j.")


if __name__ == "__main__":
    main()

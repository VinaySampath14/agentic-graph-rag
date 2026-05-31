"""Embed community summaries with BGE-M3 and store on Community nodes in Neo4j."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from FlagEmbedding import FlagModel
from neo4j import GraphDatabase

load_dotenv()


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
            RETURN c.community_id AS cid, c.theme AS theme, c.summary AS summary
        """).data()

    print(f"Embedding {len(rows)} community summaries...")

    texts = [f"{r['theme']} {r['summary']}" for r in rows]
    embeddings = model.encode(texts)

    with driver.session() as session:
        for i, row in enumerate(rows):
            session.run("""
                MATCH (c:Community {community_id: $cid})
                SET c.embedding = $embedding
            """, cid=row["cid"], embedding=embeddings[i].tolist())
            print(f"  Community {row['cid']} embedded.")

    driver.close()
    print("\nDone. All community embeddings stored in Neo4j.")


if __name__ == "__main__":
    main()

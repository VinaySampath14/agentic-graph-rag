"""Validate Neo4j graph after ingestion."""
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
)

checks = {
    "Paper nodes":       "MATCH (p:Paper) RETURN count(p) AS n",
    "Author nodes":      "MATCH (a:Author) RETURN count(a) AS n",
    "Institution nodes": "MATCH (i:Institution) RETURN count(i) AS n",
    "Method nodes":      "MATCH (m:Method) RETURN count(m) AS n",
    "AUTHORED_BY edges": "MATCH ()-[r:AUTHORED_BY]->() RETURN count(r) AS n",
    "USES_METHOD edges": "MATCH ()-[r:USES_METHOD]->() RETURN count(r) AS n",
    "FROM_INSTITUTION":  "MATCH ()-[r:FROM_INSTITUTION]->() RETURN count(r) AS n",
}

with driver.session() as session:
    for label, query in checks.items():
        n = session.run(query).single()["n"]
        print(f"{label:<22} {n:>6}")

    print("\nSample multi-hop traversal:")
    rows = session.run("""
        MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author)-[:FROM_INSTITUTION]->(i:Institution)
        RETURN p.title AS title, a.name AS author, i.name AS institution
        LIMIT 3
    """)
    for r in rows:
        print(f"  {r['title'][:50]} | {r['author']} | {r['institution']}")

driver.close()

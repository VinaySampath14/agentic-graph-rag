"""Validate community detection results."""
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
)

with driver.session() as session:

    # Check 1: papers without community_id
    n = session.run(
        "MATCH (p:Paper) WHERE p.community_id IS NULL RETURN count(p) AS n"
    ).single()["n"]
    status = "✓" if n == 0 else "✗"
    print(f"{status} Papers without community_id: {n} (expected 0)")

    # Check 2: papers with BELONGS_TO edge
    n = session.run(
        "MATCH (p:Paper)-[:BELONGS_TO]->(c:Community) RETURN count(p) AS n"
    ).single()["n"]
    status = "✓" if n == 2000 else "✗"
    print(f"{status} Papers with BELONGS_TO edge: {n} (expected 2000)")

    # Check 3: community sizes sum to 2000
    n = session.run(
        "MATCH (c:Community) RETURN sum(c.size) AS n"
    ).single()["n"]
    status = "✓" if n == 2000 else "✗"
    print(f"{status} Sum of community sizes: {n} (expected 2000)")

    # Check 4: community overview
    print("\nCommunity overview:")
    rows = session.run("""
        MATCH (c:Community)
        RETURN c.community_id AS id, c.size AS size
        ORDER BY size DESC
    """)
    for r in rows:
        print(f"  Community {r['id']:>3} — {r['size']:>4} papers")

    # Check 5: sample titles from largest community
    print("\nSample titles from largest community:")
    rows = session.run("""
        MATCH (p:Paper)-[:BELONGS_TO]->(c:Community)
        WITH c, p ORDER BY c.size DESC
        WITH c, collect(p.title)[..8] AS titles
        RETURN c.community_id AS id, titles
        LIMIT 1
    """)
    for r in rows:
        print(f"  Community {r['id']}:")
        for title in r["titles"]:
            print(f"    - {title[:70]}")

driver.close()

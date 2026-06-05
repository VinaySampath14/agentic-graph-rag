"""Update community_id in Qdrant payloads after re-running community detection."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

load_dotenv()

COLLECTION = "papers"

# Fetch updated community_ids from Neo4j
driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
)

print("Fetching community IDs from Neo4j...")
with driver.session() as session:
    rows = session.run(
        "MATCH (p:Paper) RETURN p.arxiv_id AS arxiv_id, p.community_id AS community_id"
    ).data()

community_map = {r["arxiv_id"]: r["community_id"] for r in rows}
print(f"  {len(community_map)} papers fetched")
driver.close()

# Update Qdrant payloads
client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

# Scroll all points and update community_id
offset = None
updated = 0
batch_size = 100

while True:
    results, offset = client.scroll(
        collection_name=COLLECTION,
        limit=batch_size,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    if not results:
        break

    for point in results:
        arxiv_id = point.payload.get("arxiv_id")
        if arxiv_id and arxiv_id in community_map:
            client.set_payload(
                collection_name=COLLECTION,
                payload={"community_id": community_map[arxiv_id]},
                points=[point.id],
            )
            updated += 1

    if offset is None:
        break

print(f"Updated {updated} points in Qdrant with new community_ids.")

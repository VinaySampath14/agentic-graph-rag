"""Extract project facts for updating the arXiv paper."""
import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

load_dotenv()

driver = GraphDatabase.driver(
    os.environ["NEO4J_URI"],
    auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
)

print("=" * 60)
print("1. COMMUNITY THEMES")
print("=" * 60)
with driver.session() as s:
    rows = s.run(
        "MATCH (c:Community) RETURN c.community_id AS id, c.theme AS theme, "
        "c.size AS size, c.summary AS summary ORDER BY c.community_id"
    ).data()
    for r in rows:
        summary = {}
        try:
            summary = json.loads(r["summary"] or "{}")
        except Exception:
            pass
        print(f"Community {r['id']} ({r['size']} papers)")
        print(f"  Theme: {r['theme']}")
        print(f"  Methods: {summary.get('dominant_methods', [])}")
        print(f"  Rep papers: {summary.get('representative_papers', [])[:2]}")

print()
print("=" * 60)
print("2. CORPUS DATE RANGE")
print("=" * 60)
with driver.session() as s:
    r = s.run(
        "MATCH (p:Paper) RETURN min(p.year) as earliest, max(p.year) as latest"
    ).single()
    print(f"Earliest: {r['earliest']}  Latest: {r['latest']}")

print()
print("=" * 60)
print("3. GRAPH STATISTICS")
print("=" * 60)
queries = {
    "Paper nodes":        "MATCH (p:Paper) RETURN count(p) as n",
    "Author nodes":       "MATCH (a:Author) RETURN count(a) as n",
    "Institution nodes":  "MATCH (i:Institution) RETURN count(i) as n",
    "Method nodes":       "MATCH (m:Method) RETURN count(m) as n",
    "Community nodes":    "MATCH (c:Community) RETURN count(c) as n",
    "Total relationships":"MATCH ()-[r]->() RETURN count(r) as n",
    "AUTHORED_BY":        "MATCH ()-[r:AUTHORED_BY]->() RETURN count(r) as n",
    "USES_METHOD":        "MATCH ()-[r:USES_METHOD]->() RETURN count(r) as n",
    "FROM_INSTITUTION":   "MATCH ()-[r:FROM_INSTITUTION]->() RETURN count(r) as n",
    "BELONGS_TO":         "MATCH ()-[r:BELONGS_TO]->() RETURN count(r) as n",
}
with driver.session() as s:
    for label, q in queries.items():
        n = s.run(q).single()["n"]
        print(f"{label:<25} {n:>6}")

driver.close()

print()
print("=" * 60)
print("4. LLM EXTRACTION STATS")
print("=" * 60)
llm_file = Path("data/processed/llm_extractions.jsonl")
if llm_file.exists():
    lines = llm_file.read_text(encoding="utf-8").splitlines()
    succeeded = sum(1 for l in lines if json.loads(l).get("methods"))
    print(f"Papers processed: {len(lines)}")
    print(f"Succeeded (have methods): {succeeded}")
    print(f"Failed (no methods): {len(lines) - succeeded}")
else:
    print("File not found")

print()
print("=" * 60)
print("5. NORMALISATION STATS")
print("=" * 60)
norm_file = Path("data/processed/normalisation_log.jsonl")
if norm_file.exists():
    logs = [json.loads(l) for l in norm_file.read_text(encoding="utf-8").splitlines()]
    print(f"Total decisions: {len(logs)}")
    replacements = Counter()
    for l in logs:
        if l.get("type") == "org_normalised":
            replacements[f"{l['original']} -> {l['canonical']}"] += 1
    print("Top 10 replacements:")
    for k, v in replacements.most_common(10):
        print(f"  {k}: {v}")
else:
    print("File not found")

print()
print("=" * 60)
print("6. LEIDEN PARAMETERS")
print("=" * 60)
print("Algorithm: leidenalg.ModularityVertexPartition")
print("Resolution: default (ModularityVertexPartition has no resolution param)")
print("Seed: 42")
print("Min community size kept: 5 (smaller merged into largest)")
print("Final communities: 4")

print()
print("=" * 60)
print("7. CROSS-ENCODER THRESHOLD")
print("=" * 60)
print("Threshold: 0.15 (rerank only when top-2 score margin < 0.15)")
print("Model: cross-encoder/ms-marco-MiniLM-L-6-v2")

print()
print("=" * 60)
print("8. HOP DEPTH LOGIC")
print("=" * 60)
print("Start: 1-hop AUTHORED_BY co-authorship traversal")
print("If results < 3: expand to 2-hop USES_METHOD co-occurrence")
print("Temporal filter: activates on 'after YYYY', 'since YYYY', 'recent'")
print("Venue filter: activates on NeurIPS, ACL, ICML, ICLR etc.")

print()
print("=" * 60)
print("9. QDRANT COLLECTION STATS")
print("=" * 60)
client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)
info = client.get_collection("papers")
print(f"Collection: papers")
print(f"Points: {info.points_count}")
dense = info.config.params.vectors.get("dense")
if dense:
    print(f"Dense dimension: {dense.size}")
    print(f"Dense distance: {dense.distance}")
sparse = info.config.params.sparse_vectors
print(f"Sparse vectors: {list(sparse.keys()) if sparse else 'none'}")

print()
print("=" * 60)
print("10. ROUTER KEYWORD SIGNALS")
print("=" * 60)
from src.retrievers.router import INTENT_SIGNALS
for intent, signals in INTENT_SIGNALS.items():
    print(f"{intent.upper()}: {signals}")

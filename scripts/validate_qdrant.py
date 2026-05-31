"""Validate Qdrant collection after ingestion."""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

COLLECTION = "papers"

# Check 1: collection exists and point count
info = client.get_collection(COLLECTION)
count = info.points_count
status = "✓" if count == 2000 else "✗"
print(f"{status} Points in collection: {count} (expected 2000)")

# Check 2: vector config
vec_config = info.config.params.vectors
sparse_config = info.config.params.sparse_vectors
dense_keys = list(vec_config.keys()) if vec_config else []
sparse_keys = list(sparse_config.keys()) if sparse_config else []
print(f"{'✓' if 'dense' in dense_keys else '✗'} Dense vectors: {dense_keys}")
print(f"{'✓' if 'sparse' in sparse_keys else '✗'} Sparse vectors: {sparse_keys}")

# Check 3: sample a point and check payload
sample = client.scroll(
    collection_name=COLLECTION,
    limit=3,
    with_payload=True,
    with_vectors=False,
)[0]

print(f"\nSample payloads:")
for point in sample:
    p = point.payload
    print(f"  arxiv_id={p.get('arxiv_id')} year={p.get('year')} title={p.get('title', '')[:50]}")

# Check 4: test a semantic search
from FlagEmbedding import FlagModel
print("\nTesting semantic search...")
model = FlagModel("BAAI/bge-m3", use_fp16=True, normalize_embeddings=True)
query_vec = model.encode(["graph neural networks for knowledge retrieval"])[0].tolist()

results = client.query_points(
    collection_name=COLLECTION,
    query=query_vec,
    using="dense",
    limit=3,
).points
print("Top 3 results for 'graph neural networks for knowledge retrieval':")
for r in results:
    print(f"  score={r.score:.3f} | {r.payload.get('title', '')[:60]}")

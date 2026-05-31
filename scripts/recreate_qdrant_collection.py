"""Delete and recreate Qdrant collection with both dense and sparse vector configs."""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseIndexParams,
)

load_dotenv()

client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
)

COLLECTION = "papers"

# Delete existing
client.delete_collection(COLLECTION)
print(f"Deleted collection '{COLLECTION}'.")

# Recreate with dense + sparse
client.create_collection(
    collection_name=COLLECTION,
    vectors_config={
        "dense": VectorParams(size=1024, distance=Distance.COSINE),
    },
    sparse_vectors_config={
        "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
    },
)
print(f"Recreated collection '{COLLECTION}' with dense + sparse configs.")
print("Run qdrant_loader.py to re-upload all papers.")

"""Embed abstracts and upload to Qdrant with dense + sparse vectors."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from FlagEmbedding import FlagModel
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    NamedSparseVector,
    NamedVector,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

load_dotenv()

PAPERS_DIR = Path("data/raw")
COLLECTION_NAME = "papers"
DENSE_DIM = 1024
BATCH_SIZE = 32
DATA_VERSION = "v1"


def load_papers() -> list[dict]:
    papers = []
    for path in sorted(PAPERS_DIR.glob("papers_batch_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                papers.append(json.loads(line))
    return papers


def get_client() -> QdrantClient:
    return QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )


def create_collection(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        print(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
    )
    print(f"Collection '{COLLECTION_NAME}' created.")


def ingest(papers: list[dict]) -> None:
    client = get_client()
    create_collection(client)

    print("Loading BGE-M3 embedding model...")
    dense_model = FlagModel(
        "BAAI/bge-m3",
        use_fp16=True,
        normalize_embeddings=True,
    )

    print("Loading FastEmbed BM25 sparse model...")
    sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    total = len(papers)
    uploaded = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch = papers[batch_start: batch_start + BATCH_SIZE]
        abstracts = [p["abstract"] for p in batch]

        # Dense embeddings
        dense_vecs = dense_model.encode(abstracts)

        # Sparse embeddings
        sparse_vecs = list(sparse_model.embed(abstracts))

        points = []
        for i, paper in enumerate(batch):
            sparse = sparse_vecs[i]
            points.append(
                PointStruct(
                    id=batch_start + i,
                    vector={
                        "dense": dense_vecs[i].tolist(),
                        "sparse": SparseVector(
                            indices=sparse.indices.tolist(),
                            values=sparse.values.tolist(),
                        ),
                    },
                    payload={
                        "arxiv_id": paper["arxiv_id"],
                        "title": paper["title"],
                        "abstract": paper["abstract"],
                        "year": paper.get("year", 0),
                        "venue": paper.get("venue", ""),
                        "authors": paper.get("authors", []),
                        "categories": paper.get("categories", []),
                        "data_version": DATA_VERSION,
                    },
                )
            )

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        uploaded += len(batch)
        print(f"  Uploaded {uploaded}/{total} papers")

    print(f"\nDone. {uploaded} papers uploaded to Qdrant collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    papers = load_papers()
    print(f"Loaded {len(papers)} papers.")
    ingest(papers)

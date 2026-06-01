"""Naive vector retriever — Qdrant hybrid search with conditional reranking."""
import os

from dotenv import load_dotenv
from FlagEmbedding import FlagModel
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion
from sentence_transformers import CrossEncoder

from src.retrievers.models import RetrievalResult

load_dotenv()

COLLECTION = "papers"
TOP_K_CANDIDATES = 20
TOP_K_FINAL = 5
RERANK_MARGIN_THRESHOLD = 0.15

_dense_model: FlagModel | None = None
_sparse_model: SparseTextEmbedding | None = None
_cross_encoder: CrossEncoder | None = None
_qdrant_client: QdrantClient | None = None


def _get_dense_model() -> FlagModel:
    global _dense_model
    if _dense_model is None:
        _dense_model = FlagModel("BAAI/bge-m3", use_fp16=True, normalize_embeddings=True)
    return _dense_model


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")
    return _sparse_model


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def _get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
    return _qdrant_client


def retrieve(query: str) -> RetrievalResult:
    dense_model = _get_dense_model()
    sparse_model = _get_sparse_model()
    client = _get_qdrant_client()

    # Dense embedding
    dense_vec = dense_model.encode([query])[0].tolist()

    # Sparse embedding
    sparse_vec = list(sparse_model.embed([query]))[0]

    # Hybrid search with RRF fusion
    results = client.query_points(
        collection_name=COLLECTION,
        prefetch=[
            Prefetch(query=dense_vec, using="dense", limit=TOP_K_CANDIDATES),
            Prefetch(
                query={"indices": sparse_vec.indices.tolist(),
                       "values": sparse_vec.values.tolist()},
                using="sparse",
                limit=TOP_K_CANDIDATES,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=TOP_K_CANDIDATES,
        with_payload=True,
    ).points

    if not results:
        return RetrievalResult(
            context_text="No results found.",
            source_type="vector",
        )

    # Conditional cross-encoder reranking
    top2_scores = [r.score for r in results[:2]]
    margin = abs(top2_scores[0] - top2_scores[1]) if len(top2_scores) == 2 else 1.0

    if margin < RERANK_MARGIN_THRESHOLD:
        cross_encoder = _get_cross_encoder()
        pairs = [[query, r.payload.get("abstract", "")] for r in results]
        ce_scores = cross_encoder.predict(pairs)
        results = [r for _, r in sorted(
            zip(ce_scores, results), key=lambda x: x[0], reverse=True
        )]

    top_results = results[:TOP_K_FINAL]

    # Format context text
    context_parts = []
    metadata = []
    for r in top_results:
        p = r.payload
        context_parts.append(
            f"Title: {p.get('title', '')}\n"
            f"arxiv_id: {p.get('arxiv_id', '')}\n"
            f"Year: {p.get('year', '')}\n"
            f"Abstract: {p.get('abstract', '')}"
        )
        metadata.append({
            "arxiv_id": p.get("arxiv_id"),
            "title": p.get("title"),
            "score": r.score,
        })

    return RetrievalResult(
        context_text="\n\n---\n\n".join(context_parts),
        source_type="vector",
        source_metadata={"results": metadata, "reranked": margin < RERANK_MARGIN_THRESHOLD},
    )


if __name__ == "__main__":
    query = "graph neural networks for knowledge retrieval"
    print(f"Query: {query}\n")
    result = retrieve(query)
    print(f"Source type: {result.source_type}")
    print(f"Context preview:\n{result.context_text[:500]}")

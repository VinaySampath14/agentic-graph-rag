"""Global community retriever — BGE-M3 similarity against Community embeddings."""
import json
import os

import numpy as np
from dotenv import load_dotenv
from FlagEmbedding import FlagModel
from neo4j import GraphDatabase

from src.retrievers.models import RetrievalResult

load_dotenv()

TOP_K_COMMUNITIES = 3

_dense_model: FlagModel | None = None
_driver = None


def _get_dense_model() -> FlagModel:
    global _dense_model
    if _dense_model is None:
        _dense_model = FlagModel("BAAI/bge-m3", use_fp16=True, normalize_embeddings=True)
    return _dense_model


def _get_driver():
    global _driver
    if _driver is None:
        try:
            from src.agent.connections import get_neo4j_driver
            _driver = get_neo4j_driver()
        except ImportError:
            _driver = GraphDatabase.driver(
                os.environ["NEO4J_URI"],
                auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
            )
    return _driver


def _load_community_embeddings(session) -> list[dict]:
    result = session.run("""
        MATCH (c:Community)
        WHERE c.embedding IS NOT NULL
        RETURN c.community_id AS cid, c.theme AS theme,
               c.summary AS summary, c.embedding AS embedding, c.size AS size
    """)
    return result.data()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a_arr = np.array(a)
    b_arr = np.array(b)
    return float(np.dot(a_arr, b_arr) / (np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-10))


def retrieve(query: str) -> RetrievalResult:
    model = _get_dense_model()
    driver = _get_driver()

    query_embedding = model.encode([query])[0].tolist()

    with driver.session() as session:
        communities = _load_community_embeddings(session)

    if not communities:
        return RetrievalResult(
            context_text="No community embeddings found.",
            source_type="community",
        )

    # Score each community
    scored = []
    for c in communities:
        score = _cosine_similarity(query_embedding, c["embedding"])
        scored.append((score, c))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_communities = scored[:TOP_K_COMMUNITIES]

    # Format context
    context_parts = []
    community_names = []

    for score, c in top_communities:
        summary_data = {}
        if c.get("summary"):
            try:
                summary_data = json.loads(c["summary"])
            except (json.JSONDecodeError, TypeError):
                summary_data = {}

        theme = c.get("theme", "Unknown theme")
        dominant_methods = summary_data.get("dominant_methods", [])
        key_authors = summary_data.get("key_authors", [])
        rep_papers = summary_data.get("representative_papers", [])

        community_name = f"Community {c['cid']} ({c.get('size', '?')} papers)"
        community_names.append(community_name)

        part = f"[{community_name} — similarity: {score:.3f}]\n"
        part += f"Theme: {theme}\n"
        if dominant_methods:
            part += f"Dominant methods: {', '.join(dominant_methods)}\n"
        if key_authors:
            part += f"Key authors: {', '.join(key_authors)}\n"
        if rep_papers:
            part += f"Representative papers: {'; '.join(rep_papers[:2])}\n"

        context_parts.append(part)

    return RetrievalResult(
        context_text="\n\n".join(context_parts),
        source_type="community",
        communities_used=community_names,
        source_metadata={"scores": [s for s, _ in top_communities]},
    )


if __name__ == "__main__":
    queries = [
        "What are the main trends in LLM reasoning?",
        "Overview of parameter-efficient fine-tuning research",
        "What are the dominant approaches in multimodal learning?",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        result = retrieve(q)
        print(f"Communities used: {result.communities_used}")
        print(f"Context:\n{result.context_text[:500]}")
        print("-" * 60)

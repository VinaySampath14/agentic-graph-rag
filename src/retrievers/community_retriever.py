"""Global community retriever — BGE-M3 similarity against Community embeddings."""
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase

from src.agent.connections import get_dense_model
from src.retrievers.models import RetrievalResult

load_dotenv()

TOP_K_COMMUNITIES = 3
CACHE_FILE = "data/processed/community_embeddings_cache.json"

_driver = None
_community_cache: list[dict] | None = None


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


def _get_communities() -> list[dict]:
    global _community_cache
    if _community_cache is not None:
        return _community_cache

    # Try local cache first
    cache_path = Path(CACHE_FILE)
    if cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            _community_cache = json.load(f)
        print(f"Loaded {len(_community_cache)} community embeddings from cache.")
        return _community_cache

    # Fetch from Neo4j and save to cache
    print("Fetching community embeddings from Neo4j...")
    driver = _get_driver()
    with driver.session() as session:
        communities = _load_community_embeddings(session)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(communities, f)
    print(f"Cached {len(communities)} community embeddings to {cache_path}")

    _community_cache = communities
    return _community_cache


def retrieve(query: str) -> RetrievalResult:
    model = get_dense_model()

    query_embedding = model.encode([query])[0].tolist()

    communities = _get_communities()

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

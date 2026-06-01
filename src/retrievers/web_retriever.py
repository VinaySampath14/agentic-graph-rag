"""Web retriever — Tavily search fallback."""
import os

from dotenv import load_dotenv
from tavily import TavilyClient

from src.retrievers.models import RetrievalResult

load_dotenv()

TOP_K = 5
_client: TavilyClient | None = None


def _get_client() -> TavilyClient:
    global _client
    if _client is None:
        _client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    return _client


def retrieve(query: str) -> RetrievalResult:
    client = _get_client()

    try:
        response = client.search(query, max_results=TOP_K, timeout=10)
        results = response.get("results", [])
    except Exception as e:
        return RetrievalResult(
            context_text=f"Web search unavailable: {e}",
            source_type="web",
        )

    if not results:
        return RetrievalResult(
            context_text="No web results found.",
            source_type="web",
        )

    context_parts = []
    metadata = []
    for r in results:
        context_parts.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r.get('url', '')}\n"
            f"Content: {r.get('content', '')}"
        )
        metadata.append({"title": r.get("title"), "url": r.get("url")})

    return RetrievalResult(
        context_text="\n\n---\n\n".join(context_parts),
        source_type="web",
        source_metadata={"results": metadata},
    )


if __name__ == "__main__":
    query = "graph RAG self-correcting retrieval 2024"
    print(f"Query: {query}\n")
    result = retrieve(query)
    print(f"Source type: {result.source_type}")
    print(f"Context preview:\n{result.context_text[:500]}")

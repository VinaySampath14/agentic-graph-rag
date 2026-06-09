"""Web retriever — DuckDuckGo search fallback (no API key required)."""
import logging

from ddgs import DDGS

from src.retrievers.models import RetrievalResult

TOP_K = 5


def retrieve(query: str) -> RetrievalResult:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=TOP_K))
    except Exception as e:
        logging.warning("Web retriever error: %s", e)
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
            f"URL: {r.get('href', '')}\n"
            f"Content: {r.get('body', '')}"
        )
        metadata.append({"title": r.get("title"), "url": r.get("href")})

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

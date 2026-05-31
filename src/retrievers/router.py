"""Rule-based query router — classifies intent and dispatches to retrieval mode."""

INTENT_SIGNALS: dict[str, list[str]] = {
    "graph": [
        "who", "which", "cites", "cited by", "collaborated", "co-author",
        "institution", "author", "wrote", "published by", "works at",
        "from", "affiliation", "paper by",
    ],
    "community": [
        "trend", "trends", "overview", "main topics", "across", "landscape",
        "survey", "state of", "dominant", "popular", "broadly", "generally",
        "research area", "field", "discipline", "what are the main",
        "what is the current state", "summarise", "summarize",
    ],
    "vector": [
        "what is", "define", "how does", "explain", "what does",
        "what did", "describe", "method", "technique", "approach",
        "algorithm", "model", "architecture", "dataset", "benchmark",
        "papers about", "papers on", "research on", "fine-tuning",
        "fine tuning", "training", "evaluation", "performance",
    ],
}

LOW_CONFIDENCE_THRESHOLD = 2


def classify(query: str, mode_history: list[str] | None = None) -> dict:
    """
    Classify query intent and return routing decision.

    Returns:
        {
            "primary_mode": str,
            "confidence": int,
            "low_confidence": bool,
            "fallback_mode": str | None,
        }
    """
    if mode_history is None:
        mode_history = []

    query_lower = query.lower()
    scores: dict[str, int] = {"graph": 0, "community": 0, "vector": 0}

    for intent, signals in INTENT_SIGNALS.items():
        for signal in signals:
            if signal in query_lower:
                scores[intent] += 1

    # Remove modes already tried
    available = {k: v for k, v in scores.items() if k not in mode_history}

    if not available:
        return {
            "primary_mode": "web",
            "confidence": 0,
            "low_confidence": True,
            "fallback_mode": None,
        }

    primary_mode = max(available, key=lambda k: available[k])
    confidence = available[primary_mode]
    low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD

    # Fallback mode for low confidence — second best available mode
    fallback_mode = None
    if low_confidence:
        remaining = {k: v for k, v in available.items() if k != primary_mode}
        if remaining:
            fallback_mode = max(remaining, key=lambda k: remaining[k])

    return {
        "primary_mode": primary_mode,
        "confidence": confidence,
        "low_confidence": low_confidence,
        "fallback_mode": fallback_mode,
        "all_scores": scores,
    }


if __name__ == "__main__":
    test_queries = [
        ("Who are the authors of papers using LoRA?", []),
        ("What are the main trends in LLM safety research?", []),
        ("What is the Transformer architecture?", []),
        ("Tell me about attention", []),  # low confidence
        ("Who at Google Brain worked on BERT?", ["graph"]),  # graph already tried
        ("Papers about fine-tuning after 2023", []),
    ]

    for query, history in test_queries:
        result = classify(query, history)
        conf_label = "LOW" if result["low_confidence"] else "HIGH"
        print(f"Query:   {query}")
        print(f"Mode:    {result['primary_mode']} ({conf_label} confidence: {result['confidence']})")
        if result["fallback_mode"]:
            print(f"Fallback: {result['fallback_mode']}")
        print(f"Scores:  {result.get('all_scores', {})}")
        print()

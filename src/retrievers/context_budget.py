"""Hard token limit on context before passing to generator."""


MAX_TOKENS = 6000
# Rough approximation: 1 token ≈ 4 characters
CHARS_PER_TOKEN = 4


def count_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def apply_budget(
    graph_context: str = "",
    community_context: str = "",
    vector_context: str = "",
) -> tuple[str, bool]:
    """
    Combine contexts in priority order (graph > community > vector),
    truncating to MAX_TOKENS. Returns (combined_text, was_truncated).
    """
    truncated = False
    parts = []
    tokens_used = 0

    for label, text in [
        ("Graph context", graph_context),
        ("Community context", community_context),
        ("Vector context", vector_context),
    ]:
        if not text:
            continue
        tokens = count_tokens(text)
        if tokens_used + tokens <= MAX_TOKENS:
            parts.append(f"[{label}]\n{text}")
            tokens_used += tokens
        else:
            remaining_chars = (MAX_TOKENS - tokens_used) * CHARS_PER_TOKEN
            if remaining_chars > 200:
                parts.append(f"[{label} — truncated]\n{text[:remaining_chars]}")
                truncated = True
            break

    return "\n\n".join(parts), truncated

"""AgentState — shared state passed between all LangGraph nodes."""
from typing import Any
from typing_extensions import TypedDict

from src.retrievers.models import RetrievalResult, GradeResult


class AgentState(TypedDict):
    # Core query fields
    query: str
    rewritten_query: str
    intent: str  # "graph" | "vector" | "community"

    # Retrieval + grading
    retrieved_context: RetrievalResult | None
    grade_result: GradeResult | None

    # Generation output
    answer: str
    citations: list[str]
    confidence_proxy: float

    # Loop control — define these first, they are the most critical fields
    loop_count: int        # starts 0, hard guard at 3
    mode_history: list[str]  # modes already tried this query

    # Explainability trace — every node appends one entry
    agent_trace: list[dict[str, Any]]

    # Routing metadata
    low_confidence: bool
    fallback_mode: str | None

    # Terminal state
    refused: bool
    refusal_reason: str

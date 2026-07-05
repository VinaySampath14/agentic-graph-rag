"""Pydantic models shared across all retrievers and the agent."""
from typing import Any
from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    context_text: str
    source_type: str  # "vector" | "graph" | "community" | "web"
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    cypher_query_used: str | None = None
    sparql_query_used: str | None = None
    communities_used: list[str] = Field(default_factory=list)
    truncated: bool = False


class GradeResult(BaseModel):
    passed: bool
    reason: str


class GeneratorOutput(BaseModel):
    answer: str
    citations: list[str] = Field(default_factory=list)
    confidence_proxy: float = 0.0
    mode_used: str = ""
    loop_count: int = 0


class QueryLog(BaseModel):
    query: str
    rewritten_query: str = ""
    intent: str = ""
    mode_used: str = ""
    loop_count: int = 0
    modes_tried: list[str] = Field(default_factory=list)
    rewrite_triggered: bool = False
    first_mode_success: bool = False
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    latency_ms: float = 0.0
    answer: str = ""
    timestamp: str = ""

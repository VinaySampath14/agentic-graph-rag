"""FastAPI backend for Agentic Graph RAG.

Endpoints:
    POST /query   — run a query through the agentic pipeline
    GET  /health  — liveness check
    GET  /stats   — eval summary (coverage + RAGAS per version)

Usage:
    uvicorn src.api.main:app --reload
"""
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm all models and connections at startup so the first request
    # doesn't pay model-load time (BGE-M3 + SPLADE can take 20-30s on CPU)
    print("Pre-warming models and connections...")
    from src.agent.connections import get_dense_model, get_neo4j_driver, get_qdrant_client
    from src.retrievers.naive_retriever import _get_sparse_model, _get_cross_encoder
    from src.agent.nodes import _get_groq
    get_dense_model()      # shared BGE-M3 — used by both naive and community retrievers
    _get_sparse_model()
    _get_cross_encoder()
    get_neo4j_driver()
    get_qdrant_client()
    _get_groq()
    get_graph()
    print("Pre-warm complete.")
    yield


app = FastAPI(
    title="Agentic Graph RAG",
    description="Self-correcting RAG over 2,000 arXiv CS papers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EVAL_DIR = Path("data/eval")
VERSIONS = ["v1", "v2", "v3", "v4"]


# ── Pydantic models ────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="Question to answer")


class TraceEntry(BaseModel):
    node: str
    decision: str | None = None
    reason: str | None = None
    timestamp: str | None = None
    extras: dict[str, Any] | None = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    refused: bool
    refusal_reason: str | None
    loop_count: int
    mode_history: list[str]
    agent_trace: list[dict[str, Any]]
    latency_ms: float


class VersionStats(BaseModel):
    version: str
    n_total: int
    n_answered: int
    coverage: float
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


class StatsResponse(BaseModel):
    versions: list[VersionStats]


# ── Graph (compiled once at startup) ──────────────────────────────────────

@lru_cache(maxsize=1)
def get_graph():
    from src.agent.graph import compile_graph
    return compile_graph()


# ── Helpers ────────────────────────────────────────────────────────────────

def _avg(lst: list[float]) -> float:
    return round(sum(lst) / len(lst), 4) if lst else 0.0


def _load_stats() -> list[VersionStats]:
    results = []
    for v in VERSIONS:
        raw_path    = EVAL_DIR / f"eval_results_{v}.jsonl"
        scores_path = EVAL_DIR / f"ragas_scores_{v}.jsonl"
        if not raw_path.exists() or not scores_path.exists():
            continue
        raw    = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines()]
        scores = [json.loads(l) for l in scores_path.read_text(encoding="utf-8").splitlines()]
        answered = [r for r in raw if not r.get("refused", False)]
        results.append(VersionStats(
            version=v,
            n_total=len(raw),
            n_answered=len(answered),
            coverage=round(len(answered) / len(raw), 4) if raw else 0.0,
            faithfulness=_avg([s["faithfulness"] for s in scores]),
            answer_relevancy=_avg([s["answer_relevancy"] for s in scores]),
            context_precision=_avg([s["context_precision"] for s in scores]),
            context_recall=_avg([s["context_recall"] for s in scores]),
        ))
    return results


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats", response_model=StatsResponse)
def stats():
    try:
        return StatsResponse(versions=_load_stats())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    graph = get_graph()

    initial_state = {
        "query": request.query,
        "rewritten_query": request.query,
        "intent": "",
        "retrieved_context": None,
        "grade_result": None,
        "answer": "",
        "citations": [],
        "confidence_proxy": 0.0,
        "loop_count": 0,
        "mode_history": [],
        "agent_trace": [],
        "low_confidence": False,
        "fallback_mode": None,
        "refused": False,
        "refusal_reason": "",
    }

    t0 = time.perf_counter()
    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    return QueryResponse(
        query=request.query,
        answer=final_state.get("answer", ""),
        refused=final_state.get("refused", False),
        refusal_reason=final_state.get("refusal_reason") or None,
        loop_count=final_state.get("loop_count", 0),
        mode_history=final_state.get("mode_history", []),
        agent_trace=final_state.get("agent_trace", []),
        latency_ms=latency_ms,
    )

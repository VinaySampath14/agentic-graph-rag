"""Unit tests for LangGraph conditional edge functions — no network required."""
import pytest
from langgraph.graph import END

from src.agent.graph import (
    route_after_analyser,
    route_after_router,
    route_after_grade_context,
    route_after_grade_answer,
    LOOP_GUARD,
)
from src.retrievers.models import RetrievalResult, GradeResult


def _state(**overrides):
    base = {
        "query": "test query",
        "rewritten_query": "test query",
        "intent": "vector",
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
    base.update(overrides)
    return base


# ── route_after_analyser ───────────────────────────────────────────────────

class TestRouteAfterAnalyser:
    def test_normal_goes_to_router(self):
        assert route_after_analyser(_state()) == "router"

    def test_refused_goes_to_end(self):
        assert route_after_analyser(_state(refused=True)) == END


# ── route_after_router ─────────────────────────────────────────────────────

class TestRouteAfterRouter:
    def test_graph_intent_goes_to_graph_retriever(self):
        assert route_after_router(_state(intent="graph")) == "local_graph_retriever"

    def test_community_intent_goes_to_global_retriever(self):
        assert route_after_router(_state(intent="community")) == "global_retriever"

    def test_vector_intent_goes_to_naive_retriever(self):
        assert route_after_router(_state(intent="vector")) == "naive_retriever"

    def test_web_intent_goes_to_web_retriever(self):
        assert route_after_router(_state(intent="web")) == "web_retriever"

    def test_unknown_intent_defaults_to_naive(self):
        assert route_after_router(_state(intent="unknown")) == "naive_retriever"


# ── route_after_grade_context ──────────────────────────────────────────────

def _vector_context():
    return RetrievalResult(context_text="some context", source_type="vector")

def _web_context():
    return RetrievalResult(context_text="web result", source_type="web")


class TestRouteAfterGradeContext:
    def test_pass_goes_to_generator(self):
        state = _state(
            grade_result=GradeResult(passed=True, reason="good"),
            retrieved_context=_vector_context(),
        )
        assert route_after_grade_context(state) == "generator"

    def test_fail_low_loop_goes_to_rewrite(self):
        state = _state(
            grade_result=GradeResult(passed=False, reason="poor"),
            retrieved_context=_vector_context(),
            loop_count=0,
        )
        assert route_after_grade_context(state) == "rewrite_query"

    def test_fail_at_loop_guard_goes_to_web(self):
        state = _state(
            grade_result=GradeResult(passed=False, reason="poor"),
            retrieved_context=_vector_context(),
            loop_count=LOOP_GUARD,
        )
        assert route_after_grade_context(state) == "web_retriever"

    def test_fail_after_web_goes_to_force_refusal(self):
        state = _state(
            grade_result=GradeResult(passed=False, reason="poor"),
            retrieved_context=_web_context(),
            loop_count=1,
        )
        assert route_after_grade_context(state) == "force_refusal"

    def test_fail_with_web_in_history_goes_to_force_refusal(self):
        state = _state(
            grade_result=GradeResult(passed=False, reason="poor"),
            retrieved_context=_vector_context(),
            mode_history=["graph", "web"],
            loop_count=2,
        )
        assert route_after_grade_context(state) == "force_refusal"

    def test_loop_guard_constant_is_3(self):
        assert LOOP_GUARD == 3


# ── route_after_grade_answer ───────────────────────────────────────────────

class TestRouteAfterGradeAnswer:
    def test_normal_goes_to_end(self):
        assert route_after_grade_answer(_state()) == END

    def test_refused_also_goes_to_end(self):
        assert route_after_grade_answer(_state(refused=True)) == END

"""Unit tests for Pydantic data models — no network required."""
import pytest
from src.retrievers.models import RetrievalResult, GradeResult, GeneratorOutput, QueryLog


class TestRetrievalResult:
    def test_minimal_construction(self):
        r = RetrievalResult(context_text="ctx", source_type="vector")
        assert r.context_text == "ctx"
        assert r.source_type == "vector"
        assert r.truncated is False
        assert r.communities_used == []
        assert r.cypher_query_used is None

    def test_full_construction(self):
        r = RetrievalResult(
            context_text="ctx",
            source_type="graph",
            cypher_query_used="MATCH (n) RETURN n",
            communities_used=["c1", "c2"],
            truncated=True,
            source_metadata={"score": 0.9},
        )
        assert r.cypher_query_used == "MATCH (n) RETURN n"
        assert r.truncated is True
        assert len(r.communities_used) == 2

    def test_source_types(self):
        for stype in ("vector", "graph", "community", "web"):
            r = RetrievalResult(context_text="x", source_type=stype)
            assert r.source_type == stype


class TestGradeResult:
    def test_pass(self):
        g = GradeResult(passed=True, reason="sufficient context")
        assert g.passed is True
        assert g.reason == "sufficient context"

    def test_fail(self):
        g = GradeResult(passed=False, reason="no relevant content found")
        assert g.passed is False


class TestGeneratorOutput:
    def test_defaults(self):
        g = GeneratorOutput(answer="The answer is 42.")
        assert g.answer == "The answer is 42."
        assert g.citations == []
        assert g.confidence_proxy == 0.0
        assert g.loop_count == 0

    def test_with_citations(self):
        g = GeneratorOutput(answer="ans", citations=["paper1", "paper2"], confidence_proxy=0.85)
        assert len(g.citations) == 2
        assert g.confidence_proxy == 0.85


class TestQueryLog:
    def test_minimal(self):
        q = QueryLog(query="What is RAG?")
        assert q.query == "What is RAG?"
        assert q.rewrite_triggered is False
        assert q.modes_tried == []
        assert q.latency_ms == 0.0

    def test_full(self):
        q = QueryLog(
            query="Who wrote BERT?",
            rewritten_query="Authors of the BERT paper",
            intent="graph",
            mode_used="graph",
            loop_count=1,
            modes_tried=["graph", "vector"],
            rewrite_triggered=True,
            first_mode_success=False,
            faithfulness=0.9,
            answer_relevancy=0.85,
            latency_ms=1234.5,
            answer="Devlin et al.",
        )
        assert q.rewrite_triggered is True
        assert q.loop_count == 1
        assert q.faithfulness == 0.9

"""Unit tests for the rule-based query router — no network required."""
import pytest
from src.retrievers.router import classify, INTENT_SIGNALS, LOW_CONFIDENCE_THRESHOLD


# ── classify: primary mode ─────────────────────────────────────────────────

class TestClassifyPrimaryMode:
    def test_authorship_query_routes_graph(self):
        result = classify("Who are the authors of papers using LoRA?")
        assert result["primary_mode"] == "graph"

    def test_institution_query_routes_graph(self):
        result = classify("Which institution published the most papers on RLHF?")
        assert result["primary_mode"] == "graph"

    def test_trend_query_routes_community(self):
        result = classify("What are the main trends in LLM safety research?")
        assert result["primary_mode"] == "community"

    def test_overview_query_routes_community(self):
        result = classify("Give me an overview of the NLP research landscape")
        assert result["primary_mode"] == "community"

    def test_definition_query_routes_vector(self):
        result = classify("What is the Transformer architecture?")
        assert result["primary_mode"] == "vector"

    def test_method_query_routes_vector(self):
        result = classify("Explain the LoRA fine-tuning technique")
        assert result["primary_mode"] == "vector"

    def test_returns_all_required_keys(self):
        result = classify("What is attention?")
        assert {"primary_mode", "confidence", "low_confidence", "fallback_mode"} <= result.keys()


# ── classify: mode_history exclusion ──────────────────────────────────────

class TestModeHistoryExclusion:
    def test_excludes_already_tried_mode(self):
        result = classify("Who collaborated on RAG research?", mode_history=["graph"])
        assert result["primary_mode"] != "graph"

    def test_all_modes_tried_falls_back_to_web(self):
        result = classify("What is BERT?", mode_history=["vector", "graph", "community"])
        assert result["primary_mode"] == "web"

    def test_web_mode_has_zero_confidence(self):
        result = classify("Any query", mode_history=["vector", "graph", "community"])
        assert result["confidence"] == 0
        assert result["low_confidence"] is True

    def test_empty_history_uses_best_mode(self):
        result_no_hist = classify("What are the trends in multimodal AI?", mode_history=[])
        result_with_hist = classify("What are the trends in multimodal AI?", mode_history=["vector"])
        assert result_no_hist["primary_mode"] == result_with_hist["primary_mode"] or \
               result_with_hist["primary_mode"] != "vector"


# ── classify: confidence and low_confidence ───────────────────────────────

class TestConfidence:
    def test_strong_signal_is_high_confidence(self):
        result = classify("Which authors collaborated on attention and RAG?")
        assert result["confidence"] >= LOW_CONFIDENCE_THRESHOLD
        assert result["low_confidence"] is False

    def test_weak_signal_is_low_confidence(self):
        result = classify("Tell me about AI")
        assert result["low_confidence"] is True

    def test_low_confidence_provides_fallback(self):
        result = classify("Tell me about AI")
        assert result["low_confidence"] is True
        assert result["fallback_mode"] is not None

    def test_high_confidence_has_no_fallback(self):
        result = classify("Which authors collaborated on attention mechanisms and RLHF?")
        if not result["low_confidence"]:
            assert result["fallback_mode"] is None


# ── classify: edge cases ───────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_query_returns_a_mode(self):
        result = classify("")
        assert result["primary_mode"] in {"graph", "community", "vector", "web"}

    def test_case_insensitive(self):
        lower = classify("what is rag?")
        upper = classify("WHAT IS RAG?")
        assert lower["primary_mode"] == upper["primary_mode"]

    def test_none_history_treated_as_empty(self):
        result_none = classify("What is attention?", mode_history=None)
        result_empty = classify("What is attention?", mode_history=[])
        assert result_none["primary_mode"] == result_empty["primary_mode"]


# ── INTENT_SIGNALS sanity ──────────────────────────────────────────────────

class TestIntentSignals:
    def test_all_three_modes_have_signals(self):
        assert set(INTENT_SIGNALS.keys()) == {"graph", "community", "vector"}

    def test_each_mode_has_at_least_five_signals(self):
        for mode, signals in INTENT_SIGNALS.items():
            assert len(signals) >= 5, f"{mode} has fewer than 5 signals"

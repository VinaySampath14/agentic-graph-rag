"""Unit tests for the v4 failure -> rewrite -> reroute state transition."""
from types import SimpleNamespace

from src.agent import nodes
from src.retrievers.models import GradeResult, RetrievalResult


def test_failed_vector_context_is_rewritten_for_an_untried_mode(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(
            content="Which graph entities describe the BERT method?"
        ))]
    )
    completions = SimpleNamespace(create=lambda **_: response)
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    monkeypatch.setattr(nodes, "_get_groq", lambda: fake_client)

    state = {
        "query": "What is BERT?",
        "rewritten_query": "What is BERT?",
        "intent": "vector",
        "graph_backend": None,
        "retrieved_context": RetrievalResult(
            context_text="irrelevant context", source_type="vector"
        ),
        "grade_result": GradeResult(passed=False, reason="insufficient evidence"),
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

    result = nodes.node_rewrite_query(state)

    assert result["rewritten_query"] == "Which graph entities describe the BERT method?"
    assert result["mode_history"] == ["vector"]
    assert result["loop_count"] == 1
    assert result["agent_trace"][-1]["node"] == "rewrite_query"
    assert result["agent_trace"][-1]["extras"]["next_mode"] == "graph"
    assert result["agent_trace"][-1]["extras"]["prompt_version"] == "rewrite_query_v3"

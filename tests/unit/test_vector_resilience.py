"""Tests for graceful fallback in vector retrieval."""

from types import SimpleNamespace

import pytest

from src.retrievers import naive_retriever


class _Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def query_points(self, **kwargs):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_qdrant_connection_reset_is_retried(monkeypatch):
    response = SimpleNamespace(points=[])
    client = _Client([ConnectionResetError(104, "Connection reset by peer"), response])
    monkeypatch.setattr(naive_retriever.time, "sleep", lambda _: None)

    result = naive_retriever._query_points_with_retry(client, query=[0.1])

    assert result is response
    assert client.calls == 2


def test_non_transient_qdrant_error_is_not_retried(monkeypatch):
    client = _Client([ValueError("invalid vector dimensions")])
    monkeypatch.setattr(naive_retriever.time, "sleep", lambda _: None)

    with pytest.raises(ValueError, match="invalid vector dimensions"):
        naive_retriever._query_points_with_retry(client, query=[0.1])

    assert client.calls == 1

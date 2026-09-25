"""Tests for explicit method evidence in Neo4j retrieval context."""

from src.retrievers.graph_retriever import _serialise_results, _traverse_from_method


class _Result:
    def data(self):
        return []


class _Session:
    def __init__(self):
        self.query = ""
        self.parameters = {}

    def run(self, query, **parameters):
        self.query = query
        self.parameters = parameters
        return _Result()


def test_method_traversal_returns_the_matched_method():
    session = _Session()

    _traverse_from_method("LoRA", (None, None), session)

    assert "m.name AS matched_method" in session.query
    assert session.parameters == {"name": "LoRA"}


def test_method_context_states_explicit_relationship():
    context = _serialise_results(
        [
            {
                "arxiv_id": "2605.30232",
                "title": "Example paper",
                "year": 2026,
                "matched_method": "LoRA",
                "authors": ["Author One", "Author Two"],
            }
        ],
        "LoRA",
    )

    assert "Explicit USES_METHOD link: LoRA" in context
    assert "Authors: Author One, Author Two" in context

"""Tests for safe Neo4j/Lucene full-text query construction."""
from src.retrievers.graph_retriever import _fuzzy_entity_search, _lucene_search_term


def test_hyphenated_entity_is_quoted_as_literal_tokens():
    assert _lucene_search_term("ZebraTune-X") == '"ZebraTune"'


def test_lucene_control_characters_are_removed():
    term = _lucene_search_term('method: (LoRA+QLoRA) / "test"')
    assert term == '"method" AND "LoRA" AND "QLoRA" AND "test"'


def test_empty_search_does_not_call_neo4j():
    class SessionThatMustNotRun:
        def run(self, *_args, **_kwargs):
            raise AssertionError("Neo4j should not be called for an empty search term")

    assert _fuzzy_entity_search("---", SessionThatMustNotRun()) == []

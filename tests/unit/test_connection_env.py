"""Regression tests for whitespace accidentally saved in deployment secrets."""

from src.agent.connections import clean_env_value


def test_clean_env_value_removes_real_line_breaks(monkeypatch):
    monkeypatch.setenv("TEST_SERVICE_URL", "https://example.test\n")

    assert clean_env_value("TEST_SERVICE_URL") == "https://example.test"


def test_clean_env_value_removes_literal_newline_sequence(monkeypatch):
    monkeypatch.setenv("TEST_SERVICE_URL", "https://example.test\\n")

    assert clean_env_value("TEST_SERVICE_URL") == "https://example.test"

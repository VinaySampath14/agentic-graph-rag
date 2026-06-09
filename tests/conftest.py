"""Stub out heavy optional dependencies so unit tests run without GPU/ML installs."""
import sys
from unittest.mock import MagicMock

_STUBS = [
    "fastembed",
    "fastembed.sparse",
    "fastembed.sparse.splade",
    "FlagEmbedding",
    "sentence_transformers",
    "qdrant_client",
    "qdrant_client.models",
    "qdrant_client.http",
    "qdrant_client.http.models",
    "neo4j",
    "groq",
    "spacy",
    "tavily",
    "ddgs",
    "langchain_community",
    "langchain_community.tools",
    "langchain_community.tools.tavily_search",
    "numpy",
]

for _mod in _STUBS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

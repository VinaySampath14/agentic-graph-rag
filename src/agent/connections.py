"""Singleton connections — initialised once at startup, reused across all requests."""
import os

from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

load_dotenv()

_neo4j_driver = None
_qdrant_client = None


def get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
        )
        _neo4j_driver.verify_connectivity()
    return _neo4j_driver


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
        )
    return _qdrant_client


def close_all() -> None:
    global _neo4j_driver
    if _neo4j_driver is not None:
        _neo4j_driver.close()
        _neo4j_driver = None

"""Singleton connections and shared models initialized once at startup."""
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient

load_dotenv()

_neo4j_driver = None
_qdrant_client = None
_dense_model = None
_ontology_graph = None

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONTOLOGY_FILE = PROJECT_ROOT / "ontology" / "arxiv_cs_populated.ttl"


def clean_env_value(name: str) -> str:
    """Return an environment value without accidental line breaks."""
    return (
        os.environ[name]
        .replace("\\n", "")
        .replace("\r", "")
        .replace("\n", "")
        .strip()
    )


def get_neo4j_driver():
    global _neo4j_driver
    if _neo4j_driver is None:
        _neo4j_driver = GraphDatabase.driver(
            clean_env_value("NEO4J_URI"),
            auth=(clean_env_value("NEO4J_USER"), clean_env_value("NEO4J_PASSWORD")),
        )
        _neo4j_driver.verify_connectivity()
    return _neo4j_driver


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=clean_env_value("QDRANT_URL"),
            api_key=clean_env_value("QDRANT_API_KEY"),
        )
    return _qdrant_client


def get_dense_model():
    global _dense_model
    if _dense_model is None:
        from FlagEmbedding import FlagModel
        _dense_model = FlagModel("BAAI/bge-m3", use_fp16=True, normalize_embeddings=True)
    return _dense_model


def get_ontology_graph():
    global _ontology_graph
    if _ontology_graph is None:
        from rdflib import Graph
        _ontology_graph = Graph()
        _ontology_graph.parse(ONTOLOGY_FILE, format="turtle")
        print(f"Ontology graph loaded: {len(_ontology_graph)} triples")
    return _ontology_graph


def close_all() -> None:
    global _neo4j_driver
    if _neo4j_driver is not None:
        _neo4j_driver.close()
        _neo4j_driver = None

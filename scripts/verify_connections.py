"""Verify all four service connections. Run before any ingestion or eval work."""
import os
from dotenv import load_dotenv

load_dotenv()


def check_neo4j() -> None:
    from neo4j import GraphDatabase

    uri = os.environ["NEO4J_URI"]
    user = os.environ["NEO4J_USER"]
    password = os.environ["NEO4J_PASSWORD"]
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    driver.close()


def check_qdrant() -> None:
    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )
    client.get_collections()


def check_groq() -> None:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    client.models.list()


def check_tavily() -> None:
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    client.search("test", max_results=1)


CHECKS = {
    "Neo4j": check_neo4j,
    "Qdrant": check_qdrant,
    "Groq": check_groq,
    "Tavily": check_tavily,
}

if __name__ == "__main__":
    for name, fn in CHECKS.items():
        try:
            fn()
            print(f"✓ {name}")
        except Exception as e:
            print(f"✗ {name}: {e}")

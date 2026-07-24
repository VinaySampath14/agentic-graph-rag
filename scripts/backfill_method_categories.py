"""Write method category classifications back onto Neo4j Method nodes.

Closes the gap discussed in the ontology rebuild: category-lookup questions
("what are the fine-tuning methods?") don't need SPARQL/the ontology at all —
they need a property on the Method node. This backfills m.category from
data/processed/method_classifications.json so plain Cypher can answer them.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

CLASSIFICATIONS_FILE = Path("data/processed/method_classifications.json")


def get_driver():
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )


def backfill(session) -> tuple[int, int]:
    classifications = json.loads(CLASSIFICATIONS_FILE.read_text(encoding="utf-8"))

    matched = 0
    unmatched = []
    for method_name, info in classifications.items():
        result = session.run(
            "MATCH (m:Method {name: $name}) SET m.category = $category RETURN m",
            name=method_name, category=info["subclass"],
        )
        if result.peek() is not None:
            matched += 1
        else:
            unmatched.append(method_name)

    return matched, len(unmatched)


def main() -> None:
    driver = get_driver()
    with driver.session() as session:
        matched, unmatched_count = backfill(session)

        total = session.run("MATCH (m:Method) RETURN count(m) AS n").single()["n"]
        categorized = session.run(
            "MATCH (m:Method) WHERE m.category IS NOT NULL RETURN count(m) AS n"
        ).single()["n"]

    driver.close()

    print(f"Matched and updated: {matched} methods")
    print(f"Classifications with no matching Neo4j node: {unmatched_count}")
    print(f"Total Method nodes: {total}")
    print(f"Method nodes now with a category: {categorized} ({categorized/total:.0%})")


if __name__ == "__main__":
    main()

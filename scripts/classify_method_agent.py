"""Agentic propose-validate-write loop for method classification.

This is the pattern the write/validate piece of the ontology work was scoped
around: an LLM proposes a graph write (a method's category), the proposal is
checked against ontology/shapes.ttl (SHACL) BEFORE anything is written, and on
rejection the violation is fed back to the LLM for a corrected retry — the
same shape as node_rewrite_query in the main agent, applied to a write
instead of a retrieval mode.

Usage:
    python scripts/classify_method_agent.py "Method Name" ["Another Method"]
"""
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase
from pyshacl import validate
from rdflib import Graph, Namespace, RDF

load_dotenv()

EX = Namespace("http://arxiv-cs.org/ontology#")
POPULATED_FILE = Path("ontology/arxiv_cs_populated.ttl")
SHAPES_FILE = Path("ontology/shapes.ttl")
CLASSIFICATIONS_FILE = Path("data/processed/method_classifications.json")

VALID_CATEGORIES = [
    "FineTuningMethod", "AttentionMethod", "AlignmentMethod", "ReasoningMethod",
    "RetrievalMethod", "PersonalizationMethod", "AgentSkillLearningMethod",
]
MAX_ATTEMPTS = 3

_groq: Groq | None = None


def get_groq() -> Groq:
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.environ["GROQ_API_KEY"].strip())
    return _groq


def get_neo4j_driver():
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )


def uri_method(name: str):
    return EX[f"method_{quote(name.strip(), safe='')}"]


def propose_category(method_name: str, feedback: str | None = None) -> str:
    prompt = (
        f"Classify this AI/ML method into exactly one of these 7 categories:\n"
        f"{', '.join(VALID_CATEGORIES)}\n\n"
        f'Method: "{method_name}"\n'
    )
    if feedback:
        prompt += f"\nYour previous proposal was rejected: {feedback}\nPropose a corrected category.\n"
    prompt += "\nRespond with only the category name, nothing else."

    response = get_groq().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()


def validate_proposal(method_name: str, category: str) -> tuple[bool, str]:
    """Add the proposed triples to a copy of the real graph and check SHACL
    conformance for this method specifically — catches hallucinated category
    names and exclusivity violations before anything gets written."""
    data_graph = Graph()
    data_graph.parse(POPULATED_FILE, format="turtle")

    method_uri = uri_method(method_name)
    data_graph.add((method_uri, RDF.type, EX.Method))
    data_graph.add((method_uri, RDF.type, EX[category]))

    shapes_graph = Graph()
    shapes_graph.parse(SHAPES_FILE, format="turtle")

    _, results_graph, _ = validate(data_graph, shacl_graph=shapes_graph, advanced=True)

    query = f"""
    PREFIX sh: <http://www.w3.org/ns/shacl#>
    SELECT ?message WHERE {{
        ?r a sh:ValidationResult ; sh:focusNode <{method_uri}> ; sh:resultMessage ?message .
    }}
    """
    messages = [str(row.message) for row in results_graph.query(query)]
    if messages:
        return False, "; ".join(messages)
    return True, ""


def write_classification(method_name: str, category: str) -> None:
    driver = get_neo4j_driver()
    with driver.session() as session:
        session.run(
            "MATCH (m:Method {name: $name}) SET m.category = $category",
            name=method_name, category=category,
        )
    driver.close()

    data = json.loads(CLASSIFICATIONS_FILE.read_text(encoding="utf-8"))
    data[method_name] = {
        "method": method_name,
        "subclass": category,
        "confidence": 1.0,
        "reason": "Classified by the propose-validate-write agent (scripts/classify_method_agent.py).",
        "needs_review": False,
    }
    CLASSIFICATIONS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def classify_method(method_name: str) -> None:
    feedback = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        category = propose_category(method_name, feedback)
        print(f"  Attempt {attempt}: proposed '{category}'")

        valid, report = validate_proposal(method_name, category)
        if valid:
            write_classification(method_name, category)
            print(f"  VALID -- written: {method_name} -> {category}")
            return

        print(f"  REJECTED: {report}")
        feedback = report

    print(f"  FAILED after {MAX_ATTEMPTS} attempts -- leaving '{method_name}' unclassified.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python scripts/classify_method_agent.py "Method Name" [...]')
        sys.exit(1)
    for name in sys.argv[1:]:
        print(f"\nClassifying: {name}")
        classify_method(name)

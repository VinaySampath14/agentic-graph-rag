"""Validate the populated ontology against ontology/shapes.ttl.

This is the SHACL data-quality gate — distinct from the OWL reasoning step in
build_ontology.py. OWL states what should be true; this checks whether the
actual instance data satisfies it, and reports every violation found instead
of declaring the whole graph inconsistent.
"""
from pathlib import Path

from pyshacl import validate
from rdflib import Graph

DATA_FILE = Path("ontology/arxiv_cs_populated.ttl")
SHAPES_FILE = Path("ontology/shapes.ttl")


def main() -> None:
    print(f"Loading data graph from {DATA_FILE}...")
    data_graph = Graph()
    data_graph.parse(DATA_FILE, format="turtle")
    print(f"  {len(data_graph)} triples")

    print(f"Loading shapes graph from {SHAPES_FILE}...")
    shapes_graph = Graph()
    shapes_graph.parse(SHAPES_FILE, format="turtle")
    print(f"  {len(shapes_graph)} triples")

    print("\nRunning SHACL validation...\n")
    conforms, results_graph, results_text = validate(
        data_graph,
        shacl_graph=shapes_graph,
        advanced=True,  # required for sh:xone at node-shape level
        debug=False,
    )

    # Summarise by shape + severity rather than dumping the raw (huge) report
    query = """
    PREFIX sh: <http://www.w3.org/ns/shacl#>
    SELECT ?sourceShape ?severity (COUNT(*) AS ?n) WHERE {
        ?r a sh:ValidationResult ;
           sh:sourceShape ?sourceShape ;
           sh:resultSeverity ?severity .
    } GROUP BY ?sourceShape ?severity
    """
    print(f"conforms: {conforms}\n")
    print("Violation summary by shape:")
    for row in results_graph.query(query):
        shape = str(row.sourceShape).split("#")[-1]
        severity = str(row.severity).split("#")[-1]
        print(f"  {shape} [{severity}]: {row.n} nodes")

    # Sample a few individual messages per shape for concreteness
    sample_query = """
    PREFIX sh: <http://www.w3.org/ns/shacl#>
    SELECT ?focusNode ?sourceShape ?message WHERE {
        ?r a sh:ValidationResult ;
           sh:sourceShape ?sourceShape ;
           sh:focusNode ?focusNode ;
           sh:resultMessage ?message .
    } LIMIT 5
    """
    print("\nSample violations:")
    for row in results_graph.query(sample_query):
        node = str(row.focusNode).split("#")[-1]
        shape = str(row.sourceShape).split("#")[-1]
        print(f"  [{shape}] {node} — {row.message}")


if __name__ == "__main__":
    main()

"""Run a deterministic SPARQL smoke test against the populated ontology."""
from pathlib import Path

from rdflib import Graph


ONTOLOGY_FILE = Path(__file__).resolve().parents[1] / "ontology" / "arxiv_cs_populated.ttl"

LORA_CATEGORY_QUERY = """
PREFIX ex:   <http://arxiv-cs.org/ontology#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?method ?category ?categoryLabel ?parent
WHERE {
  ?method rdfs:label ?methodLabel ;
          rdf:type ?category .
  FILTER(LCASE(STR(?methodLabel)) = "lora")

  ?category rdfs:subClassOf ?parent .
  OPTIONAL { ?category rdfs:label ?categoryLabel }
  FILTER(?parent = ex:Method)
}
"""


def main() -> None:
    graph = Graph()
    graph.parse(ONTOLOGY_FILE, format="turtle")
    rows = list(graph.query(LORA_CATEGORY_QUERY))

    print(f"Ontology file: {ONTOLOGY_FILE}")
    print(f"Triples: {len(graph)}")
    print(f"LoRA category results: {len(rows)}")
    for row in rows:
        print(
            f"method={row.method} | category={row.category} | "
            f"category_label={row.categoryLabel} | parent={row.parent}"
        )

    if not rows:
        raise SystemExit("FAIL: LoRA has no Method subclass classification")


if __name__ == "__main__":
    main()

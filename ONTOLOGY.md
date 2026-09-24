# Ontology

This project uses an RDF/OWL ontology as one backend inside the user-facing **Graph** mode. It is intentionally separate from Neo4j at query time.

[README](README.md) | [Architecture](ARCHITECTURE.md) | [Contributing](CONTRIBUTING.md) | [Detailed scope](ontology/SCOPE.md)

## Why two graph backends?

| Backend | Query language | Purpose |
|---|---|---|
| Neo4j | Cypher | Explicit operational relationships: papers, authors, institutions, methods, and communities |
| RDFLib | SPARQL | Semantic classes, method categories, hierarchy, and inferred `relatedWork` relationships |

The router chooses one backend for each Graph request. It does not blend Neo4j and RDFLib results in a single retrieval attempt.

## Files

| File | Purpose |
|---|---|
| `ontology/arxiv_cs.ttl` | OWL/RDFS schema |
| `ontology/arxiv_cs_populated.ttl` | Schema plus exported instances and inferred triples |
| `ontology/shapes.ttl` | SHACL data-quality rules |
| `data/processed/method_classifications_reviewed.json` | Reviewed method-to-category assertions used by the build |
| `data/processed/method_classifications.json` | Original LLM classifications retained for provenance |
| `src/ingestion/build_ontology.py` | Neo4j-to-RDF export, reviewed assertions, inference, and serialization |
| `src/retrievers/ontology_retriever.py` | Schema-aware SPARQL generation and RDFLib execution |

## Class model

Top-level classes:

- `Paper`
- `Author`
- `Institution`
- `Method`
- `Community`

Reviewed method subclasses:

- `FineTuningMethod`
- `AttentionMethod`
- `AlignmentMethod`
- `ReasoningMethod`
- `RetrievalMethod`
- `PersonalizationMethod`
- `AgentSkillLearningMethod`

The seven method subclasses are declared disjoint. A reviewed method must not belong to more than one category.

## Important properties

| Property | Meaning |
|---|---|
| `authoredBy` | Paper to author |
| `authorOf` | Inverse of `authoredBy` |
| `usesMethod` | Paper to method |
| `usedByPaper` | Inverse of `usesMethod` |
| `fromInstitution` | Author to institution |
| `belongsToCommunity` | Paper to Leiden community |
| `relatedWork` | Symmetric ontology-only relationship between papers sharing a reviewed method category |

## Build and reasoning

Run:

```powershell
python -m src.ingestion.build_ontology
```

The build:

1. Loads the schema.
2. Exports the current Neo4j corpus into RDF.
3. Applies only reviewed method classifications.
4. Creates symmetric `relatedWork` assertions for papers sharing a reviewed category.
5. Runs RDFS closure with `owlrl`.
6. Writes `ontology/arxiv_cs_populated.ttl`.

Current snapshot:

- 227,077 total triples
- 2,000 papers
- 9,250 authors
- 2,988 institutions
- 286 methods
- 13 communities
- 59 reviewed category assertions across corpus methods
- 82,582 `relatedWork` triples

## SHACL policy

Validate with:

```powershell
python scripts/validate_shacl.py
```

Hard failures:

- A paper has no author.
- A method belongs to multiple disjoint categories.

Warnings:

- A method has not yet received a reviewed category.
- An author's institution is unavailable.

The current graph conforms when warnings are allowed. This policy avoids forcing uncertain method names into incorrect semantic classes.

## Example SPARQL

```sparql
PREFIX ex: <http://arxiv-cs.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT DISTINCT ?methodLabel WHERE {
  ?method rdf:type ex:FineTuningMethod ;
          rdfs:label ?methodLabel .
}
ORDER BY ?methodLabel
```

Natural-language demo questions:

- "What type of method is LoRA?"
- "What are the different fine-tuning methods?"
- "Which methods are classified as alignment methods?"

For coverage details and limitations, see [ontology/SCOPE.md](ontology/SCOPE.md).

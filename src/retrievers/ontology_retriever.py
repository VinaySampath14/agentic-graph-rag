"""SPARQL-based retriever over the RDFLib in-memory ontology graph."""
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from rdflib import Graph, OWL, RDF
from rdflib.plugins.sparql.parser import parseQuery

from src.retrievers.models import RetrievalResult

load_dotenv()

PROMPTS_DIR = Path("prompts")

SCHEMA_SUMMARY = """
Classes:
  ex:Paper, ex:Author (subClassOf foaf:Person), ex:Institution (subClassOf foaf:Organization), ex:Community
  ex:Method (parent)
    ex:FineTuningMethod        — LoRA, QLoRA, AdaLoRA, instruction tuning, optimisers
    ex:AttentionMethod         — Transformer, FlashAttention, MoE, SSM, ViT
    ex:AlignmentMethod         — RLHF, DPO, PPO, RLAIF
    ex:ReasoningMethod         — Chain-of-Thought, RAG, GraphRAG, LangGraph, GNN, benchmarks
    ex:RetrievalMethod         — BM25, DPR, ColBERT, FAISS, Qdrant, Milvus
    ex:PersonalizationMethod   — user preference modeling, personalized modeling, proactive interaction
    ex:AgentSkillLearningMethod — agent skill evolution/acquisition, trajectory-based training, verifier feedback
  The 7 Method subclasses are mutually exclusive (owl:AllDisjointClasses) — a
  Method belongs to at most one.

Object properties:
  ex:authoredBy          Paper -> Author        (inverse: ex:authorOf)
  ex:usesMethod          Paper -> Method (or subclass)  (inverse: ex:usedByPaper)
  ex:fromInstitution     Author -> Institution   (inverse: ex:hasAffiliatedAuthor)
  ex:belongsToCommunity  Paper -> Community
  ex:relatedWork         Paper -> Paper  (inferred — papers sharing a method subclass)

Datatype properties on Paper: ex:arxivId, ex:title, ex:year, ex:venue
Datatype properties on Author: ex:authorName
Datatype properties on Method: ex:methodName

Namespace prefixes: ex: <http://arxiv-cs.org/ontology#>, foaf: <http://xmlns.com/foaf/0.1/>
"""

FALLBACK_SPARQL = """
PREFIX ex:   <http://arxiv-cs.org/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
SELECT ?label ?type WHERE {
  ?entity rdf:type ?type .
  ?entity rdfs:label ?label .
  FILTER(?type IN (ex:Paper, ex:Author, ex:Method))
} LIMIT 10
"""

_groq: Groq | None = None
_known_terms: set[str] | None = None
_driver = None

EX_TERM_PATTERN = re.compile(r"\bex:(\w+)\b")

# Category-listing questions ("what are the fine-tuning methods?") are plain
# lookups against the m.category property backfilled onto Neo4j Method nodes
# (scripts/backfill_method_categories.py) — no reasoning involved, so they're
# answered directly via Cypher instead of going through LLM-generated SPARQL.
CATEGORY_KEYWORDS = {
    "fine-tuning": "FineTuningMethod", "fine tuning": "FineTuningMethod", "finetuning": "FineTuningMethod",
    "alignment": "AlignmentMethod",
    "attention": "AttentionMethod",
    "reasoning": "ReasoningMethod",
    "retrieval": "RetrievalMethod",
    "personalization": "PersonalizationMethod", "personalized": "PersonalizationMethod",
    "skill learning": "AgentSkillLearningMethod", "skill evolution": "AgentSkillLearningMethod",
    "agent skill": "AgentSkillLearningMethod",
}
LISTING_SIGNALS = ["what are", "which are", "list", "show me", "what methods", "which methods"]


def get_graph() -> Graph:
    from src.agent.connections import get_ontology_graph
    return get_ontology_graph()


def _get_driver():
    global _driver
    if _driver is None:
        from src.agent.connections import get_neo4j_driver
        _driver = get_neo4j_driver()
    return _driver


def _detect_category_listing(query: str) -> str | None:
    query_lower = query.lower()
    if not any(signal in query_lower for signal in LISTING_SIGNALS):
        return None
    for keyword, category in CATEGORY_KEYWORDS.items():
        if keyword in query_lower:
            return category
    return None


def _neo4j_category_lookup(category: str) -> RetrievalResult:
    driver = _get_driver()
    with driver.session() as session:
        rows = session.run(
            "MATCH (m:Method) WHERE m.category = $category RETURN m.name AS name ORDER BY m.name",
            category=category,
        ).data()

    names = [r["name"] for r in rows]
    if names:
        context = f"Methods in category '{category}' ({len(names)} found):\n" + "\n".join(f"- {n}" for n in names)
    else:
        context = f"No methods found with category '{category}'."

    return RetrievalResult(
        context_text=context,
        source_type="ontology",
        source_metadata={"lookup_type": "neo4j_category_property", "category": category},
    )


def get_groq() -> Groq:
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.environ["GROQ_API_KEY"].strip())
    return _groq


def _load_prompt() -> str:
    path = PROMPTS_DIR / "ontology_sparql_v1.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(l for l in lines if not l.startswith("#")).strip()


def _generate_sparql(query: str) -> str:
    prompt_template = _load_prompt()
    prompt = (
        prompt_template
        .replace("{query}", query)
        .replace("{schema}", SCHEMA_SUMMARY)
    )
    response = get_groq().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    sparql = response.choices[0].message.content.strip()
    # Strip markdown fences if model ignores the instruction
    if sparql.startswith("```"):
        sparql = "\n".join(
            l for l in sparql.splitlines()
            if not l.startswith("```")
        ).strip()
    return sparql


def _known_schema_terms() -> set[str]:
    """Local names of every ex: class/property actually declared in the ontology."""
    global _known_terms
    if _known_terms is not None:
        return _known_terms
    g = get_graph()
    terms: set[str] = set()
    for cls in (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.SymmetricProperty):
        for s in g.subjects(RDF.type, cls):
            if "arxiv-cs.org/ontology#" in str(s):
                terms.add(str(s).split("#")[-1])
    _known_terms = terms
    return terms


def _unknown_terms(sparql: str) -> list[str]:
    """ex: terms referenced in the query that aren't declared anywhere in the schema."""
    referenced = set(EX_TERM_PATTERN.findall(sparql))
    return sorted(referenced - _known_schema_terms())


def _validate_sparql(sparql: str) -> tuple[bool, list[str]]:
    try:
        parseQuery(sparql)
    except Exception:
        return False, []
    unknown = _unknown_terms(sparql)
    return not unknown, unknown


def _regenerate_sparql(query: str, unknown_terms: list[str]) -> str:
    prompt_template = _load_prompt()
    correction = (
        f"\n\nYour previous attempt referenced these terms, which do not exist "
        f"in the schema: {', '.join(unknown_terms)}. Use only classes and "
        f"properties listed in the schema above. Regenerate the query."
    )
    prompt = (
        prompt_template
        .replace("{query}", query)
        .replace("{schema}", SCHEMA_SUMMARY)
        + correction
    )
    response = get_groq().chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    sparql = response.choices[0].message.content.strip()
    if sparql.startswith("```"):
        sparql = "\n".join(l for l in sparql.splitlines() if not l.startswith("```")).strip()
    return sparql


def _execute_sparql(sparql: str) -> list[dict]:
    g = get_graph()
    try:
        results = []
        qres = g.query(sparql)
        for row in qres:
            results.append({
                str(var): str(val) if val is not None else ""
                for var, val in zip(qres.vars, row)
            })
        return results
    except Exception as e:
        return [{"error": str(e)}]


def _format_results(results: list[dict], query: str) -> str:
    if not results:
        return "Ontology query returned no results."

    if "error" in results[0]:
        return f"SPARQL execution error: {results[0]['error']}"

    lines = [f"Ontology results for: {query}", f"({len(results)} items found)\n"]
    for i, row in enumerate(results[:20], 1):
        # Clean up URIs to readable names
        clean = {}
        for k, v in row.items():
            if "#" in v:
                from urllib.parse import unquote
                v = unquote(v.split("#")[-1]).replace("_", " ")
            clean[k] = v
        lines.append(f"{i}. " + " | ".join(f"{k}: {v}" for k, v in clean.items()))

    return "\n".join(lines)


def retrieve(query: str) -> RetrievalResult:
    category = _detect_category_listing(query)
    if category:
        return _neo4j_category_lookup(category)

    sparql = _generate_sparql(query)
    valid, unknown = _validate_sparql(sparql)

    if not valid and unknown:
        # Semantically invalid (hallucinated terms) but syntactically fine —
        # give the model one corrective retry before giving up.
        sparql = _regenerate_sparql(query, unknown)
        valid, _ = _validate_sparql(sparql)

    if not valid:
        sparql = FALLBACK_SPARQL

    results = _execute_sparql(sparql)
    context = _format_results(results, query)

    return RetrievalResult(
        context_text=context,
        source_type="ontology",
        sparql_query_used=sparql,
        truncated=len(results) > 20,
    )

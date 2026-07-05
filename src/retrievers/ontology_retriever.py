"""SPARQL-based retriever over the RDFLib in-memory ontology graph."""
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from rdflib import Graph
from rdflib.plugins.sparql.parser import parseQuery

from src.retrievers.models import RetrievalResult

load_dotenv()

ONTOLOGY_FILE = Path("ontology/arxiv_cs_populated.ttl")
PROMPTS_DIR   = Path("prompts")

SCHEMA_SUMMARY = """
Classes:
  ex:Paper, ex:Author, ex:Institution, ex:Community
  ex:Method (parent)
    ex:FineTuningMethod  — LoRA, QLoRA, AdaLoRA, instruction tuning, optimisers
    ex:AttentionMethod   — Transformer, FlashAttention, MoE, SSM, ViT
    ex:AlignmentMethod   — RLHF, DPO, PPO, RLAIF
    ex:ReasoningMethod   — Chain-of-Thought, RAG, GraphRAG, LangGraph, GNN, benchmarks
    ex:RetrievalMethod   — BM25, DPR, ColBERT, FAISS, Qdrant, Milvus

Object properties:
  ex:authoredBy          Paper -> Author
  ex:usesMethod          Paper -> Method (or subclass)
  ex:fromInstitution     Author -> Institution
  ex:belongsToCommunity  Paper -> Community
  ex:relatedWork         Paper -> Paper  (inferred — papers sharing a method subclass)

Datatype properties on Paper: ex:arxivId, ex:title, ex:year, ex:venue
Datatype properties on Author: ex:authorName
Datatype properties on Method: ex:methodName

Namespace prefix: ex: <http://arxiv-cs.org/ontology#>
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

_graph: Graph | None = None
_groq:  Groq   | None = None


def get_graph() -> Graph:
    global _graph
    if _graph is None:
        print("Loading ontology graph...")
        _graph = Graph()
        _graph.parse(ONTOLOGY_FILE, format="turtle")
        print(f"Ontology loaded: {len(_graph)} triples")
    return _graph


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


def _validate_sparql(sparql: str) -> bool:
    try:
        parseQuery(sparql)
        return True
    except Exception:
        return False


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
    sparql = _generate_sparql(query)

    if not _validate_sparql(sparql):
        sparql = FALLBACK_SPARQL

    results = _execute_sparql(sparql)
    context = _format_results(results, query)

    return RetrievalResult(
        context_text=context,
        source_type="ontology",
        sparql_query_used=sparql,
        truncated=len(results) > 20,
    )

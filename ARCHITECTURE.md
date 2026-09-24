# Architecture

Agentic Graph RAG is a self-correcting retrieval engine over 2,000 arXiv CS papers. A LangGraph state machine exposes three user-facing modes—Vector, Graph, and Community—grades retrieved context, and rewrites failed queries for an untried mode. Graph mode selects exactly one of two independent backends: Neo4j/Cypher for explicit relationships or RDFLib/SPARQL for ontology semantics.

[README](README.md) | [Ontology](ONTOLOGY.md) | [Contributing](CONTRIBUTING.md) | [Detailed ontology scope](ontology/SCOPE.md)

---

## Directory Structure

```
agentic-graph-rag/
├── app.py                        # HuggingFace Spaces entrypoint (Gradio)
├── src/
│   ├── agent/
│   │   ├── state.py              # AgentState TypedDict
│   │   ├── nodes.py              # All 9 LangGraph node functions
│   │   ├── graph.py              # StateGraph wiring + conditional edges
│   │   ├── connections.py        # Shared singletons (Neo4j, Qdrant, BGE-M3)
│   │   └── ablations.py          # v1/v2/v3 ablation graph variants
│   ├── retrievers/
│   │   ├── router.py             # Rule-based intent classifier
│   │   ├── naive_retriever.py    # Qdrant hybrid search (BGE-M3 + BM25 + RRF)
│   │   ├── graph_retriever.py    # Neo4j Cypher traversal
│   │   ├── ontology_retriever.py # RDFLib/SPARQL semantic queries (Graph backend)
│   │   ├── community_retriever.py# Leiden community embedding similarity
│   │   ├── web_retriever.py      # DuckDuckGo web search fallback
│   │   ├── context_budget.py     # Token-aware context truncation
│   │   └── models.py             # Shared Pydantic models
│   ├── api/
│   │   └── main.py               # FastAPI backend (/query, /health, /stats)
│   ├── ingestion/                # One-time corpus build pipeline
│   │   ├── fetch_papers.py       # arXiv API download
│   │   ├── normalise.py          # Text cleaning + metadata extraction
│   │   ├── neo4j_loader.py       # Paper/Author/Institution graph loading
│   │   ├── extract_entities.py   # spaCy NER for method nodes
│   │   ├── llm_extraction.py     # LLM-assisted method extraction (top-50)
│   │   ├── community_detection.py# Leiden algorithm over author graph
│   │   ├── embed_communities.py  # BGE-M3 embeddings for community nodes
│   │   ├── generate_summaries.py # Groq-generated community summaries
│   │   └── qdrant_loader.py      # Dense + sparse vector ingestion
│   └── eval/                     # Evaluation scripts
├── scripts/                      # One-off utility scripts
├── tests/
│   ├── unit/                     # Pure logic tests (no network)
│   └── integration/              # Live connection tests (skipped in CI)
├── data/
│   ├── eval/                     # JSONL eval results + RAGAS scores
│   └── processed/                # Community embedding cache
└── figures/                      # Result plots (fig1–fig4)
```

---

## Agent Loop

```
query_analyser → router → [retriever] → grade_context
                    ↑                        │
                    │         pass           ▼
               rewrite_query          generator → grade_answer → END
                    ↑         fail          │
                    └──────────────────     ▼
                              (loop ≥ 3)  web_retriever
                                               │ fail
                                          force_refusal → END
```

### Nodes

| Node | Role |
|------|------|
| `query_analyser` | Classifies OOD queries (refuses immediately); extracts intent |
| `router` | Rule-based dispatch to vector / graph / community based on intent signals |
| `naive_retriever` | Qdrant hybrid: BGE-M3 dense + BM25 sparse, RRF fusion, cross-encoder rerank |
| `local_graph_retriever` | Neo4j Cypher for explicit relationships and entity traversal |
| `ontology_retriever` | RDFLib/SPARQL for ontology classes and inferred relationships; internally part of Graph mode |
| `global_retriever` | BGE-M3 cosine similarity against pre-embedded Leiden community nodes |
| `web_retriever` | DuckDuckGo search, triggered only after corpus modes are exhausted |
| `grade_context` | Binary LLM judge: is the retrieved context sufficient to answer? |
| `rewrite_query` | Mode-aware reformulation — entity-centric for graph, trend-oriented for community |
| `generator` | Configurable Groq model (default: GPT-OSS 120B) for grounded answer synthesis and citation extraction |
| `grade_answer` | Groundedness check: is the answer supported by the context? |
| `force_refusal` | Structured refusal with reason after all modes exhausted |

### Loop control

`AgentState.loop_count` increments on every `rewrite_query` call. `mode_history` records failed corpus modes and excludes them from redispatch. After Vector, Graph, and Community are exhausted, the router uses `web_retriever`; if web context is also insufficient, `force_refusal` returns a structured refusal.

---

## Retrieval Modes

### Vector (naive_retriever)
Qdrant hybrid search combining BGE-M3 dense embeddings and BM25 sparse vectors, fused with Reciprocal Rank Fusion (RRF). Results are reranked by a cross-encoder (ms-marco-MiniLM-L-6-v2). Best for factual and definitional queries.

### Graph
Graph is one user-facing mode with two separate internal backends. The router does not blend their results in one retrieval attempt.

**Neo4j/Cypher (`local_graph_retriever`)** handles explicit relationships with:
- spaCy NER + fuzzy string matching for entity resolution
- Adaptive hop depth (1–3 hops based on result count)
- Temporal filters (year, venue properties on edges)
- FULLTEXT index search as fallback when entity match fails

It is best for relational and authorship queries.

**RDFLib/SPARQL (`ontology_retriever`)** handles semantic questions about class membership, method categories, and inferred `relatedWork` relationships. SPARQL is generated from the question and validated against the declared ontology vocabulary before execution.

### Community (global_retriever)
Each of the 13 Leiden-detected research communities has a BGE-M3 embedding and a Groq-generated JSON summary (theme, dominant methods, key authors, representative papers). At query time, cosine similarity selects the top-3 communities. Best for thematic and trend queries.

### Web fallback (web_retriever)
DuckDuckGo search is used only after all three corpus modes are exhausted. Results pass through the same context and answer graders as local retrieval.

---

## Knowledge Graph

Built from 2,000 arXiv CS papers (CS.AI + CS.CL, 2026):

```
Node types:   Paper · Author · Institution · Method · Community
Edge types:   AUTHORED_BY · USES_METHOD · FROM_INSTITUTION · CITES

Counts:       2,000 Paper · 9,250 Author · 2,988 Institution
              286 Method · 13 Community
              10,651 AUTHORED_BY · 999 USES_METHOD · 2,000 BELONGS_TO
```

---

## Shared Singletons

`src/agent/connections.py` exposes lazy-loaded singletons for all heavy resources:

| Singleton | What it holds |
|-----------|--------------|
| `get_dense_model()` | BGE-M3 (FlagEmbedding) — shared by naive and community retrievers |
| `get_neo4j_driver()` | Neo4j AuraDB connection |
| `get_qdrant_client()` | Qdrant cloud client |
| `get_ontology_graph()` | Independent in-memory RDFLib graph loaded from the populated Turtle file |

The FastAPI lifespan handler (`src/api/main.py`) pre-warms all singletons at startup so the first query doesn't pay model-load time.

---

## Ablation Versions

| Version | Description |
|---------|-------------|
| v1 | Naive vector only — no routing, no loop |
| v2 | Static routing — dispatches to best mode, no loop |
| v3 | Agentic loop, no rewrite — retries with original query |
| v4 | Full system — loop + mode-aware query rewriting |

Ablation graphs live in `src/agent/ablations.py`.

---

## Evaluation

80 queries stratified by type (30 factual / 30 relational / 20 thematic). Each version is evaluated with RAGAS (faithfulness, answer relevancy, context precision, context recall) using GPT-4o-mini as judge. Results are stored in `data/eval/` as JSONL files.

See the [paper source](Agentic_graph_rag/main.tex) for full results.

---

## Current validation snapshot

- Qdrant: 2,000 paper records with dense and sparse vectors
- Neo4j: 2,000 papers, 9,250 authors, 286 methods, and 13 communities
- RDFLib: 227,077 triples and 59 reviewed method classifications
- SHACL: hard constraints conform; incomplete classifications and affiliations are warnings
- Unit tests: 59 passing

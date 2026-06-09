# Architecture

Agentic Graph RAG is a self-correcting retrieval engine over 2,000 arXiv CS papers. A LangGraph state machine routes queries between three retrieval backends, grades context quality, and rewrites failed queries into the vocabulary of the next mode — up to three correction loops before a structured refusal.

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
│   │   ├── community_retriever.py# Leiden community embedding similarity
│   │   ├── web_retriever.py      # Tavily web search fallback
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
| `local_graph_retriever` | Neo4j Cypher with fuzzy entity linking, adaptive hop depth, temporal filters |
| `global_retriever` | BGE-M3 cosine similarity against pre-embedded Leiden community nodes |
| `web_retriever` | Tavily search, triggered only at `loop_count == 3` |
| `grade_context` | Binary LLM judge: is the retrieved context sufficient to answer? |
| `rewrite_query` | Mode-aware reformulation — entity-centric for graph, trend-oriented for community |
| `generator` | Groq LLaMA 3.3 70B answer synthesis with citation extraction |
| `grade_answer` | Groundedness check: is the answer supported by the context? |
| `force_refusal` | Structured refusal with reason after all modes exhausted |

### Loop control

`AgentState.loop_count` increments on every `rewrite_query` call. At `loop_count == 3` the router diverts to `web_retriever`. At `loop_count == 4` (web also failed) `force_refusal` fires. `mode_history` tracks which retrieval modes have been tried and excludes them from re-dispatch.

---

## Retrieval Modes

### Vector (naive_retriever)
Qdrant hybrid search combining BGE-M3 dense embeddings and BM25 sparse vectors, fused with Reciprocal Rank Fusion (RRF). Results are reranked by a cross-encoder (ms-marco-MiniLM-L-6-v2). Best for factual and definitional queries.

### Graph (local_graph_retriever)
Neo4j Cypher traversal with:
- spaCy NER + fuzzy string matching for entity resolution
- Adaptive hop depth (1–3 hops based on result count)
- Temporal filters (year, venue properties on edges)
- FULLTEXT index search as fallback when entity match fails

Best for relational and authorship queries.

### Community (global_retriever)
Each of the 8 Leiden-detected research communities has a BGE-M3 embedding and a Groq-generated JSON summary (theme, dominant methods, key authors, representative papers). At query time, cosine similarity selects the top-3 communities. Best for thematic and trend queries.

### Web (web_retriever)
Tavily search, used only as a last resort at `loop_count == 3`. Results are formatted as context and passed to `grade_context` like any other retriever.

---

## Knowledge Graph

Built from 2,000 arXiv CS papers (CS.AI + CS.CL, 2026):

```
Node types:   Paper · Author · Institution · Method · Community
Edge types:   AUTHORED_BY · USES_METHOD · FROM_INSTITUTION · CITES

Counts:       2,000 Paper · 9,250 Author · 2,988 Institution
              36 Method · 8 Community
              10,651 AUTHORED_BY · 1,975 USES_METHOD · 4,532 FROM_INSTITUTION
```

---

## Shared Singletons

`src/agent/connections.py` exposes lazy-loaded singletons for all heavy resources:

| Singleton | What it holds |
|-----------|--------------|
| `get_dense_model()` | BGE-M3 (FlagEmbedding) — shared by naive and community retrievers |
| `get_neo4j_driver()` | Neo4j AuraDB connection |
| `get_qdrant_client()` | Qdrant cloud client |

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

See the [paper](Agentic_graph_rag/main.tex) for full results.

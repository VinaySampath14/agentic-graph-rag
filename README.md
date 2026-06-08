---
title: Agentic Graph RAG
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
python_version: "3.11"
---

# Agentic Graph RAG

> Self-correcting context engine over 2,000 arXiv CS papers — routes queries
> between vector, graph, and community retrieval modes, rewrites on failure,
> and explains every decision.

**[Live demo](#)** · **[arXiv preprint](#)** · **[HuggingFace Space](https://huggingface.co/spaces/VinaySampath/agentic-graph-rag)**

## Status

| Component | Status |
|-----------|--------|
| Corpus ingestion (2,000 arXiv CS papers) | ✅ Complete |
| Neo4j knowledge graph (14,282 nodes, 19,158 edges) | ✅ Complete |
| Community detection (Leiden, 8 communities) | ✅ Complete |
| Qdrant vector store (dense + sparse, BGE-M3 + BM25) | ✅ Complete |
| Three retrieval modes (vector, graph, community) | ✅ Complete |
| Rule-based router (100% on 20-query validation set) | ✅ Complete |
| LangGraph agentic loop with self-correction | ✅ Complete |
| FastAPI backend + Gradio demo | ✅ Complete |
| RAGAS evaluation + ablation study (80 queries, 4 versions) | ✅ Complete |

## What it does

Three retrieval modes wired into a LangGraph agentic loop:

- **Naive vector** — Qdrant hybrid search (BGE-M3 dense + BM25 sparse, RRF fusion)
- **Local graph** — Neo4j Cypher traversal with fuzzy entity linking, adaptive hop depth, temporal + venue filters
- **Global community** — Leiden-detected research clusters with Groq-generated summaries

When retrieval fails a binary quality grade, the agent rewrites the query to suit the next mode and re-routes — up to 3 correction loops before falling back to web search. Every routing decision is logged to an explainability trace.

Built on Neo4j · Qdrant · LangGraph · Groq LLaMA 3.3 70B · BGE-M3 · spaCy

## Knowledge Graph

```
Nodes:  2,000 Paper · 9,250 Author · 2,988 Institution · 36 Method · 8 Community
Edges:  10,651 AUTHORED_BY · 1,975 USES_METHOD · 4,532 FROM_INSTITUTION
```

Papers: arXiv CS.AI + CS.CL · Temporal CITES edges with year + venue properties

## Results

Full system (v4) — 80 queries across 3 types, GPT-4o-mini as RAGAS judge.

### Coverage by version

| Version | Overall | Factual | Relational | Thematic |
|---------|---------|---------|------------|----------|
| v1 — Naive vector | 37.5% | 93.3% | 6.7% | 0.0% |
| v2 — Static routing | 28.7% | 6.7% | 36.7% | 50.0% |
| v3 — Loop, no rewrite | 27.5% | 10.0% | 16.7% | 70.0% |
| **v4 — Full system** | **81.2%** | **96.7%** | **53.3%** | **100%** |

### v4 RAGAS scores by query type

| Query type | Coverage | Faithfulness | Answer relevancy | Context precision |
|------------|----------|-------------|-----------------|------------------|
| Factual    | 96.7%    | 0.966       | 0.890           | 0.907            |
| Relational | 53.3%    | 0.438       | 0.738           | 0.363            |
| Thematic   | 100.0%   | 0.812       | 0.685           | 0.660            |
| **Overall** | **81.2%** | **0.789** | **0.789**      | **0.697**        |

Key finding: adding a correction loop *without* query rewriting (v3) gives **no coverage gain** over naive retrieval (27.5% vs 37.5%). Adding **mode-aware rewriting** (v4) recovers coverage to **81.2%** — rewriting is the critical mechanism, not the loop structure.

## Quick start

```bash
git clone https://github.com/VinaySampath14/agentic-graph-rag.git
cd agentic-graph-rag
cp .env.example .env   # fill in your API keys
pip install -e ".[dev]"
python scripts/verify_connections.py
```

## Related work

Closest prior work: [arXiv:2508.05660](https://arxiv.org/abs/2508.05660).
This project extends it with an agentic self-correction loop, mode-aware query
rewriting, temporal graph edges, and per-query-type RAGAS evaluation.

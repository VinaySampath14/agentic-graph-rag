---
title: Agentic Graph RAG
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "4.44.1"
app_file: app.py
pinned: false
python_version: "3.11"
---

# Agentic Graph RAG: Self-Correcting Retrieval over Scientific Literature

[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg)](https://arxiv.org)
[![HF Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/VinaySampath/agentic-graph-rag)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **TL;DR** — A LangGraph agent over 2,000 arXiv CS papers that routes between vector, graph, and community retrieval, rewrites failed queries, and recovers coverage from 27.5% → 81.2% through mode-aware self-correction.

**[[Live Demo]](https://huggingface.co/spaces/VinaySampath/agentic-graph-rag) · [[Paper]](https://arxiv.org) · [[Architecture]](ARCHITECTURE.md)**

---

## Overview

Standard RAG systems pick one retrieval mode and have no recovery mechanism when it fails. This work asks: *can an agentic loop with mode-aware query rewriting recover queries that any single retrieval mode would refuse?*

We build a knowledge graph from 2,000 arXiv CS papers (CS.AI + CS.CL, 2026) and wire three retrieval backends into a LangGraph state machine. When a context quality grader rejects the retrieved context, the agent rewrites the query in the vocabulary of the next retrieval mode and re-routes — up to three correction loops. A four-version ablation isolates the contribution of each component.

**Key finding:** adding a correction loop *without* query rewriting (v3) gives no coverage improvement over naive retrieval (27.5% vs 37.5%). Adding mode-aware rewriting (v4) recovers coverage to **81.2%**. The gain is entirely attributable to rewriting, not to the loop structure or web fallback.

---

## Results

*80 queries · 30 factual / 30 relational / 20 thematic · GPT-4o-mini as RAGAS judge*

### Coverage across ablation versions

| Version | Overall | Factual | Relational | Thematic |
|---------|:-------:|:-------:|:----------:|:--------:|
| v1 — Naive vector | 37.5% | 93.3% | 6.7% | 0.0% |
| v2 — Static routing | 28.7% | 6.7% | 36.7% | 50.0% |
| v3 — Loop, no rewrite | 27.5% | 10.0% | 16.7% | 70.0% |
| **v4 — Full system** | **81.2%** | **96.7%** | **53.3%** | **100%** |

### v4 RAGAS by query type

| Query type | Coverage | Faithfulness | Ans. Relevancy | Ctx. Precision |
|------------|:--------:|:------------:|:--------------:|:--------------:|
| Factual | 96.7% | 0.966 | 0.890 | 0.907 |
| Relational | 53.3% | 0.438 | 0.738 | 0.363 |
| Thematic | 100.0% | 0.812 | 0.685 | 0.660 |
| **Overall** | **81.2%** | **0.789** | **0.789** | **0.697** |

### Loop efficiency (v4)

| Query type | Avg loops | 1st-mode success | Router accuracy |
|------------|:---------:|:----------------:|:---------------:|
| Factual | 0.28 | 82.8% | 86.7% |
| Relational | 0.63 | 62.5% | 100.0% |
| Thematic | 0.30 | 75.0% | 55.0% |
| **Overall** | **0.37** | **75.4%** | **83.8%** |

---

## System

![Agent Graph](figures/architecture.png)

| Retrieval mode | Backend | Best for |
|----------------|---------|----------|
| Vector | Qdrant hybrid (BGE-M3 dense + BM25 sparse, RRF + cross-encoder rerank) | Factual, definitional |
| Graph | Neo4j Cypher with fuzzy entity linking, adaptive hop depth, temporal filters | Relational, authorship |
| Community | BGE-M3 similarity over Leiden cluster embeddings + Groq summaries | Thematic, trend |

**Knowledge graph** — 2,000 Paper · 9,250 Author · 2,988 Institution · 36 Method · 8 Community · 17,158 edges

**Stack** — Neo4j AuraDB · Qdrant · LangGraph · Groq LLaMA 3.3 70B · BGE-M3 · spaCy · FastAPI · Gradio

---

## Quickstart

```bash
git clone https://github.com/VinaySampath14/agentic-graph-rag.git
cd agentic-graph-rag
pip install -e ".[dev]"
cp .env.example .env          # add Neo4j, Qdrant, Groq, Tavily keys
python scripts/verify_connections.py
python app.py                 # Gradio demo at localhost:7860
```

Run tests:

```bash
pytest tests/unit/ -v         # 44 unit tests, no credentials needed
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for a full walkthrough of the codebase and [CONTRIBUTING.md](CONTRIBUTING.md) for setup details.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{vudumula2026agenticgraphrag,
  title   = {Agentic Graph RAG: Self-Correcting Retrieval over Scientific Literature
             via Mode-Aware Query Rewriting and Agentic Loop Control},
  author  = {Vudumula, Vinay Sampath Kumar},
  year    = {2026},
  url     = {https://huggingface.co/spaces/VinaySampath/agentic-graph-rag}
}
```

---

## Related Work

Closest prior work: [arXiv:2508.05660](https://arxiv.org/abs/2508.05660).  
This work extends it with an agentic self-correction loop, mode-aware query rewriting, temporal graph edges, and per-query-type RAGAS evaluation across a controlled four-version ablation.

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

# Agentic Graph RAG: Self-Correcting Retrieval over Scientific Literature

[![HF Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/VinaySampath/agentic-graph-rag)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-008CC1?logo=neo4j)](https://neo4j.com/cloud/platform/aura-graph-database/)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector%20DB-red)](https://qdrant.tech/)

> **TL;DR** — A LangGraph agent over 2,000 arXiv CS papers that routes between Vector, Graph, and Community retrieval. Graph mode selects either Neo4j/Cypher for explicit relationships or RDFLib/SPARQL for ontology semantics. The agent grades context, rewrites failed queries, and recovers coverage through mode-aware self-correction.

**[Live Demo](https://huggingface.co/spaces/VinaySampath/agentic-graph-rag) · [Architecture](ARCHITECTURE.md) · [Ontology](ONTOLOGY.md) · [Contributing](CONTRIBUTING.md)**

---

## Overview

Standard RAG systems pick one retrieval mode and have no recovery mechanism when it fails. This work asks: *can an agentic loop with mode-aware query rewriting recover queries that any single retrieval mode would refuse?*

We build retrieval data from 2,000 arXiv CS papers (CS.AI + CS.CL, 2026) and expose three modes through a LangGraph state machine: Vector, Graph, and Community. Graph mode contains two deliberately separate paths: Neo4j/Cypher for stored relationships and RDFLib/SPARQL for OWL/RDFS semantics. When a context quality grader rejects the retrieved context, the agent rewrites the query for the next mode and re-routes — up to three correction loops. A four-version ablation isolates the contribution of each component.

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

![Architecture](figures/architecture.png)

| Retrieval mode | Backend | Best for |
|----------------|---------|----------|
| Vector | Qdrant hybrid (BGE-M3 dense + SPLADE sparse, RRF + cross-encoder rerank) | Factual, definitional |
| Graph | Neo4j/Cypher for explicit relationships; RDFLib/SPARQL for ontology semantics | Relational, authorship, method categories |
| Community | BGE-M3 similarity over Leiden cluster embeddings + Groq summaries | Thematic, trend |

**Knowledge graph** — 2,000 Paper · 9,250 Author · 3,003 Institution · 286 Method · 13 Community · ~50,000 edges

**Ontology** — 7 method subclasses · 227,077 RDF triples · 59/286 corpus methods have reviewed classifications · 598/2,000 papers connected to methods

**Stack** — Neo4j AuraDB · Qdrant · LangGraph · Groq GPT-OSS 120B · BGE-M3 · SPLADE · spaCy · RDFLib · owlrl · FastAPI · Gradio

---

## Ontology Layer

The OWL ontology is built from a snapshot of the same corpus but queried independently from Neo4j. Neo4j stores explicit operational relationships — which paper uses which method and who authored what. The RDF graph adds semantic classes and inferred `relatedWork` links. A single retrieval request uses one graph backend; results are not blended across Neo4j and RDFLib.

### Schema (`ontology/arxiv_cs.ttl`)

Five top-level classes are `Paper`, `Author`, `Institution`, `Method`, and `Community`. Methods can be classified into seven subclasses:

| Subclass | Examples |
|---|---|
| `FineTuningMethod` | LoRA, QLoRA, AdaLoRA, Instruction Tuning |
| `AttentionMethod` | Transformer, Flash Attention, MoE, Cross-Attention |
| `AlignmentMethod` | RLHF, DPO, PPO, RLAIF |
| `ReasoningMethod` | Chain-of-Thought, RAG, LangGraph, ReAct |
| `RetrievalMethod` | BM25, DPR, ColBERT, FAISS, Qdrant |
| `PersonalizationMethod` | Personalized adaptation and user-specific modelling methods |
| `AgentSkillLearningMethod` | Agent learning, tool-use, and skill-acquisition methods |

The schema is written in OWL/RDFS Turtle format. `relatedWork` is declared as an `owl:SymmetricProperty` — if paper A is related to paper B, the reasoner infers the reverse automatically.

### Build process (`src/ingestion/build_ontology.py`)

1. Export all nodes and edges from Neo4j as RDF triples
2. Apply reviewed method classifications across the 7 subclasses
3. Assert subclass membership (`ex:method_LoRA rdf:type ex:FineTuningMethod`)
4. Run `owlrl.DeductiveClosure(RDFS_Semantics)` to infer new triples — any two papers sharing a method subclass become `relatedWork`

Current result: **227,077 total triples**, including **82,582 `relatedWork` statements**, covering **598 / 2,000 papers (29.9%)** through method usage.

### How it integrates with retrieval

**Neo4j graph backend** — uses Cypher for explicit entity relationships such as papers, authors, institutions, and methods.

**RDFLib graph backend** — uses LLM-generated, schema-validated SPARQL for class membership and inferred relationships, for example *"What type of method is LoRA?"* The router selects this backend within Graph mode when the question is ontological.

### Known limitation

Method classification is intentionally conservative: **59 of 286** corpus method nodes currently have a reviewed category, while **227** remain unclassified instead of being forced into an incorrect class. The original LLM classification output is retained for provenance, but the populated ontology is built only from `method_classifications_reviewed.json`. SHACL treats missing classifications and affiliations as expected coverage warnings while enforcing multiple disjoint categories and papers without authors as hard violations. The current graph conforms when warnings are allowed. The interview demo uses reviewed classifications, including LoRA → `FineTuningMethod`. See [`ontology/SCOPE.md`](ontology/SCOPE.md) for details.

---

## Quickstart

```bash
git clone https://github.com/VinaySampath14/agentic-graph-rag.git
cd agentic-graph-rag
pip install -e ".[dev]"
cp .env.example .env          # add Neo4j, Qdrant, and Groq credentials
python scripts/verify_connections.py
python app.py                 # Gradio demo at localhost:7860
```

For a Hugging Face Space, configure these repository secrets (never commit
their values): `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `QDRANT_URL`,
`QDRANT_API_KEY`, and `GROQ_API_KEY`. `GROQ_MODEL` is optional and defaults to
`openai/gpt-oss-120b`. The final web fallback uses DuckDuckGo and requires no
additional search API key.

Interview demo questions:

1. Vector — “How does retrieval-augmented generation improve language models?”
2. Graph/Neo4j — “Who are the authors of papers that use LoRA?”
3. Graph/RDFLib — “What type of method is LoRA?”
4. Community — “What are the main research themes across these papers?”

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
This work extends it with an agentic self-correction loop, mode-aware query rewriting, temporal graph edges, an OWL ontology layer with automated method classification and owlrl reasoning, and per-query-type RAGAS evaluation across a controlled four-version ablation.

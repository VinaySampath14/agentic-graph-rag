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
| Three retrieval modes (vector, graph, community) | 🔄 In progress |
| LangGraph agentic loop with self-correction | 🔄 In progress |
| FastAPI backend + Gradio demo | 🔄 In progress |
| RAGAS evaluation + ablation study | 🔄 In progress |

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

| Query type | Faithfulness | Answer relevancy | Context precision |
|------------|-------------|-----------------|------------------|
| Factual    | —           | —               | —                |
| Relational | —           | —               | —                |
| Thematic   | —           | —               | —                |

*Results table will be filled after eval pipeline completes (Week 4).*

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

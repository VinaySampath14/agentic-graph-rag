# Agentic Graph RAG

> Self-correcting context engine over 2,000 arXiv CS papers — routes queries
> between vector, graph, and community retrieval modes, rewrites on failure,
> and explains every decision.

**[Live demo](#)** · **[arXiv preprint](#)** · **[HuggingFace Space](#)**

## What it does

Three retrieval modes (naive vector, local graph traversal, global community
summaries) wired into a LangGraph agentic loop. When retrieval fails a quality
grade, the agent rewrites the query and re-routes to a different mode — up to
3 correction loops before falling back to web search.

Built on Neo4j · Qdrant · LangGraph · Groq LLaMA 3.3 70B

## Results

| Query type | Faithfulness | Answer relevancy | Context precision |
|------------|-------------|-----------------|------------------|
| Factual    | —           | —               | —                |
| Relational | —           | —               | —                |
| Thematic   | —           | —               | —                |

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/agentic-graph-rag.git
cd agentic-graph-rag
cp .env.example .env   # fill in your API keys
docker build -t agr . && docker run -p 8000:8000 agr
```

## Related work

Closest prior work: [arXiv:2508.05660](https://arxiv.org/abs/2508.05660).
This project extends it with an agentic self-correction loop, mode-aware query
rewriting, temporal graph edges, and per-query-type RAGAS evaluation.

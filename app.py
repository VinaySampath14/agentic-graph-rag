"""HuggingFace Spaces entrypoint — calls agent directly (no FastAPI server needed).

For local development with the FastAPI backend, use src/demo/app.py instead.
"""
import json
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

EXAMPLE_QUERIES = [
    "What is Retrieval-Augmented Generation?",
    "Which papers use Chain-of-Thought reasoning?",
    "What are the main trends in LLM safety research?",
    "How do attention mechanisms work in transformers?",
    "What methods are used for parameter-efficient fine-tuning?",
]

EVAL_DIR = Path("data/eval")
VERSIONS = ["v1", "v2", "v3", "v4"]
VERSION_LABELS = {
    "v1": "v1 — Naive RAG",
    "v2": "v2 — Static routing",
    "v3": "v3 — Loop, no rewrite",
    "v4": "v4 — Full system",
}


@lru_cache(maxsize=1)
def get_graph():
    from src.agent.graph import compile_graph
    return compile_graph()


def _format_trace(trace: list[dict]) -> str:
    if not trace:
        return "_No trace available._"
    lines = []
    for entry in trace:
        node = entry.get("node", "?")
        decision = entry.get("decision", "")
        reason = entry.get("reason", "")
        extras = entry.get("extras", {})
        lines.append(f"**{node}** → {decision}")
        if reason:
            lines.append(f"  _{reason}_")
        if extras.get("cypher_query"):
            cypher = extras["cypher_query"].split("---")[0].strip()
            lines.append(f"  ```cypher\n  {cypher}\n  ```")
        if extras.get("communities_used"):
            lines.append(f"  Communities: {', '.join(extras['communities_used'])}")
        lines.append("")
    return "\n".join(lines)


def _load_eval_summary() -> str:
    rows = []
    rows.append("| Version | Coverage | Faithfulness | Ans. Rel. | Ctx. Prec. | Ctx. Rec. |")
    rows.append("|---|---|---|---|---|---|")
    for v in VERSIONS:
        raw_path    = EVAL_DIR / f"eval_results_{v}.jsonl"
        scores_path = EVAL_DIR / f"ragas_scores_{v}.jsonl"
        if not raw_path.exists() or not scores_path.exists():
            continue
        raw    = [json.loads(l) for l in raw_path.read_text(encoding="utf-8").splitlines()]
        scores = [json.loads(l) for l in scores_path.read_text(encoding="utf-8").splitlines()]
        answered = sum(1 for r in raw if not r.get("refused", False))
        coverage = answered / len(raw) if raw else 0

        def avg(key, scores=scores):
            vals = [s[key] for s in scores if key in s]
            return sum(vals) / len(vals) if vals else 0.0

        rows.append(
            f"| **{VERSION_LABELS[v]}** "
            f"| {coverage:.1%} ({answered}/{len(raw)}) "
            f"| {avg('faithfulness'):.3f} "
            f"| {avg('answer_relevancy'):.3f} "
            f"| {avg('context_precision'):.3f} "
            f"| {avg('context_recall'):.3f} |"
        )
    return "\n".join(rows)


def run_query(query: str):
    if not query.strip():
        return "Please enter a question.", "", ""

    graph = get_graph()
    initial_state = {
        "query": query,
        "rewritten_query": query,
        "intent": "",
        "retrieved_context": None,
        "grade_result": None,
        "answer": "",
        "citations": [],
        "confidence_proxy": 0.0,
        "loop_count": 0,
        "mode_history": [],
        "agent_trace": [],
        "low_confidence": False,
        "fallback_mode": None,
        "refused": False,
        "refusal_reason": "",
    }

    t0 = time.perf_counter()
    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        return f"**Error:** {e}", "", ""
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)

    refused = final_state.get("refused", False)
    answer = final_state.get("answer", "")
    refusal_reason = final_state.get("refusal_reason", "")
    loop_count = final_state.get("loop_count", 0)
    mode_history = final_state.get("mode_history", [])
    trace = final_state.get("agent_trace", [])

    answer_md = f"**Refused:** {refusal_reason}" if refused else answer
    meta_md = (
        f"**Loops:** {loop_count} &nbsp;|&nbsp; "
        f"**Modes tried:** {' → '.join(mode_history) if mode_history else 'none'} &nbsp;|&nbsp; "
        f"**Latency:** {latency_ms:.0f} ms"
    )
    return answer_md, meta_md, _format_trace(trace)


# ── UI ─────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Agentic Graph RAG") as demo:
    gr.Markdown(
        "# Agentic Graph RAG\n"
        "Self-correcting retrieval over 2,000 arXiv CS papers. "
        "Routes between vector, graph, and community modes — rewrites on failure.\n\n"
        "**Stack:** Neo4j · Qdrant · LangGraph · Groq LLaMA 3.3 70B · BGE-M3"
    )

    with gr.Tabs():

        with gr.Tab("Ask"):
            with gr.Row():
                query_box = gr.Textbox(
                    label="Question",
                    placeholder="e.g. What are the main trends in LLM safety research?",
                    lines=2,
                    scale=4,
                )
                submit_btn = gr.Button("Ask", variant="primary", scale=1)

            gr.Examples(examples=EXAMPLE_QUERIES, inputs=query_box, label="Example queries")

            answer_out = gr.Markdown(label="Answer")
            meta_out   = gr.Markdown(label="")

            with gr.Accordion("Agent trace", open=False):
                trace_out = gr.Markdown()

            submit_btn.click(fn=run_query, inputs=query_box, outputs=[answer_out, meta_out, trace_out])
            query_box.submit(fn=run_query, inputs=query_box, outputs=[answer_out, meta_out, trace_out])

        with gr.Tab("Eval Results"):
            gr.Markdown("## Ablation Results — 80 queries (30 factual / 30 relational / 20 thematic)")
            gr.Markdown(
                "Coverage = fraction of queries answered. "
                "RAGAS scores on answered queries only, GPT-4o-mini as judge."
            )
            gr.Markdown(_load_eval_summary())
            with gr.Row():
                with gr.Column():
                    if Path("figures/fig1_coverage.png").exists():
                        gr.Image("figures/fig1_coverage.png", label="Coverage by version")
                with gr.Column():
                    if Path("figures/fig2_ragas_overall.png").exists():
                        gr.Image("figures/fig2_ragas_overall.png", label="RAGAS scores by version")
            with gr.Row():
                with gr.Column():
                    if Path("figures/fig3_ragas_v4.png").exists():
                        gr.Image("figures/fig3_ragas_v4.png", label="v4 RAGAS by query type")
                with gr.Column():
                    if Path("figures/fig4_loop_efficiency.png").exists():
                        gr.Image("figures/fig4_loop_efficiency.png", label="Loop efficiency & router accuracy")

        with gr.Tab("About"):
            gr.Markdown("""
## System Overview

**Agentic Graph RAG** is a self-correcting retrieval engine over 2,000 arXiv CS papers (CS.AI + CS.CL, 2026).

### Retrieval Modes
| Mode | Backend | Best for |
|---|---|---|
| Vector | Qdrant hybrid (BGE-M3 + BM25 + RRF) | Factual / definitional queries |
| Graph | Neo4j Cypher traversal | Relational / authorship queries |
| Community | Leiden cluster embeddings | Thematic / trend queries |

### Key Finding
Adding a correction loop **without** query rewriting (v3) gives no coverage gain (27.5%).
Adding **mode-aware rewriting** (v4) recovers coverage to 81.2%.

### Links
- [GitHub](https://github.com/VinaySampath14/agentic-graph-rag)
""")


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

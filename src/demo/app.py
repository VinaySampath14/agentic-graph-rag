"""Gradio demo for Agentic Graph RAG.

Tabs:
  1. Ask — submit a query, see the answer + agent trace
  2. Eval Results — coverage and RAGAS scores across ablation versions
  3. About — system description and architecture

Usage:
    python src/demo/app.py
    or via HuggingFace Spaces (app.py at repo root)
"""
import json
import os
import time
from pathlib import Path

import gradio as gr
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.environ.get("API_URL", "http://localhost:8000")

EXAMPLE_QUERIES = [
    "What is Retrieval-Augmented Generation?",
    "Which papers use Chain-of-Thought reasoning?",
    "What are the main trends in LLM safety research?",
    "How do attention mechanisms work in transformers?",
    "What methods are used for parameter-efficient fine-tuning?",
    "Which authors have published on both RAG and graph neural networks?",
]

EVAL_DIR = Path("data/eval")
VERSIONS = ["v1", "v2", "v3", "v4"]
VERSION_LABELS = {
    "v1": "v1 — Naive RAG",
    "v2": "v2 — Static routing",
    "v3": "v3 — Loop, no rewrite",
    "v4": "v4 — Full system",
}


# ── Helpers ────────────────────────────────────────────────────────────────

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

        def avg(key):
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


# ── Tab 1: Ask ─────────────────────────────────────────────────────────────

def run_query(query: str):
    if not query.strip():
        return "Please enter a question.", "", ""

    try:
        t0 = time.perf_counter()
        resp = requests.post(f"{API_URL}/query", json={"query": query}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        return "**Error:** Cannot connect to the API. Make sure the FastAPI server is running.", "", ""
    except Exception as e:
        return f"**Error:** {e}", "", ""

    answer = data.get("answer", "")
    refused = data.get("refused", False)
    refusal_reason = data.get("refusal_reason", "")
    loop_count = data.get("loop_count", 0)
    mode_history = data.get("mode_history", [])
    latency_ms = data.get("latency_ms", 0)
    trace = data.get("agent_trace", [])

    if refused:
        answer_md = f"**Refused:** {refusal_reason}"
    else:
        answer_md = answer

    meta_md = (
        f"**Loops:** {loop_count} &nbsp;|&nbsp; "
        f"**Modes tried:** {' → '.join(mode_history) if mode_history else 'none'} &nbsp;|&nbsp; "
        f"**Latency:** {latency_ms:.0f} ms"
    )

    trace_md = _format_trace(trace)
    return answer_md, meta_md, trace_md


# ── Build UI ───────────────────────────────────────────────────────────────

with gr.Blocks(title="Agentic Graph RAG") as demo:
    gr.Markdown(
        "# Agentic Graph RAG\n"
        "Self-correcting retrieval over 2,000 arXiv CS papers. "
        "Routes between vector, graph, and community modes — rewrites on failure.\n\n"
        "**Stack:** Neo4j · Qdrant · LangGraph · Groq LLaMA 3.3 70B · BGE-M3"
    )

    with gr.Tabs():

        # ── Tab 1: Ask ─────────────────────────────────────────────────────
        with gr.Tab("Ask"):
            with gr.Row():
                query_box = gr.Textbox(
                    label="Question",
                    placeholder="e.g. What are the main trends in LLM safety research?",
                    lines=2,
                    scale=4,
                )
                submit_btn = gr.Button("Ask", variant="primary", scale=1)

            gr.Examples(
                examples=EXAMPLE_QUERIES,
                inputs=query_box,
                label="Example queries",
            )

            answer_out = gr.Markdown(label="Answer")
            meta_out   = gr.Markdown(label="")

            with gr.Accordion("Agent trace", open=False):
                trace_out = gr.Markdown()

            submit_btn.click(
                fn=run_query,
                inputs=query_box,
                outputs=[answer_out, meta_out, trace_out],
            )
            query_box.submit(
                fn=run_query,
                inputs=query_box,
                outputs=[answer_out, meta_out, trace_out],
            )

        # ── Tab 2: Eval Results ────────────────────────────────────────────
        with gr.Tab("Eval Results"):
            gr.Markdown("## Ablation Results — 80 queries (30 factual / 30 relational / 20 thematic)")
            gr.Markdown(
                "Coverage = fraction of queries answered (not refused). "
                "RAGAS scores computed on answered queries only with GPT-4o-mini as judge."
            )
            eval_table = gr.Markdown(_load_eval_summary())

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

        # ── Tab 3: About ───────────────────────────────────────────────────
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

### Agentic Loop
1. Router classifies query → dispatches to best mode
2. `grade_context` issues binary pass/fail on retrieved context
3. On fail: `rewrite_query` reformulates for the next mode, loop repeats (max 3)
4. On pass: `generator` produces answer, `grade_answer` checks groundedness
5. If all modes fail: structured refusal returned

### Key Finding
Adding a correction loop **without** query rewriting (v3) gives no coverage gain (27.5%).
Adding **mode-aware rewriting** (v4) recovers coverage to 81.2% — rewriting is the critical mechanism.

### Links
- [GitHub](https://github.com/VinaySampath14/agentic-graph-rag)
- [arXiv paper](https://arxiv.org/) *(coming soon)*
""")


if __name__ == "__main__":
    demo.launch(share=False, theme=gr.themes.Soft())

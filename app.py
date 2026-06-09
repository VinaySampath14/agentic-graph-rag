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


_SVG_STYLE = """<style>
@keyframes pulse {
  0%,100% { opacity:1; }
  50%      { opacity:0.65; }
}
@keyframes shimmer {
  0%   { stop-color:#3b82f6; }
  50%  { stop-color:#8b5cf6; }
  100% { stop-color:#3b82f6; }
}
.pulse { animation: pulse 0.85s ease-in-out infinite; }
</style>"""

# Node layout: name → (cx, cy, w, h)  — all rects; END nodes handled separately
_NL = {
    "query_analyser":        (285, 52,  165, 34),
    "router":                (285, 132, 110, 34),
    "naive_retriever":       (108, 218, 108, 34),
    "local_graph_retriever": (285, 218, 108, 34),
    "global_retriever":      (462, 218, 108, 34),
    "rewrite_query":         (68,  318, 118, 34),
    "grade_context":         (285, 318, 140, 34),
    "web_retriever":         (488, 318,  92, 34),
    "force_refusal":         (88,  420, 128, 34),
    "generator":             (285, 420, 110, 34),
    "grade_answer":          (285, 505, 120, 34),
}
_END_OOD    = (498, 52,  20)   # cx, cy, r
_END_FORCE  = (88,  598, 20)
_END_ANSWER = (285, 598, 20)


def _nc(name, done, active, failed, retried=None, success=None):
    if name == active:                         return "#fef3c7", "#f59e0b", "#92400e"  # yellow — running
    if retried and name in retried:            return "#ffedd5", "#f97316", "#9a3412"  # orange — failed, retried
    if success and name in success:            return "#dcfce7", "#22c55e", "#166534"  # green  — final success
    if name in failed:                         return "#fee2e2", "#ef4444", "#991b1b"  # red    — permanent fail
    if name in done:                           return "#dbeafe", "#3b82f6", "#1e40af"  # blue   — done
    return "#f9fafb", "#d1d5db", "#9ca3af"                                             # gray   — pending


def _ec(src, dst, done, failed):
    return "#9ca3af" if (src in done | failed and dst in done | failed) else "#e5e7eb"


def _draw_node(name, label, done, active, failed, retried=None, success=None):
    cx, cy, w, h = _NL[name]
    fill, stroke, tc = _nc(name, done, active, failed, retried, success)
    cls = 'class="pulse"' if name == active else ""
    lw = "2.5" if name == active else "1.5"
    return (
        f'<g {cls}>'
        f'<rect x="{cx-w//2}" y="{cy-h//2}" width="{w}" height="{h}" rx="8" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{lw}"/>'
        f'<text x="{cx}" y="{cy+5}" text-anchor="middle" font-size="12" '
        f'font-family="monospace" fill="{tc}" font-weight="500">{label}</text>'
        f'</g>'
    )


def _draw_circle(cx, cy, r, label, name_key, done, active, failed, retried=None, success=None):
    fill, stroke, tc = _nc(name_key, done, active, failed, retried, success)
    return (
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        f'<text x="{cx}" y="{cy+4}" text-anchor="middle" font-size="10" '
        f'font-family="monospace" fill="{tc}" font-weight="600">{label}</text>'
    )


def _line(x1, y1, x2, y2, color, dashed=False):
    dash = 'stroke-dasharray="6,4"' if dashed else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="1.5" {dash} marker-end="url(#ah)"/>')


def _path(d, color, dashed=False):
    dash = 'stroke-dasharray="6,4"' if dashed else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" '
            f'stroke-width="1.5" {dash} marker-end="url(#ah)"/>')


def _label(x, y, text, color="#aab0bb"):
    return f'<text x="{x}" y="{y}" font-size="10" font-family="sans-serif" fill="{color}">{text}</text>'


def _build_graph_html(done: set, active: str | None, failed: set, loading: bool,
                      retried: set | None = None, success: set | None = None) -> str:
    retried = retried or set()
    success = success or set()
    v = done | failed | retried | success
    e = lambda s, d: _ec(s, d, done, failed)

    spinner = (
        '<rect x="0" y="0" width="570" height="4" rx="2" '
        'fill="#3b82f6" opacity="0.7"><animate attributeName="x" '
        'from="-570" to="570" dur="1.2s" repeatCount="indefinite"/>'
        '<animate attributeName="opacity" values="0.7;0.3;0.7" dur="1.2s" '
        'repeatCount="indefinite"/></rect>'
    ) if loading else ""

    edges = "\n".join([
        # ── Normal flow (solid) ──────────────────────────────────────────
        _line(285, 69, 285, 115,  e("query_analyser","router")),
        _line(285, 149, 285, 201, e("router","local_graph_retriever")),
        _line(265, 149, 140, 201, e("router","naive_retriever")),
        _line(308, 149, 435, 201, e("router","global_retriever")),
        _line(108, 235, 225, 301, e("naive_retriever","grade_context")),
        _line(285, 235, 285, 301, e("local_graph_retriever","grade_context")),
        _line(462, 235, 348, 301, e("global_retriever","grade_context")),
        _line(285, 335, 285, 403, e("grade_context","generator")),
        _line(285, 437, 285, 488, e("generator","grade_answer")),
        _line(285, 522, 285, 578, e("grade_answer","END_answer")),
        _line(88,  437, 88,  578, e("force_refusal","END_force")),
        # web_retr → grade_context (U-shape under, solid)
        _path(f"M 488,335 L 488,358 L 355,358 L 355,335",
              e("web_retriever","grade_context")),
        # ── Conditional / dashed ─────────────────────────────────────────
        # query_analyser → END_ood
        _line(368, 52, 478, 52,   e("query_analyser","END_ood"), dashed=True),
        # grade_context → rewrite_query (fail)
        _line(215, 318, 127, 318, e("grade_context","rewrite_query"), dashed=True),
        # grade_context → web_retr (loop=3)
        _line(355, 310, 442, 310, e("grade_context","web_retriever"), dashed=True),
        # grade_context → force_refusal (exhausted)
        _path(f"M 230,330 C 180,370 130,390 88,403",
              e("grade_context","force_refusal"), dashed=True),
        # rewrite_query → router (curved loop back up — hugs left margin to avoid vector node)
        _path(f"M 68,301 C 15,260 15,110 230,132",
              e("rewrite_query","router"), dashed=True),
    ])

    edge_labels = "\n".join([
        _label(395, 47,  "OOD"),
        _label(155, 313, "fail"),
        _label(368, 303, "loop=3"),
        _label(290, 372, "pass"),
        _label(148, 378, "exhausted"),
        _label(55,  258, "rewrite"),
    ])

    kw = dict(done=done, active=active, failed=failed, retried=retried, success=success)
    nodes = "\n".join([
        _draw_node("query_analyser",        "query_analyser", **kw),
        _draw_node("router",                "router",         **kw),
        _draw_node("naive_retriever",       "vector",         **kw),
        _draw_node("local_graph_retriever", "graph",          **kw),
        _draw_node("global_retriever",      "community",      **kw),
        _draw_node("rewrite_query",         "rewrite_query",  **kw),
        _draw_node("grade_context",         "grade_context",  **kw),
        _draw_node("web_retriever",         "web",            **kw),
        _draw_node("force_refusal",         "force_refusal",  **kw),
        _draw_node("generator",             "generator",      **kw),
        _draw_node("grade_answer",          "grade_answer",   **kw),
        _draw_circle(*_END_OOD,    "END", "END_ood",    **kw),
        _draw_circle(*_END_FORCE,  "END", "END_force",  **kw),
        _draw_circle(*_END_ANSWER, "END", "END_answer", **kw),
    ])

    return f"""{_SVG_STYLE}
<svg viewBox="0 0 570 630" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;height:auto;">
  <defs>
    <marker id="ah" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
      <polygon points="0 0,7 2.5,0 5" fill="#b0b8c4"/>
    </marker>
  </defs>
  {spinner}
  {edges}
  {edge_labels}
  {nodes}
</svg>"""


_HERO_HTML = """
<div style="background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 55%,#312e81 100%);
            border-radius:14px;padding:26px 32px 22px;margin-bottom:2px;">
  <div style="font-size:22px;font-weight:700;color:white;letter-spacing:-0.3px;margin-bottom:6px;">
    &#128269; Agentic Graph RAG
  </div>
  <div style="color:#94a3b8;font-size:13.5px;line-height:1.65;margin-bottom:16px;">
    Self-correcting retrieval engine over <strong style="color:#cbd5e1;">2,000 arXiv CS papers</strong>
    (CS.AI + CS.CL &middot; 2026).<br>
    Routes between vector, graph &amp; community modes &mdash; rewrites on failure, explains every decision.
  </div>
  <div style="display:flex;gap:7px;flex-wrap:wrap;">
    <span style="background:rgba(255,255,255,0.1);color:#93c5fd;padding:3px 11px;border-radius:20px;font-size:11.5px;font-weight:600;border:1px solid rgba(147,197,253,0.25);">Neo4j</span>
    <span style="background:rgba(255,255,255,0.1);color:#93c5fd;padding:3px 11px;border-radius:20px;font-size:11.5px;font-weight:600;border:1px solid rgba(147,197,253,0.25);">Qdrant</span>
    <span style="background:rgba(255,255,255,0.1);color:#93c5fd;padding:3px 11px;border-radius:20px;font-size:11.5px;font-weight:600;border:1px solid rgba(147,197,253,0.25);">LangGraph</span>
    <span style="background:rgba(255,255,255,0.1);color:#93c5fd;padding:3px 11px;border-radius:20px;font-size:11.5px;font-weight:600;border:1px solid rgba(147,197,253,0.25);">Groq LLaMA 3.3 70B</span>
    <span style="background:rgba(255,255,255,0.1);color:#93c5fd;padding:3px 11px;border-radius:20px;font-size:11.5px;font-weight:600;border:1px solid rgba(147,197,253,0.25);">BGE-M3</span>
    <span style="background:rgba(255,255,255,0.1);color:#93c5fd;padding:3px 11px;border-radius:20px;font-size:11.5px;font-weight:600;border:1px solid rgba(147,197,253,0.25);">spaCy</span>
  </div>
</div>
"""

_LEGEND_HTML = """
<div style="padding:4px 2px 0;">
  <div style="font-size:10.5px;font-weight:700;letter-spacing:0.07em;text-transform:uppercase;
              color:#9ca3af;margin-bottom:7px;">Node states</div>
  <div style="display:flex;gap:5px;flex-wrap:wrap;">
    <span style="background:#f9fafb;border:1px solid #d1d5db;color:#6b7280;
                 padding:3px 9px;border-radius:20px;font-size:11px;font-weight:500;">&#9711; pending</span>
    <span style="background:#fef3c7;border:1px solid #f59e0b;color:#92400e;
                 padding:3px 9px;border-radius:20px;font-size:11px;font-weight:500;">&#9711; active</span>
    <span style="background:#dbeafe;border:1px solid #3b82f6;color:#1e40af;
                 padding:3px 9px;border-radius:20px;font-size:11px;font-weight:500;">&#9711; done</span>
    <span style="background:#ffedd5;border:1px solid #f97316;color:#9a3412;
                 padding:3px 9px;border-radius:20px;font-size:11px;font-weight:500;">&#9711; retried</span>
    <span style="background:#dcfce7;border:1px solid #22c55e;color:#166534;
                 padding:3px 9px;border-radius:20px;font-size:11px;font-weight:500;">&#9711; success</span>
    <span style="background:#fee2e2;border:1px solid #ef4444;color:#991b1b;
                 padding:3px 9px;border-radius:20px;font-size:11px;font-weight:500;">&#9711; failed</span>
  </div>
</div>
"""

_ABOUT_HTML = """
<div style="max-width:900px;">

  <div style="font-size:18px;font-weight:700;color:#0f172a;margin-bottom:4px;">System Overview</div>
  <div style="color:#64748b;font-size:14px;margin-bottom:22px;line-height:1.6;">
    A LangGraph agentic loop over a Neo4j knowledge graph + Qdrant vector store built from
    2,000 arXiv CS papers. When retrieval fails a quality grade, the agent rewrites the query
    to suit the next mode and re-routes — up to 3 loops before a structured refusal.
  </div>

  <div style="font-size:13px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;
              color:#94a3b8;margin-bottom:10px;">Retrieval Modes</div>
  <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:24px;">
    <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:16px;">
      <div style="font-size:13px;font-weight:700;color:#1e40af;margin-bottom:4px;">&#9632; Vector</div>
      <div style="font-size:12.5px;color:#1e3a8a;font-weight:600;margin-bottom:6px;">Qdrant hybrid (BGE-M3 + BM25 + RRF)</div>
      <div style="font-size:12px;color:#3730a3;">Best for <em>factual / definitional</em> queries</div>
    </div>
    <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:16px;">
      <div style="font-size:13px;font-weight:700;color:#166534;margin-bottom:4px;">&#9632; Graph</div>
      <div style="font-size:12.5px;color:#14532d;font-weight:600;margin-bottom:6px;">Neo4j Cypher traversal + fuzzy entity linking</div>
      <div style="font-size:12px;color:#15803d;">Best for <em>relational / authorship</em> queries</div>
    </div>
    <div style="background:#fdf4ff;border:1px solid #e9d5ff;border-radius:10px;padding:16px;">
      <div style="font-size:13px;font-weight:700;color:#7e22ce;margin-bottom:4px;">&#9632; Community</div>
      <div style="font-size:12.5px;color:#581c87;font-weight:600;margin-bottom:6px;">Leiden cluster embeddings + summaries</div>
      <div style="font-size:12px;color:#9333ea;">Best for <em>thematic / trend</em> queries</div>
    </div>
  </div>

  <div style="font-size:13px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;
              color:#94a3b8;margin-bottom:10px;">Knowledge Graph</div>
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;
              padding:14px 18px;font-family:monospace;font-size:12.5px;color:#334155;
              line-height:1.8;margin-bottom:24px;">
    Nodes: &nbsp;2,000 Paper &middot; 9,250 Author &middot; 2,988 Institution &middot; 36 Method &middot; 8 Community<br>
    Edges: 10,651 AUTHORED_BY &middot; 1,975 USES_METHOD &middot; 4,532 FROM_INSTITUTION
  </div>

  <div style="font-size:13px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;
              color:#94a3b8;margin-bottom:10px;">Key Finding</div>
  <div style="background:#fffbeb;border:1px solid #fde68a;border-left:4px solid #f59e0b;
              border-radius:0 10px 10px 0;padding:14px 18px;margin-bottom:24px;">
    <div style="font-size:13.5px;color:#92400e;line-height:1.65;">
      Adding a correction loop <strong>without</strong> query rewriting (v3) gives <strong>no coverage gain</strong> (27.5%).<br>
      Adding <strong>mode-aware rewriting</strong> (v4) recovers coverage to <strong>81.2%</strong> &mdash;
      rewriting is the critical mechanism.
    </div>
  </div>

  <div style="font-size:13px;font-weight:700;letter-spacing:0.06em;text-transform:uppercase;
              color:#94a3b8;margin-bottom:10px;">Links</div>
  <div style="display:flex;gap:10px;">
    <a href="https://github.com/VinaySampath14/agentic-graph-rag"
       style="background:#0f172a;color:white;padding:7px 16px;border-radius:8px;
              font-size:13px;font-weight:600;text-decoration:none;">
      GitHub &#8599;
    </a>
    <span style="background:#f1f5f9;color:#64748b;padding:7px 16px;border-radius:8px;
                 font-size:13px;font-weight:600;">arXiv (coming soon)</span>
  </div>
</div>
"""

_CSS = """
/* Widen the main container */
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; }

/* Answer output — subtle left-border card */
#answer-out .prose {
    background: #f8fafc;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    font-size: 14.5px;
    line-height: 1.72;
    min-height: 60px;
}

/* Section labels above outputs */
.out-label > .label-wrap span,
.out-label > span {
    font-size: 10.5px !important;
    font-weight: 700 !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #94a3b8 !important;
}

/* Graph container */
#graph-panel {
    background: #fafafa;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    padding: 10px 8px !important;
}

/* Submit button */
#submit-btn { min-height: 50px !important; font-size: 15px !important; font-weight: 700 !important; }

/* Trace accordion — tighten spacing */
#trace-accordion > .label-wrap { padding: 8px 14px !important; }
#trace-accordion > .label-wrap span { font-size: 13px !important; font-weight: 600 !important; color: #374151 !important; }

/* Smaller examples text */
.dataset td { font-size: 12px !important; padding: 4px 10px !important; }
.dataset thead th { font-size: 11px !important; font-weight: 600; color: #94a3b8 !important; padding: 4px 10px !important; }
.dataset { border-radius: 8px !important; overflow: hidden; }
"""


def _meta_html(loop_count: int, mode_history: list, latency_ms: float) -> str:
    modes = " &rarr; ".join(mode_history) if mode_history else "none"
    return (
        f'<div style="display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 4px;">'
        f'<span style="background:#dbeafe;color:#1e40af;padding:5px 13px;border-radius:20px;'
        f'font-size:12px;font-weight:700;">&#x1F501; {loop_count} loop{"s" if loop_count != 1 else ""}</span>'
        f'<span style="background:#f1f5f9;color:#334155;padding:5px 13px;border-radius:20px;'
        f'font-size:12px;font-weight:700;">&#x1F4E1; {modes}</span>'
        f'<span style="background:#dcfce7;color:#166534;padding:5px 13px;border-radius:20px;'
        f'font-size:12px;font-weight:700;">&#x26A1; {latency_ms:.0f} ms</span>'
        f'</div>'
    )


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
    _hide = gr.update(visible=False)
    _show = gr.update(visible=True)
    if not query.strip():
        yield "Please enter a question.", "", "", _build_graph_html(set(), None, set(), loading=False), gr.update(), gr.update()
        return

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
    final_state = None
    done_nodes:    set[str] = set()
    failed_nodes:  set[str] = set()
    retried_nodes: set[str] = set()  # failed but triggered a retry (orange)
    success_nodes: set[str] = set()  # final success (green)
    _RETRIEVERS = {"naive_retriever", "local_graph_retriever", "global_retriever", "web_retriever"}
    last_retriever: str | None = None  # tracks which retriever ran most recently

    def _graph(loading):
        return _build_graph_html(done_nodes, None, failed_nodes, loading,
                                 retried=retried_nodes, success=success_nodes)

    # Show spinner immediately on submit; hide examples, reveal answer area
    yield "", "", "", _graph(loading=True), _hide, _show

    try:
        for chunk in graph.stream(initial_state, stream_mode="updates"):
            # chunk = {node_name: state_delta}
            node_name = list(chunk.keys())[0]
            state_delta = chunk[node_name]

            # Determine if this node failed — check only the last trace entry
            # (the one just appended by this node). The full accumulated list is
            # returned, so scanning all entries would catch old grade_context failures.
            trace = state_delta.get("agent_trace", [])
            node_failed = False
            if trace:
                last = trace[-1]
                if last.get("node") == node_name:
                    decision = last.get("decision", "")
                    node_failed = decision.startswith("fail") or decision == "refused"

            # Track the most recent retriever that completed (used for orange coloring on grade_context fail)
            if node_name in _RETRIEVERS:
                last_retriever = node_name

            if node_failed:
                if node_name == "grade_context":
                    # Mark grade_context orange (retried)
                    retried_nodes.add(node_name)
                    done_nodes.discard(node_name)
                    failed_nodes.discard(node_name)
                    # Mark the retriever whose context was just rejected as orange.
                    # We use last_retriever (set when the retriever completed) rather than
                    # mode_history, because mode_history is updated by rewrite_query AFTER
                    # grade_context fails — so it would point to the previous retriever.
                    if last_retriever:
                        retried_nodes.add(last_retriever)
                        done_nodes.discard(last_retriever)
                else:
                    failed_nodes.add(node_name)
                    done_nodes.discard(node_name)
                    retried_nodes.discard(node_name)
            else:
                if node_name == "grade_answer":
                    success_nodes.add(node_name)   # green — final success
                elif node_name == "grade_context":
                    # grade_context passed — clear retried state from it and last retriever
                    retried_nodes.discard(node_name)
                    done_nodes.add(node_name)
                else:
                    done_nodes.add(node_name)
                failed_nodes.discard(node_name)

            # Peek at next active node from agent_trace extras if available
            next_active = None
            for e in reversed(trace):
                extras = e.get("extras", {})
                if "next_mode" in extras:
                    mode_map = {"vector": "naive_retriever", "graph": "local_graph_retriever",
                                "community": "global_retriever", "web": "web_retriever"}
                    next_active = mode_map.get(extras["next_mode"])
                    break

            # Accumulate full state for final answer
            if final_state is None:
                final_state = {**initial_state}
            for k, v in state_delta.items():
                if isinstance(v, list) and k == "agent_trace":
                    final_state[k] = final_state.get(k, []) + v
                else:
                    final_state[k] = v

            trace_md = _format_trace(final_state.get("agent_trace", []))
            yield "", "", trace_md, _graph(loading=True), _hide, _show

    except Exception as e:
        yield f"**Error:** {e}", "", "", _graph(loading=False), _hide, _show
        return

    if not final_state:
        yield "No response.", "", "", _graph(loading=False), _hide, _show
        return

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    refused = final_state.get("refused", False)
    refusal_reason = final_state.get("refusal_reason", "")
    loop_count = final_state.get("loop_count", 0)
    mode_history = final_state.get("mode_history", [])

    answer_md = f"**Refused:** {refusal_reason}" if refused else final_state.get("answer", "")
    # Final yield — spinner off
    yield answer_md, _meta_html(loop_count, mode_history, latency_ms), \
          _format_trace(final_state.get("agent_trace", [])), _graph(loading=False), _hide, _show


# ── UI ─────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Agentic Graph RAG", css=_CSS, theme=gr.themes.Soft()) as demo:

    gr.HTML(_HERO_HTML)

    with gr.Tabs():

        # ── Tab 1: Ask ─────────────────────────────────────────────────────
        with gr.Tab("💬 Ask"):
            with gr.Row(equal_height=False):

                # ── Left: input + examples/answer ─────────────────────────
                with gr.Column(scale=3):
                    with gr.Row():
                        query_box = gr.Textbox(
                            label="",
                            placeholder="Ask about any CS paper, author, method, or trend...",
                            lines=2,
                            scale=5,
                            show_label=False,
                        )
                        submit_btn = gr.Button("Ask →", variant="primary", scale=1, elem_id="submit-btn")

                    # Examples — visible by default, hidden once user interacts
                    with gr.Column(visible=True) as examples_col:
                        gr.Examples(
                            examples=EXAMPLE_QUERIES,
                            inputs=query_box,
                            label="Try an example",
                        )

                    # Answer section — hidden until first submit
                    with gr.Column(visible=False) as answer_col:
                        gr.HTML('<div style="font-size:10.5px;font-weight:700;letter-spacing:0.07em;'
                                'text-transform:uppercase;color:#94a3b8;margin-bottom:6px;">Answer</div>')
                        answer_out = gr.Markdown(value="", elem_id="answer-out")
                        meta_out   = gr.HTML(value="")
                        with gr.Accordion("Agent trace", open=False, elem_id="trace-accordion"):
                            trace_out = gr.Markdown()

                # ── Right: live graph + legend (wider) ────────────────────
                with gr.Column(scale=3):
                    gr.HTML('<div style="font-size:10.5px;font-weight:700;letter-spacing:0.07em;'
                            'text-transform:uppercase;color:#94a3b8;margin-bottom:6px;">Agent Graph</div>')
                    graph_out = gr.HTML(
                        value=_build_graph_html(set(), None, set(), False),
                        elem_id="graph-panel",
                    )
                    gr.HTML(_LEGEND_HTML)

            outputs = [answer_out, meta_out, trace_out, graph_out, examples_col, answer_col]
            submit_btn.click(fn=run_query, inputs=query_box, outputs=outputs)
            query_box.submit(fn=run_query, inputs=query_box, outputs=outputs)
            # Hide examples as soon as user starts typing
            query_box.change(
                fn=lambda q: gr.update(visible=not bool(q.strip())),
                inputs=query_box,
                outputs=examples_col,
            )

        # ── Tab 2: Eval Results ────────────────────────────────────────────
        with gr.Tab("📊 Eval Results"):
            gr.HTML("""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:16px 20px;margin-bottom:16px;">
  <div style="font-size:16px;font-weight:700;color:#0f172a;margin-bottom:4px;">Ablation Study</div>
  <div style="font-size:13.5px;color:#64748b;line-height:1.6;">
    80 queries &middot; 30 factual / 30 relational / 20 thematic.<br>
    <strong>Coverage</strong> = fraction answered (not refused).
    <strong>RAGAS</strong> scores on answered queries, GPT-4o-mini as judge.
  </div>
</div>""")
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

        # ── Tab 3: About ───────────────────────────────────────────────────
        with gr.Tab("ℹ️ About"):
            gr.HTML(_ABOUT_HTML)


if __name__ == "__main__":
    demo.launch()

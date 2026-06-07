"""Ablation graph variants for evaluation.

v1 — Naive RAG only: always vector retrieval, no routing, no loop.
v2 — Static routing: correct mode dispatch, single pass, no loop.
v3 — Agentic loop, no rewrite: loop retries with original query, no rewrite_query node.
v4 — Full system: use compile_graph() from graph.py directly.
"""
from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import (
    node_query_analyser,
    node_router,
    node_naive_retriever,
    node_graph_retriever,
    node_community_retriever,
    node_web_retriever,
    node_grade_context,
    node_rewrite_query,
    node_generator,
    node_grade_answer,
    node_force_refusal,
)

LOOP_GUARD = 3


# ── Shared edge helpers ────────────────────────────────────────────────────

def _route_after_analyser(state: AgentState) -> str:
    return END if state.get("refused") else "router"


def _route_after_router(state: AgentState) -> str:
    intent = state["intent"]
    if intent == "graph":
        return "local_graph_retriever"
    elif intent == "community":
        return "global_retriever"
    elif intent == "web":
        return "web_retriever"
    return "naive_retriever"


def _route_after_grade_answer(state: AgentState) -> str:
    return END


# ── v1: Naive RAG only ─────────────────────────────────────────────────────
# query_analyser (OOD check only) → naive_retriever → generator → grade_answer → END
# No routing, no graph, no community, no loop.

def compile_v1():
    graph = StateGraph(AgentState)

    graph.add_node("query_analyser",  node_query_analyser)
    graph.add_node("naive_retriever", node_naive_retriever)
    graph.add_node("generator",       node_generator)
    graph.add_node("grade_answer",    node_grade_answer)
    graph.add_node("force_refusal",   node_force_refusal)

    graph.set_entry_point("query_analyser")

    graph.add_conditional_edges("query_analyser",
        lambda s: END if s.get("refused") else "naive_retriever",
        {"naive_retriever": "naive_retriever", END: END})

    graph.add_edge("naive_retriever", "generator")
    graph.add_edge("generator",       "grade_answer")
    graph.add_conditional_edges("grade_answer", _route_after_grade_answer, {END: END})

    return graph.compile()


# ── v2: Static routing, single pass ───────────────────────────────────────
# query_analyser → router → retriever → generator → grade_answer → END
# Correct mode dispatch but no loop, no retry on failure.

def compile_v2():
    graph = StateGraph(AgentState)

    graph.add_node("query_analyser",       node_query_analyser)
    graph.add_node("router",               node_router)
    graph.add_node("naive_retriever",      node_naive_retriever)
    graph.add_node("local_graph_retriever",node_graph_retriever)
    graph.add_node("global_retriever",     node_community_retriever)
    graph.add_node("web_retriever",        node_web_retriever)
    graph.add_node("generator",            node_generator)
    graph.add_node("grade_answer",         node_grade_answer)
    graph.add_node("force_refusal",        node_force_refusal)

    graph.set_entry_point("query_analyser")

    graph.add_conditional_edges("query_analyser", _route_after_analyser,
        {"router": "router", END: END})

    graph.add_conditional_edges("router", _route_after_router, {
        "naive_retriever":       "naive_retriever",
        "local_graph_retriever": "local_graph_retriever",
        "global_retriever":      "global_retriever",
        "web_retriever":         "web_retriever",
    })

    # All retrievers → generator directly (no grade_context, no loop)
    for node in ["naive_retriever", "local_graph_retriever",
                 "global_retriever", "web_retriever"]:
        graph.add_edge(node, "generator")

    graph.add_edge("generator", "grade_answer")
    graph.add_conditional_edges("grade_answer", _route_after_grade_answer, {END: END})

    return graph.compile()


# ── v3: Agentic loop, no rewrite ──────────────────────────────────────────
# Full loop but on grade_context fail → router directly with original query.
# rewrite_query node is bypassed.

def _route_after_grade_context_v3(state: AgentState) -> str:
    grade       = state["grade_result"]
    loop_count  = state["loop_count"]
    context     = state.get("retrieved_context")
    current_mode = context.source_type if context else ""
    mode_history = state.get("mode_history", [])

    if grade and grade.passed:
        return "generator"
    if current_mode == "web" or "web" in mode_history:
        return "force_refusal"
    if loop_count >= LOOP_GUARD:
        return "web_retriever"
    # Re-route with original query — no rewrite
    return "router"


def compile_v3():
    graph = StateGraph(AgentState)

    graph.add_node("query_analyser",       node_query_analyser)
    graph.add_node("router",               node_router)
    graph.add_node("naive_retriever",      node_naive_retriever)
    graph.add_node("local_graph_retriever",node_graph_retriever)
    graph.add_node("global_retriever",     node_community_retriever)
    graph.add_node("web_retriever",        node_web_retriever)
    graph.add_node("grade_context",        node_grade_context)
    graph.add_node("generator",            node_generator)
    graph.add_node("grade_answer",         node_grade_answer)
    graph.add_node("force_refusal",        node_force_refusal)

    graph.set_entry_point("query_analyser")

    graph.add_conditional_edges("query_analyser", _route_after_analyser,
        {"router": "router", END: END})

    graph.add_conditional_edges("router", _route_after_router, {
        "naive_retriever":       "naive_retriever",
        "local_graph_retriever": "local_graph_retriever",
        "global_retriever":      "global_retriever",
        "web_retriever":         "web_retriever",
    })

    for node in ["naive_retriever", "local_graph_retriever",
                 "global_retriever", "web_retriever"]:
        graph.add_edge(node, "grade_context")

    graph.add_conditional_edges("grade_context", _route_after_grade_context_v3, {
        "generator":    "generator",
        "router":       "router",       # loop back without rewrite
        "web_retriever":"web_retriever",
        "force_refusal":"force_refusal",
    })

    graph.add_edge("force_refusal", END)
    graph.add_edge("generator",     "grade_answer")
    graph.add_conditional_edges("grade_answer", _route_after_grade_answer, {END: END})

    return graph.compile()

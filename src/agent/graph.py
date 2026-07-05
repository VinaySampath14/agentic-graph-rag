"""LangGraph state machine — wires all 9 nodes with conditional edges."""
from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import (
    node_query_analyser,
    node_router,
    node_naive_retriever,
    node_graph_retriever,
    node_community_retriever,
    node_ontology_retriever,
    node_web_retriever,
    node_grade_context,
    node_rewrite_query,
    node_generator,
    node_grade_answer,
    node_force_refusal,
)

LOOP_GUARD = 3


# ── Conditional edge functions ─────────────────────────────────────────────

def route_after_analyser(state: AgentState) -> str:
    if state.get("refused"):
        return END
    return "router"


def route_after_router(state: AgentState) -> str:
    intent = state["intent"]
    if intent == "graph":
        return "local_graph_retriever"
    elif intent == "community":
        return "global_retriever"
    elif intent == "ontology":
        return "ontology_retriever"
    elif intent == "web":
        return "web_retriever"
    return "naive_retriever"


def route_after_grade_context(state: AgentState) -> str:
    grade = state["grade_result"]
    loop_count = state["loop_count"]
    context = state.get("retrieved_context")
    current_mode = context.source_type if context else ""
    mode_history = state.get("mode_history", [])

    if grade and grade.passed:
        return "generator"

    # Web already tried and failed — structured refusal
    if current_mode == "web" or "web" in mode_history:
        return "force_refusal"

    # Loop guard — try web as last resort
    if loop_count >= LOOP_GUARD:
        return "web_retriever"

    return "rewrite_query"


def route_after_grade_answer(state: AgentState) -> str:
    if state.get("refused"):
        return END
    return END


# ── Build the graph ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("query_analyser", node_query_analyser)
    graph.add_node("router", node_router)
    graph.add_node("naive_retriever", node_naive_retriever)
    graph.add_node("local_graph_retriever", node_graph_retriever)
    graph.add_node("global_retriever", node_community_retriever)
    graph.add_node("ontology_retriever", node_ontology_retriever)
    graph.add_node("web_retriever", node_web_retriever)
    graph.add_node("grade_context", node_grade_context)
    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("generator", node_generator)
    graph.add_node("grade_answer", node_grade_answer)
    graph.add_node("force_refusal", node_force_refusal)

    # Entry point
    graph.set_entry_point("query_analyser")

    # query_analyser → router or END (if refused)
    graph.add_conditional_edges("query_analyser", route_after_analyser, {
        "router": "router",
        END: END,
    })

    # router → one of the four retrievers
    graph.add_conditional_edges("router", route_after_router, {
        "naive_retriever": "naive_retriever",
        "local_graph_retriever": "local_graph_retriever",
        "global_retriever": "global_retriever",
        "ontology_retriever": "ontology_retriever",
        "web_retriever": "web_retriever",
    })

    # All retrievers → grade_context
    graph.add_edge("naive_retriever", "grade_context")
    graph.add_edge("local_graph_retriever", "grade_context")
    graph.add_edge("global_retriever", "grade_context")
    graph.add_edge("ontology_retriever", "grade_context")
    graph.add_edge("web_retriever", "grade_context")

    # grade_context → generator | rewrite_query | web_retriever | force_refusal
    graph.add_conditional_edges("grade_context", route_after_grade_context, {
        "generator": "generator",
        "rewrite_query": "rewrite_query",
        "web_retriever": "web_retriever",
        "force_refusal": "force_refusal",
    })

    # force_refusal → END
    graph.add_edge("force_refusal", END)

    # rewrite_query → router (loop back)
    graph.add_edge("rewrite_query", "router")

    # generator → grade_answer
    graph.add_edge("generator", "grade_answer")

    # grade_answer → END
    graph.add_conditional_edges("grade_answer", route_after_grade_answer, {
        END: END,
    })

    return graph


def compile_graph():
    return build_graph().compile()

"""All LangGraph agent nodes."""
import json
import os
from datetime import datetime
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

from src.agent.state import AgentState
from src.retrievers import naive_retriever
from src.retrievers import graph_retriever
from src.retrievers import community_retriever
from src.retrievers import ontology_retriever
from src.retrievers import web_retriever
from src.retrievers.router import classify
from src.retrievers.context_budget import apply_budget
from src.retrievers.models import GradeResult, GeneratorOutput

load_dotenv()

PROMPTS_DIR = Path("prompts")
LOOP_GUARD = 3

_groq_client: Groq | None = None


def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.environ["GROQ_API_KEY"].strip())
    return _groq_client


def _load_prompt(filename: str) -> str:
    path = PROMPTS_DIR / filename
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(l for l in lines if not l.startswith("#")).strip()


def _trace_entry(node: str, decision: str, reason: str, **extras) -> dict:
    return {
        "node": node,
        "decision": decision,
        "reason": reason,
        "timestamp": datetime.utcnow().isoformat(),
        "extras": extras,
    }


def _groq_json(prompt: str, system: str = "") -> dict:
    import time
    client = _get_groq()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                wait = 30 * (attempt + 1)
                print(f"  Groq rate limit — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Groq failed after 3 attempts")


# ── Out-of-domain signals ──────────────────────────────────────────────────
OOD_SIGNALS = [
    "weather", "recipe", "cook", "sport", "football", "cricket",
    "stock", "price", "news", "politics", "movie", "music",
]

GREETING_SIGNALS = [
    "how are you", "how are u", "hello", "hi there", "hey there",
    "good morning", "good evening", "good afternoon", "what's up",
    "whats up", "sup ", "how do you do", "nice to meet",
]


def node_query_analyser(state: AgentState) -> AgentState:
    query = state["query"]
    query_lower = query.lower()
    trace = list(state.get("agent_trace", []))

    # Greeting check
    for signal in GREETING_SIGNALS:
        if signal in query_lower:
            trace.append(_trace_entry(
                "query_analyser", "refused", "Greeting detected — not a research query",
            ))
            return {
                **state,
                "refused": True,
                "refusal_reason": "Hi! I'm a research assistant for CS/AI papers. Ask me about methods, authors, trends, or specific papers.",
                "agent_trace": trace,
            }

    # Out-of-domain check
    for signal in OOD_SIGNALS:
        if signal in query_lower:
            trace.append(_trace_entry(
                "query_analyser", "refused",
                f"Out-of-domain signal detected: '{signal}'",
            ))
            return {
                **state,
                "refused": True,
                "refusal_reason": f"Query appears out of domain ('{signal}'). This system answers questions about CS/AI research papers only.",
                "agent_trace": trace,
            }

    # Classify intent
    routing = classify(query, mode_history=[])
    intent = routing["primary_mode"]

    trace.append(_trace_entry(
        "query_analyser", f"intent={intent}",
        f"Classified as '{intent}' with confidence {routing['confidence']}",
        scores=routing.get("all_scores", {}),
    ))

    return {
        **state,
        "intent": intent,
        "low_confidence": routing["low_confidence"],
        "fallback_mode": routing.get("fallback_mode"),
        "loop_count": 0,
        "mode_history": [],
        "agent_trace": trace,
        "refused": False,
        "refusal_reason": "",
        "rewritten_query": "",
        "retrieved_context": None,
        "grade_result": None,
        "answer": "",
        "citations": [],
        "confidence_proxy": 0.0,
    }


def node_router(state: AgentState) -> AgentState:
    trace = list(state["agent_trace"])
    query = state.get("rewritten_query") or state["query"]
    mode_history = list(state["mode_history"])

    routing = classify(query, mode_history=mode_history)
    intent = routing["primary_mode"]

    trace.append(_trace_entry(
        "router", f"dispatch={intent}",
        f"Routing to '{intent}' (confidence={routing['confidence']}, history={mode_history})",
        low_confidence=routing["low_confidence"],
        fallback_mode=routing.get("fallback_mode"),
    ))

    return {
        **state,
        "intent": intent,
        "low_confidence": routing["low_confidence"],
        "fallback_mode": routing.get("fallback_mode"),
        "agent_trace": trace,
    }


def node_naive_retriever(state: AgentState) -> AgentState:
    trace = list(state["agent_trace"])
    query = state.get("rewritten_query") or state["query"]

    result = naive_retriever.retrieve(query)

    trace.append(_trace_entry(
        "naive_retriever", "retrieved",
        f"Vector search returned {len(result.context_text)} chars",
        source_type=result.source_type,
        truncated=result.truncated,
    ))

    return {**state, "retrieved_context": result, "agent_trace": trace}


def node_graph_retriever(state: AgentState) -> AgentState:
    trace = list(state["agent_trace"])
    query = state.get("rewritten_query") or state["query"]

    result = graph_retriever.retrieve(query)

    trace.append(_trace_entry(
        "local_graph_retriever", "retrieved",
        f"Graph traversal returned {len(result.context_text)} chars",
        cypher_query=result.cypher_query_used,
        source_type=result.source_type,
    ))

    return {**state, "retrieved_context": result, "agent_trace": trace}


def node_community_retriever(state: AgentState) -> AgentState:
    trace = list(state["agent_trace"])
    query = state.get("rewritten_query") or state["query"]

    result = community_retriever.retrieve(query)

    trace.append(_trace_entry(
        "global_retriever", "retrieved",
        f"Community retrieval returned {len(result.communities_used)} communities",
        communities_used=result.communities_used,
        source_type=result.source_type,
    ))

    return {**state, "retrieved_context": result, "agent_trace": trace}


def node_ontology_retriever(state: AgentState) -> AgentState:
    trace = list(state["agent_trace"])
    query = state.get("rewritten_query") or state["query"]

    result = ontology_retriever.retrieve(query)

    trace.append(_trace_entry(
        "ontology_retriever", "retrieved",
        f"SPARQL query returned {len(result.context_text)} chars",
        sparql_query=result.sparql_query_used,
        source_type=result.source_type,
        truncated=result.truncated,
    ))

    return {**state, "retrieved_context": result, "agent_trace": trace}


def node_web_retriever(state: AgentState) -> AgentState:
    trace = list(state["agent_trace"])
    query = state.get("rewritten_query") or state["query"]

    result = web_retriever.retrieve(query)

    trace.append(_trace_entry(
        "web_retriever", "retrieved",
        "Fallback to web search — all corpus modes exhausted or loop guard triggered",
        source_type="web",
    ))

    return {**state, "retrieved_context": result, "agent_trace": trace}


def node_grade_context(state: AgentState) -> AgentState:
    trace = list(state["agent_trace"])
    query = state.get("rewritten_query") or state["query"]
    context = state["retrieved_context"]

    prompt_template = _load_prompt("grade_context_v2.txt")
    prompt = prompt_template.replace("{query}", query).replace(
        "{context}", context.context_text[:3000]
    )

    try:
        raw = _groq_json(prompt)
        grade = GradeResult(
            passed=bool(raw.get("pass", False)),
            reason=str(raw.get("reason", "")),
        )
    except Exception as e:
        grade = GradeResult(passed=False, reason=f"Grading failed: {e}")

    trace.append(_trace_entry(
        "grade_context",
        "pass" if grade.passed else "fail",
        grade.reason,
        loop_count=state["loop_count"],
        mode=context.source_type if context else "unknown",
        prompt_version="grade_context_v2",
    ))

    return {**state, "grade_result": grade, "agent_trace": trace}


def node_rewrite_query(state: AgentState) -> AgentState:
    trace = list(state["agent_trace"])
    original_query = state["query"]
    failed_mode = state["retrieved_context"].source_type if state["retrieved_context"] else "unknown"
    mode_history = list(state["mode_history"]) + [failed_mode]
    loop_count = state["loop_count"] + 1

    # Determine next mode
    routing = classify(original_query, mode_history=mode_history)
    next_mode = routing["primary_mode"]

    prompt_template = _load_prompt("rewrite_query_v1.txt")
    prompt = (
        prompt_template
        .replace("{failed_mode}", failed_mode)
        .replace("{next_mode}", next_mode)
        .replace("{original_query}", original_query)
        .replace("{failure_reason}", state["grade_result"].reason if state["grade_result"] else "")
    )

    try:
        import time
        client = _get_groq()
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                rewritten = response.choices[0].message.content.strip().strip('"')
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 30 * (attempt + 1)
                    print(f"  Groq rate limit — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    rewritten = original_query
                    break
    except Exception:
        rewritten = original_query

    trace.append(_trace_entry(
        "rewrite_query", f"rewritten for {next_mode}",
        f"Original: '{original_query[:60]}' → Rewritten: '{rewritten[:60]}'",
        failed_mode=failed_mode,
        next_mode=next_mode,
        loop_count=loop_count,
        prompt_version="rewrite_query_v1",
    ))

    return {
        **state,
        "rewritten_query": rewritten,
        "mode_history": mode_history,
        "loop_count": loop_count,
        "agent_trace": trace,
    }


def node_generator(state: AgentState) -> AgentState:
    trace = list(state["agent_trace"])
    query = state.get("rewritten_query") or state["query"]
    context = state["retrieved_context"]

    # Apply context budget
    if context.source_type == "graph":
        budgeted_context, truncated = apply_budget(graph_context=context.context_text)
    elif context.source_type == "community":
        budgeted_context, truncated = apply_budget(community_context=context.context_text)
    else:
        budgeted_context, truncated = apply_budget(vector_context=context.context_text)

    prompt_template = _load_prompt("generator_v1.txt")
    prompt = (
        prompt_template
        .replace("{query}", query)
        .replace("{context}", budgeted_context)
        .replace("{mode}", context.source_type)
    )

    try:
        raw = _groq_json(prompt)
        answer = str(raw.get("answer", ""))
        citations = list(raw.get("citations", []))
        confidence_proxy = float(raw.get("confidence_proxy", 0.5))
    except Exception as e:
        answer = f"Generation failed: {e}"
        citations = []
        confidence_proxy = 0.0

    trace.append(_trace_entry(
        "generator", "generated",
        f"Answer generated ({len(answer)} chars), {len(citations)} citations",
        mode=context.source_type,
        truncated=truncated,
        prompt_version="generator_v1",
    ))

    return {
        **state,
        "answer": answer,
        "citations": citations,
        "confidence_proxy": confidence_proxy,
        "agent_trace": trace,
    }


def node_force_refusal(state: AgentState) -> AgentState:
    trace = list(state["agent_trace"])
    trace.append(_trace_entry(
        "force_refusal", "refused",
        "All retrieval modes exhausted including web fallback — returning structured refusal",
    ))
    return {
        **state,
        "refused": True,
        "refusal_reason": "Unable to find sufficient context across all retrieval modes. Please rephrase your query or try a more specific question.",
        "agent_trace": trace,
    }


def node_grade_answer(state: AgentState) -> AgentState:
    trace = list(state["agent_trace"])
    query = state.get("rewritten_query") or state["query"]
    context = state["retrieved_context"]

    prompt_template = _load_prompt("grade_answer_v1.txt")
    prompt = (
        prompt_template
        .replace("{query}", query)
        .replace("{context}", context.context_text[:2000] if context else "")
        .replace("{answer}", state["answer"])
    )

    try:
        raw = _groq_json(prompt)
        passed = bool(raw.get("pass", False))
        reason = str(raw.get("reason", ""))
    except Exception as e:
        passed = False
        reason = f"Answer grading failed: {e}"

    trace.append(_trace_entry(
        "grade_answer",
        "pass" if passed else "fail",
        reason,
        prompt_version="grade_answer_v1",
    ))

    if not passed:
        return {
            **state,
            "refused": True,
            "refusal_reason": f"Answer failed quality check: {reason}",
            "agent_trace": trace,
        }

    return {**state, "refused": False, "agent_trace": trace}

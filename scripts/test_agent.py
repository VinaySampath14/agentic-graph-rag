"""Integration test — run the full agent end-to-end on test queries."""
import json
from src.agent.graph import compile_graph

graph = compile_graph()


def run_query(query: str) -> dict:
    initial_state = {"query": query, "agent_trace": []}
    result = graph.invoke(initial_state)
    return result


def print_result(query: str, result: dict) -> None:
    print(f"\n{'='*70}")
    print(f"QUERY: {query}")
    print(f"{'='*70}")

    if result.get("refused"):
        print(f"REFUSED: {result.get('refusal_reason')}")
    else:
        print(f"ANSWER: {result.get('answer', '')[:300]}")
        print(f"CITATIONS: {result.get('citations', [])}")
        print(f"MODE USED: {result.get('retrieved_context', {}).source_type if result.get('retrieved_context') else 'unknown'}")
        print(f"LOOP COUNT: {result.get('loop_count', 0)}")

    print(f"\nAGENT TRACE:")
    for entry in result.get("agent_trace", []):
        print(f"  [{entry['node']}] {entry['decision']} — {entry['reason'][:80]}")


# ── Test queries ───────────────────────────────────────────────────────────

TEST_QUERIES = [
    # Relational — should use graph mode, single loop
    ("What papers did Yang Liu write?", "relational"),

    # Thematic — should use community mode, single loop
    ("What are the main trends in LLM reasoning research?", "thematic"),

    # Out-of-domain — should refuse immediately, no retrieval
    ("What is the weather in London today?", "ood"),
]

if __name__ == "__main__":
    print("Running agent integration tests...\n")
    for query, qtype in TEST_QUERIES:
        print(f"\nTesting [{qtype}]: {query}")
        try:
            result = run_query(query)
            print_result(query, result)
        except Exception as e:
            print(f"ERROR: {e}")

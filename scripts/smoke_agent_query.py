"""Run one or more questions through the compiled agent and print decision traces."""
import argparse
import json

from src.agent.graph import compile_graph


def initial_state(query: str) -> dict:
    return {
        "query": query,
        "rewritten_query": query,
        "intent": "",
        "graph_backend": "neo4j",
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="+")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print compact routing and grading results instead of full answers/traces.",
    )
    args = parser.parse_args()

    graph = compile_graph()
    for query in args.query:
        result = graph.invoke(initial_state(query))
        print(f"\nQuestion: {query}")
        print(f"Intent: {result.get('intent')}")
        print(f"Graph backend: {result.get('graph_backend')}")
        print(f"Modes tried: {result.get('mode_history')}")
        print(f"Refused: {result.get('refused')}")
        if args.summary:
            decisions = [
                f"{entry.get('node')}={entry.get('decision')}"
                for entry in result.get("agent_trace", [])
            ]
            print(f"Decisions: {', '.join(decisions)}")
            print(f"Citations: {len(result.get('citations', []))}")
            if result.get("refusal_reason"):
                print(f"Refusal reason: {result['refusal_reason']}")
            continue

        print(f"Answer: {result.get('answer')}")
        if result.get("refusal_reason"):
            print(f"Refusal reason: {result['refusal_reason']}")
        print("Trace:")
        print(json.dumps(result.get("agent_trace", []), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

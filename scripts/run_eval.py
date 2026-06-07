"""Run all 80 eval queries through the agent and save raw outputs.

Checkpoints every 10 queries so a crash mid-run doesn't lose work.
Resume by re-running — already-completed query ids are skipped.

Usage:
    python scripts/run_eval.py           # v4 (full system, default)
    python scripts/run_eval.py --version v1
    python scripts/run_eval.py --version v2
    python scripts/run_eval.py --version v3

Output: data/eval/eval_results_{version}.jsonl
"""
import argparse
import json
import time
from pathlib import Path

QUERIES_FILE  = Path("data/eval/eval_queries_validated.jsonl")
ANSWERS_FILE  = Path("data/eval/reference_answers.jsonl")
SLEEP_BETWEEN = 2        # seconds — stay within Groq 30 req/min
CHECKPOINT_N  = 10       # flush to disk every N queries


def get_graph(version: str):
    if version == "v1":
        from src.agent.ablations import compile_v1
        return compile_v1()
    elif version == "v2":
        from src.agent.ablations import compile_v2
        return compile_v2()
    elif version == "v3":
        from src.agent.ablations import compile_v3
        return compile_v3()
    else:
        from src.agent.graph import compile_graph
        return compile_graph()


def load_queries() -> list[dict]:
    queries = []
    with open(QUERIES_FILE, encoding="utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries


def load_reference_answers() -> dict[int, str]:
    refs: dict[int, str] = {}
    with open(ANSWERS_FILE, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            refs[rec["id"]] = rec["reference_answer"]
    return refs


def load_completed_ids(output_file: Path) -> set[int]:
    if not output_file.exists():
        return set()
    done = set()
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            done.add(rec["id"])
    return done


def run_query(graph, query: str) -> tuple[dict, float]:
    start = time.perf_counter()
    state = graph.invoke({"query": query, "agent_trace": []})
    elapsed_ms = (time.perf_counter() - start) * 1000
    return state, elapsed_ms


def state_to_record(qid: int, query: str, qtype: str, ref: str,
                    state: dict, latency_ms: float) -> dict:
    context = state.get("retrieved_context")
    mode_history = state.get("mode_history") or []

    rewrite_triggered = any(
        entry.get("node") == "rewrite_query"
        for entry in (state.get("agent_trace") or [])
    )

    # first_mode_success: answered in one pass without any rewrites
    first_mode_success = (
        state.get("loop_count", 0) == 0
        and not bool(state.get("refused"))
    )

    return {
        "id":               qid,
        "query":            query,
        "query_type":       qtype,
        "reference_answer": ref,
        "answer":           state.get("answer", ""),
        "refused":          bool(state.get("refused")),
        "refusal_reason":   state.get("refusal_reason", ""),
        "citations":        state.get("citations") or [],
        "loop_count":       state.get("loop_count", 0),
        "mode_history":     mode_history,
        "first_mode":       mode_history[0] if mode_history else "",
        "final_mode":       (context.source_type if context else ""),
        "context_text":     (context.context_text if context else ""),
        "rewrite_triggered": rewrite_triggered,
        "first_mode_success": first_mode_success,
        "latency_ms":       round(latency_ms, 1),
        "agent_trace":      state.get("agent_trace") or [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v4", choices=["v1","v2","v3","v4"])
    args = parser.parse_args()

    version     = args.version
    output_file = Path(f"data/eval/eval_results_{version}.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    queries  = load_queries()
    refs     = load_reference_answers()
    done_ids = load_completed_ids(output_file)

    pending = [q for q in queries if q["id"] not in done_ids]
    total   = len(queries)

    print(f"Version: {version} | Output: {output_file}")
    print(f"Total queries: {total} | Already done: {len(done_ids)} | Remaining: {len(pending)}")

    if not pending:
        print(f"All queries already complete. Delete {output_file} to re-run.")
        return

    graph = get_graph(version)
    print(f"Agent ({version}) compiled. Starting eval...\n")

    buffer: list[dict] = []
    completed = 0

    for q in pending:
        qid   = q["id"]
        query = q["query"]
        qtype = q.get("query_type", "unknown")
        ref   = refs.get(qid, "")

        print(f"[{len(done_ids) + completed + 1}/{total}] id={qid} ({qtype}): {query[:70]}")

        try:
            state, latency_ms = run_query(graph, query)
            record = state_to_record(qid, query, qtype, ref, state, latency_ms)
            print(f"  → mode={record['final_mode']} loops={record['loop_count']} "
                  f"refused={record['refused']} latency={record['latency_ms']}ms")
        except Exception as e:
            print(f"  ERROR: {e}")
            record = {
                "id": qid, "query": query, "query_type": qtype,
                "reference_answer": ref, "answer": "", "refused": True,
                "refusal_reason": f"EVAL_ERROR: {e}", "citations": [],
                "loop_count": 0, "mode_history": [], "first_mode": "",
                "final_mode": "", "rewrite_triggered": False,
                "first_mode_success": False, "latency_ms": 0.0, "agent_trace": [],
            }

        buffer.append(record)
        completed += 1

        # Checkpoint every N queries
        if completed % CHECKPOINT_N == 0:
            with open(output_file, "a", encoding="utf-8") as f:
                for rec in buffer:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  [checkpoint] flushed {len(buffer)} records to disk")
            buffer.clear()

        if completed < len(pending):
            time.sleep(SLEEP_BETWEEN)

    # Flush any remaining
    if buffer:
        with open(output_file, "a", encoding="utf-8") as f:
            for rec in buffer:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [final flush] {len(buffer)} records")

    print(f"\nDone. Results saved to {output_file}")
    print(f"Total completed: {len(done_ids) + completed}/{total}")


if __name__ == "__main__":
    main()

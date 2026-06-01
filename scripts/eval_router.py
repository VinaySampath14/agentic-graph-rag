"""Evaluate router accuracy on 20 hand-labelled queries."""
import json
from pathlib import Path
from src.retrievers.router import classify

TEST_SET = Path("data/eval/router_test_set.jsonl")

queries = [json.loads(l) for l in TEST_SET.read_text(encoding="utf-8").splitlines()]

correct = 0
wrong = []

print(f"{'Query':<55} {'Expected':<12} {'Predicted':<12} {'OK'}")
print("-" * 90)

for q in queries:
    result = classify(q["query"])
    predicted = result["primary_mode"]
    expected = q["expected_mode"]
    ok = predicted == expected
    if ok:
        correct += 1
    else:
        wrong.append(q)
    print(f"{q['query'][:54]:<55} {expected:<12} {predicted:<12} {'✓' if ok else '✗'}")

accuracy = correct / len(queries) * 100
print(f"\nAccuracy: {correct}/{len(queries)} = {accuracy:.1f}%")

if wrong:
    print("\nMisclassified:")
    for q in wrong:
        print(f"  - {q['query']} (expected: {q['expected_mode']})")

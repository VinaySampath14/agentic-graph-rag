"""Experiment: derive method categories bottom-up via Leiden clustering on
method co-occurrence, then compare against the handpicked 5-category scheme
in data/processed/method_classifications.json.

Pulls live from Neo4j (falls back to local backups if the DB is unreachable):
    data/backups/method_nodes_backup.jsonl
    data/backups/uses_method_edges_backup.jsonl
"""
import json
import os
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import igraph as ig
import leidenalg
from dotenv import load_dotenv

load_dotenv()

METHOD_NODES_BACKUP = Path("data/backups/method_nodes_backup.jsonl")
USES_EDGES_BACKUP = Path("data/backups/uses_method_edges_backup.jsonl")
CLASSIFICATIONS = Path("data/processed/method_classifications.json")

MIN_CLUSTER_SIZE = 3  # clusters smaller than this are reported separately as "unclustered"


def load_from_neo4j() -> tuple[list[str], dict[str, set[str]]] | None:
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.environ["NEO4J_URI"],
            auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
        )
        driver.verify_connectivity()
    except Exception as e:
        print(f"Neo4j unreachable ({e}) — falling back to local backups.")
        return None

    with driver.session() as session:
        methods = [r["name"] for r in session.run("MATCH (m:Method) RETURN m.name AS name")]
        paper_methods: dict[str, set[str]] = defaultdict(set)
        for r in session.run(
            "MATCH (p:Paper)-[:USES_METHOD]->(m:Method) RETURN p.arxiv_id AS pid, m.name AS name"
        ):
            paper_methods[r["pid"]].add(r["name"])
    driver.close()
    print(f"Loaded live from Neo4j: {len(methods)} methods, {len(paper_methods)} papers")
    return methods, paper_methods


def load_from_backup() -> tuple[list[str], dict[str, set[str]]]:
    with open(METHOD_NODES_BACKUP, encoding="utf-8") as f:
        methods = [json.loads(l)["name"] for l in f]
    paper_methods: dict[str, set[str]] = defaultdict(set)
    with open(USES_EDGES_BACKUP, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            paper_methods[r["arxiv_id"]].add(r["method"])
    print(f"Loaded from backup: {len(methods)} methods, {len(paper_methods)} papers")
    return methods, paper_methods


def build_cooccurrence(paper_methods: dict[str, set[str]]) -> Counter:
    weights: Counter = Counter()
    for methods in paper_methods.values():
        if len(methods) < 2:
            continue
        for m1, m2 in combinations(sorted(methods), 2):
            weights[(m1, m2)] += 1
    return weights


def run_leiden(methods: list[str], weights: Counter) -> dict[str, int]:
    idx = {m: i for i, m in enumerate(methods)}
    edges, edge_weights = [], []
    for (m1, m2), w in weights.items():
        if m1 in idx and m2 in idx:
            edges.append((idx[m1], idx[m2]))
            edge_weights.append(w)

    g = ig.Graph(n=len(methods), edges=edges)
    g.vs["name"] = methods
    g.es["weight"] = edge_weights

    print(f"Graph: {g.vcount()} method nodes, {g.ecount()} co-occurrence edges")

    partition = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, weights="weight", seed=42,
    )
    print(f"Leiden found {len(partition)} raw clusters")

    return {methods[v]: cid for cid, community in enumerate(partition) for v in community}


def main() -> None:
    loaded = load_from_neo4j()
    methods, paper_methods = loaded if loaded else load_from_backup()

    print(f"{len(methods)} unique methods, {len(paper_methods)} papers with >=1 method")

    weights = build_cooccurrence(paper_methods)
    print(f"{len(weights)} unique co-occurring method pairs")

    cluster_map = run_leiden(methods, weights)

    clusters: dict[int, list[str]] = defaultdict(list)
    for m, cid in cluster_map.items():
        clusters[cid].append(m)

    sizes = Counter(len(v) for v in clusters.values())
    print(f"\nCluster size distribution: {dict(sorted(sizes.items()))}")

    real_clusters = {cid: ms for cid, ms in clusters.items() if len(ms) >= MIN_CLUSTER_SIZE}
    singleton_or_small = {cid: ms for cid, ms in clusters.items() if len(ms) < MIN_CLUSTER_SIZE}
    total_small = sum(len(v) for v in singleton_or_small.values())

    print(f"\n{len(real_clusters)} clusters with >= {MIN_CLUSTER_SIZE} methods "
          f"({sum(len(v) for v in real_clusters.values())} methods)")
    print(f"{len(singleton_or_small)} tiny/singleton clusters, "
          f"{total_small} methods total with no strong co-occurrence signal")

    handpicked_raw = json.loads(CLASSIFICATIONS.read_text(encoding="utf-8"))
    handpicked = {m: v["subclass"] for m, v in handpicked_raw.items()}

    print("\n" + "=" * 70)
    print("REAL CLUSTERS (co-occurrence-derived) vs. handpicked categories")
    print("=" * 70)

    for cid, ms in sorted(real_clusters.items(), key=lambda x: -len(x[1])):
        labeled = [(m, handpicked[m]) for m in ms if m in handpicked]
        label_counts = Counter(l for _, l in labeled)
        purity = (label_counts.most_common(1)[0][1] / len(labeled)) if labeled else None

        print(f"\n--- Cluster {cid} ({len(ms)} methods, {len(labeled)} handpicked-labeled) ---")
        print(f"Methods: {', '.join(sorted(ms)[:15])}{' ...' if len(ms) > 15 else ''}")
        if labeled:
            print(f"Handpicked label distribution: {dict(label_counts)}")
            print(f"Purity (majority-label agreement): {purity:.0%}")
        else:
            print("No methods in this cluster were in the handpicked 96 — entirely new territory.")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_purities = []
    for cid, ms in real_clusters.items():
        labeled = [handpicked[m] for m in ms if m in handpicked]
        if labeled:
            label_counts = Counter(labeled)
            all_purities.append(label_counts.most_common(1)[0][1] / len(labeled))
    if all_purities:
        print(f"Average cluster purity vs. handpicked scheme: {sum(all_purities)/len(all_purities):.0%} "
              f"(across {len(all_purities)} clusters that had >=1 handpicked method)")
    print(f"Methods covered by handpicked scheme: {sum(1 for m in methods if m in handpicked)}/{len(methods)}")
    print(f"Methods covered by data-driven real clusters: "
          f"{sum(len(v) for v in real_clusters.values())}/{len(methods)}")


if __name__ == "__main__":
    main()

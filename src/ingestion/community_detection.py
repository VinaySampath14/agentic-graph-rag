"""Detect communities using Leiden algorithm and write results back to Neo4j."""
import json
import os
from pathlib import Path

import igraph as ig
import leidenalg
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

OUTPUT_FILE = Path("data/processed/communities.jsonl")


def get_driver():
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )


def export_graph(session) -> tuple[list[str], list[tuple[int, int]]]:
    """Export Paper nodes and AUTHORED_BY co-authorship edges from Neo4j."""
    print("Exporting paper nodes...")
    result = session.run("MATCH (p:Paper) RETURN p.arxiv_id AS arxiv_id")
    arxiv_ids = [r["arxiv_id"] for r in result]
    id_to_idx = {arxiv_id: i for i, arxiv_id in enumerate(arxiv_ids)}

    print("Exporting co-authorship edges...")
    result = session.run("""
        MATCH (p1:Paper)-[:AUTHORED_BY]->(a:Author)<-[:AUTHORED_BY]-(p2:Paper)
        WHERE p1.arxiv_id < p2.arxiv_id
        RETURN p1.arxiv_id AS id1, p2.arxiv_id AS id2
    """)
    edge_set = set()
    for r in result:
        if r["id1"] in id_to_idx and r["id2"] in id_to_idx:
            edge_set.add((id_to_idx[r["id1"]], id_to_idx[r["id2"]]))

    print("Exporting method co-occurrence edges...")
    result = session.run("""
        MATCH (p1:Paper)-[:USES_METHOD]->(m:Method)<-[:USES_METHOD]-(p2:Paper)
        WHERE p1.arxiv_id < p2.arxiv_id
        RETURN p1.arxiv_id AS id1, p2.arxiv_id AS id2
    """)
    for r in result:
        if r["id1"] in id_to_idx and r["id2"] in id_to_idx:
            edge_set.add((id_to_idx[r["id1"]], id_to_idx[r["id2"]]))

    edges = list(edge_set)
    print(f"  {len(arxiv_ids)} nodes, {len(edges)} total edges (co-authorship + method co-occurrence)")
    return arxiv_ids, edges


def run_leiden(arxiv_ids: list[str], edges: list[tuple[int, int]]) -> dict[str, int]:
    """Run Leiden community detection, return arxiv_id → community_id mapping."""
    print("Building igraph...")
    g = ig.Graph(n=len(arxiv_ids), edges=edges)
    g.vs["arxiv_id"] = arxiv_ids

    if g.ecount() == 0:
        print("  No edges found — assigning all papers to community 0")
        return {arxiv_id: 0 for arxiv_id in arxiv_ids}

    print("Running Leiden algorithm...")
    partition = leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition,
        seed=42,
    )

    community_map = {}
    for community_id, community in enumerate(partition):
        for vertex_idx in community:
            community_map[arxiv_ids[vertex_idx]] = community_id

    sizes = [len(c) for c in partition]
    print(f"  Found {len(partition)} communities")
    print(f"  Sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")
    return community_map


def write_communities_to_neo4j(community_map: dict[str, int], session) -> None:
    print("Writing community IDs back to Neo4j...")
    for arxiv_id, community_id in community_map.items():
        session.run(
            "MATCH (p:Paper {arxiv_id: $arxiv_id}) SET p.community_id = $community_id",
            arxiv_id=arxiv_id,
            community_id=community_id,
        )
    print(f"  Updated {len(community_map)} Paper nodes with community_id")


def merge_small_communities(
    community_map: dict[str, int], min_size: int = 5
) -> dict[str, int]:
    """Merge communities smaller than min_size into the largest community."""
    from collections import Counter
    counts = Counter(community_map.values())
    large = {cid for cid, size in counts.items() if size >= min_size}

    if not large:
        return community_map

    # Find the largest community to absorb singletons
    largest = max(large, key=lambda cid: counts[cid])

    merged = {}
    for arxiv_id, cid in community_map.items():
        merged[arxiv_id] = cid if cid in large else largest

    before = len(counts)
    after = len(set(merged.values()))
    print(f"  Merged {before - after} small communities into community {largest}")
    print(f"  Communities before: {before} → after: {after}")
    return merged


def create_community_nodes(community_map: dict[str, int], session) -> None:
    """Create Community nodes with paper lists for summary generation."""
    print("Creating Community nodes...")
    from collections import defaultdict
    communities: dict[int, list[str]] = defaultdict(list)
    for arxiv_id, cid in community_map.items():
        communities[cid].append(arxiv_id)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    records = []

    for community_id, arxiv_ids in communities.items():
        # Fetch paper titles for this community
        result = session.run("""
            MATCH (p:Paper) WHERE p.arxiv_id IN $ids
            RETURN p.arxiv_id AS arxiv_id, p.title AS title
        """, ids=arxiv_ids)
        papers = [{"arxiv_id": r["arxiv_id"], "title": r["title"]} for r in result]

        session.run("""
            MERGE (c:Community {community_id: $community_id})
            SET c.size = $size, c.theme = '', c.summary = ''
        """, community_id=community_id, size=len(arxiv_ids))

        session.run("""
            MATCH (p:Paper) WHERE p.arxiv_id IN $ids
            MATCH (c:Community {community_id: $community_id})
            MERGE (p)-[:BELONGS_TO]->(c)
        """, ids=arxiv_ids, community_id=community_id)

        records.append({
            "community_id": community_id,
            "size": len(arxiv_ids),
            "papers": papers,
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  {len(communities)} Community nodes created")
    print(f"  Community data saved to {OUTPUT_FILE}")


def main() -> None:
    driver = get_driver()
    with driver.session() as session:
        arxiv_ids, edges = export_graph(session)
        community_map = run_leiden(arxiv_ids, edges)
        community_map = merge_small_communities(community_map, min_size=5)
        write_communities_to_neo4j(community_map, session)
        create_community_nodes(community_map, session)

    driver.close()
    print("\nCommunity detection complete.")


if __name__ == "__main__":
    main()

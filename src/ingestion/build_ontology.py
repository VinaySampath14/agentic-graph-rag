"""
Export Neo4j graph to RDF triples, apply method subclass hierarchy,
run OWL reasoner to infer relatedWork edges, serialise to Turtle.
"""
import json
import os
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from neo4j import GraphDatabase
from rdflib import Graph, Namespace, RDF, RDFS, OWL, Literal, URIRef, XSD
import owlrl

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────
SCHEMA_FILE  = Path("ontology/arxiv_cs.ttl")
CLASSIF_FILE = Path("data/processed/method_classifications.json")
OUTPUT_FILE  = Path("ontology/arxiv_cs_populated.ttl")

# ── Namespaces ────────────────────────────────────────────────────────────────
EX = Namespace("http://arxiv-cs.org/ontology#")


# ── URI helpers ───────────────────────────────────────────────────────────────

def uri_paper(arxiv_id: str) -> URIRef:
    return EX[f"paper_{quote(arxiv_id, safe='')}"]

def uri_author(name: str) -> URIRef:
    return EX[f"author_{quote(name.strip(), safe='')}"]

def uri_method(name: str) -> URIRef:
    return EX[f"method_{quote(name.strip(), safe='')}"]

def uri_institution(name: str) -> URIRef:
    return EX[f"inst_{quote(name.strip(), safe='')}"]

def uri_community(cid: str) -> URIRef:
    return EX[f"community_{quote(str(cid), safe='')}"]


# ── Neo4j export ──────────────────────────────────────────────────────────────

def export_from_neo4j(g: Graph) -> dict:
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )
    stats = {
        "papers": 0, "authors": 0, "institutions": 0,
        "methods": 0, "communities": 0,
        "authored_by": 0, "from_institution": 0,
        "uses_method": 0, "belongs_to_community": 0,
    }

    with driver.session() as session:

        # Papers
        print("  Exporting Paper nodes...")
        for row in session.run("MATCH (p:Paper) RETURN p"):
            p = row["p"]
            uri = uri_paper(p["arxiv_id"])
            g.add((uri, RDF.type, EX.Paper))
            g.add((uri, RDF.type, OWL.NamedIndividual))
            g.add((uri, RDFS.label, Literal(p["title"])))
            g.add((uri, EX.arxivId, Literal(p["arxiv_id"])))
            g.add((uri, EX.title, Literal(p["title"])))
            if p.get("year"):
                g.add((uri, EX.year, Literal(int(p["year"]), datatype=XSD.integer)))
            if p.get("venue"):
                g.add((uri, EX.venue, Literal(p["venue"])))
            stats["papers"] += 1

        # Authors
        print("  Exporting Author nodes...")
        for row in session.run("MATCH (a:Author) RETURN a"):
            a = row["a"]
            uri = uri_author(a["name"])
            g.add((uri, RDF.type, EX.Author))
            g.add((uri, RDF.type, OWL.NamedIndividual))
            g.add((uri, RDFS.label, Literal(a["name"])))
            g.add((uri, EX.authorName, Literal(a["name"])))
            stats["authors"] += 1

        # Institutions
        print("  Exporting Institution nodes...")
        for row in session.run("MATCH (i:Institution) RETURN i"):
            inst = row["i"]
            uri = uri_institution(inst["name"])
            g.add((uri, RDF.type, EX.Institution))
            g.add((uri, RDF.type, OWL.NamedIndividual))
            g.add((uri, RDFS.label, Literal(inst["name"])))
            g.add((uri, EX.institutionName, Literal(inst["name"])))
            stats["institutions"] += 1

        # Method nodes (from Neo4j — just the ones actually used in papers)
        print("  Exporting Method nodes...")
        for row in session.run("MATCH (m:Method) RETURN m"):
            m = row["m"]
            uri = uri_method(m["name"])
            g.add((uri, RDF.type, EX.Method))
            g.add((uri, RDF.type, OWL.NamedIndividual))
            g.add((uri, RDFS.label, Literal(m["name"])))
            g.add((uri, EX.methodName, Literal(m["name"])))
            stats["methods"] += 1

        # Community nodes
        print("  Exporting Community nodes...")
        for row in session.run("MATCH (c:Community) RETURN c"):
            c = row["c"]
            cid = str(c.get("community_id", c.element_id))
            uri = uri_community(cid)
            g.add((uri, RDF.type, EX.Community))
            g.add((uri, RDF.type, OWL.NamedIndividual))
            if c.get("theme"):
                g.add((uri, EX.communityTheme, Literal(c["theme"])))
            if c.get("size"):
                g.add((uri, EX.communitySize, Literal(int(c["size"]), datatype=XSD.integer)))
            stats["communities"] += 1

        # AUTHORED_BY edges
        print("  Exporting AUTHORED_BY edges...")
        for row in session.run(
            "MATCH (p:Paper)-[:AUTHORED_BY]->(a:Author) RETURN p.arxiv_id AS pid, a.name AS aname"
        ):
            g.add((uri_paper(row["pid"]), EX.authoredBy, uri_author(row["aname"])))
            stats["authored_by"] += 1

        # FROM_INSTITUTION edges
        print("  Exporting FROM_INSTITUTION edges...")
        for row in session.run(
            "MATCH (a:Author)-[:FROM_INSTITUTION]->(i:Institution) RETURN a.name AS aname, i.name AS iname"
        ):
            g.add((uri_author(row["aname"]), EX.fromInstitution, uri_institution(row["iname"])))
            stats["from_institution"] += 1

        # USES_METHOD edges
        print("  Exporting USES_METHOD edges...")
        for row in session.run(
            "MATCH (p:Paper)-[:USES_METHOD]->(m:Method) RETURN p.arxiv_id AS pid, m.name AS mname"
        ):
            g.add((uri_paper(row["pid"]), EX.usesMethod, uri_method(row["mname"])))
            stats["uses_method"] += 1

        # BELONGS_TO community edges (via community_id on Paper)
        print("  Exporting community membership...")
        for row in session.run(
            "MATCH (p:Paper) WHERE p.community_id IS NOT NULL RETURN p.arxiv_id AS pid, p.community_id AS cid"
        ):
            g.add((uri_paper(row["pid"]), EX.belongsToCommunity, uri_community(str(row["cid"]))))
            stats["belongs_to_community"] += 1

    driver.close()
    return stats


# ── Apply method subclass assertions ─────────────────────────────────────────

def apply_method_subclasses(g: Graph, classifications: dict) -> int:
    count = 0
    for method_name, info in classifications.items():
        uri = uri_method(method_name)
        subclass_uri = EX[info["subclass"]]
        # Only assert if this method actually exists in the graph (was used in a paper)
        if (uri, RDF.type, EX.Method) in g:
            g.add((uri, RDF.type, subclass_uri))
            count += 1
    return count


# ── Coverage metric ───────────────────────────────────────────────────────────

def compute_coverage(g: Graph) -> None:
    papers_with_methods = set(s for s, _, _ in g.triples((None, EX.usesMethod, None)))
    total_papers = sum(1 for _ in g.triples((None, RDF.type, EX.Paper)))
    covered = len(papers_with_methods)
    pct = 100 * covered / total_papers if total_papers > 0 else 0
    print(f"\nMethod coverage: {covered}/{total_papers} papers have >= 1 method ({pct:.1f}%)")

    # Break down by subclass
    for subclass in ["FineTuningMethod", "AttentionMethod", "AlignmentMethod",
                     "ReasoningMethod", "RetrievalMethod",
                     "PersonalizationMethod", "AgentSkillLearningMethod"]:
        sc_uri = EX[subclass]
        methods_in_class = set(s for s, _, _ in g.triples((None, RDF.type, sc_uri)))
        papers_using = set()
        for m_uri in methods_in_class:
            for p_uri, _, _ in g.triples((None, EX.usesMethod, m_uri)):
                papers_using.add(p_uri)
        print(f"  {subclass}: {len(methods_in_class)} methods, {len(papers_using)} papers")


# ── Main ──────────────────────────────────────────────────────────────────────

def build_ontology() -> None:
    print("Loading schema...")
    g = Graph()
    g.parse(SCHEMA_FILE, format="turtle")
    g.bind("ex", EX)
    print(f"  Schema loaded: {len(g)} triples")

    print("\nLoading method classifications...")
    classifications = json.loads(CLASSIF_FILE.read_text(encoding="utf-8"))
    print(f"  {len(classifications)} methods loaded")

    print("\nExporting Neo4j graph to RDF...")
    stats = export_from_neo4j(g)
    print(f"\nNeo4j export complete:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  Total triples so far: {len(g)}")

    print("\nApplying method subclass assertions...")
    count = apply_method_subclasses(g, classifications)
    print(f"  Subclass assertions added: {count}")
    print(f"  Total triples so far: {len(g)}")

    print("\nRunning OWL reasoner (RDFS semantics)...")
    before = len(g)
    owlrl.DeductiveClosure(owlrl.RDFS_Semantics).expand(g)
    after = len(g)
    print(f"  Triples before reasoning: {before}")
    print(f"  Triples after reasoning:  {after}")
    print(f"  Inferred triples added:   {after - before}")

    compute_coverage(g)

    print(f"\nSerialising to {OUTPUT_FILE}...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(OUTPUT_FILE, format="turtle")
    print(f"Done. {len(g)} total triples written to {OUTPUT_FILE}")


if __name__ == "__main__":
    build_ontology()

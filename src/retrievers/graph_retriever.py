"""Local graph retriever — Neo4j Cypher traversal with fuzzy entity linking."""
import os
import re

import spacy
from dotenv import load_dotenv
from neo4j import GraphDatabase
from src.retrievers.models import RetrievalResult

load_dotenv()

_nlp = None
_driver = None

TEMPORAL_PATTERNS = [
    (r"after (\d{4})", "after"),
    (r"since (\d{4})", "after"),
    (r"before (\d{4})", "before"),
    (r"in (\d{4})", "exact"),
    (r"recent|latest|new", "recent"),
]

VENUE_KEYWORDS = [
    "neurips", "nips", "icml", "iclr", "acl", "emnlp", "naacl",
    "cvpr", "iccv", "eccv", "aaai", "ijcai", "arxiv",
]

GENERIC_ENTITY_WORDS = {
    "a", "an", "the", "that", "which", "who", "what", "paper", "papers",
    "author", "authors", "wrote", "written", "use", "uses", "using", "about",
    "related", "to", "of", "by", "method", "methods",
}


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_lg")
    return _nlp


def _get_driver():
    global _driver
    if _driver is None:
        try:
            from src.agent.connections import get_neo4j_driver
            _driver = get_neo4j_driver()
        except ImportError:
            _driver = GraphDatabase.driver(
                os.environ["NEO4J_URI"],
                auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
            )
    return _driver


def _traverse_from_method(
    method_name: str,
    temporal_filter: tuple,
    session,
) -> tuple[list[dict], str]:
    """Find papers and authors for a given method name via Neo4j."""
    filter_type, year = temporal_filter
    time_clause = f"AND p.year >= {year}" if filter_type == "after" and year else ""

    cypher = f"""
        MATCH (m:Method {{name: $name}})<-[:USES_METHOD]-(p:Paper)-[:AUTHORED_BY]->(a:Author)
        WHERE 1=1 {time_clause}
        RETURN DISTINCT p.arxiv_id AS arxiv_id, p.title AS title,
               p.year AS year, p.venue AS venue,
               m.name AS matched_method,
               collect(DISTINCT a.name)[..3] AS authors
        ORDER BY p.year DESC
        LIMIT 10
    """
    result = session.run(cypher, name=method_name)
    return result.data(), cypher.strip()


def _extract_entities(query: str) -> list[str]:
    nlp = _get_nlp()
    doc = nlp(query)
    entities = [ent.text.strip() for ent in doc.ents
                if ent.label_ in ("PERSON", "ORG", "PRODUCT", "WORK_OF_ART")]
    # Also add noun chunks as fallback
    if not entities:
        entities = [chunk.text.strip() for chunk in doc.noun_chunks
                    if len(chunk.text.strip()) > 3]
    informative = []
    for entity in entities:
        words = {word.lower() for word in re.findall(r"[^\W_]+", entity)}
        if words and not words.issubset(GENERIC_ENTITY_WORDS):
            informative.append(entity)
    return list(dict.fromkeys(informative))[:5]


def _detect_temporal_filter(query: str) -> tuple[str | None, int | None]:
    query_lower = query.lower()
    for pattern, filter_type in TEMPORAL_PATTERNS:
        match = re.search(pattern, query_lower)
        if match:
            if filter_type == "recent":
                return "after", 2023
            year = int(match.group(1))
            return filter_type, year
    return None, None


def _detect_venue_filter(query: str) -> str | None:
    query_lower = query.lower()
    for venue in VENUE_KEYWORDS:
        if venue in query_lower:
            return venue.upper()
    return None


def _lucene_search_term(entity: str) -> str:
    """Convert user text to a literal full-text query without Lucene operators."""
    # Quoting alphanumeric tokens prevents punctuation such as '-', ':', '/',
    # parentheses, or a trailing backslash from becoming Lucene query syntax.
    tokens = [
        token for token in re.findall(r"[^\W_]+", entity, flags=re.UNICODE)
        if len(token) > 1
    ]
    return " AND ".join(f'"{token}"' for token in tokens)


def _fuzzy_entity_search(entity: str, session) -> list[dict]:
    search_term = _lucene_search_term(entity)
    if not search_term:
        return []

    result = session.run("""
        CALL db.index.fulltext.queryNodes('paperTitleIndex', $search_term)
        YIELD node, score
        WHERE score > 0.3
        RETURN node.arxiv_id AS arxiv_id, node.title AS title, score
        ORDER BY score DESC
        LIMIT 5
    """, search_term=search_term)
    papers = result.data()

    if not papers:
        result = session.run("""
            CALL db.index.fulltext.queryNodes('authorNameIndex', $search_term)
            YIELD node, score
            WHERE score > 0.3
            RETURN node.name AS name, score
            ORDER BY score DESC
            LIMIT 3
        """, search_term=search_term)
        authors = result.data()
        return [{"type": "author", **a} for a in authors]

    return [{"type": "paper", **p} for p in papers]


def _methods_mentioned_in_query(query: str, session) -> list[str]:
    """Find known Neo4j method names explicitly mentioned in the query."""
    rows = session.run("MATCH (m:Method) RETURN m.name AS name")
    matches = []
    for row in rows:
        name = row["name"]
        if not name or len(name) < 3:
            continue
        pattern = rf"(?<!\w){re.escape(name)}(?!\w)"
        if re.search(pattern, query, flags=re.IGNORECASE):
            matches.append(name)
    return sorted(set(matches), key=len, reverse=True)


def _traverse_from_paper(
    arxiv_id: str,
    hops: int,
    temporal_filter: tuple,
    venue_filter: str | None,
    session,
) -> tuple[list[dict], str]:
    filter_type, year = temporal_filter

    if filter_type == "after" and year:
        time_clause = f"AND p2.year >= {year}"
    elif filter_type == "before" and year:
        time_clause = f"AND p2.year <= {year}"
    elif filter_type == "exact" and year:
        time_clause = f"AND p2.year = {year}"
    else:
        time_clause = ""

    venue_clause = f'AND toLower(p2.venue) CONTAINS "{venue_filter.lower()}"' if venue_filter else ""

    if hops == 1:
        cypher = f"""
            MATCH (p1:Paper {{arxiv_id: $arxiv_id}})-[:AUTHORED_BY]->(a:Author)<-[:AUTHORED_BY]-(p2:Paper)
            WHERE p1 <> p2 {time_clause} {venue_clause}
            RETURN DISTINCT p2.arxiv_id AS arxiv_id, p2.title AS title,
                   p2.year AS year, p2.venue AS venue,
                   collect(DISTINCT a.name)[..3] AS shared_authors
            LIMIT 10
        """
    else:
        cypher = f"""
            MATCH (p1:Paper {{arxiv_id: $arxiv_id}})-[:USES_METHOD]->(m:Method)<-[:USES_METHOD]-(p2:Paper)
            WHERE p1 <> p2 {time_clause} {venue_clause}
            RETURN DISTINCT p2.arxiv_id AS arxiv_id, p2.title AS title,
                   p2.year AS year, p2.venue AS venue,
                   collect(DISTINCT m.name)[..3] AS shared_methods
            LIMIT 15
        """

    result = session.run(cypher, arxiv_id=arxiv_id)
    return result.data(), cypher.strip()


def _traverse_from_author(author_name: str, temporal_filter: tuple, session) -> tuple[list[dict], str]:
    filter_type, year = temporal_filter
    time_clause = f"AND p.year >= {year}" if filter_type == "after" and year else ""

    cypher = f"""
        MATCH (a:Author {{name: $name}})<-[:AUTHORED_BY]-(p:Paper)
        WHERE 1=1 {time_clause}
        RETURN p.arxiv_id AS arxiv_id, p.title AS title,
               p.year AS year, p.venue AS venue
        ORDER BY p.year DESC
        LIMIT 10
    """
    result = session.run(cypher, name=author_name)
    return result.data(), cypher.strip()


def _serialise_results(results: list[dict], entity: str) -> str:
    if not results:
        return ""
    lines = [f"Neo4j results for '{entity}':"]
    for r in results:
        line = f"- {r.get('title', 'Unknown')} (arxiv:{r.get('arxiv_id', '')}, {r.get('year', '')})"
        if r.get("matched_method"):
            line += f". Explicit USES_METHOD link: {r['matched_method']}"
        if r.get("shared_authors"):
            line += f". Shared authors: {', '.join(r['shared_authors'])}"
        if r.get("shared_methods"):
            line += f". Shared methods: {', '.join(r['shared_methods'])}"
        if r.get("authors"):
            line += f". Authors: {', '.join(r['authors'])}"
        lines.append(line)
    return "\n".join(lines)


def retrieve(query: str) -> RetrievalResult:
    entities = _extract_entities(query)
    temporal_filter = _detect_temporal_filter(query)
    venue_filter = _detect_venue_filter(query)

    driver = _get_driver()
    all_context_parts = []
    cypher_queries_used = []

    with driver.session() as session:

        # Resolve method names explicitly mentioned in the question directly
        # against Neo4j. This keeps ordinary graph retrieval independent from
        # the RDF ontology and avoids relying on spaCy to label names like LoRA.
        mentioned_methods = _methods_mentioned_in_query(query, session)
        for method_name in mentioned_methods[:5]:
            results, cypher = _traverse_from_method(
                method_name, temporal_filter, session
            )
            cypher_queries_used.append(cypher)
            context = _serialise_results(results, method_name)
            if context:
                all_context_parts.append(context)

        # Fall back to standard entity extraction when no named method matched.
        if not all_context_parts:
            if not entities:
                return RetrievalResult(
                    context_text="No entities found in query for graph traversal.",
                    source_type="graph",
                    source_metadata={"graph_backend": "neo4j"},
                )

            for entity in entities[:5]:
                matches = _fuzzy_entity_search(entity, session)
                if not matches:
                    continue

                for match in matches[:2]:
                    if match.get("type") == "paper":
                        results, cypher = _traverse_from_paper(
                            match["arxiv_id"], hops=1,
                            temporal_filter=temporal_filter,
                            venue_filter=venue_filter,
                            session=session,
                        )
                        if len(results) < 3:
                            results, cypher = _traverse_from_paper(
                                match["arxiv_id"], hops=2,
                                temporal_filter=temporal_filter,
                                venue_filter=venue_filter,
                                session=session,
                            )
                        cypher_queries_used.append(cypher)
                        context = _serialise_results(results, entity)
                        if context:
                            all_context_parts.append(context)

                    elif match.get("type") == "author":
                        results, cypher = _traverse_from_author(
                            match["name"], temporal_filter, session
                        )
                        cypher_queries_used.append(cypher)
                        context = _serialise_results(results, match["name"])
                        if context:
                            all_context_parts.append(context)

    if not all_context_parts:
        return RetrievalResult(
            context_text="No graph results found for the given entities.",
            source_type="graph",
            source_metadata={"graph_backend": "neo4j"},
            cypher_query_used=cypher_queries_used[0] if cypher_queries_used else None,
        )

    return RetrievalResult(
        context_text="\n\n".join(all_context_parts),
        source_type="graph",
        cypher_query_used="\n---\n".join(cypher_queries_used[:2]),
        source_metadata={
            "graph_backend": "neo4j",
            "entities_found": entities,
            "temporal_filter": temporal_filter,
            "venue_filter": venue_filter,
        },
    )


if __name__ == "__main__":
    queries = [
        "What papers use the Transformer method?",
        "Papers about LoRA after 2023",
        "What did Ashish Vaswani work on?",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        result = retrieve(q)
        print(f"Source: {result.source_type}")
        print(f"Cypher: {result.cypher_query_used}")
        print(f"Context:\n{result.context_text[:400]}")
        print("-" * 60)

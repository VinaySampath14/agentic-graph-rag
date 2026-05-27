// Schema reference:
// Nodes: Paper(arxiv_id, title, year, venue, community_id)
//        Author(name)
//        Institution(name)
//        Method(name)
//        Community(community_id, theme, summary)
// Edges: AUTHORED_BY, FROM_INSTITUTION, CITES(year, venue), USES_METHOD, BELONGS_TO

// 1. FULLTEXT search on paper title — finds papers matching a keyword phrase
CALL db.index.fulltext.queryNodes("paperTitleIndex", "attention mechanism")
YIELD node, score
RETURN node.arxiv_id, node.title, score
ORDER BY score DESC
LIMIT 10;

// 2. Multi-hop author traversal — find all papers by an author's co-authors (2-hop)
MATCH (a:Author {name: "Yoshua Bengio"})-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]-(coauthor:Author)
WHERE coauthor.name <> "Yoshua Bengio"
WITH DISTINCT coauthor
MATCH (coauthor)-[:AUTHORED_BY]-(p2:Paper)
RETURN coauthor.name, collect(p2.title) AS papers
LIMIT 20;

// 3. Temporal filter by year — papers published after 2022
MATCH (p:Paper)
WHERE p.year >= 2022
RETURN p.arxiv_id, p.title, p.year
ORDER BY p.year DESC
LIMIT 20;

// 4. Venue filter — papers from a specific conference
MATCH (p:Paper)
WHERE p.venue CONTAINS "NeurIPS"
RETURN p.arxiv_id, p.title, p.year, p.venue
ORDER BY p.year DESC
LIMIT 20;

// 5. Community lookup — all papers in a given community with their theme
MATCH (p:Paper)-[:BELONGS_TO]->(c:Community {community_id: 3})
RETURN c.theme AS theme, c.summary AS summary, collect(p.title) AS papers
LIMIT 1;

// 6. Method usage across papers — which papers use a given method
MATCH (p:Paper)-[:USES_METHOD]->(m:Method)
WHERE m.name CONTAINS "Transformer"
RETURN m.name, collect(p.title) AS papers, count(p) AS paper_count
ORDER BY paper_count DESC
LIMIT 10;

// 7. Citation chain traversal — papers citing a specific paper (1-hop)
MATCH (citing:Paper)-[:CITES]->(cited:Paper {arxiv_id: "1706.03762"})
RETURN citing.arxiv_id, citing.title, citing.year
ORDER BY citing.year DESC
LIMIT 20;

// 8. Author collaboration — find pairs of authors who co-authored multiple papers
MATCH (a1:Author)-[:AUTHORED_BY]-(p:Paper)-[:AUTHORED_BY]-(a2:Author)
WHERE a1.name < a2.name
WITH a1, a2, count(p) AS shared_papers
WHERE shared_papers >= 2
RETURN a1.name, a2.name, shared_papers
ORDER BY shared_papers DESC
LIMIT 20;

// 9. Papers in a date range — papers published between two years
MATCH (p:Paper)
WHERE p.year >= 2021 AND p.year <= 2023
RETURN p.arxiv_id, p.title, p.year, p.venue
ORDER BY p.year, p.title
LIMIT 30;

// 10. Top cited papers by in-degree — most-cited papers in the corpus
MATCH (cited:Paper)
WITH cited, count{ (citing:Paper)-[:CITES]->(cited) } AS citation_count
WHERE citation_count > 0
RETURN cited.arxiv_id, cited.title, cited.year, citation_count
ORDER BY citation_count DESC
LIMIT 20;

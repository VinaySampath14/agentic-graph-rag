"""Generate community summaries using Groq and store in Neo4j."""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase

load_dotenv()

PROMPT_FILE = Path("prompts/community_summary_v1.txt")
COMMUNITIES_FILE = Path("data/processed/communities.jsonl")


def get_driver():
    return GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]),
    )


def load_prompt() -> str:
    with open(PROMPT_FILE, encoding="utf-8") as f:
        lines = f.readlines()
    # skip the comment line at top
    return "".join(lines[1:]).strip()


def summarise_community(
    client: Groq, prompt_template: str, papers: list[dict]
) -> dict:
    titles = "\n".join(f"- {p['title']}" for p in papers[:50])
    prompt = prompt_template.replace("{paper_titles_and_abstracts}", titles)

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            time.sleep(10)

    return {
        "theme": "Unknown",
        "dominant_methods": [],
        "key_authors": [],
        "representative_papers": [],
    }


def main() -> None:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    driver = get_driver()
    prompt_template = load_prompt()

    with open(COMMUNITIES_FILE, encoding="utf-8") as f:
        communities = [json.loads(line) for line in f]

    print(f"Generating summaries for {len(communities)} communities...")

    with driver.session() as session:
        for i, community in enumerate(communities):
            cid = community["community_id"]
            papers = community["papers"]
            print(f"  Community {cid} ({len(papers)} papers)...")

            summary = summarise_community(client, prompt_template, papers)

            theme = summary.get("theme", "")
            summary_text = json.dumps(summary)

            session.run("""
                MATCH (c:Community {community_id: $cid})
                SET c.theme = $theme, c.summary = $summary
            """, cid=cid, theme=theme, summary=summary_text)

            print(f"    Theme: {theme}")
            time.sleep(3)  # respect Groq rate limits

    driver.close()
    print("\nAll community summaries generated.")


if __name__ == "__main__":
    main()

"""Fetch arXiv CS.AI + CS.CL papers and save to data/raw/ as JSONL batches."""
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx


CATEGORIES = ["cs.AI", "cs.CL"]
TOTAL_PAPERS = 2000
BATCH_SIZE = 100
OUTPUT_DIR = Path("data/raw")
ARXIV_API = "https://export.arxiv.org/api/query"
NS = "{http://www.w3.org/2005/Atom}"


def _fetch_page(query: str, start: int, max_results: int) -> list[dict]:
    """Fetch one page from arxiv API with exponential backoff on 429/503."""
    params = {
        "search_query": query,
        "start": start,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    backoff = 60
    for attempt in range(6):
        try:
            resp = httpx.get(ARXIV_API, params=params, timeout=60)
        except (httpx.ReadTimeout, httpx.ConnectTimeout, httpx.ConnectError) as e:
            print(f"  Connection error ({e}) — waiting {backoff}s (attempt {attempt + 1}/6)...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 600)
            continue
        if resp.status_code == 200:
            return _parse_feed(resp.text)
        if resp.status_code in (429, 503):
            print(f"  HTTP {resp.status_code} — waiting {backoff}s (attempt {attempt + 1}/6)...")
            time.sleep(backoff)
            backoff = min(backoff * 2, 600)
        else:
            resp.raise_for_status()
    raise RuntimeError(f"Failed after 6 attempts at offset {start}")


def _parse_feed(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    papers = []
    for entry in root.findall(f"{NS}entry"):
        entry_id = entry.findtext(f"{NS}id", "")
        arxiv_id = entry_id.split("/abs/")[-1].split("v")[0]

        published_raw = entry.findtext(f"{NS}published", "")
        year = int(published_raw[:4]) if published_raw else 0

        authors = [
            a.findtext(f"{NS}name", "")
            for a in entry.findall(f"{NS}author")
        ]

        categories = [
            t.get("term", "")
            for t in entry.findall("{http://arxiv.org/schemas/atom}primary_category")
        ] + [
            t.get("term", "")
            for t in entry.findall(f"{NS}category")
        ]
        categories = list(dict.fromkeys(filter(None, categories)))

        papers.append({
            "arxiv_id": arxiv_id,
            "title": (entry.findtext(f"{NS}title", "") or "").replace("\n", " ").strip(),
            "authors": authors,
            "abstract": (entry.findtext(f"{NS}summary", "") or "").replace("\n", " ").strip(),
            "published": published_raw,
            "year": year,
            "categories": categories,
            "journal_ref": "",
        })
    return papers


def fetch_papers(
    categories: list[str] = CATEGORIES,
    total: int = TOTAL_PAPERS,
    batch_size: int = BATCH_SIZE,
    output_dir: Path = OUTPUT_DIR,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume: load IDs already saved
    seen_ids: set[str] = set()
    existing_batches = sorted(output_dir.glob("papers_batch_*.jsonl"))
    for existing in existing_batches:
        with open(existing, encoding="utf-8") as f:
            for line in f:
                seen_ids.add(json.loads(line)["arxiv_id"])
    if seen_ids:
        print(f"Resuming — {len(seen_ids)} papers already saved")

    query = " OR ".join(f"cat:{c}" for c in categories)
    batch_num = len(existing_batches)
    total_saved = len(seen_ids)
    offset = len(seen_ids)  # skip already-fetched pages

    while total_saved < total:
        print(f"Fetching offset {offset}...")
        page = _fetch_page(query, start=offset, max_results=batch_size)
        if not page:
            print("No more results from arxiv.")
            break

        new_papers = [p for p in page if p["arxiv_id"] not in seen_ids]
        for p in new_papers:
            seen_ids.add(p["arxiv_id"])

        if new_papers:
            _save_batch(new_papers, batch_num, output_dir)
            total_saved += len(new_papers)
            print(f"  Saved batch {batch_num} (+{len(new_papers)} papers, {total_saved} total)")
            batch_num += 1

        offset += batch_size
        time.sleep(15)  # polite delay between pages

    print(f"\nDone. {total_saved} unique papers saved to {output_dir}/")
    return total_saved


def _save_batch(batch: list[dict], batch_num: int, output_dir: Path) -> None:
    path = output_dir / f"papers_batch_{batch_num:03d}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for paper in batch:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fetch_papers()

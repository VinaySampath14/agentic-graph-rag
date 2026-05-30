"""Extract named entities from paper abstracts using spaCy."""
import json
from pathlib import Path

import spacy


INPUT_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/entities.jsonl")

METHOD_PATTERNS = [
    "BERT", "GPT", "GPT-2", "GPT-3", "GPT-4", "LLaMA", "LLaMA-2",
    "Transformer", "Attention", "LoRA", "QLoRA", "RLHF", "PPO",
    "RAG", "GraphRAG", "LightRAG", "FAISS", "BM25", "DPR",
    "T5", "RoBERTa", "ALBERT", "XLNet", "ELECTRA", "DeBERTa",
    "CLIP", "DALL-E", "Stable Diffusion", "ViT", "ResNet",
    "Adam", "AdamW", "SGD", "Dropout", "BatchNorm", "LayerNorm",
    "Chain-of-Thought", "ReAct", "LangChain", "LangGraph",
    "Neo4j", "Qdrant", "Pinecone", "Weaviate", "ChromaDB",
    "Leiden", "Louvain", "PageRank", "Node2Vec",
]


def load_papers() -> list[dict]:
    papers = []
    for path in sorted(INPUT_DIR.glob("papers_batch_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                papers.append(json.loads(line))
    return papers


def extract_method_patterns(text: str) -> list[str]:
    found = []
    for method in METHOD_PATTERNS:
        if method.lower() in text.lower():
            found.append(method)
    return found


def extract_entities(batch_size: int = 32) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_lg")
    # only need NER, disable unused components for speed
    nlp.select_pipes(enable=["ner"])

    papers = load_papers()
    print(f"Processing {len(papers)} abstracts...")

    abstracts = [p["abstract"] for p in papers]
    results = []

    for i, (paper, doc) in enumerate(
        zip(papers, nlp.pipe(abstracts, batch_size=batch_size))
    ):
        persons = list({
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ == "PERSON" and len(ent.text.strip()) > 2
        })
        orgs = list({
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ == "ORG" and len(ent.text.strip()) > 2
        })
        methods = extract_method_patterns(paper["abstract"])

        results.append({
            "arxiv_id": paper["arxiv_id"],
            "persons": persons,
            "orgs": orgs,
            "methods": methods,
        })

        if (i + 1) % 200 == 0:
            print(f"  Processed {i + 1}/{len(papers)} papers")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone. Entities saved to {OUTPUT_FILE}")
    print(f"Sample: {results[0]}")


if __name__ == "__main__":
    extract_entities()

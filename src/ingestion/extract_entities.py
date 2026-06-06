"""Extract named entities from paper abstracts using spaCy."""
import json
import re
from pathlib import Path

import spacy


INPUT_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/processed/entities.jsonl")

METHOD_PATTERNS = [
    # Architecture components
    "Transformer", "Attention", "Multi-head Attention",
    "Flash Attention", "Sparse Attention", "Cross-Attention",
    "MoE", "Mixture of Experts", "SSM", "State Space Model",

    # Fine-tuning / adaptation
    "LoRA", "QLoRA", "AdaLoRA", "DoRA", "LoRAX",
    "RLHF", "PPO", "DPO", "GRPO", "RLAIF",
    "Instruction Tuning", "FLAN", "Alpaca", "Vicuna",
    "Prefix Tuning", "Prompt Tuning", "P-Tuning",

    # Reasoning / agents
    "Chain-of-Thought", "CoT", "Tree of Thoughts", "ToT",
    "ReAct", "Reflexion", "Self-Consistency",
    "RAG", "GraphRAG", "LightRAG", "HippoRAG",
    "LangChain", "LangGraph", "AutoGen", "CrewAI",

    # Vision / multimodal
    "CLIP", "DALL-E", "DALL-E 2", "DALL-E 3",
    "Stable Diffusion", "ViT", "ResNet", "DINO",
    "LLaVA", "Flamingo", "BLIP", "InstructBLIP",
    "GPT-4V", "Gemini Vision",

    # Retrieval / search
    "FAISS", "BM25", "DPR", "ColBERT", "Contriever",
    "Qdrant", "Pinecone", "Weaviate", "ChromaDB", "Milvus",

    # Training techniques
    "Adam", "AdamW", "SGD", "Lion",
    "Dropout", "BatchNorm", "LayerNorm", "RMSNorm",
    "Flash Attention", "Gradient Checkpointing",
    "DeepSpeed", "FSDP", "ZeRO",

    # Evaluation
    "BLEU", "ROUGE", "BERTScore", "RAGAS",
    "MMLU", "HumanEval", "GSM8K", "MATH", "HellaSwag",

    # Graph / community
    "Neo4j", "Leiden", "Louvain", "PageRank", "Node2Vec",
    "GraphSAGE", "GAT", "GCN", "GNN",
]


def load_papers() -> list[dict]:
    papers = []
    for path in sorted(INPUT_DIR.glob("papers_batch_*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                papers.append(json.loads(line))
    return papers


# Acronyms whose lowercase/mixed-case form collides with an unrelated common
# English word or phrase (e.g. "ZeRO" the optimizer vs. "zero-shot"; "MATH" the
# benchmark vs. the generic subject "math") — these require exact-case matching
# to avoid false positives that case-insensitive matching would otherwise produce.
CASE_SENSITIVE_METHODS = {"ZeRO", "MATH"}


def extract_method_patterns(text: str) -> list[str]:
    found = []
    for method in METHOD_PATTERNS:
        flags = 0 if method in CASE_SENSITIVE_METHODS else re.IGNORECASE
        if re.search(r"\b" + re.escape(method) + r"\b", text, flags):
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

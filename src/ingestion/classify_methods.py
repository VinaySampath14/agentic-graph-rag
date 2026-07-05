"""LLM-assisted classification of METHOD_PATTERNS into OWL subclasses."""
import json
import os
import time
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

# Import directly to avoid pulling in spaCy at module level
METHOD_PATTERNS = [
    "Transformer", "Attention", "Multi-head Attention",
    "Flash Attention", "Sparse Attention", "Cross-Attention",
    "MoE", "Mixture of Experts", "SSM", "State Space Model",
    "LoRA", "QLoRA", "AdaLoRA", "DoRA", "LoRAX",
    "RLHF", "PPO", "DPO", "GRPO", "RLAIF",
    "Instruction Tuning", "FLAN", "Alpaca", "Vicuna",
    "Prefix Tuning", "Prompt Tuning", "P-Tuning",
    "Chain-of-Thought", "CoT", "Tree of Thoughts", "ToT",
    "ReAct", "Reflexion", "Self-Consistency",
    "RAG", "GraphRAG", "LightRAG", "HippoRAG",
    "LangChain", "LangGraph", "AutoGen", "CrewAI",
    "CLIP", "DALL-E", "DALL-E 2", "DALL-E 3",
    "Stable Diffusion", "ViT", "ResNet", "DINO",
    "LLaVA", "Flamingo", "BLIP", "InstructBLIP",
    "GPT-4V", "Gemini Vision",
    "FAISS", "BM25", "DPR", "ColBERT", "Contriever",
    "Qdrant", "Pinecone", "Weaviate", "ChromaDB", "Milvus",
    "Adam", "AdamW", "SGD", "Lion",
    "Dropout", "BatchNorm", "LayerNorm", "RMSNorm",
    "Flash Attention", "Gradient Checkpointing",
    "DeepSpeed", "FSDP", "ZeRO",
    "BLEU", "ROUGE", "BERTScore", "RAGAS",
    "MMLU", "HumanEval", "GSM8K", "MATH", "HellaSwag",
    "Neo4j", "Leiden", "Louvain", "PageRank", "Node2Vec",
    "GraphSAGE", "GAT", "GCN", "GNN",
]

load_dotenv()

OUTPUT = Path("data/processed/method_classifications.json")

SUBCLASSES = [
    "FineTuningMethod",
    "AttentionMethod",
    "AlignmentMethod",
    "ReasoningMethod",
    "RetrievalMethod",
]

SUBCLASS_DESCRIPTIONS = """
- FineTuningMethod: adapting pretrained models — LoRA, QLoRA, PEFT, instruction tuning, prefix tuning, adapters
- AttentionMethod: attention-based architectures — Transformer, FlashAttention, sparse attention, MoE, SSM
- AlignmentMethod: aligning models with human preferences — RLHF, DPO, PPO, RLAIF, constitutional AI
- ReasoningMethod: augmenting reasoning or agentic behaviour — Chain-of-Thought, RAG, ReAct, LangChain, AutoGen
- RetrievalMethod: information retrieval and vector search — BM25, DPR, ColBERT, FAISS, Qdrant, Milvus
"""

LOW_CONFIDENCE_THRESHOLD = 0.75


def _build_prompt(method: str) -> str:
    return f"""Classify this AI/ML method into exactly one subclass from the list below.

Subclasses and their scope:
{SUBCLASS_DESCRIPTIONS}

Method to classify: "{method}"

Rules:
- Choose exactly one subclass from the list above
- Do not invent new subclass names
- If the method fits multiple categories, pick the most specific one
- Training optimisers (Adam, SGD) → FineTuningMethod
- Benchmarks and evaluation metrics (BLEU, MMLU) → use ReasoningMethod
- Graph neural networks (GNN, GCN, GAT) → ReasoningMethod
- Vector databases (Qdrant, Pinecone) → RetrievalMethod

Output JSON only:
{{"method": "{method}", "subclass": "<one of the 5 subclasses>", "confidence": <0.0-1.0>, "reason": "<one sentence>"}}"""


def classify_methods() -> dict:
    client = Groq(api_key=os.environ["GROQ_API_KEY"].strip())
    results = {}
    needs_review = []

    # Deduplicate — METHOD_PATTERNS has "Flash Attention" twice
    unique_methods = list(dict.fromkeys(METHOD_PATTERNS))
    print(f"Classifying {len(unique_methods)} methods...\n")

    for i, method in enumerate(unique_methods):
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": _build_prompt(method)}],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                raw = json.loads(response.choices[0].message.content)
                break
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 30 * (attempt + 1)
                    print(f"  Rate limit — waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raw = {
                        "method": method,
                        "subclass": "FineTuningMethod",
                        "confidence": 0.0,
                        "reason": f"Classification failed: {e}",
                    }

        # Validate subclass is from allowed list
        if raw.get("subclass") not in SUBCLASSES:
            raw["subclass"] = "FineTuningMethod"
            raw["confidence"] = 0.0
            raw["reason"] = "Invalid subclass returned — defaulted to FineTuningMethod"

        raw["needs_review"] = raw["confidence"] < LOW_CONFIDENCE_THRESHOLD
        results[method] = raw

        flag = " *** REVIEW" if raw["needs_review"] else ""
        print(f"  [{i+1:2d}/{len(unique_methods)}] {method:<30} -> {raw['subclass']:<22} ({raw['confidence']:.2f}){flag}")

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    # Summary
    print(f"\nSaved to {OUTPUT}")
    print(f"Total methods classified: {len(results)}")

    by_subclass = {s: [] for s in SUBCLASSES}
    for m, v in results.items():
        by_subclass[v["subclass"]].append(m)

    print("\nDistribution:")
    for subclass, methods in by_subclass.items():
        print(f"  {subclass}: {len(methods)}")
        for m in methods:
            print(f"    - {m}")

    print(f"\nNeeds manual review ({sum(1 for v in results.values() if v['needs_review'])}):")
    for m, v in results.items():
        if v["needs_review"]:
            print(f"  {m} -> {v['subclass']} ({v['confidence']:.2f}) | {v['reason']}")

    return results


if __name__ == "__main__":
    classify_methods()

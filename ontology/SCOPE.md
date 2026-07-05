# Ontology Scope and Coverage

## What This Ontology Covers

The ontology (`arxiv_cs_populated.ttl`) is built from the 2,000-paper arXiv CS.AI + CS.CL corpus (2026) loaded into Neo4j.

### Classes
- `Paper` — 2,000 arXiv papers
- `Author` — 9,250 authors
- `Institution` — 3,003 institutions
- `Method` — 286 method nodes extracted from abstracts
- `Community` — 13 Leiden-detected research communities

### Method Subclass Hierarchy (new — not in Neo4j)
| Subclass | Description | Example methods |
|---|---|---|
| `FineTuningMethod` | Model adaptation techniques | LoRA, QLoRA, PEFT, Instruction Tuning |
| `AttentionMethod` | Attention-based architectures | Transformer, FlashAttention, MoE, SSM |
| `AlignmentMethod` | Human preference alignment | RLHF, DPO, PPO, RLAIF |
| `ReasoningMethod` | Reasoning and agentic methods | Chain-of-Thought, RAG, LangGraph, GNN |
| `RetrievalMethod` | Information retrieval | BM25, DPR, ColBERT, FAISS, Qdrant |

### Inferred Relationships
- `relatedWork` — derived by the OWL reasoner between papers sharing a method subclass. Not stored in Neo4j; created at build time by owlrl.

### Scale
- **130,294 total triples** (82,573 explicit + 47,721 inferred)
- **598 / 2,000 papers (29.9%)** have at least one method represented in the ontology

## Known Limitation

Method coverage is bounded by the predefined `METHOD_PATTERNS` list in `src/ingestion/extract_entities.py`. This list covers 96 high-frequency methods chosen to represent the most common techniques in CS.AI and CS.CL literature.

Neo4j contains 286 unique method strings extracted from the corpus. Of these, 62 matched the predefined list and received subclass assertions. The remaining 224 method strings are free-text extractions that did not match the vocabulary — they are present in the graph as `ex:Method` instances but have no subclass classification.

Papers not using any of the 96 predefined methods are retrievable via vector and graph modes but are not represented in the ontology hierarchy.

## Future Work

Replace the regex-based `METHOD_PATTERNS` with LLM-assisted method extraction from abstracts (Option A in the project plan). This would expand coverage from 96 predefined terms to the full method vocabulary in the corpus, estimated at 150-200 unique methods appearing in 5+ papers.

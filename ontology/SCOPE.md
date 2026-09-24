# Ontology Scope and Coverage

## What This Ontology Covers

The ontology (`arxiv_cs_populated.ttl`) is built from the 2,000-paper arXiv CS.AI + CS.CL corpus (2026) loaded into Neo4j.

### Classes
- `Paper` — 2,000 arXiv papers
- `Author` — 9,250 authors
- `Institution` — 3,003 institutions
- `Method` — 286 method nodes extracted from abstracts
- `Community` — 13 Leiden-detected research communities

### Reviewed Method Subclass Hierarchy (not used by Neo4j retrieval)
| Subclass | Description | Example methods |
|---|---|---|
| `FineTuningMethod` | Model adaptation techniques | LoRA, QLoRA, AdaLoRA, Instruction Tuning |
| `AttentionMethod` | Attention-based architectures | Transformer, Flash Attention, Cross-Attention |
| `AlignmentMethod` | Human preference alignment | RLHF, DPO, PPO, RLAIF |
| `ReasoningMethod` | Reasoning and agentic methods | Chain-of-Thought, RAG, LangGraph, ReAct |
| `RetrievalMethod` | Information retrieval | BM25, DPR, ColBERT, FAISS, Qdrant |
| `PersonalizationMethod` | User-specific adaptation | Preference inference, personalized modeling |
| `AgentSkillLearningMethod` | Reusable agent skill acquisition | Skill extraction, verifier feedback |

### Inferred Relationships
- `relatedWork` — derived by the OWL reasoner between papers sharing a method subclass. Not stored in Neo4j; created at build time by owlrl.

### Scale
- **227,077 total triples**
- **82,582 `relatedWork` triples**
- **598 / 2,000 papers (29.9%)** have at least one method represented in the ontology

## Known Limitation

Neo4j contains 286 method strings extracted from the corpus. The populated ontology applies reviewed categories to 59 methods that actually occur in the graph. The remaining 227 stay valid `ex:Method` instances but intentionally have no subclass assertion until reviewed.

The original LLM classifications remain in `data/processed/method_classifications.json` for provenance. The build uses only `data/processed/method_classifications_reviewed.json`, preventing models, optimizers, and normalization layers from being forced into unrelated semantic categories.

Papers without a reviewed method category remain retrievable through Vector and Neo4j Graph retrieval but do not participate in category-derived `relatedWork` inference.

## Future Work

Expand the reviewed classification set with human-in-the-loop proposals. Each accepted category should pass SHACL before it is added to the build input.

See the repository-level [Ontology guide](../ONTOLOGY.md) for architecture, build instructions, validation policy, and example SPARQL.

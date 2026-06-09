# Contributing

## Local Setup

**Requirements:** Python 3.11, a Neo4j AuraDB instance, a Qdrant cloud cluster, and API keys for Groq and Tavily.

```bash
git clone https://github.com/VinaySampath14/agentic-graph-rag.git
cd agentic-graph-rag
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env   # fill in your credentials
python scripts/verify_connections.py
```

`.env` keys required:

```
NEO4J_URI=neo4j+s://...
NEO4J_USER=neo4j
NEO4J_PASSWORD=...
QDRANT_URL=https://...
QDRANT_API_KEY=...
GROQ_API_KEY=...
TAVILY_API_KEY=...
```

---

## Running the Demo

```bash
# Gradio demo (direct, no FastAPI)
python app.py

# FastAPI backend + Gradio frontend separately
uvicorn src.api.main:app --reload
python src/demo/app.py
```

---

## Running Tests

```bash
# Unit tests only (no network required)
pytest tests/unit/ -v

# All tests including integration (requires live connections)
pytest tests/ -v
```

Unit tests stub all heavy dependencies (Neo4j, Qdrant, BGE-M3, Groq) via `tests/conftest.py` — they run anywhere without credentials.

---

## Branch Naming

| Prefix | Use for |
|--------|---------|
| `feat/` | New features |
| `fix/` | Bug fixes |
| `chore/` | Dependencies, config, CI |
| `docs/` | Documentation only |

---

## Pull Requests

- One logical change per PR
- All unit tests must pass (`pytest tests/unit/`)
- No hardcoded credentials — use environment variables
- Update `ARCHITECTURE.md` if you add a new node, retriever, or change data flow

---

## Repo Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for a full walkthrough of the codebase.

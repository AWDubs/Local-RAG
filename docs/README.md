# Local-RAG — Documentation

Documentation for this repo is split in two:

| Folder | Audience | Contents |
|---|---|---|
| [**`implementation/`**](implementation/README.md) | Anyone running or modifying *this* codebase | Architecture, ingestion, query flow, and setup as actually wired up in `app.py`, `agent.py`, `rag.py`, `ingest.py`. |
| [**`theory/`**](theory/README.md) | Anyone learning RAG end-to-end | Conceptual reference: what RAG is, tokens & embeddings, chunking strategies, sampling, the wider Ollama / ChromaDB / Hugging Face ecosystem, plus a reference design for a fuller Streamlit RAG app. Some material is **aspirational** — see the implementation-status table at the top of [`theory/README.md`](theory/README.md). |

Start with [`implementation/`](implementation/README.md) if you just want to run the app, or [`theory/`](theory/README.md) if you want to understand the moving parts first.

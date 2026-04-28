# Local Proposal RAG — Documentation

A fully local Retrieval-Augmented Generation (RAG) application for querying engineering proposal PDFs, powered by **Strands Agents**. No data leaves the machine after the initial model download.

| | |
|---|---|
| **Embedding model** | `embeddinggemma` (300M params, 768-dim, via Ollama) |
| **Generation model** | `gemma4:e2b` (2.3B effective params, 128K context, via Ollama) |
| **Vector store** | ChromaDB (persistent, on-disk) |
| **Agent SDK** | Strands Agents |
| **UI** | Streamlit (`localhost:8501`) |
| **Package manager** | uv |

---

## System Context

The app consists entirely of local processes. The only external traffic is the one-time `ollama pull` to download model weights.

```mermaid
C4Context
    accTitle: Local Proposal RAG system context
    accDescr: An engineer opens a browser on localhost:8501; all processing stays on the local PC via Ollama and ChromaDB.

    title System Context — Local Proposal RAG

    Person(engineer, "Engineer", "Asks questions about engineering proposal PDFs")

    System_Boundary(pc, "Local PC") {
        System(app, "Local Proposal RAG", "Streamlit web app — orchestrates ingestion, retrieval, and generation via a Strands agent")
        System(ollama_sys, "Ollama", "Local model server on localhost:11434 — serves EmbeddingGemma and Gemma 4 e2b")
        System(chroma_sys, "ChromaDB", "On-disk vector store — holds 768-dim embeddings and chunk metadata")
    }

    Rel(engineer, app, "Opens browser", "localhost:8501")
    Rel(app, ollama_sys, "Embeds text and generates answers", "HTTP localhost:11434")
    Rel(app, chroma_sys, "Reads and writes vectors", "Python SDK")
```

---

## Documents

| File | What it covers |
|---|---|
| [architecture.md](architecture.md) | Container diagram and service topology |
| [ingestion.md](ingestion.md) | PDF-to-vector pipeline (flowchart + sequence) |
| [query-flow.md](query-flow.md) | End-to-end query and answer sequence via Strands |
| [setup.md](setup.md) | First-time setup decision tree and command reference |

---

## Quick Start

> Prerequisites: Ollama installed, both models pulled, uv installed.

```powershell
# From Local-RAG/
uv sync                              # rebuild .venv from pyproject.toml
uv run streamlit run app.py          # open http://localhost:8501
```

Then click **Re-ingest PDFs** in the sidebar once to index all PDFs in `raw-files/`.

---

## Project Layout

```
Local-RAG/
├── app.py            # Streamlit UI
├── ingest.py         # Ingestion pipeline
├── rag.py            # Retrieval (embedding + ChromaDB lookup)
├── agent.py          # Strands Agent — search_documents tool + agent factory
├── pyproject.toml    # uv project definition
├── uv.lock           # Reproducible lockfile
├── chroma_db/        # ChromaDB data (git-ignored)
├── raw-files/        # Source PDFs (git-ignored)
├── .venv/            # uv-managed venv (git-ignored)
└── docs/             # This documentation
    ├── README.md         # Hub — points here and to theory/
    ├── implementation/   # You are here
    │   ├── README.md
    │   ├── architecture.md
    │   ├── ingestion.md
    │   ├── query-flow.md
    │   └── setup.md
    └── theory/           # General RAG / ecosystem reference
```

---

## Privacy and Network Boundaries

```mermaid
flowchart LR
    accTitle: Network boundary — what leaves the PC vs what stays local
    accDescr: Only model downloads cross the network boundary; all inference and data stays on localhost.

    internet@{ shape: cloud, label: "Internet" }
    ollama_reg@{ shape: subproc, label: "Ollama Registry\n(one-time pull)" }
    pypi@{ shape: subproc, label: "PyPI\n(one-time uv sync)" }

    subgraph local["Local PC — localhost only after setup"]
        browser@{ shape: stadium, label: "Browser\nlocalhost:8501" }
        streamlit@{ shape: rect, label: "Streamlit\napp.py" }
        strands@{ shape: rect, label: "Strands Agent\nagent.py" }
        ollama_d@{ shape: rect, label: "Ollama daemon\nlocalhost:11434" }
        chroma_d@{ shape: cyl, label: "ChromaDB\nchroma_db/" }
        pdfs@{ shape: docs, label: "PDFs\nraw-files/" }
    end

    internet -. "model weights (one-time)" .-> ollama_reg
    internet -. "packages (one-time)" .-> pypi
    ollama_reg --> ollama_d
    pypi --> streamlit
    browser --> streamlit
    streamlit --> strands
    strands --> ollama_d
    strands --> chroma_d
    streamlit --> pdfs
```

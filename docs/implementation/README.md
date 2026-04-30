# Local Proposal RAG — Documentation

A Retrieval-Augmented Generation (RAG) application for querying engineering proposal PDFs, powered by **Strands Agents**. Embeddings and the vector store run on-device; only the user question and the top retrieved chunks are sent to the hosted **Google Gemini API** for answer generation.

| | |
|---|---|
| **Embedding model** | `embeddinggemma` (300M params, 768-dim, via Ollama — runs on-device) |
| **Generation model** | `gemini-2.5-flash-lite` (default; via the hosted Google Gemini API) — `gemini-2.5-flash` is a higher-quality option, `gemini-2.5-pro` is the highest-quality option. Override with the `GEMINI_MODEL` env var. |
| **Vector store** | ChromaDB (persistent, on-disk) |
| **Agent SDK** | Strands Agents |
| **UI** | Streamlit (`localhost:8501`) |
| **Package manager** | uv |

---

## System Context

Ingestion, retrieval, and the embedding model are local. Generation is delegated to the hosted Gemini API — only the user question and the retrieved chunks leave the machine.

```mermaid
C4Context
    accTitle: Local Proposal RAG system context
    accDescr: An engineer opens a browser on localhost:8501; ingestion, retrieval, and embeddings run locally via Ollama and ChromaDB. Answer generation calls the hosted Gemini API.

    title System Context — Local Proposal RAG

    Person(engineer, "Engineer", "Asks questions about engineering proposal PDFs")

    System_Boundary(pc, "Local PC") {
        System(app, "Local Proposal RAG", "Streamlit web app — orchestrates ingestion, retrieval, and generation via a Strands agent")
        System(ollama_sys, "Ollama", "Local model server on localhost:11434 — serves EmbeddingGemma")
        System(chroma_sys, "ChromaDB", "On-disk vector store — holds 768-dim embeddings and chunk metadata")
    }

    System_Ext(gemini_sys, "Google Gemini API", "Hosted LLM — generates grounded answers (default model: gemini-2.5-flash-lite)")

    Rel(engineer, app, "Opens browser", "localhost:8501")
    Rel(app, ollama_sys, "Embeds text", "HTTP localhost:11434")
    Rel(app, chroma_sys, "Reads and writes vectors", "Python SDK")
    Rel(app, gemini_sys, "Generates answers (question + retrieved chunks)", "HTTPS")
```

---

## Documents

| File | What it covers |
|---|---|
| [architecture.md](architecture.md) | Container diagram and service topology |
| [ingestion.md](ingestion.md) | PDF-to-vector pipeline (flowchart + sequence) |
| [query-flow.md](query-flow.md) | End-to-end query and answer sequence via Strands |
| [setup.md](setup.md) | First-time setup decision tree and command reference |
| [gemini-api-key.md](gemini-api-key.md) | How to obtain, install, and rotate a Google Gemini API key |

---

## Quick Start

> Prerequisites: Ollama installed, `embeddinggemma` pulled, a Gemini API key in `.env`, and uv installed. See [setup.md](setup.md) and [gemini-api-key.md](gemini-api-key.md).

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
    accDescr: After setup, the only ongoing external traffic is the Gemini generation call (question + retrieved chunks). PDFs, embeddings, and the vector store stay on-device.

    internet@{ shape: cloud, label: "Internet" }
    ollama_reg@{ shape: subproc, label: "Ollama Registry\n(one-time pull)" }
    pypi@{ shape: subproc, label: "PyPI\n(one-time uv sync)" }
    gemini_api@{ shape: subproc, label: "Google Gemini API\n(every question:\nquery + retrieved chunks)" }

    subgraph local["Local PC"]
        browser@{ shape: stadium, label: "Browser\nlocalhost:8501" }
        streamlit@{ shape: rect, label: "Streamlit\napp.py" }
        strands@{ shape: rect, label: "Strands Agent\nagent.py" }
        ollama_d@{ shape: rect, label: "Ollama daemon\nlocalhost:11434\n(embeddings only)" }
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
    strands ==> gemini_api
    streamlit --> pdfs
```

# Local-RAG

A fully local Retrieval-Augmented Generation (RAG) app for asking questions about your own PDF documents (e.g. engineering proposals). Nothing leaves your machine.

## Stack

- **Streamlit** — chat UI
- **Ollama** — runs the embedding and generation models locally
  - `embeddinggemma` — 768-dim embeddings
  - `gemma4:e2b` — generation model (active default; `gemma4:e4b` adds quality at higher cost, `gemma4:31b` is the high-end workstation option)
- **ChromaDB** — local persistent vector store
- **Strands Agents** — agent loop that calls a `search_documents` tool before answering
- **uv** — Python project + dependency manager

## Project layout

```
Local-RAG/
├── app.py          # Streamlit UI
├── agent.py        # Strands agent + search_documents tool
├── rag.py          # Query embedding + Chroma retrieval
├── ingest.py       # PDF -> chunks -> embeddings -> Chroma
├── pyproject.toml  # uv-managed dependencies
├── raw-files/      # <- put your PDFs here
└── chroma_db/      # created on first ingest
```

## Prerequisites

1. **Python 3.11+**
2. **[uv](https://docs.astral.sh/uv/)** — install on Windows:
   ```powershell
   winget install --id=astral-sh.uv -e
   ```
3. **[Ollama](https://ollama.com/download)** running locally on `http://localhost:11434`. After installing, pull the embedding model and **one** of the generation models:
   ```powershell
   ollama pull embeddinggemma

   # Pick ONE generation model (active default is gemma4:e2b):
   ollama pull gemma4:e2b      # ~1.6 GB, fast on CPU / iGPU — default
   ollama pull gemma4:e4b      # ~5 GB, modest quality lift — needs decent GPU
   ollama pull gemma4:31b      # ~19 GB, needs ~24 GB VRAM — best quality
   ```
   If you pick a non-default option, update `GEN_MODEL` in [agent.py](agent.py) accordingly.
   Verify Ollama is running:
   ```powershell
   ollama list
   ```

## Setup

Clone and enter the repo:

```powershell
git clone https://github.com/AWDubs/Local-RAG.git
cd Local-RAG
```

Install dependencies into a project-local virtual environment:

```powershell
uv sync
```

## 1. Add your PDFs

Drop any PDF files you want to query into the `raw-files/` folder:

```powershell
Copy-Item "C:\path\to\your\proposal.pdf" .\raw-files\
```

Subfolders are fine — every `.pdf` under `raw-files/` will be picked up.

## 2. Run the UI

Start the Streamlit chat app:

```powershell
uv run streamlit run app.py
```

Streamlit will open a browser tab (typically <http://localhost:8501>).

## 3. Ingest (from the app)

Ingestion is triggered **from the Streamlit sidebar**, not from the command line. Click **Re-ingest PDFs** to:

1. Scan every `.pdf` under `raw-files/`
2. Chunk the text (~1200 chars with 200 overlap)
3. Embed each chunk with `embeddinggemma`
4. Write the vectors to `chroma_db/` (created on first run)

Re-click the button any time you add, remove, or change PDFs in `raw-files/`.

> **Note:** `ingest.py` only **defines** the ingestion pipeline — it does not run it on import. Running `uv run python ingest.py` directly does nothing visible (no progress, no `chroma_db/`). Always trigger ingestion from the app sidebar.

Once ingestion completes, ask questions in the chat box — the agent will call the search tool, retrieve the top matching chunks, and answer with `[source.pdf #chunk_index]` citations. The right-hand **Thinking** panel shows the per-turn trace (tool calls, retrieval results, generation), and the sidebar **Temperature** slider tunes generation randomness (lower = more grounded).

## Troubleshooting

- **"Connection refused" / model errors** — make sure Ollama is running and both models are pulled (`ollama list`).
- **"Indexed chunks: 0"** in the sidebar — you haven't ingested yet, or `raw-files/` is empty. Click **Re-ingest PDFs**.
- **No `chroma_db/` folder** — ingestion has not been run yet. Use the **Re-ingest PDFs** button in the app sidebar (running `ingest.py` from the CLI will not create it).
- **Changed PDFs but answers are stale** — click **Re-ingest PDFs** in the sidebar.

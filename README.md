# Local-RAG

A Retrieval-Augmented Generation (RAG) app for asking questions about your own PDF documents (e.g. engineering proposals). Embeddings and the vector store run on-device; only the user question and the retrieved chunks are sent to the Gemini API for answer generation.

## Stack

- **Streamlit** — chat UI
- **Ollama** — runs the embedding model locally (`embeddinggemma`, 768-dim)
- **Google Gemini API** — generates answers (`gemini-2.5-flash-lite` by default, overridable via the `GEMINI_MODEL` env var; free tier on Google AI Studio)
- **ChromaDB** — local persistent vector store
- **Strands Agents** — agent loop that calls a `search_documents` tool before answering
- **uv** — Python project + dependency manager

## Privacy note

The PDF text never leaves your machine in bulk. At query time, however, the retrieved chunks (typically 20 passages) **are sent to Google as part of the prompt**. On the free tier of Google AI Studio, that input may be used to improve Google's products. If your documents are confidential, switch to the paid tier (which opts out) or revert to a fully local generation backend.

## Project layout

```
Local-RAG/
├── app.py          # Streamlit UI
├── agent.py        # Strands agent + search_documents tool (uses Gemini)
├── rag.py          # Query embedding + Chroma retrieval
├── ingest.py       # PDF -> chunks -> embeddings -> Chroma
├── pyproject.toml  # uv-managed dependencies
├── .env.example    # Template for GEMINI_API_KEY
├── raw-files/      # <- put your PDFs here
└── chroma_db/      # created on first ingest
```

## Prerequisites

1. **Python 3.11+**
2. **[uv](https://docs.astral.sh/uv/)** — install on Windows:
   ```powershell
   winget install --id=astral-sh.uv -e
   ```
3. **[Ollama](https://ollama.com/download)** running locally on `http://localhost:11434`. After installing, pull the embedding model:
   ```powershell
   ollama pull embeddinggemma
   ollama list
   ```
4. **Gemini API key** — get a free key at <https://aistudio.google.com/apikey>. Free tier covers the Gemini 2.5 Flash family (including `gemini-2.5-flash-lite`, the app's default) for both input and output (rate-limited; see [pricing](https://ai.google.dev/gemini-api/docs/pricing)). Step-by-step walkthrough: [docs/implementation/gemini-api-key.md](docs/implementation/gemini-api-key.md).

## Setup

Clone and enter the repo:

```powershell
git clone https://github.com/AWDubs/Local-RAG.git
cd Local-RAG
```

Install dependencies:

```powershell
uv sync
```

Configure your API key by copying the example env file and editing it:

```powershell
Copy-Item .env.example .env
notepad .env
```

`.env` is git-ignored. Alternatively export `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in your shell.

## 1. Add your PDFs

Drop any PDF files you want to query into the `raw-files/` folder:

```powershell
Copy-Item "C:\path\to\your\proposal.pdf" .\raw-files\
```

Subfolders are fine — every `.pdf` under `raw-files/` will be picked up.

## 2. Run the UI

```powershell
uv run streamlit run app.py
```

Streamlit will open a browser tab (typically <http://localhost:8501>).

## 3. Ingest (from the app)

Ingestion is triggered **from the Streamlit sidebar**, not from the command line. Click **Re-ingest PDFs** to:

1. Scan every `.pdf` under `raw-files/`
2. Chunk the text (~1200 chars with 200 overlap)
3. Embed each chunk locally with `embeddinggemma` via Ollama
4. Write the vectors to `chroma_db/` (created on first run)

Re-click the button any time you add, remove, or change PDFs in `raw-files/`.

> **Note:** `ingest.py` only **defines** the ingestion pipeline — it does not run it on import. Always trigger ingestion from the app sidebar.

Once ingestion completes, ask questions in the chat box — the agent calls the search tool, retrieves the top matching chunks, and Gemini answers with `[title · p.PAGE]` citations. The right-hand **Thinking** panel shows the per-turn trace, and the sidebar **Temperature** slider tunes generation randomness (lower = more grounded).

## Choosing a Gemini model

Set the `GEMINI_MODEL` environment variable (in `.env` or your shell) to override the default. The fallback in [agent.py](agent.py) is `gemini-2.5-flash-lite`.

| Model | Notes |
|---|---|
| `gemini-2.5-flash-lite` *(default)* | Cheapest / fastest, highest free-tier RPM/RPD |
| `gemini-2.5-flash` | Higher quality, lower free-tier RPM/RPD |
| `gemini-2.5-pro` | Highest quality, tightest free-tier RPD |

## Troubleshooting

- **"GEMINI_API_KEY is not set"** — create `.env` from `.env.example` and put your key in it, or export `GEMINI_API_KEY` in your shell.
- **Rate-limit / 429 errors from Gemini** — you've hit the free-tier RPM/RPD cap. Wait a minute, switch back to the default `gemini-2.5-flash-lite` (if you'd overridden `GEMINI_MODEL`), or upgrade to the paid tier in Google AI Studio.
- **"Connection refused" on embeddings** — Ollama isn't running. Start it (system tray on Windows) and confirm with `ollama list`.
- **"Indexed chunks: 0"** in the sidebar — you haven't ingested yet, or `raw-files/` is empty. Click **Re-ingest PDFs**.
- **No `chroma_db/` folder** — ingestion has not been run yet. Use the **Re-ingest PDFs** button in the sidebar.
- **Changed PDFs but answers are stale** — click **Re-ingest PDFs** in the sidebar.

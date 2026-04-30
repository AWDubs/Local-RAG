# Setup and Operations

---

## First-Time Setup

Follow the decision tree to get from zero to a running app.

```mermaid
flowchart TD
    accTitle: First-time setup decision tree
    accDescr: Walks through installing Ollama, pulling the two models, syncing the uv environment, and launching the Streamlit app.

    begin_n@{ shape: stadium, label: "Begin setup" }

    q_ollama@{ shape: diam, label: "Ollama\ninstalled?" }
    install_ollama@{ shape: lean-r, label: "Download installer\nollama.com/download\nRun and follow prompts" }

    q_embed@{ shape: diam, label: "embeddinggemma\npulled?" }
    pull_embed@{ shape: subproc, label: "ollama pull embeddinggemma\n622 MB" }

    q_gen@{ shape: diam, label: "Gemini API key\nset?" }
    pull_gen@{ shape: subproc, label: "Get free key at\naistudio.google.com/apikey\nCopy .env.example -> .env\nSet GEMINI_API_KEY=..." }

    q_uv@{ shape: diam, label: "uv\ninstalled?" }
    install_uv@{ shape: lean-r, label: "winget install astral-sh.uv\nor: pip install uv" }

    sync_n@{ shape: subproc, label: "cd Local-RAG\nuv sync" }
    run_n@{ shape: subproc, label: "uv run streamlit run app.py" }
    ingest_n@{ shape: hex, label: "Click Re-ingest PDFs\nin sidebar — wait for completion" }
    ready_n@{ shape: dbl-circ, label: "App ready\nask questions" }

    begin_n ==> q_ollama
    q_ollama -- "no" --> install_ollama
    q_ollama -- "yes" --> q_embed
    install_ollama --> q_embed
    q_embed -- "no" --> pull_embed
    q_embed -- "yes" --> q_gen
    pull_embed --> q_gen
    q_gen -- "no" --> pull_gen
    q_gen -- "yes" --> q_uv
    pull_gen --> q_uv
    q_uv -- "no" --> install_uv
    q_uv -- "yes" --> sync_n
    install_uv --> sync_n
    sync_n --> run_n
    run_n --> ingest_n
    ingest_n ==> ready_n
```

---

## Command Reference

### Ollama

```powershell
# Pull the embedding model (one-time). Generation runs on Gemini, not Ollama.
ollama pull embeddinggemma        # 622 MB — embedding model

# Verify it's present
ollama list

# Check what is loaded in VRAM right now
ollama ps
```

Generation uses the **hosted Gemini API** (free tier on Google AI Studio).
Get a key at <https://aistudio.google.com/apikey> and put it in `.env`:

```powershell
Copy-Item .env.example .env
notepad .env   # set GEMINI_API_KEY=...
```

To switch which Gemini model the app uses, set the `GEMINI_MODEL` env var (in `.env` or your shell) and restart Streamlit — e.g. `GEMINI_MODEL=gemini-2.5-flash`. The fallback in [`agent.py`](../../agent.py) is `gemini-2.5-flash-lite`.

### uv (from `Local-RAG/`)

```powershell
# Install / restore dependencies from pyproject.toml + uv.lock
uv sync

# Run the app (no activation needed)
uv run streamlit run app.py

# NOTE: Ingestion is triggered from the app sidebar ("Re-ingest PDFs"),
# not from the CLI. `ingest.py` only defines the pipeline; running it
# directly with `uv run python ingest.py` will not produce output or
# create `chroma_db/`.

# Add a new package
uv add package-name

# Show installed packages
uv pip list
```

---

## Model Resource Requirements

Only the embedding model runs locally. Generation runs in Google's data centres.

```mermaid
quadrantChart
    accTitle: Gemini model size vs quality tradeoff
    accDescr: Plots each Gemini 2.5 variant by latency / cost vs quality so you can pick the right one for your workload.

    title Gemini 2.5 variants — Cost/latency vs Quality
    x-axis Lower cost / faster --> Higher cost / slower
    y-axis Lower Quality --> Higher Quality
    quadrant-1 High quality, higher cost
    quadrant-2 High quality, low cost
    quadrant-3 Low quality, low cost
    quadrant-4 Low quality, high cost
    flash-lite (this app): [0.20, 0.45]
    flash: [0.45, 0.70]
    pro: [0.85, 0.92]
```

| Model | Where it runs | Notes |
|---|---|---|
| `embeddinggemma` | Local via Ollama | Always needed; ~1 GB VRAM, stays loaded |
| `gemini-2.5-flash-lite` *(this app)* | Gemini API | Default — cheapest / fastest, generous free-tier RPM/RPD |
| `gemini-2.5-flash` | Gemini API | Higher quality, lower free-tier RPM/RPD |
| `gemini-2.5-pro` | Gemini API | Highest quality, tightest free-tier RPD |

To switch generation models, set `GEMINI_MODEL` in `.env` (or your shell) and restart the app.

---

## Troubleshooting

```mermaid
flowchart TD
    accTitle: Troubleshooting decision tree
    accDescr: Common failure modes and their fixes.

    prob@{ shape: stadium, label: "Problem encountered" }

    q1@{ shape: diam, label: "App won't start?" }
    fix1@{ shape: subproc, label: "Run: uv sync\nCheck pyproject.toml exists" }

    q2@{ shape: diam, label: "Ollama connection\nerror?" }
    fix2@{ shape: subproc, label: "Check system tray — is Ollama running?\nOr run: ollama serve\nVerify: curl http://localhost:11434" }

    q3@{ shape: diam, label: "No chunks indexed?" }
    fix3@{ shape: subproc, label: "Check raw-files/ contains *.pdf\nClick Re-ingest PDFs again" }

    q4@{ shape: diam, label: "Embedding model\nnot found error?" }
    fix4@{ shape: subproc, label: "ollama pull embeddinggemma" }

    q5@{ shape: diam, label: "Agent not calling\nthe tool?" }
    fix5@{ shape: subproc, label: "Check SYSTEM_PROMPT in agent.py\nEnsure model supports tool calling\nTry a different Gemini model" }

    q6@{ shape: diam, label: "Ingestion is\nvery slow?" }
    fix6@{ shape: subproc, label: "Normal — each chunk is one\nHTTP round-trip to Ollama\n10 PDFs takes 2-5 minutes" }

    q7@{ shape: diam, label: "Gemini 429 / rate\nlimit error?" }
    fix7@{ shape: subproc, label: "Wait a minute\nor stay on the default gemini-2.5-flash-lite\nor upgrade to paid tier" }

    q8@{ shape: diam, label: "GEMINI_API_KEY\nnot set error?" }
    fix8@{ shape: subproc, label: "Copy .env.example to .env\nSet GEMINI_API_KEY=...\nGet a key: aistudio.google.com/apikey" }

    done_t@{ shape: dbl-circ, label: "Resolved" }

    prob --> q1
    prob --> q2
    prob --> q3
    prob --> q4
    prob --> q5
    prob --> q6
    prob --> q7
    prob --> q8
    q1 --> fix1
    q2 --> fix2
    q3 --> fix3
    q4 --> fix4
    q5 --> fix5
    q6 --> fix6
    q7 --> fix7
    q8 --> fix8
    fix1 & fix2 & fix3 & fix4 & fix5 & fix6 & fix7 & fix8 ==> done_t
```

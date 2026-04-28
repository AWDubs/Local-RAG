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

    q_gen@{ shape: diam, label: "gemma4:e2b\npulled?" }
    pull_gen@{ shape: subproc, label: "ollama pull gemma4:e2b\n7.2 GB" }

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
# Pull models (one-time)
ollama pull embeddinggemma        # 622 MB — embedding model
ollama pull gemma4:e2b            # 7.2 GB — generation model

# Verify models are present
ollama list

# Check what is loaded in VRAM right now
ollama ps

# Smoke-test generation
ollama run gemma4:e2b "What is RAG? One sentence."
```

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

```mermaid
quadrantChart
    accTitle: Model size vs quality tradeoff for Gemma 4 variants
    accDescr: Plots each Gemma 4 variant by VRAM requirement and output quality so you can pick the right one for your hardware.

    title Gemma 4 variants — VRAM vs Quality
    x-axis Low VRAM --> High VRAM
    y-axis Lower Quality --> Higher Quality
    quadrant-1 High quality, high VRAM
    quadrant-2 High quality, low VRAM
    quadrant-3 Low quality, low VRAM
    quadrant-4 Low quality, high VRAM
    e2b (this app): [0.35, 0.45]
    e4b (default): [0.45, 0.55]
    26b MoE: [0.65, 0.80]
    31b Dense: [0.85, 0.90]
```

| Model | VRAM | Context | Notes |
|---|---|---|---|
| `embeddinggemma` | ~1 GB | 2K | Always needed; stays loaded |
| `gemma4:e2b` *(this app)* | ~7 GB | 128K | Good balance for laptops |
| `gemma4:e4b` | ~9.6 GB | 128K | Default tag; slightly higher quality |
| `gemma4:26b` | ~18 GB | 256K | MoE — high quality, workstation GPU |
| `gemma4:31b` | ~20 GB | 256K | Dense — top accuracy, 24 GB VRAM |

To switch generation models, change `GEN_MODEL` in [agent.py](../agent.py) and restart the app.

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

    q4@{ shape: diam, label: "Model not found\nerror?" }
    fix4@{ shape: subproc, label: "ollama pull embeddinggemma\nollama pull gemma4:e2b" }

    q5@{ shape: diam, label: "Agent not calling\nthe tool?" }
    fix5@{ shape: subproc, label: "Check SYSTEM_PROMPT in agent.py\nEnsure model supports tool calling\nTry a different model variant" }

    q6@{ shape: diam, label: "Ingestion is\nvery slow?" }
    fix6@{ shape: subproc, label: "Normal — each chunk is one\nHTTP round-trip to Ollama\n10 PDFs takes 2-5 minutes" }

    done_t@{ shape: dbl-circ, label: "Resolved" }

    prob --> q1
    prob --> q2
    prob --> q3
    prob --> q4
    prob --> q5
    prob --> q6
    q1 --> fix1
    q2 --> fix2
    q3 --> fix3
    q4 --> fix4
    q5 --> fix5
    q6 --> fix6
    fix1 & fix2 & fix3 & fix4 & fix5 & fix6 ==> done_t
```

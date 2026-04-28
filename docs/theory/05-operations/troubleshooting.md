# Troubleshooting

A decision tree for every common error you'll hit running the local RAG app — from Ollama not starting to ChromaDB file locks and Streamlit port conflicts.

---

## Master Error Decision Tree

```mermaid
flowchart TD
    accTitle: Local RAG troubleshooting decision tree
    accDescr: Routes common errors to their fix through a series of yes/no questions about Ollama, ChromaDB, Streamlit, and Python environment.

    start@{ shape: stadium, label: "Something broke" }
    q_sidebar@{ shape: diam, label: "Sidebar shows\n🔴 Ollama not reachable?" }
    q_model@{ shape: diam, label: "Error says\n'model not found'?" }
    q_chroma@{ shape: diam, label: "ChromaDB error?" }
    q_streamlit@{ shape: diam, label: "Streamlit fails\nto start?" }
    q_import@{ shape: diam, label: "ModuleNotFoundError?" }

    fix_ollama@{ shape: hex, label: "Start Ollama:\n'ollama serve'\nor restart system-tray app" }
    fix_model@{ shape: hex, label: "Pull missing model:\n'ollama pull gemma4:e2b'\n'ollama pull embeddinggemma'" }
    chroma_sub@{ shape: diam, label: "ChromaDB error type?" }
    fix_lock@{ shape: hex, label: "DB locked / PermissionError:\nStop all app instances\nDelete chroma_db/chroma.sqlite3-wal" }
    fix_dim@{ shape: hex, label: "Dimension mismatch:\nDelete chroma_db/ folder\nRe-embed with new model" }
    fix_seg@{ shape: hex, label: "SegmentationFault:\nuv lock --upgrade-package chromadb\nuv sync" }
    fix_port@{ shape: hex, label: "Port 8501 busy:\nuv run streamlit run app.py --server.port 8502" }
    fix_import@{ shape: hex, label: "Re-sync env:\nuv sync\n(or run via 'uv run ...')" }

    q_other@{ shape: diam, label: "Slow generation?" }
    fix_slow@{ shape: hex, label: "See Performance Tuning doc:\nReduce num_ctx, use gemma4:e2b,\nor enable GPU offload" }
    q_halluc@{ shape: diam, label: "Model hallucinating\n(ignoring context)?" }
    fix_halluc@{ shape: hex, label: "Lower temperature to 0.1\nStrengthen system prompt\nIncrease top_k" }

    done@{ shape: dbl-circ, label: "Problem resolved" }

    start ==> q_sidebar
    q_sidebar -- "yes" --> fix_ollama --> done
    q_sidebar -- "no" --> q_model
    q_model -- "yes" --> fix_model --> done
    q_model -- "no" --> q_chroma
    q_chroma -- "yes" --> chroma_sub
    chroma_sub -- "locked / Permission" --> fix_lock --> done
    chroma_sub -- "dimension mismatch" --> fix_dim --> done
    chroma_sub -- "SegFault / crash" --> fix_seg --> done
    q_chroma -- "no" --> q_streamlit
    q_streamlit -- "yes" --> fix_port --> done
    q_streamlit -- "no" --> q_import
    q_import -- "yes" --> fix_import --> done
    q_import -- "no" --> q_other
    q_other -- "yes" --> fix_slow --> done
    q_other -- "no" --> q_halluc
    q_halluc -- "yes" --> fix_halluc --> done
    q_halluc -- "no" --> done
```

---

## Error Reference

### Ollama errors

| Error message | Cause | Fix |
|---|---|---|
| `ConnectionError: [Errno 111] Connection refused` | Ollama daemon not running | `ollama serve` in a terminal |
| `ResponseError: model "gemma4:e2b" not found` | Model not pulled | `ollama pull gemma4:e2b` |
| `ResponseError: model "embeddinggemma" not found` | Embedding model not pulled | `ollama pull embeddinggemma` |
| `CUDA out of memory` | VRAM exhausted | Switch to `gemma4:e2b` or set `OLLAMA_NUM_GPU=0` for CPU |
| Ollama hangs / no response | Model loading (first run) | Wait 30–60 s for GGUF to load into VRAM |

### ChromaDB errors

| Error | Cause | Fix |
|---|---|---|
| `sqlite3.OperationalError: database is locked` | Two app instances open | Kill all Python processes: `pkill -f streamlit` |
| `InvalidDimensionException` | Switched embedding models after first embed | Delete `./chroma_db/` and re-embed |
| `SegmentationFault` on startup | Stale hnswlib binary | `uv lock --upgrade-package chromadb && uv sync` |
| `FileNotFoundError: chroma_db/` | Path doesn't exist | `mkdir chroma_db` or check `CHROMA_PATH` in `.env` |

### Streamlit errors

| Error | Cause | Fix |
|---|---|---|
| `OSError: [Errno 98] Address already in use` | Port 8501 occupied | `--server.port 8502` flag |
| `DuplicateWidgetID` | Widgets created inside loops without unique keys | Add `key=f"widget_{i}"` to each widget |
| Page resets on every interaction | `st.rerun()` called unconditionally | Guard rerun calls with a state condition |
| `StreamlitAPIException: no active script run ctx` | Calling Streamlit outside the main thread | Move Ollama/Chroma calls into the main thread |

### Python environment errors

```mermaid
flowchart TD
    accTitle: Python import error resolution
    accDescr: Steps to resolve ModuleNotFoundError for any package in the RAG app.

    err@{ shape: doc, label: "ModuleNotFoundError:\nNo module named 'X'" }
    q_run@{ shape: diam, label: "Running via\n'uv run …' ?" }
    q_installed@{ shape: diam, label: "Is package in\npyproject.toml?" }

    use_uv@{ shape: hex, label: "Run via uv:\n'uv run python ...'\nor 'uv run streamlit ...'" }
    sync@{ shape: hex, label: "uv sync" }
    add@{ shape: hex, label: "uv add <package>\n(updates pyproject.toml + uv.lock)" }
    done2@{ shape: dbl-circ, label: "Import works" }

    err ==> q_run
    q_run -- "no" --> use_uv --> q_installed
    q_run -- "yes" --> q_installed
    q_installed -- "yes" --> sync --> done2
    q_installed -- "no" --> add --> done2
```

---

## Diagnostic Commands

```bash
# Is Ollama running?
curl http://localhost:11434/api/tags

# List loaded models (in VRAM)
ollama ps

# Check ChromaDB file sizes
du -sh ./chroma_db/

# What Python is active?
uv run python --version

# What packages are installed?
uv pip list | Select-String "chromadb|ollama|streamlit"

# Streamlit logs
uv run streamlit run app.py --logger.level debug
```

---

## Next Steps

- [Performance Tuning →](performance-tuning.md) — if the app is slow but working  
- [Running & Testing →](../04-build-the-app/05-running-and-testing.md) — smoke tests to verify your setup  
- [Evaluating RAG →](evaluating-rag.md) — if answers are poor quality

# System Architecture

This page gives you two views of the same system: an **`architecture-beta` topology** (where things live and how data flows) and a **C4 Container diagram** (what each deployable unit does).

---

## Topology Diagram

```mermaid
architecture-beta
    accTitle: Full local RAG application topology
    accDescr: Four groups — Streamlit UI, Ingest Pipeline, Query Pipeline, and Local Storage — showing services and their connections.

    group ui(internet)["Streamlit UI — localhost:8501"]
    group ctrl(server)[Sidebar Controls]
    group ingest(server)[Ingest Pipeline]
    group qpipe(server)[Query Pipeline]
    group storage(disk)[Local Storage]

    service user(internet)["Browser / User"] in ui
    service uploader(server)[File Uploader Tab] in ui
    service embed_btn(server)[Embed Button] in ui
    service chroma_browser(disk)[Browse ChromaDB Tab] in ui
    service chat_panel(server)[Chat Tab] in ui

    service temp_ctrl(server)[Temperature Slider] in ctrl
    service topk_ctrl(server)[top_k Slider] in ctrl
    service topp_ctrl(server)[top_p Slider] in ctrl
    service chunk_ctrl(server)[Chunk Size Slider] in ctrl
    service model_ctrl(server)[Model Picker] in ctrl

    service file_loader(server)["File Loader / Parser"] in ingest
    service chunker(server)[Text Chunker] in ingest
    service embed_model(server)["embeddinggemma via Ollama"] in ingest

    service query_embed(server)[Query Embedder] in qpipe
    service sim_search(database)[Similarity Search] in qpipe
    service prompt_bldr(server)[Prompt Builder] in qpipe
    service gemma(server)[Gemma 4 via Ollama] in qpipe

    service ollama_api(server)["Ollama REST :11434"] in storage
    service chromadb(database)[ChromaDB] in storage
    service chroma_files(disk)["./chroma_db/"] in storage
    service uploads_dir(disk)["./uploads/"] in storage

    user:R --> L:uploader
    user:R --> L:chat_panel
    uploader:R --> L:file_loader
    file_loader:B --> T:uploads_dir
    file_loader:R --> L:chunker
    embed_btn:R --> L:chunker
    chunker:R --> L:embed_model
    embed_model:R --> L:ollama_api
    ollama_api:R --> L:chromadb
    chromadb:B --> T:chroma_files
    chroma_browser:R --> L:chromadb

    chat_panel:R --> L:query_embed
    query_embed:R --> L:ollama_api
    ollama_api:R --> L:sim_search
    sim_search:R --> L:prompt_bldr
    prompt_bldr:R --> L:gemma
    gemma:R --> L:ollama_api
    ollama_api:B --> T:chat_panel

    temp_ctrl:B --> T:gemma
    topk_ctrl:B --> T:sim_search
    chunk_ctrl:B --> T:chunker
    model_ctrl:B --> T:gemma
```

---

## C4 Container View

```mermaid
C4Container
    accTitle: C4 Container diagram — Local RAG Application
    accDescr: Shows the Streamlit UI, Ingest Service, RAG Agent, Ollama daemon with model store, and ChromaDB with its persistence directory.
    title C4 Container — Local RAG Application
    Person(user, "User", "Uploads docs, browses ChromaDB, chats")

    System_Boundary(app, "Local RAG App") {
        Container(ui, "Streamlit UI", "Python / Streamlit", "File upload, Embed button, ChromaDB browser, Chat panel, Sidebar controls")
        Container(ingest, "Ingest Service", "Python", "Parses files, chunks text, calls Ollama /api/embeddings, writes to ChromaDB")
        Container(rag, "RAG Agent", "Python", "Embeds query, searches ChromaDB, assembles prompt, calls Ollama /api/generate, streams tokens")
    }

    System_Boundary(ollama_sys, "Ollama Daemon") {
        Container(ollama, "Ollama Server", "Go binary / :11434", "Serves /api/embeddings and /api/generate over HTTP")
        ContainerDb(models, "Model Store", "GGUF files on disk", "gemma4:e2b (default; gemma4:e4b or gemma4:31b also supported), embeddinggemma")
    }

    System_Boundary(db_sys, "ChromaDB") {
        ContainerDb(chroma, "ChromaDB", "Python / SQLite + HNSW", "Stores vectors, metadata, and raw chunk text")
        Container(chroma_dir, "Persistent Dir", "./chroma_db/ on disk", "SQLite WAL files, HNSW index segments")
    }

    Rel(user, ui, "Opens browser", "HTTP")
    Rel(ui, ingest, "Triggers ingestion", "in-process call")
    Rel(ui, rag, "Sends query", "in-process call")
    Rel(ingest, ollama, "POST /api/embeddings", "HTTP / localhost")
    Rel(rag, ollama, "POST /api/embeddings + /api/generate", "HTTP / localhost")
    Rel(ingest, chroma, "collection.add()", "Python client")
    Rel(rag, chroma, "collection.query()", "Python client")
    Rel(ollama, models, "Loads GGUF shards")
    Rel(chroma, chroma_dir, "Persists index")
```

---

## Data Flow Narratives

### Ingest Flow

1. User selects one or more files (PDF, Markdown, plain text) in the **File Uploader** tab.
2. The **File Loader** saves raw bytes to `./uploads/` and extracts plain text.
3. The **Chunker** splits text into overlapping windows (chunk size and overlap configurable in the sidebar).
4. Each chunk is sent to **Ollama `/api/embeddings`** with model `embeddinggemma`.
5. The resulting 768-d vectors are written to **ChromaDB** together with metadata (source filename, page number, chunk index).
6. The **Browse ChromaDB** tab in the UI shows the new entries immediately.

### Query Flow

1. User types a question in the **Chat** tab.
2. The question is embedded by `embeddinggemma` via Ollama.
3. ChromaDB performs an **HNSW approximate nearest-neighbour search** and returns the top-k most similar chunks (k is configurable in the sidebar).
4. The **Prompt Builder** wraps the retrieved chunks in a system prompt with citations.
5. Gemma 4 (`gemma4:31b` via Ollama; `gemma4:e2b` is the lightweight alternative) generates an answer using the provided temperature and top_p values from the sidebar.
6. Tokens are streamed back to the **Chat** panel as they are produced.

---

## Sidebar Controls Reference

| Control | Default | Effect |
|---------|---------|--------|
| **Temperature** | `0.2` | Higher → more creative; lower → more deterministic |
| **top_p** | `0.9` | Nucleus sampling threshold |
| **top_k** | `5` | Number of ChromaDB chunks to retrieve |
| **Chunk size** | `512` | Tokens per chunk (approximate) |
| **Chunk overlap** | `64` | Overlap between adjacent chunks |
| **Inference model** | `gemma4:31b` (default) — `gemma4:e2b` for laptops | Any Ollama model tag |
| **Embedding model** | `embeddinggemma` | Any Ollama embedding-capable model |

---

## Next Steps

- [What is RAG? →](what-is-rag.md) — concepts and motivation  
- [Tokens & Embeddings →](../01-foundations/tokens-and-embeddings.md) — the math behind the arrows  
- [Build the App →](../04-build-the-app/01-project-layout.md) — start coding

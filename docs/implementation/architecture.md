# Architecture

---

## Containers

Embeddings and the vector store run on a single developer workstation. Generation is delegated to the hosted **Google Gemini API** — only the user question and the retrieved chunks ever leave the machine.

```mermaid
C4Deployment
    accTitle: Local RAG deployment — hybrid local/cloud
    accDescr: The Python process, Ollama daemon, and local filesystem are on one developer workstation. Generation calls go out to the Gemini API; everything else stays on-device.

    title Deployment — Proposal RAG (Strands + Gemini edition)

    Deployment_Node(local, "Local Machine", "Developer workstation — only query text and retrieved chunks leave the box") {

        Deployment_Node(python_proc, "Python Process", "Single process, Streamlit :8501") {
            Container(ui, "Streamlit UI", "Python / Streamlit", "Chat interface, sidebar controls (temperature slider, re-ingest, clear chat), Sources/Chunks panels, live Thinking panel — app.py")
            Container(ingest, "Ingest Module", "Python", "Reads PDFs page-by-page, parses filename metadata (proposal_id, customer, year, title), chunks text, calls Ollama for embeddings, writes to ChromaDB — ingest.py")
            Container(rag, "RAG Module", "Python", "Hybrid retrieval: caches the corpus, builds a BM25 index over docs + metadata text, applies optional customer/year/proposal_id filters, fuses BM25 + dense cosine scores, and MMR re-ranks to top-K chunks — rag.py")
            Container(agent, "Strands Agent", "Python / Strands", "Agent loop: calls search_documents tool (with optional metadata filters), synthesises grounded answer with [title · p.N] citations — agent.py")
        }

        Deployment_Node(ollama_proc, "Ollama Daemon", "localhost:11434") {
            Container(embed_model, "EmbeddingGemma", "Ollama model", "Produces 768-dim embeddings")
        }

        Deployment_Node(disk, "Local Filesystem", "Persistent storage") {
            ContainerDb(chroma, "ChromaDB", "Embedded vector DB", "Embeddings and chunk metadata at ./chroma_db/")
            Container(pdfs, "PDF Files", "Raw documents", "Source documents at ./raw-files/")
        }
    }

    Deployment_Node(google, "Google AI Studio", "generativelanguage.googleapis.com") {
        Container(gen_model, "Gemini 2.5 Flash Lite", "Hosted LLM (default; override with GEMINI_MODEL)", "Generates answers from retrieved context")
    }

    Rel(ui, ingest, "Calls ingest()", "Python import")
    Rel(ui, agent, "Calls agent(question)", "Python import")
    Rel(agent, rag, "Calls retrieve() via search_documents tool", "Python import")
    Rel(ingest, pdfs, "Reads", "Filesystem")
    Rel(ingest, embed_model, "POST /api/embeddings", "HTTP localhost")
    Rel(ingest, chroma, "Writes chunks and embeddings", "Python client")
    Rel(rag, embed_model, "POST /api/embeddings", "HTTP localhost")
    Rel(rag, chroma, "Loads full corpus once (cached); fuses BM25 + dense and MMR re-ranks in-process", "Python client")
    Rel(agent, gen_model, "generateContent via GeminiModel", "HTTPS")
```

---

## Service Topology

End-to-end flows for ingestion (one-time setup) and query (every question). Both flows share the same Ollama embedding endpoint and ChromaDB collection.

```mermaid
sequenceDiagram
    accTitle: Local RAG service topology — ingestion and query flows
    accDescr: Ingestion embeds PDFs and stores them in ChromaDB. The query flow runs via the Strands agent which calls the search_documents tool then generates a grounded answer.

    actor Engineer
    participant UI as Streamlit UI
    participant Ingest as Ingest Module
    participant Agent as Strands Agent
    participant RAG as RAG Module
    participant Embed as EmbeddingGemma<br/>(Ollama)
    participant DB as ChromaDB
    participant LLM as Gemini 2.5 Flash Lite<br/>(AI Studio)

    rect rgb(240, 248, 255)
        note over Engineer,DB: Ingestion flow (one-time / on demand)
        Engineer->>UI: Click "Re-ingest PDFs"
        UI->>Ingest: ingest()
        Ingest->>Ingest: Read & chunk PDF files
        Ingest->>Embed: POST /api/embeddings (chunks)
        Embed-->>Ingest: Embedding vectors
        Ingest->>DB: Write chunks + embeddings
    end

    rect rgb(240, 255, 240)
        note over Engineer,LLM: Query flow (every question)
        Engineer->>UI: Ask a question
        UI->>Agent: agent(question)
        Agent->>LLM: Initial reasoning turn
        LLM-->>Agent: Tool call: search_documents(query)
        Agent->>RAG: retrieve(query, top_k=20)
        RAG->>Embed: POST /api/embeddings (query)
        Embed-->>RAG: Query embedding
        RAG->>RAG: Score corpus: BM25 + cosine, fuse, MMR re-rank
        Note right of RAG: Corpus + BM25 index<br/>cached after first call;<br/>invalidated by re-ingest
        RAG-->>Agent: Top-20 diverse chunks
        Agent->>LLM: generateContent (context + question)
        LLM-->>Agent: Generated answer
        Agent-->>UI: AgentResult
        UI-->>Engineer: Display answer and sources
    end
```

---

## Module Responsibilities

```mermaid
classDiagram
    accTitle: Module responsibilities and key functions
    accDescr: Four Python modules with their public functions and shared constants.

    class ingest {
        +RAW_DIR: Path
        +DB_DIR: Path
        +COLLECTION: str
        +CHUNK_SIZE: int
        +CHUNK_OVERLAP: int
        +EMBED_MODEL: str
        +read_pdf_pages(pdf_path) list
        +parse_filename_metadata(filename) dict
        +chunk_text(text, size, overlap) list
        +embed_batch(texts, progress_cb) list
        +ingest(progress_cb) dict
    }

    class rag {
        +DB_DIR: Path
        +COLLECTION: str
        +EMBED_MODEL: str
        +DEFAULT_TOP_K: int
        +CANDIDATE_POOL: int
        +HYBRID_ALPHA: float
        +MMR_LAMBDA: float
        +embed_query(question) list
        +get_collection() Collection
        +retrieve(question, top_k, alpha, candidate_pool, mmr_lambda, customer, year, proposal_id) list
        +invalidate_cache() None
        -_load_corpus() tuple
        -_filter_indices(metas, customer, year, proposal_id) ndarray
        -_cosine_scores(q, M) ndarray
        -_minmax(x) ndarray
        -_mmr(...) list
    }

    class agent {
        +GEN_MODEL: str
        +GEMINI_API_KEY: str | None
        +SYSTEM_PROMPT: str
        +_last_chunks: list
        +_log_event: Callable
        +set_event_logger(fn) None
        +search_documents(query, customer, year, proposal_id) str
        +create_agent(temperature, callback_handler) Agent
    }

    class app {
        +get_chunk_count() int
        +_make_event_sink() list
        +_render_thinking_panel(placeholder, log) None
        +_make_callback_handler(sink) Callable
    }

    app --> ingest : "calls ingest()"
    app --> agent : "calls create_agent() / agent(question)"
    app --> rag : "calls get_collection() for chunk count"
    agent --> rag : "calls retrieve() inside search_documents tool"
    ingest ..> rag : "shares DB_DIR and COLLECTION constants"
```

---

## Data Shapes

The key data structures that flow between modules.

```mermaid
flowchart LR
    accTitle: Data shapes flowing between modules
    accDescr: PDF text becomes chunks, which become embeddings stored in ChromaDB; at query time the Strands agent calls the retrieval tool and produces a grounded answer.

    subgraph ingest_shapes["ingest.py produces"]
        chunk_shape@{ shape: doc, label: "Chunk\n{id: str\ndoc: str\nmeta: {source, title,\nproposal_id, customer, year,\npage_number, chunk_index,\ntotal_chunks, char_count,\ningested_at}}" }
        vec_shape@{ shape: lin-cyl, label: "Embedding\nlist[float] — 768 dims" }
    end

    subgraph rag_shapes["rag.py produces"]
        result_shape@{ shape: doc, label: "Retrieved chunk\n{doc, source, title,\nproposal_id, customer, year,\npage_number, chunk_index,\ntotal_chunks, char_count,\ningested_at, distance,\ndense_score, sparse_score,\nfused_score}" }
    end

    subgraph agent_shapes["agent.py produces"]
        tool_result@{ shape: das, label: "Tool result string\nFormatted context blocks [1]...[N]" }
        answer_shape@{ shape: doc, label: "AgentResult\nstr(response) = final answer" }
    end

    chunk_shape --> vec_shape
    vec_shape -. "stored in ChromaDB" .-> result_shape
    result_shape --> tool_result
    tool_result ==> answer_shape
```

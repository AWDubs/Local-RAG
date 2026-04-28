# Ingestion Pipeline

The ingestion pipeline converts raw PDF files into searchable embeddings stored in ChromaDB. It runs on-demand via the **Re-ingest PDFs** button and always rebuilds the collection from scratch.

---

## Pipeline Stages

Four stages from raw files to indexed vectors. The embedding stage processes all chunks collected across every PDF before touching ChromaDB.

```mermaid
flowchart TD
    accTitle: Ingestion pipeline stages
    accDescr: Four stages — scan, extract and chunk per PDF, embed all chunks, then store — with the skip path for unreadable PDFs.

    start_n@{ shape: stadium, label: "Ingest triggered" }
    scan_n@{ shape: subproc, label: "Scan raw-files/ for *.pdf" }
    q_found@{ shape: diam, label: "PDFs found?" }
    abort_n@{ shape: dbl-circ, label: "Abort: no PDFs" }

    subgraph per_pdf["For each PDF"]
        direction TB
        read_n@{ shape: docs, label: "PdfReader: extract text\npage by page with markers" }
        q_text@{ shape: diam, label: "Text\nextracted?" }
        skip_n@{ shape: brace, label: "Log warning\nand skip file" }
        chunk_n@{ shape: das, label: "chunk_text()\n1200-char window\n200-char overlap" }
    end

    prefix_n@{ shape: hex, label: "Prepend EmbeddingGemma\ndocument prefix:\ntitle: none | text: {chunk}" }
    embed_n@{ shape: subproc, label: "ollama.embeddings()\nper chunk — 768-dim vector" }
    wipe_n@{ shape: lean-r, label: "Delete old collection\nproposals_gemma" }
    store_n@{ shape: cyl, label: "ChromaDB create_collection\nadd() in batches of 500" }
    done_n@{ shape: dbl-circ, label: "Return {pdfs, chunks, errors}" }

    start_n ==> scan_n
    scan_n --> q_found
    q_found -- "no" --> abort_n
    q_found -- "yes" --> read_n
    read_n --> q_text
    q_text -- "no text" --> skip_n
    skip_n -. "next PDF" .-> read_n
    q_text -- "yes" --> chunk_n
    chunk_n --> prefix_n
    prefix_n --> embed_n
    embed_n --> wipe_n
    wipe_n --> store_n
    store_n ==> done_n
```

---

## Chunking Strategy

Overlapping windows preserve context that would otherwise be split at a boundary.

```mermaid
flowchart LR
    accTitle: Sliding-window chunking with overlap
    accDescr: Shows how the 200-char overlap between consecutive 1200-char chunks ensures no sentence is lost at a boundary.

    raw@{ shape: doc, label: "Raw document text\n(all pages joined)" }

    subgraph window["Sliding window — 1000-char stride"]
        c1@{ shape: das, label: "Chunk 0\nchars 0 – 1200" }
        c2@{ shape: das, label: "Chunk 1\nchars 1000 – 2200" }
        c3@{ shape: das, label: "Chunk 2\nchars 2000 – 3200" }
        overlap_a@{ shape: brace, label: "200-char overlap\nbetween chunk 0 and 1" }
        overlap_b@{ shape: brace, label: "200-char overlap\nbetween chunk 1 and 2" }
    end

    raw --> c1
    raw --> c2
    raw --> c3
    c1 -. "overlap" .-> overlap_a
    c2 -. "overlap" .-> overlap_a
    c2 -. "overlap" .-> overlap_b
    c3 -. "overlap" .-> overlap_b
```

---

## EmbeddingGemma Prompt Prefixes

EmbeddingGemma uses task-specific prompt prefixes to optimize embedding quality. The ingestion pipeline uses the **document** prefix; the query pipeline uses the **query** prefix. Using mismatched prefixes degrades retrieval quality.

| Context | Prefix | Used in |
|---|---|---|
| Document (ingestion) | `title: none \| text: {chunk}` | `ingest.py → _doc_prompt()` |
| Query (retrieval) | `task: question answering \| query: {question}` | `rag.py → embed_query()` |

---

## Ingestion Sequence

Detailed call flow from the Streamlit button click through Ollama and into ChromaDB.

```mermaid
sequenceDiagram
    accTitle: Ingestion pipeline call sequence
    accDescr: Shows progress callbacks to the UI, one Ollama embedding call per chunk, and the final ChromaDB batch write.
    autonumber

    actor U as User
    participant UI as app.py
    participant I as ingest.py
    participant O as Ollama :11434
    participant C as ChromaDB

    U->>UI: Click Re-ingest PDFs
    UI->>I: ingest(progress_cb)

    loop For each PDF in raw-files/
        I->>I: read_pdf_text() — PdfReader page loop
        I->>I: chunk_text() — sliding-window split
    end

    Note over I: All chunks accumulated in memory

    loop For each chunk
        I->>O: POST /api/embeddings {model: "embeddinggemma", prompt: "title: none | text: ..."}
        O-->>I: {embedding: [768 floats]}
        I->>UI: progress_cb(done, total)
        UI-->>U: Progress bar update
    end

    I->>C: delete_collection("proposals_gemma")
    I->>C: create_collection("proposals_gemma", cosine distance)

    loop Batch of up to 500 chunks
        I->>C: collection.add(ids, documents, metadatas, embeddings)
    end

    C-->>I: success
    I-->>UI: {pdfs: N, chunks: M, errors: [...]}
    UI-->>U: "Indexed N PDFs into M chunks"
```

---

## Key Constants

| Constant | Value | Reason |
|---|---|---|
| `CHUNK_SIZE` | 1200 chars | ≈ 300–400 tokens — safely under EmbeddingGemma's 2048-token input limit |
| `CHUNK_OVERLAP` | 200 chars | Preserves sentence context across chunk boundaries |
| `EMBED_MODEL` | `embeddinggemma` | 768-dim output; purpose-built for retrieval |
| `COLLECTION` | `proposals_gemma` | Separate from the old `proposal_docs` collection (different vector dim: 768 vs 384) |
| Batch size | 500 | Below ChromaDB's default per-call item limit |

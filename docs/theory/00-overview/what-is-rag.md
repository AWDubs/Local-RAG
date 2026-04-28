# What is RAG?

Retrieval-Augmented Generation (RAG) lets a language model answer questions about documents it was never trained on by **fetching relevant text at query time** and injecting it into the prompt. This gives you an up-to-date, private knowledge base without fine-tuning a model or sending your data to the cloud.

---

## LLM-Only vs RAG Pipeline

The diagram below contrasts a plain LLM call (left) against a full RAG pipeline (right).

```mermaid
flowchart LR
    accTitle: LLM-only versus RAG pipeline comparison
    accDescr: Left path shows a user question going straight to an LLM; right path shows question going through embedding, vector search, and context injection before the LLM.

    subgraph plain ["LLM-Only"]
        direction TB
        q1@{ shape: stadium, label: "User Question" }
        llm1@{ shape: rect, label: "LLM\n(training data only)" }
        a1@{ shape: stadium, label: "Answer\n(may hallucinate)" }
        q1 ==> llm1 ==> a1
    end

    subgraph rag ["RAG Pipeline"]
        direction TB
        q2@{ shape: stadium, label: "User Question" }
        embed@{ shape: subproc, label: "Embed Query\n(embeddinggemma)" }
        db@{ shape: cyl, label: "Vector DB\n(ChromaDB)" }
        chunks@{ shape: docs, label: "Retrieved\nChunks" }
        decide@{ shape: diam, label: "Enough\nContext?" }
        prompt@{ shape: hex, label: "Assemble\nPrompt" }
        llm2@{ shape: rect, label: "LLM\n(Gemma 4)" }
        a2@{ shape: stadium, label: "Grounded\nAnswer" }
        q2 ==> embed
        embed -- "nearest neighbours" --> db
        db -- "top-k docs" --> chunks
        chunks --> decide
        decide -- "yes" --> prompt
        decide -- "no, expand k" --> db
        prompt ==> llm2
        llm2 ==> a2
    end
```

---

## Why RAG Instead of Fine-Tuning?

| Concern | Fine-Tuning | RAG |
|---------|-------------|-----|
| **Data freshness** | Re-train for new docs | Re-index in minutes |
| **Cost** | GPU hours + expertise | Commodity hardware |
| **Privacy** | Model weights encode data | Data stays in your DB |
| **Explainability** | Black-box | Cited source chunks |
| **Scope** | Global knowledge update | Targeted retrieval |

RAG wins when you have **a defined document corpus that changes over time** and you need **verifiable, cited answers**.

---

## The Three Phases

```mermaid
flowchart TD
    accTitle: Three phases of a RAG system — ingest, index, query
    accDescr: Documents flow through ingestion and indexing during setup; queries flow through retrieval and generation at runtime.

    subgraph ingest ["1 — Ingest"]
        raw@{ shape: docs, label: "Raw Documents\n(PDF, MD, TXT)" }
        load@{ shape: subproc, label: "Load & Parse" }
        chunk@{ shape: docs, label: "Text Chunks" }
        raw ==> load ==> chunk
    end

    subgraph index_phase ["2 — Index"]
        emb@{ shape: subproc, label: "Embed Chunks\n(embeddinggemma)" }
        vec@{ shape: cyl, label: "Vector Store\n(ChromaDB)" }
        chunk ==> emb ==> vec
    end

    subgraph query_phase ["3 — Query"]
        uq@{ shape: stadium, label: "User Query" }
        qemb@{ shape: subproc, label: "Embed Query" }
        ret@{ shape: subproc, label: "Retrieve top-k" }
        gen@{ shape: rect, label: "Generate Answer\n(Gemma 4)" }
        ans@{ shape: stadium, label: "Answer + Sources" }
        uq ==> qemb
        qemb -- "similarity search" --> vec
        vec -- "top-k chunks" --> ret
        ret ==> gen ==> ans
    end
```

**Phase 1 — Ingest:** Load and parse raw documents into plain text.  
**Phase 2 — Index:** Split text into chunks, embed each chunk into a vector, and store in ChromaDB.  
**Phase 3 — Query:** Embed the user's question, search for the most similar chunks, build a prompt from those chunks, and generate an answer with Gemma 4.

---

## When NOT to Use RAG

- You need the model to **learn new skills** (reasoning patterns, coding style) → fine-tune instead.
- Your corpus is tiny (< 20 short documents) → just stuff the full text in the context window.
- Latency is critical (< 100 ms) → RAG adds retrieval overhead; consider keyword search or caching.

---

## Next Steps

- [Architecture diagram →](architecture.md) — the full system design  
- [Tokens & Embeddings →](../01-foundations/tokens-and-embeddings.md) — how text becomes vectors  
- [Ollama →](../02-ecosystem/ollama.md) — running models locally

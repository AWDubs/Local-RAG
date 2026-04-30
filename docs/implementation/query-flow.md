# Query and Answer Flow

Every question submitted to the app goes through a **Strands Agent loop**: the agent decides to call the `search_documents` tool, receives the retrieved context, then generates a grounded answer. Retrieval (embeddings + ChromaDB) runs on-device; generation calls out to the **Gemini API**.

Retrieval is **hybrid**: BM25 (keyword) and dense (EmbeddingGemma cosine) scores are min-max normalised and fused. **MMR** then picks the top 20 from the top-60 candidates to balance precision against diversity across source PDFs. The agent can also pass **metadata filters** (`customer`, `year`, `proposal_id`) which restrict the corpus *before* scoring.

---

## End-to-End Sequence

```mermaid
sequenceDiagram
    accTitle: Full query and answer sequence — hybrid retrieval edition
    accDescr: A user question is passed to the Strands agent, which calls the search_documents tool. The tool embeds the query locally via Ollama, then in rag.py runs hybrid BM25 + dense scoring, fuses the scores, and MMR re-ranks the top candidates to top_k=20 chunks. Gemini 2.5 Flash Lite (the default; configurable via GEMINI_MODEL) then generates a grounded answer via the hosted API.
    autonumber

    actor U as User
    participant UI as app.py
    participant SA as Strands Agent<br/>(agent.py)
    participant Tool as search_documents<br/>tool
    participant R as rag.py
    participant O as Ollama :11434
    participant G as Gemini API
    participant C as ChromaDB

    U->>UI: Type question and press Enter

    UI->>SA: agent(question)

    Note over SA,G: Agent loop — Turn 1: decide to use tool

    SA->>G: generateContent {model: "gemini-2.5-flash-lite", messages: [system + question]}
    G-->>SA: Tool call: search_documents(query="...", customer?, year?, proposal_id?)

    Note over SA,C: Tool execution — hybrid retrieval + MMR

    SA->>Tool: search_documents(query, filters)
    Tool->>R: retrieve(query, top_k=20, customer?, year?, proposal_id?)

    Note over R,C: First call per process loads & caches the corpus
    R->>C: collection.get(documents, metadatas, embeddings)
    C-->>R: full corpus (ids, docs, metas, embedding matrix)
    R->>R: Build BM25Okapi index over [doc tokens + title/proposal_id/customer/year tokens] (cached)

    R->>O: POST /api/embeddings {model: "embeddinggemma", prompt: "task: question answering | query: ..."}
    O-->>R: {embedding: [768 floats]}
    R->>R: dense = cosine(query_vec, all chunk embeddings)
    R->>R: sparse = bm25.get_scores(tokenised query)
    R->>R: (optional) mask out chunks failing customer/year/proposal_id filter
    R->>R: fused = 0.6 * minmax(dense) + 0.4 * minmax(sparse)
    R->>R: candidate_pool = top 60 by fused score
    R->>R: MMR re-rank → top 20 chunks (λ = 0.5) using fused scores
    R-->>Tool: 20 chunks (doc, source, title, proposal_id, customer, year, page_number, chunk_index, total_chunks, char_count, ingested_at, distance, dense/sparse/fused scores)
    Tool-->>SA: Formatted context string ([title · p.N (source: file)] headers) + updates _last_chunks

    Note over SA,G: Agent loop — Turn 2: generate grounded answer

    SA->>G: generateContent {model: "gemini-2.5-flash-lite", messages: [system + question + tool_result]}
    G-->>SA: Generated answer with citations

    SA-->>UI: AgentResult

    UI->>UI: str(response) → answer text
    UI-->>U: Display answer (st.markdown)
    UI-->>U: Sources expander (deduped doc titles) + Chunks expander (full per-chunk text) from _last_chunks
    UI-->>U: Thinking panel populated with tool/retrieval/generation events
```

---

## Agent Loop

The Strands agent orchestrates multi-turn reasoning. For a typical RAG question this is two turns: one to decide to retrieve, one to generate the answer.

```mermaid
flowchart TD
    accTitle: Strands agent loop for RAG
    accDescr: The agent receives the question, calls search_documents, receives context, then generates a final answer.

    start_n@{ shape: stadium, label: "agent(question)" }
    turn1@{ shape: subproc, label: "Turn 1 — Reasoning\nGemini sees system prompt + question\nDecides to call search_documents,\noptionally with customer/year/proposal_id filters" }
    tool_exec@{ shape: hex, label: "Execute search_documents(query, filters)\n→ embed query via EmbeddingGemma\n→ hybrid score (BM25 + dense) all chunks\n→ mask filtered-out chunks (if any)\n→ fuse → top 60 candidates\n→ MMR → top 20\n→ return formatted context" }
    turn2@{ shape: subproc, label: "Turn 2 — Generation\nGemini sees context + question\nGenerates grounded answer\nciting [title · p.N]" }
    done_n@{ shape: dbl-circ, label: "AgentResult\nstr(response) → answer text" }

    start_n ==> turn1
    turn1 --> tool_exec
    tool_exec --> turn2
    turn2 ==> done_n
```

---

## Retrieval Distance

The `distance` field surfaced in the UI is `1 - cosine_similarity` (lower = more similar) so it stays comparable to the old Chroma-only behaviour. The hybrid retriever also exposes the raw `dense_score`, `sparse_score`, and `fused_score` for diagnostics.

| Distance range | Interpretation |
|---|---|
| `0.00 – 0.15` | Very strong match — likely the exact answer |
| `0.15 – 0.35` | Good match — relevant context |
| `0.35 – 0.55` | Weak match — tangentially related |
| `> 0.55` | Poor match — may introduce noise |

Note that with hybrid + MMR enabled, a chunk can land in the top 20 because of a strong **BM25** keyword hit even if its dense distance is mediocre — inspect `sparse_score` and `fused_score` if the `distance` value alone looks surprising.

---

## Hybrid Retrieval + MMR

`rag.retrieve()` runs five steps per query, all in-process after the corpus is loaded once.

```mermaid
flowchart TD
    accTitle: Hybrid retrieval and MMR pipeline
    accDescr: Embed query, score all chunks with BM25 and cosine, min-max normalise and fuse, take top 60 candidates, then MMR re-rank to top 20.

    q@{ shape: stadium, label: "User query" }
    embed@{ shape: subproc, label: "Embed query\nEmbeddingGemma" }
    dense@{ shape: das, label: "Dense scores\ncosine(q_vec, all chunks)" }
    sparse@{ shape: das, label: "Sparse scores\nBM25Okapi.get_scores()" }
    fuse@{ shape: hex, label: "Fuse\nfused = 0.6*minmax(dense)\n+ 0.4*minmax(sparse)" }
    pool@{ shape: lean-r, label: "Top 60 candidates\nby fused score" }
    mmr@{ shape: subproc, label: "MMR re-rank (λ=0.5)\nuses fused scores as relevance,\ndiversity from selected" }
    out@{ shape: dbl-circ, label: "Top 20 chunks\nback to Strands tool" }

    q ==> embed
    embed --> dense
    q --> sparse
    dense --> fuse
    sparse --> fuse
    fuse --> pool
    pool --> mmr
    mmr ==> out
```

### Why hybrid?

- **Dense (EmbeddingGemma cosine)** is great at semantic paraphrase ("what does it cost?" ↔ "net price is…") but weak on rare tokens like proposal IDs (`P-118231-24C`), part numbers, and customer names.
- **BM25** is the opposite: it loves exact tokens but ignores meaning. Together they cover each other's blind spots.
- The BM25 index is built over **document tokens + filename-derived metadata tokens** (`title`, `proposal_id`, `customer`, `year`). This way a query like *"Hexcel proposal cost"* gets a strong sparse score on every chunk of the Hexcel PDF even if the word "Hexcel" never appears in the body text.
- Fusion happens on min-max normalised scores so neither retriever dominates by virtue of using a different scale.

### Why MMR?

The top-N nearest neighbours of any business-style query tend to collapse onto cover letters and boilerplate from one or two PDFs (high cosine similarity to almost any prompt). **Maximal Marginal Relevance** greedily picks the most relevant candidate that is *least* similar to what's already been picked, spreading the final 20 chunks across multiple source documents.

---

## Metadata Filters

`search_documents` accepts three optional arguments that the agent can populate from the user's question:

| Argument | Type | Match style | Example query |
|---|---|---|---|
| `customer` | string | case-insensitive substring | *"What did we quote Hexcel?"* → `customer="Hexcel"` |
| `year` | string | exact 4-digit year | *"Show me 2025 proposals about ovens"* → `year="2025"` |
| `proposal_id` | string | exact match | *"Summarise P-118231-24C"* → `proposal_id="P-118231-24C"` |

Filters are applied in `rag.retrieve()` *before* score fusion: non-matching chunks have their dense score forced to `-inf` and are dropped from the candidate pool. If every chunk is filtered out the tool returns the no-results sentinel and the system prompt instructs the model to say so.

---

## Corpus Cache

Hybrid retrieval needs the full corpus (documents + metadata + embeddings + BM25 index) in memory. `rag._load_corpus()` is wrapped in `functools.lru_cache(maxsize=1)` so it loads once per process. `ingest.ingest()` calls `rag.invalidate_cache()` on success so the next query reloads the freshly written collection.

---

## Chat Session Lifecycle

The Strands agent is stored in `st.session_state["strands_agent"]` so its internal conversation history persists across Streamlit reruns. Clearing chat history destroys the agent and creates a fresh one.

```mermaid
stateDiagram-v2
    accTitle: Chat session lifecycle — Strands edition
    accDescr: Session moves from Empty through Retrieving/Generating (both inside the agent loop) to Answered.

    [*] --> Empty: App starts or history cleared\n(agent recreated)

    Empty --> AgentRunning: User submits question
    note right of AgentRunning
        Strands agent loop:
        1. Reasoning turn
        2. search_documents tool call
        3. Generation turn
    end note

    AgentRunning --> Answered: AgentResult returned
    note right of Answered
        Both turns appended
        to session_state["messages"]
        _last_chunks updated
    end note

    Answered --> AgentRunning: User submits next question
    Answered --> Empty: Clear chat history clicked
    Answered --> [*]: Browser tab closed
```

---

## Generation Parameters

The Strands `GeminiModel` is configured per session via `create_agent(temperature=...)`. Temperature is exposed in the Streamlit sidebar as a live slider (default `0.20`); changing it rebuilds the agent.

| Parameter | Default | Notes |
|---|---|---|
| `temperature` | `0.20` | Sidebar slider (`0.00–1.50`); rebuilds the agent on change and resets model conversation memory |
| Model | `gemini-2.5-flash-lite` (default) — `gemini-2.5-flash` is higher-quality; `gemini-2.5-pro` is the highest-quality option | Set `GEMINI_MODEL` in `.env` (overrides the fallback in `agent.py`) |
| API key | from `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) env var | Loaded via `python-dotenv` from `.env` if present |
| Tool `top_k` chunks | `20` | Hardcoded in `search_documents` tool body — increased from 4 now that hybrid + MMR keeps the pool diverse |
| `HYBRID_ALPHA` | `0.6` | rag.py — weight on dense vs. BM25 in fusion (1.0 = pure dense, 0.0 = pure BM25) |
| `CANDIDATE_POOL` | `60` | rag.py — number of chunks fed into MMR |
| `MMR_LAMBDA` | `0.5` | rag.py — 1.0 = pure relevance, 0.0 = pure diversity |
| Citation format | `[title · p.PAGE]` | Enforced by `SYSTEM_PROMPT` in agent.py |

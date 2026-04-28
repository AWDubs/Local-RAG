# Query and Answer Flow

Every question submitted to the app goes through a **Strands Agent loop**: the agent decides to call the `search_documents` tool, receives the retrieved context, then generates a grounded answer — all on-device via Ollama and ChromaDB.

---

## End-to-End Sequence

```mermaid
sequenceDiagram
    accTitle: Full query and answer sequence — Strands Agents edition
    accDescr: A user question is passed to the Strands agent, which calls the search_documents tool to embed the query and retrieve chunks from ChromaDB, then generates a grounded answer via Gemma 4 e2b — all on localhost.
    autonumber

    actor U as User
    participant UI as app.py
    participant SA as Strands Agent<br/>(agent.py)
    participant Tool as search_documents<br/>tool
    participant R as rag.py
    participant O as Ollama :11434
    participant C as ChromaDB

    U->>UI: Type question and press Enter

    UI->>SA: agent(question)

    Note over SA,O: Agent loop — Turn 1: decide to use tool

    SA->>O: POST /api/chat {model: "gemma4:e2b", messages: [system + question]}
    O-->>SA: Tool call: search_documents(query="...")

    Note over SA,C: Tool execution — retrieve relevant chunks

    SA->>Tool: search_documents(query)
    Tool->>R: retrieve(query, top_k=4)
    R->>O: POST /api/embeddings {model: "embeddinggemma", prompt: "task: question answering | query: ..."}
    O-->>R: {embedding: [768 floats]}
    R->>C: collection.query(query_embeddings, n_results=4, cosine distance)
    C-->>R: [{doc, source, chunk_index, distance}, ...]
    R-->>Tool: top-4 chunks
    Tool-->>SA: Formatted context string + updates _last_chunks

    Note over SA,O: Agent loop — Turn 2: generate grounded answer

    SA->>O: POST /api/chat {model: "gemma4:e2b", messages: [system + question + tool_result]}
    O-->>SA: Generated answer with citations

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
    turn1@{ shape: subproc, label: "Turn 1 — Reasoning\nGemma 4 sees system prompt + question\nDecides to call search_documents" }
    tool_exec@{ shape: hex, label: "Execute search_documents(query)\n→ embed query via EmbeddingGemma\n→ query ChromaDB top-4\n→ return formatted context" }
    turn2@{ shape: subproc, label: "Turn 2 — Generation\nGemma 4 sees context + question\nGenerates grounded answer with citations" }
    done_n@{ shape: dbl-circ, label: "AgentResult\nstr(response) → answer text" }

    start_n ==> turn1
    turn1 --> tool_exec
    tool_exec --> turn2
    turn2 ==> done_n
```

---

## Retrieval Distance

ChromaDB uses cosine distance. Lower values mean higher similarity.

| Distance range | Interpretation |
|---|---|
| `0.00 – 0.15` | Very strong match — likely the exact answer |
| `0.15 – 0.35` | Good match — relevant context |
| `0.35 – 0.55` | Weak match — tangentially related |
| `> 0.55` | Poor match — may introduce noise |

The Sources expander shows the distance for each retrieved chunk so you can judge relevance at a glance.

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

The Strands `OllamaModel` is configured per session via `create_agent(temperature=...)`. Temperature is exposed in the Streamlit sidebar as a live slider (default `0.20`); changing it rebuilds the agent.

| Parameter | Default | Notes |
|---|---|---|
| `temperature` | `0.20` | Sidebar slider (`0.00–1.50`); rebuilds the agent on change and resets model conversation memory |
| `top_p` | Ollama default | Configurable via `OllamaModel` kwargs |
| Model | `gemma4:e2b` | Change `GEN_MODEL` in `agent.py` |
| Tool `top_k` chunks | `4` | Hardcoded in `search_documents` tool body |

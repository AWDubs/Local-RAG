"""
Strands Agents integration for the local RAG app.

Provides:
  - search_documents  @tool — retrieves relevant chunks from ChromaDB
  - create_agent()    — factory that returns a configured Strands Agent
  - _last_chunks      — module-level list; populated after every tool call
                        so app.py can render source citations

Strands is a small "agent loop" framework: you give it a model, a system
prompt, and a list of tools. The model decides when to call tools, Strands
executes them, feeds the results back, and loops until the model emits a
final answer.
"""

# `Agent` is the orchestrator. `tool` is a decorator that exposes a Python
# function to the LLM as a callable tool (with auto-generated JSON schema
# derived from the type hints + docstring).
from strands import Agent, tool
# OllamaModel is a Strands adapter that speaks to a local Ollama server,
# letting the agent reason with a self-hosted LLM.
from strands.models.ollama import OllamaModel

# Our retrieval module — provides `retrieve(question, top_k)`.
import rag

# Shared state — populated by the tool on every call so the UI can display sources.
# Safe for single-user local use; do not share across threads.
# This is a deliberate convenience: the agent loop is hidden from the UI, so
# we expose its most recent retrieval results via this module-level variable.
_last_chunks: list[dict] = []

# Optional event logger. The UI swaps this out per-turn to receive structured
# events (tool calls, retrieval results) for the live "thinking" panel. Default
# is a no-op so non-UI callers (tests, scripts) work unchanged.
_log_event = lambda kind, payload: None  # noqa: E731


def set_event_logger(fn) -> None:
    """Install a logger callable invoked as ``fn(kind: str, payload: dict)``.

    Pass ``None`` (or a no-op) to disable. The UI calls this at the start of
    every turn so events flow into the side panel for that turn only.
    """
    global _log_event
    _log_event = fn if fn is not None else (lambda kind, payload: None)


# The local generation model used to *answer* questions. Must already be
# pulled in Ollama (`ollama pull gemma4:e2b`).
GEN_MODEL = "gemma4:e2b"
# Default Ollama server URL. The Ollama daemon listens on this port by default.
OLLAMA_HOST = "http://localhost:11434"

# The system prompt is the highest-priority instruction the model sees on
# every turn. It hard-codes the RAG discipline: always retrieve first, never
# answer from parametric memory, always cite.
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about engineering proposals. "
    "Always call the search_documents tool first to retrieve relevant context before answering. "
    "Answer ONLY using the information returned by the tool. "
    "If the tool returns no relevant content, say 'I could not find that in the provided documents.' "
    "Cite your sources using the format [source_filename.pdf #chunk_index] "
    "after each factual statement."
)


# `@tool` registers this function with Strands. The function's name, type
# hints, and docstring are automatically converted into the JSON schema that
# the LLM sees when deciding whether to call it.
@tool
def search_documents(query: str) -> str:
    """Search the indexed proposal documents for relevant information.

    Use this tool whenever a question is asked about the engineering proposals.
    Retrieves the most semantically similar passages from the local vector store.

    Args:
        query: The search query used to find relevant document passages.

    Returns:
        Formatted document passages with source file names and chunk indices,
        ready to use as context for answering the question.
    """
    # `global` lets us *reassign* the module-level name from inside the function.
    # Without this we'd just create a local variable shadowing the outer one.
    global _last_chunks
    # Notify the UI that a tool call has started, with the model-supplied query.
    _log_event("tool_call", {"name": "search_documents", "query": query})
    # Delegate the actual vector search to rag.py. `top_k=4` is a small,
    # focused context window — large enough to find the answer, small enough
    # to keep generation fast and on-topic.
    chunks = rag.retrieve(query, top_k=4)
    # Stash the chunks so app.py can render them in the "Sources" expander.
    _last_chunks = chunks
    # Stream the retrieval result to the thinking panel: which docs were hit,
    # how many chunks, and their distances. The UI decides how to display it.
    _log_event(
        "tool_result",
        {
            "name": "search_documents",
            "chunk_count": len(chunks),
            "chunks": chunks,
        },
    )

    # Defensive: if retrieval returned nothing, tell the model so it can apply
    # the "I could not find that" fallback baked into the system prompt.
    if not chunks:
        return "No relevant documents found in the index."

    # Build a single newline-joined string for the LLM. Numbering ([1], [2]…)
    # makes it easier for the model to refer to specific passages in its answer.
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        # Each block: a header line with citation info, then the chunk text.
        parts.append(
            f"[{i}] Source: {chunk['source']} | Chunk #{chunk['chunk_index']}\n{chunk['doc']}"
        )
    # Two newlines between blocks gives the model a clean visual separator.
    return "\n\n".join(parts)


def create_agent(temperature: float = 0.2, callback_handler=None) -> Agent:
    """Instantiate a fresh Strands Agent backed by a local Ollama model.

    Args:
        temperature: Sampling temperature for generation. 0.0 = deterministic,
            higher values = more creative/random. Defaults to 0.2 for grounded
            RAG answers.
        callback_handler: Optional Strands callback handler. If provided, the
            agent will stream events (tokens, tool calls) to it. Used by the
            UI to power the live thinking panel.

    Returns a new Agent each time — call once per session and store the
    returned object to preserve conversation history across turns.
    """
    # Construct the model adapter. Strands handles the chat protocol details;
    # we just hand it a host URL, a model id, and any sampling params.
    model = OllamaModel(
        host=OLLAMA_HOST,
        model_id=GEN_MODEL,
        temperature=temperature,
    )
    # Assemble the agent. `tools=[...]` is the list of callables the LLM may
    # invoke; Strands will inject their JSON schemas into the prompt so the
    # model knows what's available. `callback_handler` receives streaming
    # events; passing None lets Strands fall back to its own default.
    agent_kwargs = dict(
        model=model,
        tools=[search_documents],
        system_prompt=SYSTEM_PROMPT,
    )
    if callback_handler is not None:
        agent_kwargs["callback_handler"] = callback_handler
    return Agent(**agent_kwargs)

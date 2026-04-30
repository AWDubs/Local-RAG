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
# Other supported tags: `gemma4:e4b` (~5 GB VRAM, modest quality lift),
# `gemma4:26b`, `gemma4:31b` (best quality, ~24 GB VRAM).
# See docs/theory/07-next-steps/02-model-upgrades-within-gemma-4.md.
GEN_MODEL = "gemma4:e2b"
# Default Ollama server URL. The Ollama daemon listens on this port by default.
OLLAMA_HOST = "http://localhost:11434"

# Number of chunks the search_documents tool returns to the LLM each turn.
# Exposed as a sidebar slider in app.py so users can trade prompt size
# (and therefore latency on CPU-only machines) against retrieval breadth.
TOP_K = 20

# The system prompt is the highest-priority instruction the model sees on
# every turn. It hard-codes the RAG discipline: always retrieve first, never
# answer from parametric memory, always cite.
SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions about engineering proposals. "
    "Always call the search_documents tool first to retrieve relevant context before answering. "
    "When the user mentions a specific customer, year, or proposal id (e.g. 'P-118231-24C'), "
    "pass it to the tool as the `customer`, `year`, or `proposal_id` argument so results are filtered. "
    "Answer ONLY using the information returned by the tool. "
    "If the tool returns no relevant content, say 'I could not find that in the provided documents.' "
    "Cite your sources inline using the format [title \u00b7 p.PAGE] (or [source.pdf \u00b7 p.PAGE] "
    "if no title is available) after each factual statement."
)


# `@tool` registers this function with Strands. The function's name, type
# hints, and docstring are automatically converted into the JSON schema that
# the LLM sees when deciding whether to call it.
@tool
def search_documents(
    query: str,
    customer: str | None = None,
    year: str | None = None,
    proposal_id: str | None = None,
) -> str:
    """Search the indexed proposal documents for relevant information.

    Use this tool whenever a question is asked about the engineering proposals.
    Retrieves the most semantically similar passages from the local vector store,
    optionally restricted to a specific customer, year, or proposal id.

    Args:
        query: The search query used to find relevant document passages.
        customer: Optional case-insensitive substring filter on customer name
            (e.g. "Hexcel", "General Dynamics").
        year: Optional 4-digit year filter (e.g. "2025").
        proposal_id: Optional exact proposal id filter (e.g. "P-118231-24C").

    Returns:
        Formatted document passages with citation info (title, source file,
        page number, chunk index), ready to use as context for answering.
    """
    # `global` lets us *reassign* the module-level name from inside the function.
    # Without this we'd just create a local variable shadowing the outer one.
    global _last_chunks
    # Notify the UI that a tool call has started, with the model-supplied query.
    _log_event(
        "tool_call",
        {
            "name": "search_documents",
            "query": query,
            "customer": customer,
            "year": year,
            "proposal_id": proposal_id,
        },
    )
    # Delegate the actual vector search to rag.py. `TOP_K` is a module-level
    # value the UI can override per-session via the sidebar slider; rag.retrieve()
    # is hybrid (BM25 + dense) with MMR re-ranking so larger pools stay diverse
    # instead of collapsing onto one PDF.
    chunks = rag.retrieve(
        query,
        top_k=TOP_K,
        customer=customer,
        year=year,
        proposal_id=proposal_id,
    )
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
        # Title + page in the citation header gives the model everything it
        # needs to follow the cite-as-you-write rule in SYSTEM_PROMPT.
        title = chunk.get("title") or chunk["source"]
        page = chunk.get("page_number")
        page_str = f"p.{page}" if page else f"chunk #{chunk['chunk_index']}"
        parts.append(
            f"[{i}] {title} \u00b7 {page_str} (source: {chunk['source']})\n{chunk['doc']}"
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
    # `options={"num_ctx": ...}` is forwarded to Ollama. Without it, Ollama
    # defaults to a 2048-token context window, which silently truncates the
    # tool result (≈4K tokens for top_k=20) and the model then produces a
    # generic "ask me a question" non-answer because it never sees the chunks.
    model = OllamaModel(
        host=OLLAMA_HOST,
        model_id=GEN_MODEL,
        temperature=temperature,
        options={"num_ctx": 8192},
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

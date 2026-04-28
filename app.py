"""
Streamlit UI for the local RAG app — Strands Agents edition.
Run from Local-RAG/ with:  uv run streamlit run app.py

Streamlit re-runs this entire script top-to-bottom on every user interaction
(button click, chat input, etc.). Anything that should survive a re-run must
be stored in `st.session_state`. Anything declared at module scope is rebuilt
on every run.
"""

# Standard library — used for live timestamps in the thinking panel.
from datetime import datetime

# Streamlit is the web UI framework. Importing it once gives us the full
# component library plus the `st.session_state` persistence dict.
import streamlit as st

# Our three project modules. Importing them at module scope is fine — Python
# caches imports, so the heavy work only happens on the first run per process.
import ingest
import rag
# We alias `agent` to `rag_agent` to avoid shadowing any future variable named
# `agent` and to make call sites self-documenting.
import agent as rag_agent

# ── Page config ──────────────────────────────────────────────────────────────
# `set_page_config` MUST be the first Streamlit call on the page. It controls
# the browser tab title/icon and the overall layout (`wide` uses full width).
st.set_page_config(
    page_title="Local RAG",
    page_icon="📄",
    layout="wide",
)


# ── Session state defaults ────────────────────────────────────────────────────
# `st.session_state` is a dict-like object scoped to the current browser tab.
# We seed default values only if they don't already exist so reruns don't wipe them.
if "messages" not in st.session_state:
    # Chat transcript for rendering. Each entry is {"role": "user"|"assistant", "content": str}.
    st.session_state["messages"] = []   # list of {role, content} dicts for display
if "temperature" not in st.session_state:
    # Default sampling temperature. Low values keep RAG answers grounded.
    st.session_state["temperature"] = 0.2
if "thinking_log" not in st.session_state:
    # List of event dicts shown in the right-hand "Thinking" panel. Each entry
    # is {"ts": "HH:MM:SS", "kind": str, "text": str}. Survives reruns so the
    # last turn's trace stays visible until the next question is asked.
    st.session_state["thinking_log"] = []
if "strands_agent" not in st.session_state:
    # The Strands Agent persists across reruns to maintain conversation history.
    # Creating it once means follow-up questions can reference earlier turns.
    st.session_state["strands_agent"] = rag_agent.create_agent(
        temperature=st.session_state["temperature"]
    )
# Track which temperature the live agent was built with so we can rebuild on change.
if "agent_temperature" not in st.session_state:
    st.session_state["agent_temperature"] = st.session_state["temperature"]


# ── Helper: get chunk count safely ──────────────────────────────────────────
def get_chunk_count() -> int:
    # We wrap this in try/except because `get_collection()` raises if the
    # collection doesn't exist yet (i.e. ingest has never been run).
    try:
        # `.count()` is a cheap metadata lookup, not a full scan.
        return rag.get_collection().count()
    except Exception:
        # On any failure (no collection, locked DB, etc.) report zero — the
        # UI will then prompt the user to run ingest.
        return 0


# ── Thinking-panel helpers ──────────────────────────────────────────────────
# Strands runs the agent loop (and its callbacks) on a worker thread, where
# `st.session_state` is unavailable. We therefore collect events into a plain
# list during the run and flush it into session_state after the agent returns.
def _now() -> str:
    # Short clock string for log entries. Seconds resolution is enough.
    return datetime.now().strftime("%H:%M:%S")


def _make_event_sink() -> list:
    """Return a thread-safe list used to collect events for one agent turn."""
    # A bare list is safe enough here: appends from a single Strands worker
    # thread are atomic in CPython and the main thread only reads it AFTER
    # the agent call has returned.
    return []


def _push_to_sink(sink: list, kind: str, text: str) -> None:
    """Append a timestamped event to the per-turn sink."""
    sink.append({"ts": _now(), "kind": kind, "text": text})


def _render_thinking_panel(placeholder, log: list) -> None:
    """Render the supplied log into the supplied placeholder."""
    # Build a single Markdown block: cheaper than many widget calls and keeps
    # the panel scroll position stable.
    icons = {
        "user": "🗨️",
        "tool_call": "🔧",
        "tool_result": "📦",
        "token": "✏️",
        "info": "ℹ️",
        "done": "✅",
        "error": "❌",
    }
    if not log:
        placeholder.markdown("_Idle — ask a question to see the agent think._")
        return
    lines = []
    for entry in log:
        icon = icons.get(entry["kind"], "•")
        lines.append(f"`{entry['ts']}` {icon} {entry['text']}  ")
    placeholder.markdown("\n".join(lines))


def _make_callback_handler(sink: list):
    """Build a Strands callback handler that funnels events into ``sink``.

    We coalesce streamed text tokens into a single rolling line per assistant
    turn so the panel doesn't get spammed with one entry per character.
    The handler runs on a Strands worker thread, so it must NOT touch
    ``st.session_state`` — append-only writes to ``sink`` are safe.
    """
    state = {"buffer": ""}

    def handler(**kwargs) -> None:
        # Tool-call detection: Strands emits a contentBlockStart event whose
        # `start.toolUse` payload names the tool the model wants to invoke.
        tool_use = (
            kwargs.get("event", {})
            .get("contentBlockStart", {})
            .get("start", {})
            .get("toolUse")
        )
        if tool_use:
            _push_to_sink(
                sink, "tool_call", f"Model invoked **{tool_use.get('name', '?')}**"
            )
            return

        # Streaming text tokens — accumulate, then flush on `complete`.
        data = kwargs.get("data", "")
        complete = kwargs.get("complete", False)
        if data:
            state["buffer"] += data
        if complete and state["buffer"]:
            preview = state["buffer"].strip().replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "…"
            _push_to_sink(sink, "token", f"Generated: _{preview}_")
            state["buffer"] = ""

    return handler


# ── Sidebar ───────────────────────────────────────────────────────────────────
# `with st.sidebar:` routes every nested component into the left-hand panel.
with st.sidebar:
    st.title("⚙️ Settings")

    # Model status
    # Re-read the count on every rerun so the sidebar always shows fresh state.
    chunk_count = get_chunk_count()
    # Markdown supports backticks for inline code and `  \n` for line breaks.
    st.markdown(
        f"**Embed model:** `{ingest.EMBED_MODEL}`  \n"
        f"**Gen model:** `{rag_agent.GEN_MODEL}`  \n"
        f"**Agent:** Strands  \n"
        f"**Indexed chunks:** `{chunk_count}`"
    )
    # Visual divider between sidebar sections.
    st.divider()

    # Generation tuning
    st.subheader("Generation")
    # The slider's value is auto-stored under `key` in session_state. We read
    # it from there below to decide whether the agent needs to be rebuilt.
    st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        step=0.05,
        key="temperature",
        help=(
            "Lower = more deterministic, factual answers (recommended for RAG). "
            "Higher = more creative, varied phrasing."
        ),
    )
    # If the user dragged the slider, rebuild the agent so the new temperature
    # actually takes effect on the next question. This also resets conversation
    # memory — call it out so the user isn't surprised.
    if st.session_state["temperature"] != st.session_state["agent_temperature"]:
        st.session_state["strands_agent"] = rag_agent.create_agent(
            temperature=st.session_state["temperature"]
        )
        st.session_state["agent_temperature"] = st.session_state["temperature"]
        st.caption(
            f"🔁 Agent rebuilt at temperature `{st.session_state['temperature']:.2f}` "
            "(chat history preserved in UI; model context reset)."
        )

    st.divider()

    # Re-ingest button
    st.subheader("Index")
    # `st.button` returns True only on the rerun triggered by the click.
    # `use_container_width=True` makes the button fill the sidebar width.
    if st.button("🔄 Re-ingest PDFs", use_container_width=True):
        # `st.status` is a collapsible container that shows a spinner + state.
        status_box = st.status("Ingesting PDFs…", expanded=True)
        # `progress` returns a handle we can update with new percentages.
        progress_bar = status_box.progress(0.0)
        # `empty()` reserves a slot we can overwrite repeatedly with new text.
        progress_text = status_box.empty()

        def _progress(done: int, total: int):
            # Fraction in [0.0, 1.0] for the progress bar widget.
            pct = done / total
            progress_bar.progress(pct)
            # Replace the previous status line in-place (no scroll spam).
            progress_text.text(f"Embedding chunk {done}/{total}")

        try:
            # Kick off the pipeline; `_progress` is called after each chunk embed.
            result = ingest.ingest(progress_cb=_progress)
            # Mark the status box complete and collapse it.
            status_box.update(label="✅ Ingestion complete!", state="complete", expanded=False)
            # Toast-style success summary above the chat area.
            st.success(
                f"Indexed **{result['pdfs']}** PDFs into **{result['chunks']}** chunks."
            )
            # Surface non-fatal issues (e.g. unreadable PDFs) as warnings.
            if result["errors"]:
                for err in result["errors"]:
                    st.warning(err)
        except FileNotFoundError as exc:
            # The raw-files folder is missing — actionable error for the user.
            status_box.update(label="❌ Error", state="error", expanded=True)
            st.error(str(exc))
        except Exception as exc:
            # Catch-all so unexpected failures don't crash the whole UI.
            status_box.update(label="❌ Error", state="error", expanded=True)
            st.error(f"Ingestion failed: {exc}")
        # Force a rerun so the chunk-count display reflects the new index.
        st.rerun()

    st.divider()

    if st.button("🗑️ Clear chat history", use_container_width=True):
        # Reset the visible transcript.
        st.session_state["messages"] = []
        # Reset the agent so its internal conversation history is also cleared.
        # Without this, the model would still "remember" earlier turns even
        # though the UI looks empty.
        st.session_state["strands_agent"] = rag_agent.create_agent(
            temperature=st.session_state["temperature"]
        )
        # Wipe the thinking log too — it's tied to the cleared conversation.
        st.session_state["thinking_log"] = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("📄 Local RAG")
st.caption(
    "Ask questions about your proposal PDFs. "
    "Everything runs locally via Ollama and Strands Agents."
)

# Two-column layout: chat on the left, live thinking panel on the right.
# Streamlit lays columns out side-by-side; the [2, 1] ratio gives the chat
# roughly two-thirds of the width.
chat_col, think_col = st.columns([2, 1], gap="large")

# Render the thinking panel first so the placeholder exists before the agent
# starts streaming events into it during this rerun.
with think_col:
    st.subheader("🧠 Thinking")
    st.caption("Trace of tool calls, retrieval, and generation for the last turn.")
    # `st.empty()` is the canonical "I'll fill this in later" widget — we hand
    # the handle to `_render_thinking_panel` once now (showing prior turn) and
    # again after the agent returns (showing the new turn).
    thinking_placeholder = st.empty()
    _render_thinking_panel(thinking_placeholder, st.session_state["thinking_log"])

with chat_col:
    # Render existing chat history
    # On each rerun we replay the transcript so the chat appears continuous.
    for msg in st.session_state["messages"]:
        # `st.chat_message` produces a styled bubble keyed by role.
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# Chat input
# `st.chat_input` is sticky to the bottom of the page and returns the submitted
# string on the rerun triggered by Enter (None otherwise). It must live at the
# top level (outside columns) — Streamlit only allows one global chat input.
question = st.chat_input("Ask a question about the proposals…")

if question:
    # Guard: make sure there's something indexed
    # Without this, the agent would call the tool, get nothing back, and
    # respond with the "could not find" fallback — confusing for new users.
    if get_chunk_count() == 0:
        st.warning("No documents indexed yet. Click **Re-ingest PDFs** in the sidebar first.")
        # `st.stop()` halts the rest of the script for this rerun only.
        st.stop()

    # New turn → fresh per-turn sink. The agent loop runs on a worker thread,
    # so we keep events in a plain list and copy them into session_state once
    # the call returns (touching session_state from a worker thread crashes).
    event_sink: list = _make_event_sink()
    _push_to_sink(event_sink, "user", f"Question: _{question}_")
    _push_to_sink(
        event_sink, "info", f"Temperature: `{st.session_state['temperature']:.2f}`"
    )

    # Wire the agent's tool to push retrieval events into the same sink.
    rag_agent.set_event_logger(
        lambda kind, payload: _push_to_sink(
            event_sink,
            kind,
            (
                f"Searching index for: _{payload['query']}_"
                if kind == "tool_call"
                else (
                    f"Retrieved **{payload['chunk_count']}** chunks "
                    f"from {len({c['source'] for c in payload['chunks']})} document(s)"
                )
            ),
        )
    )

    # Show the user's question
    # Echo the question immediately so the UI feels responsive while the
    # model thinks (we append to history *after* the response arrives).
    with chat_col:
        with st.chat_message("user"):
            st.markdown(question)

    # Run the Strands agent — it will call search_documents internally,
    # then synthesise an answer grounded in the retrieved chunks.
    agent = st.session_state["strands_agent"]
    # Install the streaming callback for THIS turn so token/tool events flow
    # into the per-turn sink. Reassigning is supported by Strands.
    agent.callback_handler = _make_callback_handler(event_sink)

    with chat_col:
        with st.chat_message("assistant"):
            # Spinner gives the user feedback during the multi-second LLM call.
            with st.spinner("Thinking…"):
                try:
                    # Calling the agent like a function runs one full loop:
                    # decide → (maybe) call tool → read tool output → answer.
                    response = agent(question)
                    # Strands returns a rich response object; `str()` extracts the
                    # final assistant text suitable for display.
                    answer = str(response)
                except Exception as exc:
                    # Surface model/tool errors instead of silently failing.
                    _push_to_sink(event_sink, "error", f"Agent error: {exc}")
                    st.session_state["thinking_log"] = event_sink
                    _render_thinking_panel(thinking_placeholder, event_sink)
                    st.error(f"Agent error: {exc}")
                    st.stop()
            # Render the final answer as Markdown (supports lists, code, etc.).
            st.markdown(answer)

    _push_to_sink(event_sink, "done", "Answer complete.")
    # Detach the per-turn logger so non-UI callers don't accidentally inherit it.
    rag_agent.set_event_logger(None)
    # Persist this turn's trace and repaint the side panel with the full log.
    st.session_state["thinking_log"] = event_sink
    _render_thinking_panel(thinking_placeholder, event_sink)

    # Persist both turns for display
    # We add user + assistant together so the next rerun replays them in order.
    st.session_state["messages"].append({"role": "user", "content": question})
    st.session_state["messages"].append({"role": "assistant", "content": answer})

    # Sources (deduped document titles only) and full chunks live below the chat.
    chunks = rag_agent._last_chunks
    if chunks:
        with chat_col:
            # Deduplicate by source filename, preserving first-seen order so
            # the most-relevant document (lowest distance) appears first.
            seen: set[str] = set()
            unique_sources: list[str] = []
            for c in chunks:
                src = c["source"]
                if src not in seen:
                    seen.add(src)
                    unique_sources.append(src)

            with st.expander(f"📚 Sources ({len(unique_sources)})", expanded=True):
                # One bullet per document — concise citation surface for the user.
                for src in unique_sources:
                    st.markdown(f"- **{src}**")

            # Full per-chunk detail moves into its own expander, collapsed by
            # default so the cleaner Sources view stays the headline view.
            with st.expander(f"🧩 Chunks ({len(chunks)})", expanded=False):
                # `start=1` makes the displayed numbering 1-based (human-friendly).
                for i, chunk in enumerate(chunks, start=1):
                    st.markdown(
                        f"**[{i}] {chunk['source']}** — chunk #{chunk['chunk_index']}  \n"
                        f"Distance: `{chunk['distance']:.4f}`"
                    )
                    # Show the first 500 chars; truncation marker if longer.
                    st.text(chunk["doc"][:500] + ("…" if len(chunk["doc"]) > 500 else ""))
                    # Skip the divider after the very last item for a cleaner look.
                    if i < len(chunks):
                        st.divider()


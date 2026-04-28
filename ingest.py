"""
Ingestion pipeline: reads PDFs from raw-files/, chunks text,
embeds with EmbeddingGemma via Ollama, and stores in ChromaDB.
Run directly as: python ingest.py

This file is heavily commented as a learning resource. The pipeline below is
the classic "R" (Retrieval) preparation step in Retrieval-Augmented Generation:
    PDFs  ->  raw text  ->  overlapping chunks  ->  vectors  ->  vector DB
"""

# `os` is imported in case future code needs environment variables or path ops.
import os
# `pathlib.Path` gives us OS-agnostic, object-oriented filesystem paths.
from pathlib import Path
# `ssl` is the standard-library TLS module; we tweak its defaults below.
import ssl

# `ollama` is the official Python client for the local Ollama server.
# Ollama runs LLMs and embedding models on your own machine over HTTP (port 11434).
import ollama
# `chromadb` is an embedded, file-backed vector database used to store/search embeddings.
import chromadb
# `pypdf` is a pure-Python PDF parser; `PdfReader` opens a PDF for text extraction.
from pypdf import PdfReader

# Some corporate networks intercept HTTPS with self-signed certs which break
# Python's default cert validation. This line disables verification globally.
# WARNING: this is convenient for a local learning project but is insecure in
# production — it exposes you to man-in-the-middle attacks.
ssl._create_default_https_context = ssl._create_unverified_context

# --- Constants ---
# `__file__` is the path of *this* module; `.parent` strips the filename so
# RAW_DIR resolves to <repo>/raw-files regardless of where the script is run from.
RAW_DIR = Path(__file__).parent / "raw-files"
# Where ChromaDB will persist its SQLite + HNSW index files on disk.
DB_DIR = Path(__file__).parent / "chroma_db"
# A "collection" in Chroma is roughly equivalent to a table in SQL: a named
# bucket of vectors + documents + metadata.
COLLECTION = "proposals_gemma"
# Maximum characters per chunk. ~1200 chars ≈ ~300 tokens — small enough to
# keep embeddings focused, large enough to retain useful context.
CHUNK_SIZE = 1200
# Adjacent chunks share this many characters so a sentence split across the
# boundary still appears whole in at least one chunk.
CHUNK_OVERLAP = 200
# The embedding model name as registered in Ollama (`ollama pull embeddinggemma`).
# EmbeddingGemma produces 768-dimensional vectors.
EMBED_MODEL = "embeddinggemma"


# --- PDF reading ---

def read_pdf_text(pdf_path: str) -> str:
    """Extract full text from a PDF, one page at a time, with page markers."""
    # Accumulate per-page text fragments here; we join at the end (faster than
    # repeated string concatenation in a loop).
    text_parts = []
    try:
        # Open and parse the PDF. `PdfReader` is lazy — it doesn't read pages
        # until you iterate them.
        reader = PdfReader(pdf_path)
        # `enumerate` gives us (index, value) pairs so we can build human-friendly
        # 1-based page numbers in the marker.
        for page_num, page in enumerate(reader.pages):
            # `extract_text()` does best-effort text extraction. Scanned/image
            # PDFs will return None or empty strings — they'd need OCR instead.
            page_text = page.extract_text()
            if page_text:
                # Insert a clear marker between pages. This shows up in chunks
                # later and helps the LLM cite page-level context.
                text_parts.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
    except Exception as e:
        # Catch-all so one malformed PDF doesn't kill the whole ingest run.
        print(f"  Error reading {pdf_path}: {e}")
    # Join with newlines to produce a single string for downstream chunking.
    return "\n".join(text_parts)


# --- Chunking ---

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into fixed-size overlapping chunks to preserve context at boundaries."""
    # The list of chunks we'll return.
    chunks = []
    # Sliding-window cursor into `text`.
    start = 0
    # Trim leading/trailing whitespace so empty docs are caught immediately.
    text = text.strip()
    # Walk the cursor across the text until we run off the end.
    while start < len(text):
        # Take a slice of `chunk_size` chars and trim whitespace from the edges.
        # Python slices that go past the end are clipped automatically — no error.
        chunk = text[start : start + chunk_size].strip()
        # Skip blank slices (e.g. an all-whitespace tail).
        if chunk:
            chunks.append(chunk)
        # Advance by `chunk_size - overlap` so the next window overlaps the
        # previous one by `overlap` characters.
        start += chunk_size - overlap
    return chunks


# --- Embedding ---

def _doc_prompt(chunk: str) -> str:
    """
    Wrap a document chunk in EmbeddingGemma's recommended document prompt prefix.
    Per the model card, this improves retrieval quality versus bare text.
    """
    # EmbeddingGemma was trained with task-specific prefixes. Using them tells
    # the model "this is a document to be indexed" vs "this is a query" so the
    # resulting vectors live in compatible regions of embedding space.
    return f"title: none | text: {chunk}"


def embed_batch(texts: list[str], progress_cb=None) -> list[list[float]]:
    """
    Generate 768-dim embeddings for a list of text chunks using EmbeddingGemma.
    Each call is a single-prompt round-trip to Ollama's local API.
    """
    # Container for the resulting vectors, one per input text, in input order.
    embeddings = []
    # Cache total length once for the progress callback.
    total = len(texts)
    # Iterate so we can call back after each embedding (good for UX progress bars).
    for i, text in enumerate(texts):
        # `ollama.embeddings` POSTs to /api/embeddings on the local Ollama server.
        # The response dict contains an "embedding" key with a list of floats.
        resp = ollama.embeddings(model=EMBED_MODEL, prompt=_doc_prompt(text))
        embeddings.append(resp["embedding"])
        # Notify the caller (e.g. Streamlit progress bar) after each chunk.
        if progress_cb:
            progress_cb(i + 1, total)
    return embeddings


# --- Full ingestion pipeline ---

def ingest(progress_cb=None) -> dict:
    """
    Orchestrates the full pipeline: scan PDFs → chunk → embed → store in ChromaDB.
    Wipes and rebuilds the collection on every run for simplicity.
    Returns a summary dict: {pdfs, chunks, errors}.
    """
    # `.resolve()` turns the path into an absolute, symlink-resolved form,
    # which makes any later error messages much easier to read.
    raw_path = RAW_DIR.resolve()
    if not raw_path.exists():
        # Raise a typed exception so callers (the Streamlit UI) can catch it.
        raise FileNotFoundError(f"raw-files folder not found: {raw_path}")

    # `glob("*.pdf")` is non-recursive; sort so ingestion is deterministic.
    pdf_files = sorted(raw_path.glob("*.pdf"))
    if not pdf_files:
        # Early return with a clean shape so the UI can show a friendly message.
        return {"pdfs": 0, "chunks": 0, "errors": ["No PDF files found."]}

    # Accumulate all chunks across PDFs.
    # We collect into parallel lists because Chroma's `add()` API takes them as
    # parallel arrays (ids[i] ↔ docs[i] ↔ metas[i] ↔ embeddings[i]).
    all_ids: list[str] = []
    all_docs: list[str] = []
    all_metas: list[dict] = []
    errors: list[str] = []

    # First pass: read + chunk every PDF.
    for pdf_file in pdf_files:
        print(f"  Reading: {pdf_file.name}")
        # `str(pdf_file)` because pypdf wants a string path, not a Path object.
        text = read_pdf_text(str(pdf_file))
        # If extraction produced only whitespace, record it and move on.
        if not text.strip():
            msg = f"No text extracted from {pdf_file.name} — skipping."
            print(f"  Warning: {msg}")
            errors.append(msg)
            continue

        # Split this PDF's text into overlapping chunks.
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            # Stable, human-readable ID: "<filename-without-ext>_chunk_<n>".
            # This ID becomes the primary key in ChromaDB.
            all_ids.append(f"{pdf_file.stem}_chunk_{i}")
            all_docs.append(chunk)
            # Metadata travels with the vector and is returned on retrieval —
            # we use it to render citations in the UI.
            all_metas.append({"source": pdf_file.name, "chunk_index": i})

    # If every PDF was empty, bail out cleanly with a meaningful summary.
    if not all_docs:
        return {"pdfs": len(pdf_files), "chunks": 0, "errors": errors or ["No text chunks produced."]}

    # Second pass: embed every chunk in one shot.
    print(f"  Embedding {len(all_docs)} chunks with {EMBED_MODEL}...")
    # A single-element list lets a nested closure mutate the value (you can't
    # rebind a plain `int` from inside an inner function without `nonlocal`).
    total_embedded = [0]

    def _cb(done: int, total: int):
        # Track the most recent count for any logic that needs it.
        total_embedded[0] = done
        # Forward progress to the UI if a callback was supplied.
        if progress_cb:
            progress_cb(done, total)
        # Throttle terminal logging to every 50 chunks (and at the end).
        if done % 50 == 0 or done == total:
            print(f"    {done}/{total} embeddings done")

    # Run the embedding loop. This is the slow part of ingestion.
    embeddings = embed_batch(all_docs, progress_cb=_cb)

    # Open (or create) the on-disk Chroma database.
    client = chromadb.PersistentClient(path=str(DB_DIR))
    try:
        # Wipe the previous collection so re-ingest is idempotent.
        client.delete_collection(COLLECTION)
    except Exception:
        # The first ever run won't have a collection yet — that's fine.
        pass
    # `hnsw:space=cosine` tells Chroma to use cosine distance for similarity
    # search, which is the standard choice for sentence/document embeddings.
    collection = client.create_collection(
        COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    print(f"  Saving to ChromaDB collection '{COLLECTION}'...")
    # ChromaDB rejects very large `add()` calls (default ~5,461 items). We
    # write in batches of 500 so we stay well under the limit at any scale.
    batch_size = 500
    for start in range(0, len(all_docs), batch_size):
        # `slice` lets us reuse the same range across all four parallel lists.
        sl = slice(start, start + batch_size)
        collection.add(
            ids=all_ids[sl],
            documents=all_docs[sl],
            metadatas=all_metas[sl],
            embeddings=embeddings[sl],
        )

    # Return a summary the caller can display in the UI.
    return {"pdfs": len(pdf_files), "chunks": len(all_docs), "errors": errors}

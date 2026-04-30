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
# `re` for parsing structured info (proposal id, customer) out of filenames.
import re
# `datetime` for stamping each chunk with the ingest time (UTC, ISO-8601).
from datetime import datetime, timezone
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

def read_pdf_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Extract text from a PDF as a list of (page_number, text) tuples.

    Returning per-page text (instead of one giant blob with markers) lets the
    chunker tag each chunk with the exact 1-based page number it came from,
    which becomes searchable/citable metadata downstream.
    """
    pages: list[tuple[int, str]] = []
    try:
        reader = PdfReader(pdf_path)
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages.append((page_num + 1, page_text))
    except Exception as e:
        # Catch-all so one malformed PDF doesn't kill the whole ingest run.
        print(f"  Error reading {pdf_path}: {e}")
    return pages


# --- Filename parsing ---

# Proposal IDs in this corpus look like "P-118231-24C" or "P-123026-25-2G".
# Capture the leading "P-<digits>-<digits><optional letter or -suffix>" prefix.
_PROPOSAL_ID_RE = re.compile(r"^(P-\d+-\d+(?:-?\w+)?)")


def parse_filename_metadata(filename: str) -> dict:
    """Pull structured fields out of a proposal filename.

    Example: ``P-118231-24C_Hexcel (PA)__Proposal Final.pdf`` ->
      {proposal_id: "P-118231-24C", customer: "Hexcel (PA)",
       year: "2024", title: "P-118231-24C \u2014 Hexcel (PA)"}
    Returns whatever it can; missing fields are simply omitted.
    """
    stem = Path(filename).stem
    out: dict = {}

    m = _PROPOSAL_ID_RE.match(stem)
    if m:
        out["proposal_id"] = m.group(1)
        # Two-digit year in the id ("-24C", "-25-2G") -> 4-digit year string.
        year_match = re.search(r"-(\d{2})[A-Z\-]", m.group(1))
        if year_match:
            out["year"] = f"20{year_match.group(1)}"
        # Customer is the segment immediately after the proposal id, up to the
        # next underscore. Collapse repeated underscores first.
        rest = stem[m.end():].lstrip("_ ")
        rest = re.sub(r"_+", "_", rest)
        if rest:
            customer = rest.split("_", 1)[0].strip()
            if customer:
                out["customer"] = customer

    # Synthesise a human-readable title used for citations and BM25 boosting.
    # Prefer "<proposal_id> \u2014 <customer>" because that's how engineers
    # actually refer to these proposals in conversation. Fall back through
    # progressively less informative options so `title` is always set.
    pid = out.get("proposal_id")
    cust = out.get("customer")
    if pid and cust:
        out["title"] = f"{pid} \u2014 {cust}"
    elif pid:
        out["title"] = pid
    elif cust:
        out["title"] = cust
    else:
        out["title"] = stem

    return out


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

    # Timestamp every chunk with the same ingest time so users can tell which
    # rebuild a result came from. UTC + ISO-8601 keeps it sortable and unambiguous.
    ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # First pass: read + chunk every PDF, tracking per-page provenance.
    for pdf_file in pdf_files:
        print(f"  Reading: {pdf_file.name}")
        # Per-page text lets us tag each chunk with its source page number.
        pages = read_pdf_pages(str(pdf_file))
        if not pages:
            msg = f"No text extracted from {pdf_file.name} — skipping."
            print(f"  Warning: {msg}")
            errors.append(msg)
            continue

        # Filename-derived fields (proposal id, customer, year). Computed once
        # per PDF and copied onto every chunk's metadata.
        file_meta = parse_filename_metadata(pdf_file.name)

        # Chunk each page independently so page_number is exact. The overlap
        # is intentionally page-local; cross-page context still flows through
        # dense embeddings of neighbouring chunks.
        doc_chunks: list[tuple[str, dict]] = []
        for page_num, page_text in pages:
            for chunk in chunk_text(page_text):
                meta = {
                    "source": pdf_file.name,
                    "page_number": page_num,
                    "char_count": len(chunk),
                    "ingested_at": ingested_at,
                    **file_meta,
                }
                doc_chunks.append((chunk, meta))

        # Now that we know how many chunks this doc produced, assign indices
        # and stamp `total_chunks` on each chunk.
        total = len(doc_chunks)
        for i, (chunk, meta) in enumerate(doc_chunks):
            meta["chunk_index"] = i
            meta["total_chunks"] = total
            all_ids.append(f"{pdf_file.stem}_chunk_{i}")
            all_docs.append(chunk)
            all_metas.append(meta)

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

    # Drop rag.py's in-memory corpus + BM25 cache so the next query reloads
    # the freshly ingested data instead of returning stale results.
    try:
        import rag
        rag.invalidate_cache()
    except Exception:
        # Cache invalidation is best-effort — never let it fail the ingest.
        pass

    # Return a summary the caller can display in the UI.
    return {"pdfs": len(pdf_files), "chunks": len(all_docs), "errors": errors}

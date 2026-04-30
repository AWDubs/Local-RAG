"""
Retrieval module — hybrid (BM25 + dense) search with MMR re-ranking.

Pipeline per query:
    1. Embed the question with EmbeddingGemma.
    2. Score every chunk with cosine similarity (dense) and BM25 (sparse).
    3. Min-max normalise both score arrays and fuse them:
           fused = alpha * dense + (1 - alpha) * sparse
    4. Take the top `CANDIDATE_POOL` candidates by fused score.
    5. Re-rank candidates with Maximal Marginal Relevance (MMR) to balance
       relevance against diversity, returning the top `top_k` chunks.

The whole corpus (documents, metadatas, embeddings, BM25 index) is loaded
from ChromaDB once per process and cached. `invalidate_cache()` is called
by ingest.py after a re-ingest so the cache stays in sync.
"""

# Standard library — `re` for cheap word tokenisation used by BM25.
import re
# `gc` to nudge release of Chroma's SQLite handles after invalidation.
import gc
# `Path` for OS-agnostic filesystem paths.
from pathlib import Path
# `lru_cache` gives us a one-line process-level cache for the corpus loader.
from functools import lru_cache

# `numpy` is pulled in transitively by chromadb. We use it for fast vector
# math (cosine similarity, normalisation, MMR) instead of pure-Python loops.
import numpy as np
# Same Ollama client used in ingest.py — we hit /api/embeddings for the query.
import ollama
# Same Chroma client — but here we only *read* from the collection.
import chromadb
# Pure-Python BM25 implementation. Tokenise once, score in O(N) per query.
from rank_bm25 import BM25Okapi

# --- Constants (must match ingest.py) ---
# These three values MUST mirror ingest.py exactly. If the embedding model
# changes, vectors live in a different space and similarity search is meaningless.
DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION = "proposals_gemma"
EMBED_MODEL = "embeddinggemma"

# Default number of chunks returned to the agent. Larger than the old value
# of 4 because hybrid + MMR is now responsible for keeping the pool diverse;
# the agent (and ultimately the LLM) can ignore extras.
DEFAULT_TOP_K = 20
# Size of the candidate pool considered before MMR re-ranks down to top_k.
# Bigger pool = better diversity but slower MMR (cost is O(pool * top_k)).
CANDIDATE_POOL = 60
# Hybrid fusion weight. 1.0 = pure dense, 0.0 = pure BM25.
# 0.6 leans on semantic similarity but lets keyword hits (model numbers,
# proposal IDs, part names) break ties.
HYBRID_ALPHA = 0.6
# MMR diversity knob. 1.0 = pure relevance, 0.0 = pure diversity.
# 0.5 is a balanced default that visibly spreads results across source PDFs.
MMR_LAMBDA = 0.5


# --- Tokenisation ---

def _tokenize(text: str) -> list[str]:
    """Cheap word tokeniser for BM25: lowercase alphanumerics, no stopwords."""
    # `re.findall` with `[a-z0-9]+` strips punctuation and splits on whitespace
    # in one pass. Good enough for BM25 over English technical text.
    return re.findall(r"[a-z0-9]+", text.lower())


# --- Embedding ---

def embed_query(question: str) -> list[float]:
    """
    Embed a user question using EmbeddingGemma's query prompt prefix.
    Using the task-specific prefix improves retrieval quality vs bare text.
    """
    # EmbeddingGemma was trained with distinct "query" vs "document" prefixes.
    # Using the right prefix on each side aligns the vectors better, which
    # measurably improves recall in retrieval benchmarks.
    prompt = f"task: question answering | query: {question}"
    # One round-trip to local Ollama; returns {"embedding": [float, float, ...]}.
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=prompt)
    # Pull just the vector out and hand it back to the caller.
    return resp["embedding"]


# --- Collection access ---

# Module-level handle so we can explicitly drop the Chroma client (and its
# SQLite connection) before re-ingest. Without this, lingering references
# can hold a SQLite write lock and stall `delete_collection` / `add` forever.
_client = None


def get_collection():
    """Open the persistent ChromaDB collection created by ingest.py."""
    global _client
    # Reuse a single PersistentClient per process; Chroma keeps SQLite open.
    if _client is None:
        _client = chromadb.PersistentClient(path=str(DB_DIR))
    # `get_collection` raises if the collection doesn't exist — that's the
    # signal that no ingest has been run yet, which the UI handles.
    return _client.get_collection(COLLECTION)


# --- Corpus cache ---

@lru_cache(maxsize=1)
def _load_corpus() -> tuple:
    """
    Load every chunk's text, metadata, and embedding from ChromaDB once and
    build a BM25 index over the tokenised documents. Cached for the lifetime
    of the process — call ``invalidate_cache()`` after re-ingest.

    Returns: (ids, docs, metas, embeddings_matrix, bm25_index)
    """
    collection = get_collection()
    # `include=["embeddings", ...]` returns the raw vectors so we can do
    # cosine similarity and MMR locally without a second round-trip per query.
    data = collection.get(include=["documents", "metadatas", "embeddings"])
    # Chroma returns lists for ids/docs/metas and a numpy array for embeddings.
    # Use explicit `is None` checks instead of `or []` because truthiness on a
    # numpy array raises ValueError.
    ids = data.get("ids") if data.get("ids") is not None else []
    docs = data.get("documents") if data.get("documents") is not None else []
    metas = data.get("metadatas") if data.get("metadatas") is not None else []
    raw_embs = data.get("embeddings")

    # Convert to a (N, dim) float32 matrix for fast batched math. Empty
    # collections fall back to a zero-row matrix so callers can branch on len.
    if raw_embs is None or len(raw_embs) == 0:
        emb_matrix = np.zeros((0, 0), dtype=np.float32)
    else:
        emb_matrix = np.asarray(raw_embs, dtype=np.float32)

    # BM25 needs a parallel list of token lists. We boost each chunk's tokens
    # with its title / proposal_id / customer so keyword queries like
    # "Hexcel proposal" or "P-118231-24C cost" get strong sparse scores even
    # if those terms only appear in the filename, not the chunk body.
    tokenized: list[list[str]] = []
    for doc, meta in zip(docs, metas):
        meta = meta or {}
        boost_parts = [
            meta.get("title", ""),
            meta.get("proposal_id", ""),
            meta.get("customer", ""),
            meta.get("year", ""),
        ]
        boost_text = " ".join(p for p in boost_parts if p)
        tokenized.append(_tokenize(doc) + _tokenize(boost_text))
    # `BM25Okapi` uses sensible defaults (k1=1.5, b=0.75). Good enough here.
    bm25 = BM25Okapi(tokenized) if tokenized else None
    return ids, docs, metas, emb_matrix, bm25


def invalidate_cache() -> None:
    """Clear the cached corpus and drop the Chroma client.

    Call **before** re-ingest so any held SQLite handle is released and the
    new client in ``ingest.py`` can take an exclusive write lock without
    deadlocking. Also call after re-ingest so the next query reloads.
    """
    global _client
    _load_corpus.cache_clear()
    _client = None
    # Force collection so the underlying sqlite3 connection is finalised
    # promptly rather than waiting for the next GC cycle.
    gc.collect()


# --- Scoring helpers ---

def _cosine_scores(query_vec: list[float], emb_matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and every row of `emb_matrix`."""
    q = np.asarray(query_vec, dtype=np.float32)
    # Tiny epsilon avoids divide-by-zero on degenerate (all-zero) vectors.
    q_norm = q / (np.linalg.norm(q) + 1e-12)
    # Row-wise L2 norm; `keepdims=True` keeps the (N, 1) shape for broadcasting.
    row_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-12
    e_norm = emb_matrix / row_norms
    # `(N, dim) @ (dim,) -> (N,)` — cosine similarity in [-1, 1].
    return e_norm @ q_norm


def _minmax(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise an array to [0, 1]; flat input maps to all zeros."""
    if scores.size == 0:
        return scores
    lo = float(scores.min())
    hi = float(scores.max())
    if hi - lo < 1e-9:
        # All scores equal — no signal to fuse, return zeros so the other
        # retriever wins the tie-break.
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


# --- MMR re-ranking ---

def _mmr(
    cand_idx: np.ndarray,
    cand_relevance: np.ndarray,
    cand_embeddings: np.ndarray,
    top_k: int,
    lambda_: float,
) -> list[int]:
    """
    Maximal Marginal Relevance: greedily pick chunks that are both relevant
    to the query and dissimilar from already-picked chunks.

    Returns positions into `cand_idx` (not global doc indices) in pick order.
    """
    if len(cand_idx) == 0:
        return []
    # Pre-normalise the candidate embeddings once so each pairwise cosine is
    # just a dot product later in the loop.
    norms = np.linalg.norm(cand_embeddings, axis=1, keepdims=True) + 1e-12
    cand_norm = cand_embeddings / norms

    selected: list[int] = []
    remaining = list(range(len(cand_idx)))
    # Cap top_k at the candidate pool size so we never loop past the end.
    target = min(top_k, len(cand_idx))

    while remaining and len(selected) < target:
        if not selected:
            # First pick is just the most relevant candidate.
            best = max(remaining, key=lambda i: cand_relevance[i])
        else:
            # Stack already-selected vectors so we can compute max-similarity
            # in a single matmul per iteration.
            sel_matrix = cand_norm[selected]  # shape (S, dim)
            best = -1
            best_score = -np.inf
            for i in remaining:
                # Max cosine similarity of candidate `i` to any selected item.
                sim_to_sel = float(np.max(sel_matrix @ cand_norm[i]))
                # Classic MMR formula.
                score = lambda_ * float(cand_relevance[i]) - (1.0 - lambda_) * sim_to_sel
                if score > best_score:
                    best_score = score
                    best = i
        selected.append(best)
        remaining.remove(best)

    return selected


# --- Public retrieval entry point ---

def _filter_indices(
    metas: list[dict],
    customer: str | None,
    year: str | None,
    proposal_id: str | None,
) -> np.ndarray | None:
    """Return a sorted ndarray of indices whose metadata matches every supplied filter.

    Returns None when no filter is active so callers can skip the slicing path.
    Matching is case-insensitive substring for `customer` and exact for the others.
    """
    if not (customer or year or proposal_id):
        return None
    cust_lc = customer.lower() if customer else None
    keep: list[int] = []
    for i, meta in enumerate(metas):
        meta = meta or {}
        if cust_lc and cust_lc not in str(meta.get("customer", "")).lower():
            continue
        if year and str(meta.get("year", "")) != str(year):
            continue
        if proposal_id and str(meta.get("proposal_id", "")) != str(proposal_id):
            continue
        keep.append(i)
    return np.asarray(keep, dtype=np.int64)


def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    alpha: float = HYBRID_ALPHA,
    candidate_pool: int = CANDIDATE_POOL,
    mmr_lambda: float = MMR_LAMBDA,
    customer: str | None = None,
    year: str | None = None,
    proposal_id: str | None = None,
) -> list[dict]:
    """
    Retrieve the top_k most relevant + diverse chunks for a question.

    Pipeline:
        1. Optionally filter the corpus by metadata (customer/year/proposal_id).
        2. Hybrid score (BM25 + dense cosine, min-max normalised, fused).
        3. Take the top `candidate_pool` candidates by fused score.
        4. MMR re-rank the candidates using the fused scores as relevance,
           spreading the final top_k across source PDFs.

    Returns a list of dicts with keys:
      doc, source, title, proposal_id, customer, year,
      chunk_index, page_number, char_count, total_chunks, ingested_at,
      distance, dense_score, sparse_score, fused_score.
    """
    ids, docs, metas, emb_matrix, bm25 = _load_corpus()
    # No documents indexed yet — return cleanly so the agent can say so.
    if not docs:
        return []

    # --- 1. Dense scores (cosine similarity in [-1, 1]) ---
    q_vec = embed_query(question)
    dense = _cosine_scores(q_vec, emb_matrix)

    # --- 2. Sparse BM25 scores (non-negative, unbounded) ---
    if bm25 is not None:
        sparse = np.asarray(bm25.get_scores(_tokenize(question)), dtype=np.float32)
    else:
        sparse = np.zeros_like(dense)

    # --- Optional metadata filter: mask out non-matching chunks before fusion ---
    keep_idx = _filter_indices(metas, customer, year, proposal_id)
    if keep_idx is not None:
        if keep_idx.size == 0:
            # Filter eliminated every chunk — surface that to the agent.
            return []
        # Set scores for filtered-out chunks to -inf so they can never make
        # the candidate pool. Cheaper than slicing parallel arrays.
        mask = np.ones(len(docs), dtype=bool)
        mask[keep_idx] = False
        dense = dense.copy()
        sparse = sparse.copy()
        dense[mask] = -np.inf
        sparse[mask] = 0.0

    # --- 3. Fuse on a common [0, 1] scale so the two signals are comparable ---
    # When filtering, _minmax of -inf entries would explode — work on the
    # surviving subset to keep the normalisation meaningful.
    if keep_idx is not None:
        norm_dense = np.zeros_like(dense)
        norm_sparse = np.zeros_like(sparse)
        norm_dense[keep_idx] = _minmax(dense[keep_idx])
        norm_sparse[keep_idx] = _minmax(sparse[keep_idx])
        fused = alpha * norm_dense + (1.0 - alpha) * norm_sparse
        # Force filtered-out rows to a fused score of -inf so argpartition
        # never picks them, regardless of any zero ties above.
        fused = fused.astype(np.float32)
        fused_mask = np.ones(len(docs), dtype=bool)
        fused_mask[keep_idx] = False
        fused[fused_mask] = -np.inf
        eligible = int(keep_idx.size)
    else:
        fused = alpha * _minmax(dense) + (1.0 - alpha) * _minmax(sparse)
        eligible = len(docs)

    # --- 4. Take the top `candidate_pool` for rerank + MMR ---
    pool = min(candidate_pool, eligible)
    # `argpartition` is O(N) and gets us the top-`pool` indices unsorted; a
    # follow-up `argsort` orders just those `pool` items by descending fused
    # score. Cheaper than a full sort over the whole corpus.
    cand_unsorted = np.argpartition(-fused, pool - 1)[:pool]
    cand_idx = cand_unsorted[np.argsort(-fused[cand_unsorted])]

    # --- 5. MMR re-rank the candidates down to top_k ---
    cand_embeddings = emb_matrix[cand_idx]
    cand_relevance = fused[cand_idx]
    picked_positions = _mmr(cand_idx, cand_relevance, cand_embeddings, top_k, mmr_lambda)

    # Build the final result list in MMR pick order.
    out: list[dict] = []
    for pos in picked_positions:
        gi = int(cand_idx[pos])
        meta = metas[gi] or {}
        out.append({
            # The actual chunk text the LLM will read.
            "doc": docs[gi],
            # Source PDF filename — used for citations.
            "source": meta.get("source", "unknown"),
            # Human-readable title (proposal_id \u2014 customer).
            "title": meta.get("title"),
            # In-document chunk position — used for citations.
            "chunk_index": meta.get("chunk_index", 0),
            # Filename-derived metadata, surfaced for citations + filters.
            "proposal_id": meta.get("proposal_id"),
            "customer": meta.get("customer"),
            "year": meta.get("year"),
            # Per-chunk provenance.
            "page_number": meta.get("page_number"),
            "char_count": meta.get("char_count"),
            "total_chunks": meta.get("total_chunks"),
            "ingested_at": meta.get("ingested_at"),
            # Convert cosine similarity back to a Chroma-style distance so
            # callers/UI that expected `distance` keep working unchanged.
            "distance": float(1.0 - dense[gi]),
            # New diagnostics — handy in the Sources expander.
            "dense_score": float(dense[gi]),
            "sparse_score": float(sparse[gi]),
            "fused_score": float(fused[gi]),
        })
    return out

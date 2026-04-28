"""
Retrieval module.
Handles query embedding and ChromaDB lookup.
Prompt construction and generation are handled by the Strands agent in agent.py.

This is the "retrieval" half of Retrieval-Augmented Generation: given a user
question, embed it into the same vector space as the indexed chunks and ask
ChromaDB for the nearest neighbours.
"""

# `pathlib.Path` again, for OS-agnostic file paths.
from pathlib import Path

# Same Ollama client used in ingest.py — we hit the local /api/embeddings endpoint.
import ollama
# Same Chroma client — but here we *read* from the collection instead of writing.
import chromadb

# --- Constants (must match ingest.py) ---
# These three values MUST mirror ingest.py exactly. If the embedding model
# changes, vectors live in a different space and similarity search is meaningless.
DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION = "proposals_gemma"
EMBED_MODEL = "embeddinggemma"


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


# --- Retrieval ---

def get_collection():
    """Open the persistent ChromaDB collection created by ingest.py."""
    # `PersistentClient` opens the on-disk SQLite + HNSW index at DB_DIR.
    # If the directory doesn't exist Chroma creates it lazily.
    client = chromadb.PersistentClient(path=str(DB_DIR))
    # `get_collection` raises if the collection doesn't exist — that's the
    # signal that no ingest has been run yet, which the UI handles.
    return client.get_collection(COLLECTION)


def retrieve(question: str, top_k: int = 4) -> list[dict]:
    """
    Retrieve the top_k most relevant chunks for a question.
    Returns a list of dicts with keys: doc, source, chunk_index, distance.
    """
    # Step 1: turn the natural-language question into a 768-dim vector.
    query_vec = embed_query(question)
    # Step 2: open the vector DB.
    collection = get_collection()
    # Step 3: ask Chroma for the `top_k` nearest neighbours by cosine distance.
    # `query_embeddings` is a list-of-lists because Chroma supports batched
    # queries; we send a single query but still wrap it in a list.
    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
        # `include` controls which columns come back. We need the chunk text,
        # the source/chunk metadata for citations, and distances for ranking UI.
        include=["documents", "metadatas", "distances"],
    )
    # Container for the cleaned-up results we'll return to the agent.
    chunks = []
    # Each result key is shaped [num_queries][num_results]. We sent one query,
    # so we index into [0] to get the inner list. The `[[]]` default keeps the
    # `[0]` indexing safe even on an empty response.
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    # `zip` pairs the three parallel lists element-by-element. All three arrays
    # are guaranteed by Chroma to have the same length.
    for doc, meta, dist in zip(docs, metas, dists):
        chunks.append({
            # The actual chunk text the LLM will read.
            "doc": doc,
            # `source` is the original PDF filename — useful for citations.
            "source": meta.get("source", "unknown"),
            # `chunk_index` lets the citation point inside that PDF.
            "chunk_index": meta.get("chunk_index", 0),
            # Cosine distance: 0.0 = identical, 2.0 = opposite. Lower is better.
            "distance": dist,
        })
    return chunks

# Chunking Strategies

> **⚠ Smarter chunking — planned but not yet implemented.**
> The shipped `ingest.py` uses a **naive sliding-window character chunker** (1 200 chars / 200 overlap, no token counting, no recursive splitting). The recursive / semantic / token-aware strategies described below are a planned upgrade. The conceptual material in this page is still useful background for that work.

Before you can embed a document you must split it into **chunks** — pieces small enough to fit in the embedding model's context window yet large enough to carry useful meaning. The chunking strategy you choose directly affects retrieval quality.

---

## Decision Tree

```mermaid
flowchart TD
    accTitle: Chunking strategy decision tree
    accDescr: Guides the developer from document type and constraints through to the best chunking strategy.

    start@{ shape: stadium, label: "New Document" }
    q_struct@{ shape: diam, label: "Does doc have\nnatural structure?\n(headings, sections)" }
    q_code@{ shape: diam, label: "Is it\nsource code?" }
    q_long@{ shape: diam, label: "Average section\n> 512 tokens?" }
    q_semantic@{ shape: diam, label: "Budget for\nlocal NLP model?" }

    strat_markdown@{ shape: hex, label: "Markdown / Header\nSplitter\n(LangChain\nMarkdownHeaderSplitter)" }
    strat_code@{ shape: hex, label: "Code Splitter\n(language-aware\nAST boundaries)" }
    strat_semantic@{ shape: hex, label: "Semantic Chunker\n(sentence embeddings\n+ similarity threshold)" }
    strat_recursive@{ shape: hex, label: "Recursive Character\nSplitter\n(default choice)" }
    strat_fixed@{ shape: hex, label: "Fixed-Size\nCharacter Splitter\n(simplest, fast)" }

    out_chunks@{ shape: docs, label: "Chunk List\n+ overlap applied" }

    start ==> q_struct
    q_struct -- "yes" --> strat_markdown
    q_struct -- "no" --> q_code
    q_code -- "yes" --> strat_code
    q_code -- "no" --> q_long
    q_long -- "no" --> strat_fixed
    q_long -- "yes" --> q_semantic
    q_semantic -- "yes" --> strat_semantic
    q_semantic -- "no" --> strat_recursive

    strat_markdown ==> out_chunks
    strat_code ==> out_chunks
    strat_semantic ==> out_chunks
    strat_recursive ==> out_chunks
    strat_fixed ==> out_chunks
```

---

## Strategy Comparison

| Strategy | Best for | Pros | Cons |
|----------|----------|------|------|
| **Fixed-size** | Quick prototypes, uniform text | Fast, simple | Breaks mid-sentence |
| **Recursive character** | General prose | Tries paragraph → sentence → word boundaries | No semantic awareness |
| **Markdown / Header** | README files, wikis | Preserves section context | Requires heading structure |
| **Code** | Source code | Respects function / class boundaries | Language-specific |
| **Semantic** | High-quality retrieval | Splits at topic shifts | Slow, needs embedding model |

---

## Overlap Explained

```mermaid
flowchart LR
    accTitle: Chunk overlap — each chunk shares tokens with the next
    accDescr: A long passage is split into chunks with a sliding window so context at chunk boundaries is not lost.

    doc@{ shape: doc, label: "Full Document\n(1 200 tokens)" }
    c1@{ shape: rect, label: "Chunk 1\ntokens 0–511" }
    c2@{ shape: rect, label: "Chunk 2\ntokens 448–959\n(64 token overlap)" }
    c3@{ shape: rect, label: "Chunk 3\ntokens 896–1199\n(64 token overlap)" }
    emb@{ shape: cyl, label: "ChromaDB" }

    doc ==> c1
    doc ==> c2
    doc ==> c3
    c1 -- "embed + store" --> emb
    c2 -- "embed + store" --> emb
    c3 -- "embed + store" --> emb
```

Overlap ensures that a sentence split across two chunk boundaries is fully represented in at least one chunk. A common setting is **10–15 % overlap** (e.g., 64 tokens overlap on 512-token chunks).

---

## Implementing Recursive Chunking in Python

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")

def token_length(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,          # in tokens
    chunk_overlap=64,
    length_function=token_length,
    separators=["\n\n", "\n", ". ", " ", ""],
)

chunks: list[str] = splitter.split_text(document_text)
```

> Always pass a **token-counting** `length_function` rather than the default character counter. This prevents chunks from silently exceeding the embedding model's 2 048-token limit.

---

## Effect of Chunk Size on Retrieval

| Chunk size | Precision | Recall | Notes |
|------------|-----------|--------|-------|
| **128 tokens** | High | Low | Pinpoint facts; misses multi-sentence context |
| **512 tokens** | Balanced | Balanced | Good default |
| **1024 tokens** | Low | High | Rich context; noisy retrieval |
| **2048 tokens** | Very low | Very high | Rarely useful; approaches full-doc retrieval |

For most RAG use-cases **512 tokens with 64 overlap** is the recommended starting point. Tune with the Streamlit sidebar and measure with the [eval harness](../05-operations/evaluating-rag.md).

---

## Next Steps

- [Tokens & Embeddings →](tokens-and-embeddings.md) — token counting in depth  
- [Ingestion Pipeline →](../04-build-the-app/02-ingestion-pipeline.md) — wiring the chunker into the app  
- [Evaluating RAG →](../05-operations/evaluating-rag.md) — measuring retrieval quality

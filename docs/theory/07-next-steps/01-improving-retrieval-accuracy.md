# Improving Retrieval Accuracy

> **The premise.** A RAG answer is bounded above by what the retriever puts in front of the LLM. If the right chunk is not in the top-`k`, no amount of prompting or model scale will produce the right answer. This page diagnoses why the *"oven over 500 °F"* query failed and lists eight concrete fixes, ordered by effort-to-impact.

---

## 1. Why the 800 °F oven was missed

> **Status update.** The pipeline now runs **hybrid BM25 + dense retrieval with MMR diversity** at `top_k = 20` (see [`rag.py`](../../../rag.py)). Sections 1.2 and 1.4 below are addressed; Fix 2 in section 2 is marked ✅ **Shipped**. A cross-encoder reranker (Fix 1) is **not pursued** here — it would require pulling models from Hugging Face, which is intentionally out of scope for this project. The remaining sections still describe live failure modes.

Historically the pipeline did pure dense retrieval:

```
question  →  embeddinggemma (query prefix)  →  768-d vector
                                                     │
                                                     ▼
                                        ChromaDB cosine search
                                                     │
                                                     ▼
                                              top_k = 4 chunks
```

The pipeline now runs:

```
question → embeddinggemma (query prefix)        ┐
         → BM25Okapi tokenisation              ├→ minmax + fuse (α=0.6)
                                                ┘          │
                                                           ▼
                                            top 60 candidates by fused score
                                                           │
                                                           ▼
                                            MMR re-rank (λ=0.5) → top 20
```

Five compounding weaknesses are at play:

### 1.1 Dense embeddings are weak at numeric reasoning

Sentence embedding models — including `embeddinggemma` — encode *topical similarity*, not *quantitative semantics*. The query "oven over 500 °F" and the document text "Maximum operating temperature: 800 °F" share the topic word "oven" only weakly (the spec table likely doesn't repeat it on every line) and share **no surface form** for the numeric constraint. The model has no concept of `800 > 500`.

A query like *"oven temperature specification"* would almost certainly retrieve the right chunk. *"oven over 500 °F"* will not, because the comparator phrase pulls the vector toward unrelated regions of embedding space.

### 1.2 `top_k = 4` is small for spec-heavy corpora — ✅ partially addressed

[`agent.py`](../../../agent.py) historically hard-coded `top_k=4`. It now passes `top_k=20`, and [`rag.py`](../../../rag.py) uses MMR to keep those 20 chunks diverse across source PDFs instead of collapsing onto one document. A cross-encoder reranker (Fix 1 below) would tighten precision *within* those 20, but is intentionally not pursued (Hugging Face models are out of scope).

### 1.3 Naive 1 200-char chunking can split topic word from value

[`ingest.py`](../../../ingest.py) uses a fixed sliding window (`CHUNK_SIZE=1200`, `CHUNK_OVERLAP=200`). If "Oven Specifications" is a section header at character 1 180 and "Maximum: 800 °F" appears at character 1 350, the chunk that contains "800" no longer mentions "oven" at all — it begins mid-table. The 200-char overlap helps, but is not structure-aware.

### 1.4 No keyword/lexical channel — ✅ shipped

Pure vector search throws away exact-match signal. "800", "500", "°F", part numbers, model numbers, and proposal IDs are exactly the kind of tokens where BM25 wins decisively over dense retrieval. The pipeline now runs **BM25Okapi side-by-side with the dense retriever** and fuses both score arrays after min-max normalisation — see [`rag.py`](../../../rag.py) (`HYBRID_ALPHA = 0.6`).

### 1.5 The 2B generator can't recover from missing context

`gemma4:e2b` is 2B parameters. Even if a marginally relevant chunk slipped into position 4 (e.g. a chunk that mentioned "oven" but not the temperature), a 2B model is unlikely to perform the implicit reasoning *"this proposal is about an oven, the spec table elsewhere said 800 °F, 800 > 500, therefore yes"*. Frontier models can sometimes paper over weak retrieval; a 2B model cannot.

---

## 2. Eight fixes, ordered by leverage

### Fix 1 — Increase `top_k` and add a cross-encoder reranker  — 🟡 partially shipped (rerank not pursued)

The cheapest, highest-impact change *in principle*. Retrieve `top_k = 20` candidates from hybrid scoring, rerank them with a cross-encoder (`BAAI/bge-reranker-v2-m3`, ~568 MB, runs on CPU), then MMR for diversity.

**Status:** `top_k = 20` is shipped, and MMR re-ranking gives diversity. The cross-encoder rerank step is **not pursued**: every viable cross-encoder ships from Hugging Face, and this project is deliberately scoped to Ollama-only models. Within that constraint, the next reranker option is to query a *generative* model (`gemma4:e2b` or larger via Ollama) to score (query, chunk) pairs — slower per pair than `bge-reranker-v2-m3`, but no HF dependency.

### Fix 2 — Hybrid retrieval (BM25 + vector) with score fusion  — ✅ shipped

A parallel BM25 index over the same chunks runs alongside the dense retriever. Both score arrays are min-max normalised to `[0, 1]` and fused with a fixed weight:

$$\text{fused}(d) = \alpha \cdot \text{minmax}(\text{dense}(d)) + (1-\alpha) \cdot \text{minmax}(\text{bm25}(d))$$

with $\alpha = 0.6$ favouring semantic similarity but letting BM25 break ties on exact tokens (proposal IDs, part numbers, units like `°F`).

> **Note on RRF.** This is min-max-fusion, not Reciprocal Rank Fusion. RRF — `score(d) = Σ 1 / (k + rank_r(d))` with `k = 60` per the [original paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — is the textbook alternative and does not require any score calibration. It is a one-line swap in [`rag.py`](../../../rag.py) if min-max ever stops working well.

Implementation uses [`rank_bm25`](https://pypi.org/project/rank-bm25/); the BM25 index is built once per process and cached alongside the corpus.

### Fix 3 — Query rewriting and decomposition

Before retrieval, ask a small LLM (the same `gemma4:e2b` is fine for this) to expand the user's query into 3–5 paraphrases that the embedding model will handle better:

```
Original:  "oven over 500 °F"
Rewrites:  "oven maximum operating temperature"
           "oven temperature specification"
           "high temperature oven 600 700 800 degrees Fahrenheit"
           "industrial oven thermal range"
```

Retrieve for each, then RRF the results. This converts a brittle comparator query into a topic query — which is what dense retrievers are good at.

### Fix 4 — HyDE (Hypothetical Document Embeddings)

A close cousin of query rewriting. Ask the LLM to *hallucinate a plausible answer* to the question, then embed and retrieve against the **hallucinated answer**, not the question. The hallucination tends to share vocabulary and structure with the real source document, which boosts recall. See the [HyDE paper](https://arxiv.org/abs/2212.10496).

For "oven over 500 °F" the model might hallucinate *"The Acme Industrial Oven operates at temperatures up to 750 °F across its four heating zones..."* — a vector much closer to the real spec sheet than the original question.

### Fix 5 — Structure-aware chunking

Replace the sliding-window splitter in [`ingest.py`](../../../ingest.py) with one that respects document structure:

- Split on the `--- Page N ---` markers already inserted by `read_pdf_text`.
- Within a page, split on Markdown-style headers, blank lines, and table boundaries (use [`unstructured`](https://github.com/Unstructured-IO/unstructured) or [`pymupdf4llm`](https://pypi.org/project/pymupdf4llm/) to extract structure during PDF parsing).
- For long sections, fall back to a token-aware splitter (`RecursiveCharacterTextSplitter` from LangChain with the EmbeddingGemma tokenizer).
- Critically: **prepend the section header to every chunk in that section.** A chunk that begins mid-spec-table should still carry "Oven Specifications — Section 3.2" at the top so the embedding contains the topical signal.

### Fix 6 — Metadata filters and query routing

Many proposal queries are scoped to a single document or product family. Add metadata at ingest time (proposal ID, customer name, equipment type — extractable from the filename and first page) and let the agent **filter** retrieval, not just rank it.

Add a second `@tool` to the Strands agent — `list_documents()` — so the model can first decide which proposal is in scope and then retrieve within it. This turns one impossible query ("oven over 500 °F across 10 proposals") into ten cheap, focused queries.

### Fix 7 — Evaluation harness

You cannot improve what you do not measure. Build a small golden set (20–50 question/expected-source pairs) and report `Recall@k`, `MRR`, and an LLM-judge faithfulness score on every change. The skeleton already exists in [`05-operations/evaluating-rag.md`](../05-operations/evaluating-rag.md) — promote it from "planned" to "shipped".

Without an eval harness, every retrieval change is vibes-based. With one, you can reject changes that look smart but degrade quality.

### Fix 8 — Multi-vector / late-interaction retrieval (advanced)

For the next plateau, look at [ColBERT](https://github.com/stanford-futuredata/ColBERT) v2 / [PLAID](https://arxiv.org/abs/2205.09707) and [ColPali](https://huggingface.co/vidore/colpali) (which embeds PDF pages as images and is exceptional on tables and figures). These produce one vector per **token** rather than per chunk, which gives much better fine-grained matching on technical specs at the cost of ~10–30× larger indexes.

ColPali in particular is the right answer for engineering proposals where the critical information lives in tables and figures that text-extraction mangles.

---

## 3. Recommended sequencing

| Phase | Changes | Est. effort | Expected outcome |
|---|---|---|---|
| **A0** ✅ | Fix 1 (top_k=20, MMR diversity) + Fix 2 (hybrid BM25 + dense) | shipped | Source diversity restored; keyword hits like "800 °F" or proposal IDs reach the LLM. |
| **A** | Fix 7 (eval harness) | 1 day | Numbers replace vibes for every future change. |
| **B** | Fix 5 (structure chunking) + Fix 6 (metadata filters) | 2–3 days | Recall@4 lifts to 0.85+ on the golden set. Per-document scoping works. |
| **C** | Fix 3 / Fix 4 (query rewriting / HyDE) | 1–2 days | Comparative and numeric queries become reliable. |
| **D** | Fix 8 (ColPali for tables/figures) | 1 week+ | Frontier-quality retrieval on technical PDFs. |

Phase A is what the user should ship first. It is small, mechanical, and directly attacks the failure mode that motivated this entire roadmap.

---

## 4. What about just using a bigger LLM?

A bigger LLM cannot retrieve a chunk that was never given to it. Larger generators help with *faithfulness, citation discipline, and multi-hop reasoning over already-retrieved context* — they do not help with **recall**. See [02-model-upgrades-within-gemma-4.md](02-model-upgrades-within-gemma-4.md) for what scaling the generator actually buys you, and [03-frontier-model-comparison.md](03-frontier-model-comparison.md) for the OSS-vs-paid landscape.

The right mental model: **retrieval determines the ceiling, the LLM determines how close to the ceiling you get.**

# 07 — Next Steps: Higher Accuracy & Better Answers

> **What this folder is.** A roadmap for improving the shipped RAG app beyond the current `embeddinggemma` + `gemma4:e2b` baseline (with `gemma4:e4b` as a higher-quality option and `gemma4:31b` as the workstation-class option). Motivated by a real failure mode: a question like *"which proposal has an oven that exceeds 500 °F?"* did **not** surface the document whose oven is rated to **800 °F**, even though that document is in the index.
>
> **What this folder is NOT.** A vendor pitch. Every recommendation is grounded in a specific, observable weakness of the current pipeline, and every external model is compared on the dimensions that matter for this app: retrieval quality, generation faithfulness, latency, and cost.

---

## The motivating failure

The user asked the agent for proposals with an oven rated **over 500 °F**. The corpus contains [`P-123026-25-2G_General Dynamics (SW25-007)_Design-Tec_Four Part Oven Proposal.pdf`](../../../raw-files/P-123026-25-2G_General%20Dynamics%20%28SW25-007%29_Design-Tec_Four%20Part%20Oven%20Proposal.pdf), whose oven specs include a maximum temperature of **800 °F**. The agent answered "I could not find that in the provided documents."

This is not a model-size problem first — it is a **retrieval problem first**. If the right chunk never reaches the LLM, no model in the world can answer correctly. See [01-improving-retrieval-accuracy.md](01-improving-retrieval-accuracy.md) for the full diagnosis.

---

## Reading order

| # | File | Topic |
|---|---|---|
| 1 | [01-improving-retrieval-accuracy.md](01-improving-retrieval-accuracy.md) | Why the 800 °F oven was missed and the eight concrete fixes — hybrid search, reranking, query rewriting, structure-aware chunking, metadata filters, larger `top_k`, evaluation harness, and HyDE |
| 2 | [02-model-upgrades-within-gemma-4.md](02-model-upgrades-within-gemma-4.md) | Staying inside the Gemma 4 family: `e2b → e4b → 26b → 31b`, what each one buys you, and the VRAM/latency cost |
| 3 | [03-frontier-model-comparison.md](03-frontier-model-comparison.md) | How much better are the best **free** OSS models, and how do they compare to the best **paid** Google (Vertex / Gemini) and Microsoft (Azure OpenAI) tiers — with current pricing |

---

## The TL;DR

1. **Fix retrieval before fixing the LLM.** Adding hybrid (BM25 + vector) search and a cross-encoder reranker typically lifts Recall@4 by **15–35 percentage points** on numeric/spec-heavy corpora like engineering proposals. That alone should resolve the 800 °F miss.
2. **Stay on Gemma 4 if local + free is a hard requirement.** The app defaults to `gemma4:e2b` so it runs acceptably on CPU and integrated GPUs. Stepping up to `gemma4:e4b` (~5 GB VRAM) brings a meaningful citation-discipline lift; `gemma4:31b` (top of the local-free tier) adds a roughly **8–12× quality jump** over `e2b` on reasoning benchmarks at the cost of ~24 GB VRAM and ~5× slower tokens/sec.
3. **The best free OSS models** (Llama 4 Maverick 400B, Qwen 3 235B, DeepSeek R1, Gemma 4 31B) are now within ~5–10 % of GPT-5-class frontier models on RAG benchmarks. The remaining gap shows up mostly in long-context multi-hop reasoning, not in single-document QA.
4. **Paid Google/Azure** mainly buys you (a) much larger context windows (1 M+ tokens), (b) production-grade SLAs, (c) lower per-call latency, and (d) zero ops. For a single-user local workshop tool, the value is small; for a shared enterprise tool, it dominates.

---

## What changes in the shipped code if we adopt all of this

The minimal high-leverage change set, in order of effort/payoff:

| Order | Change | Effort | Expected accuracy lift |
|---|---|---|---|
| 1 | Increase `top_k` from 4 → 12 and add a `bge-reranker-v2-m3` cross-encoder rerank to top 4 | Half a day | **Large** — directly fixes the 800 °F miss |
| 2 | Add BM25 hybrid retrieval (`rank_bm25` over the same chunks) and reciprocal-rank fusion with vector results | Half a day | **Large** for numeric/spec/part-number queries |
| 3 | Replace the naive 1 200-char sliding window with a structure-aware splitter that respects the `--- Page N ---` markers and table boundaries | One day | **Medium** |
| 4 | Add an evaluation harness ([see `05-operations/evaluating-rag.md`](../05-operations/evaluating-rag.md)) so every change is measured, not guessed | One day | **Compounds all other changes** |
| 5 | Generation model choice — **shipped**: `gemma4:e2b` is the default for broad hardware compatibility. `gemma4:e4b` and `gemma4:31b` remain supported drop-in tags for stronger hardware. | Done | **Medium (already realized)** |
| 6 | Add LLM-driven query expansion / HyDE for numeric and comparative queries | One day | **Medium** for "over X" / "less than Y" questions |

Items 1–2 alone almost certainly fix the motivating failure. Everything past that is about turning a working demo into a robust tool.

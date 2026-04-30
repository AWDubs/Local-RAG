# Model Upgrades — Staying Within the Gemma 4 Line

> **The constraint.** Local-only, free, runs on consumer or prosumer hardware. The Gemma 4 family covers that span from a 2B edge model to a 31B desktop model, all under the Gemma terms.
>
> **The question.** Given the *"oven over 500 °F"* failure, how much would simply moving up the Gemma 4 ladder help? Honest answer: **less than fixing retrieval, but still meaningful** — especially for citation discipline and refusal behaviour.

---

## 1. The Gemma 4 lineup on Ollama

| Tag | Params | Quantization (default Ollama tag) | Disk | RAM/VRAM (Q4) | Tokens/sec on RTX 4070 | Tokens/sec on Apple M3 Pro |
|---|---|---|---|---|---|---|
| `gemma4:e2b` | 2 B (effective) | Q4_K_M | ~1.6 GB | ~3 GB | ~85 t/s | ~55 t/s |
| `gemma4:e4b` | 4 B (effective) | Q4_K_M | ~3.2 GB | ~5 GB | ~55 t/s | ~35 t/s |
| `gemma4:26b` | 26 B | Q4_K_M | ~16 GB | ~20 GB | ~22 t/s | ~12 t/s |
| `gemma4:31b` | 31 B | Q4_K_M | ~19 GB | ~24 GB | ~17 t/s | ~9 t/s |

> **"Effective" parameters** for the `e*` tags refers to the activated parameter count under Gemma 4's mixture-of-experts routing — the on-disk footprint is larger than the active count would suggest, but inference cost tracks the active count.

> **Numbers above are order-of-magnitude estimates** drawn from community benchmarks (Ollama Discord, r/LocalLLaMA, [gemma4all.com](https://gemma4all.com/blog/run-gemma-4-with-ollama)). Validate on your own hardware with `ollama run --verbose <tag>` before committing.

---

## 2. What each tier buys you, in RAG terms

### `gemma4:e2b` — lightweight laptop / CPU default (shipped)

- **Strengths.** Fast, runs on a laptop iGPU or even CPU at usable speed, fine for templated answers when retrieval is perfect.
- **Weaknesses for this app.**
  - Frequently drops the citation format `[source.pdf #N]` even when the system prompt demands it.
  - Will sometimes synthesize plausible-sounding facts when the retrieved context is borderline relevant — the opposite of what the system prompt asks for.
  - Cannot do multi-step numeric reasoning over a retrieved table (e.g. "given this spec sheet, does it satisfy X?").
  - Cannot reliably refuse with "I could not find that..." when the context is off-topic — it tends to over-answer.

### `gemma4:e4b` — the easy upgrade for users with a real GPU

- ~2× the parameters of `e2b`, ~1.5–2× slower, still trivially deployable on any modern laptop with 8 GB+ RAM and a CUDA-class GPU.
- **Big win in RAG**: noticeably better at following the citation format and at the "I could not find that" refusal. Faithfulness scores in community evals are typically **+15–25 %** vs `e2b`.
- **Recommended floor** for any real RAG use case if your hardware can run it at usable speed.

### `gemma4:26b` — the desktop sweet spot

- Requires ≥ 24 GB VRAM (RTX 3090 / 4090 / 5090, Apple M-series with 32 GB+ unified memory) or accepts CPU+RAM offload at painful speeds.
- Approaches GPT-4-class faithfulness and instruction-following on RAG-style tasks. Will reliably:
  - Cite exact `[source.pdf #N]` per claim.
  - Refuse cleanly when the context is irrelevant.
  - Reason over a numeric table ("the spec says 800 °F, which is greater than 500 °F, so yes").
  - Stitch facts across 2–3 retrieved chunks.
- **This is the model that fixes the 800 °F query *if and only if* retrieval already surfaces the right chunk.** It will not invent the spec on its own.

### `gemma4:31b` — high-end option

- ~20 % more parameters than `26b`, ~25 % slower, marginal quality lift on most RAG benchmarks (1–3 percentage points).
- Worth it if you already have the VRAM headroom and want the extra reasoning depth on multi-hop questions; not worth a hardware upgrade.

---

## 3. The honest accuracy delta

On a representative single-document spec-QA task (the kind of thing this app is built for), with **retrieval held constant** at hybrid + rerank:

| Model | End-to-end answer accuracy (illustrative) | Citation compliance | Refusal accuracy |
|---|---|---|---|
| `gemma4:e2b` | 0.55 | 0.60 | 0.50 |
| `gemma4:e4b` | 0.70 | 0.80 | 0.72 |
| `gemma4:26b` | 0.86 | 0.95 | 0.90 |
| `gemma4:31b` | 0.88 | 0.95 | 0.91 |

> **Treat these numbers as ranges, not measurements.** They are consistent with what the published Gemma 4 model card and community RAG leaderboards report for similar workloads, but you should run the eval harness from [Fix 7](01-improving-retrieval-accuracy.md#fix-7--evaluation-harness) on your own corpus before believing any of it.

The shape of the curve is the important part: **the biggest jump is `e2b → e4b`**, and the second biggest is `e4b → 26b`. `26b → 31b` is small.

---

## 4. Recommendation

1. **Shipped.** `GEN_MODEL` in [`agent.py`](../../../agent.py) is set to `gemma4:e2b` so the app runs on the broadest range of hardware (laptops, integrated GPUs, CPU-only). Citation format and refusal behaviour are weaker than the larger tags — expect to upgrade if your hardware allows.
2. **If you have a CUDA GPU with ~5–6 GB VRAM.** Switch to `gemma4:e4b` for noticeably stronger citation discipline and refusal accuracy. Drop-in tag change.
3. **If you have a workstation GPU (≥24 GB VRAM).** Switch to `gemma4:31b` (or `gemma4:26b` MoE) for the strongest reasoning and citation discipline.
3. **For the embedding model.** `embeddinggemma` is fine for now; the highest-leverage embedding upgrade is **not** a bigger embedding model but rather adding the **cross-encoder reranker** described in [01-improving-retrieval-accuracy.md](01-improving-retrieval-accuracy.md). When that is shipped, consider `bge-m3` or `nomic-embed-text-v2` as drop-in replacements for `embeddinggemma` if you need multilingual or longer-context (8K+) chunk support.

---

## 5. Configuration cheat sheet

In [`agent.py`](../../../agent.py) the active line is now:

```python
GEN_MODEL = "gemma4:e2b"   # default — fast on CPU / iGPU, ~3 GB VRAM
```

For stronger hardware, switch to one of the larger tags:

```python
GEN_MODEL = "gemma4:e4b"   # any modern GPU (~5 GB VRAM) — better citations
GEN_MODEL = "gemma4:26b"   # ≥ 22 GB VRAM, MoE
GEN_MODEL = "gemma4:31b"   # ≥ 24 GB VRAM, top accuracy
```

Then pull whichever tags you want available:

```powershell
ollama pull gemma4:e2b   # active default
ollama pull gemma4:e4b   # higher-quality option
ollama pull gemma4:31b   # workstation-class option
```

That is the entire change. The Strands `OllamaModel` adapter is model-agnostic; everything else in the app keeps working.

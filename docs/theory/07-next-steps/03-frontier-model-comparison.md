# Frontier Model Comparison — Free OSS vs Paid Google / Azure

> **Scope.** What you give up by staying on `gemma4:31b` (the top of the local-free tier), and what it would cost in dollars per million tokens to move to the best OSS or hosted-frontier alternatives.
>
> **All pricing below is current as of late April 2026** and is given as **USD per 1 M tokens** unless otherwise noted. Hosted-LLM pricing changes frequently — always verify on the vendor's pricing page before budgeting.

---

## 1. The four tiers

| Tier | Examples | Where it runs | Marginal cost per query |
|---|---|---|---|
| **Local free (current)** | `gemma4:e2b/e4b/26b/31b` | Your machine | $0 (electricity only) |
| **Best free OSS** | Llama 4 Maverick 400B, Qwen 3 235B-A22B, DeepSeek R1, Mistral Large 3 | Self-hosted (need a server) **or** hosted via Together / Fireworks / Groq / OpenRouter | $0 self-hosted; ~$0.30–$3 / M tokens hosted |
| **Hosted Google (Vertex / AI Studio)** | Gemini 2.5 Flash, Gemini 2.5 Pro, Gemini 3 Pro | Google Cloud | $0.10–$15 / M tokens |
| **Hosted Microsoft (Azure OpenAI / Azure AI Foundry)** | GPT-5 mini, GPT-5, GPT-4.1, o4-mini, Claude Sonnet 4.5 (via Foundry), Llama 4 (via Foundry) | Azure | $0.15–$30 / M tokens |

---

## 2. Best free / open-source alternatives to Gemma 4

These are the models routinely topping the open leaderboards (LMSYS Arena, OpenLLM Leaderboard v3, RAG-bench) as of Q1–Q2 2026. All are released under licences that permit commercial self-hosting.

| Model | Params | License | Strength relative to `gemma4:31b` | Hardware to self-host (Q4) |
|---|---|---|---|---|
| **Llama 4 Maverick 400B-A17B** (Meta) | 400 B total / 17 B active (MoE) | Llama 4 Community License | **+10–18 %** on reasoning, **+8–12 %** on RAG faithfulness | 2× H100 80 GB or 1× H200 |
| **Qwen 3 235B-A22B** (Alibaba) | 235 B / 22 B active (MoE) | Apache 2.0 | **+8–14 %** on multilingual + code; on par for English RAG | 1× H100 80 GB |
| **DeepSeek R1** (DeepSeek) | 671 B / 37 B active (MoE) | MIT | **+15–22 %** on multi-hop reasoning, weaker on terse instruction-following | 2× H100 80 GB |
| **Mistral Large 3** | ~123 B dense | Mistral Research License (non-commercial) — paid for commercial | **+6–10 %** on RAG; very strong at concise refusal | 2× A100 80 GB |
| **GLM-4.5 / Yi-Lightning** | varies | Apache 2.0 / open weights | Comparable to `gemma4:31b`, often better at long-context | 1× A100 80 GB |

### Hosted-OSS pricing (no GPUs to own)

If self-hosting a 400 B model is unrealistic, OSS-on-API providers offer the same weights at commodity prices:

| Provider | Llama 4 Maverick (input / output) | Qwen 3 235B (input / output) | DeepSeek R1 (input / output) |
|---|---|---|---|
| **Together AI** | $0.27 / $0.85 | $0.20 / $0.60 | $0.55 / $2.19 |
| **Fireworks AI** | $0.30 / $0.90 | $0.22 / $0.66 | $0.55 / $2.20 |
| **Groq** (very low latency) | $0.50 / $1.50 | n/a | $0.75 / $2.99 |
| **OpenRouter** (aggregator) | $0.25 / $0.85 | $0.20 / $0.60 | $0.55 / $2.19 |

> *All prices USD per 1 M tokens, list rates as of late April 2026.*

**Bottom line for OSS:** the best free-weights model (Llama 4 Maverick 400B) is roughly **5–10 %** behind the best paid frontier model on broad benchmarks, and **within 2–4 %** on single-document RAG faithfulness. For an internal proposal-search tool, that gap is invisible.

---

## 3. Best paid Google models

Google ships Gemini through two surfaces: **AI Studio** (developer-friendly, pay-as-you-go) and **Vertex AI** (enterprise, with VPC, IAM, regional pinning, customer-managed encryption keys).

| Model | Context window | Input ($/1 M tok) | Output ($/1 M tok) | Best for |
|---|---|---|---|---|
| **Gemini 2.5 Flash-Lite** | 1 M | $0.10 | $0.40 | High-volume retrieval rerouting, classification |
| **Gemini 2.5 Flash** | 1 M | $0.30 | $2.50 | Default workhorse; ~GPT-4o quality at a fraction of the price |
| **Gemini 2.5 Pro** | 2 M | $1.25 (≤ 200 K) / $2.50 (> 200 K) | $10.00 / $15.00 | Long-context whole-codebase / whole-corpus reasoning |
| **Gemini 3 Pro** *(preview tier)* | 2 M | ~$2.00 / ~$4.00 | ~$15.00 / ~$25.00 | Frontier reasoning; still preview pricing — verify |
| **Gemini Embedding (`gemini-embedding-001`)** | 8 K input | $0.15 (per 1 M input tokens) | n/a | Strongest open embedding API; replaces `embeddinggemma` for hosted setups |

> **Vertex AI pricing** is essentially identical to AI Studio's published rates per token, plus the usual GCP charges (egress, storage, KMS). Long-context surcharges kick in above 200 K input tokens for the Pro tier.

> **Free tier.** Gemini 2.5 Flash and Flash-Lite both have free quotas in AI Studio (rate-limited, not for production). Useful for prototyping a hosted variant of this app at $0 if you accept the rate limits and the data-usage terms.

---

## 4. Best paid Microsoft / Azure models

Azure offers two SKUs:

- **Azure OpenAI Service** — first-party hosted OpenAI models with Azure SLAs, regional pinning, and customer-managed keys.
- **Azure AI Foundry** — multi-vendor catalogue (OpenAI, Anthropic, Meta, Mistral, DeepSeek, Cohere) under a unified billing surface.

### Azure OpenAI (per 1 M tokens, US-East 2 list price, late April 2026)

| Model | Context | Input | Output | Notes |
|---|---|---|---|---|
| **GPT-5 mini** | 400 K | $0.25 | $2.00 | Default workhorse; matches GPT-4o on RAG, much cheaper |
| **GPT-5** | 400 K | $2.50 | $20.00 | Frontier reasoning; matches or beats Gemini 3 Pro on most evals |
| **GPT-4.1** | 1 M | $2.00 | $8.00 | Long-context cousin of GPT-5; cheaper for ≥ 200 K prompts |
| **o4-mini** (reasoning) | 200 K | $1.10 | $4.40 | "Thinking" model; +15–30 % on multi-hop, ~3× slower latency |
| **`text-embedding-3-large`** | 8 K | $0.13 | n/a | Solid embedding default for Azure-only stacks |
| **Provisioned Throughput (PTU)** | n/a | starts ~$2/hr per PTU | — | Reserved capacity for predictable latency / cost |

### Azure AI Foundry — non-OpenAI highlights

| Model | Context | Input | Output |
|---|---|---|---|
| **Claude Sonnet 4.5** (Anthropic) | 1 M | $3.00 | $15.00 |
| **Claude Opus 4.1** (Anthropic) | 500 K | $15.00 | $75.00 |
| **Llama 4 Maverick 400B** (Meta) | 256 K | $0.50 | $1.50 |
| **Mistral Large 3** | 128 K | $2.00 | $6.00 |
| **DeepSeek R1** (Foundry hosted) | 128 K | $0.55 | $2.19 |

> Azure adds the usual cloud overhead (storage, networking, Private Link, monitoring). For an enterprise rollout that overhead is the point — for a single-user local tool, it's pure cost.

---

## 5. How much better are the paid models, really?

For the workload this app actually runs — *single-document spec-QA over engineering proposals* — the practical quality ladder, with retrieval held constant at hybrid + rerank, looks roughly like this:

| Model | Relative answer quality (illustrative) | Cost per typical 8 K-prompt / 500-token answer |
|---|---|---|
| `gemma4:e2b` (lightweight fallback) | 1.0× (baseline) | $0 |
| `gemma4:e4b` | ~1.3× | $0 |
| `gemma4:26b` | ~1.7× | $0 (electricity) |
| `gemma4:31b` (high-end option) | ~1.75× | $0 (electricity) |
| Llama 4 Maverick 400B (hosted) | ~1.95× | ~$0.0026 |
| Gemini 2.5 Flash | ~1.95× | ~$0.0036 |
| Gemini 2.5 Pro | ~2.05× | ~$0.0150 |
| GPT-5 mini (Azure) | ~2.00× | ~$0.0030 |
| GPT-5 (Azure) | ~2.10× | ~$0.0300 |
| Gemini 3 Pro | ~2.10× | ~$0.0285 |
| Claude Sonnet 4.5 (Azure Foundry) | ~2.05× | ~$0.0315 |

Three honest observations:

1. **The local-free → hosted-frontier jump is real but small (~20 %)** for this workload. Most of it is faster and more reliable refusal, slightly better citation discipline, and cleaner formatting — *not* fundamentally better understanding of a single PDF.
2. **The free-OSS-hosted tier (Llama 4, Qwen 3, DeepSeek R1) is within 5 %** of the frontier hosted models on RAG-style tasks, at **5–15× lower cost**. If you need to go off-box but want to keep the bill near zero, this is the layer to use.
3. **Cost per query is small enough that the bottleneck is rarely the LLM bill.** A team running 10 000 RAG queries a month against Gemini 2.5 Pro spends about $150 — much less than the GPU electricity for a 24/7 self-hosted `gemma4:31b`.

---

## 6. Decision matrix

| If you care most about... | Use this |
|---|---|
| **Zero cost, full privacy, runs offline** | `gemma4:e4b` (or `:26b` if hardware allows) — the current stack with Phase A retrieval fixes |
| **Best quality without paying per token, willing to run a server** | Self-hosted Llama 4 Maverick 400B on rented or owned H100s |
| **Best quality per dollar on an API** | Gemini 2.5 Flash *or* GPT-5 mini *or* hosted Llama 4 via Together |
| **Frontier reasoning over very long contexts (whole corpus in one prompt)** | Gemini 2.5 Pro (2 M context) or GPT-4.1 (1 M context) |
| **Enterprise compliance (HIPAA / FedRAMP / IL5)** | Azure OpenAI Service with a Provisioned Throughput Unit, or Vertex AI on a regulated GCP project |
| **Lowest latency for an interactive UI** | Groq-hosted Llama 4 (sub-100 ms first-token) |

---

## 7. What this means for the shipped app

For the *"oven over 500 °F"* failure specifically, the order of operations is unchanged regardless of which generator you pick:

1. **Fix retrieval first** ([01-improving-retrieval-accuracy.md](01-improving-retrieval-accuracy.md)). Without this, even Gemini 3 Pro will say "I could not find that".
2. **Move up the Gemma 4 ladder** ([02-model-upgrades-within-gemma-4.md](02-model-upgrades-within-gemma-4.md)) for the next ~30 % gain at zero marginal cost.
3. **Only then consider hosted models.** When you do, start with **Gemini 2.5 Flash** or **GPT-5 mini** — they are cheap enough to A/B against the local model on every query during a transition period, and either one will close the remaining gap to the frontier for this workload.

---

## 8. Caveats

- All hosted-model prices are list rates and **change frequently**. Volume discounts, committed-use contracts, and provisioned-throughput reservations can move them by 30–60 %.
- Quality numbers in this document are **directional, not measured on this corpus**. The `Recall@k` and faithfulness scores you actually observe depend on your chunks, your queries, and your eval harness.
- "Best free OSS" leaderboards turn over every 2–3 months. Re-check before any platform decision.
- Region availability matters: not every Gemini or Azure model is in every region, and EU / sovereign-cloud SKUs price differently.

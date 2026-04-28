# Further Reading

Curated links to the official documentation, model cards, and community resources for every tool and concept in this documentation set.

---

## Gemma 4

| Resource | URL | What it covers |
|----------|-----|----------------|
| Gemma 4 model card (HF) | https://huggingface.co/google/gemma-4 | Architecture, benchmarks, license (Apache 2.0) |
| Google DeepMind — Gemma 4 | https://deepmind.google/models/gemma/gemma-4/ | Official announcement and technical overview |
| Ollama Gemma 4 library page | https://ollama.com/library/gemma4:e2b | All available tags (e2b, e4b, 26b, 31b) |
| Gemma 4 on Ollama — setup guide | https://meshworld.in/blog/ai/tooling/gemma4:e2b-local-ollama/ | Step-by-step install and hardware guide |
| All Gemma 4 sizes compared | https://gemma4all.com/blog/run-gemma-4-with-ollama | Variant comparison, VRAM requirements |

---

## Ollama

| Resource | URL | What it covers |
|----------|-----|----------------|
| Ollama official site | https://ollama.com | Downloads, documentation, model library |
| Ollama GitHub | https://github.com/ollama/ollama | Source, issues, Modelfile reference |
| Ollama REST API reference | https://github.com/ollama/ollama/blob/main/docs/api.md | All endpoints: `/api/chat`, `/api/embeddings`, `/api/pull` |
| Ollama Python library | https://github.com/ollama/ollama-python | SDK source and examples |
| Ollama FAQ | https://github.com/ollama/ollama/blob/main/docs/faq.md | GPU setup, Windows WSL, environment variables |

---

## ChromaDB

| Resource | URL | What it covers |
|----------|-----|----------------|
| ChromaDB docs | https://docs.trychroma.com | Getting started, Python client, filtering |
| ChromaDB GitHub | https://github.com/chroma-core/chroma | Source, changelog, issues |
| HNSW parameters | https://docs.trychroma.com/guides#changing-the-distance-function | `hnsw:space`, `hnsw:M`, `construction_ef`, `search_ef` |
| ChromaDB collections guide | https://docs.trychroma.com/guides/multimodal | Collections, metadata, filtering operators |

---

## embeddinggemma

| Resource | URL | What it covers |
|----------|-----|----------------|
| Model card (HF) | https://huggingface.co/google/embeddinggemma-300m | Architecture, benchmarks, 2 048-token context, 768-d output |
| Ollama library page | https://ollama.com/library/embeddinggemma | Pull command and tag list |
| Task prompt prefixes | https://ai.google.dev/gemma/docs/embeddinggemma | Official `title:` / `task:` prefix grammar |

---

## Streamlit

| Resource | URL | What it covers |
|----------|-----|----------------|
| Streamlit docs | https://docs.streamlit.io | API reference for all widgets |
| `st.chat_input` / `st.chat_message` | https://docs.streamlit.io/develop/api-reference/chat | Chat UI components used in this app |
| Session state guide | https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state | Persisting state across reruns |
| Streamlit GitHub | https://github.com/streamlit/streamlit | Source, issues, examples |

---

## Hugging Face

| Resource | URL | What it covers |
|----------|-----|----------------|
| HF Model Hub | https://huggingface.co/models | Browse all open models |
| `huggingface_hub` CLI | https://huggingface.co/docs/huggingface_hub/guides/cli | `huggingface-cli download` reference |
| `AutoTokenizer` docs | https://huggingface.co/docs/transformers/main_classes/tokenizer | Loading tokenizers locally |
| Model card guide | https://huggingface.co/docs/hub/model-cards | How to read and write model cards |

---

## RAG Concepts

| Resource | URL | What it covers |
|----------|-----|----------------|
| Original RAG paper (Lewis et al. 2020) | https://arxiv.org/abs/2005.11401 | The foundational RAG paper |
| LangChain text splitters | https://python.langchain.com/docs/how_to/#text-splitters | `RecursiveCharacterTextSplitter` and all variants |
| MTEB Leaderboard | https://huggingface.co/spaces/mteb/leaderboard | Embedding model benchmarks for retrieval tasks |
| RAGAS evaluation framework | https://docs.ragas.io | Automated RAG evaluation (faithfulness, answer relevancy, context recall) |

---

## Python Tooling

| Resource | URL | What it covers |
|----------|-----|----------------|
| `uv` — fast Python package manager | https://docs.astral.sh/uv/ | `uv venv`, `uv pip install` |
| `pypdf` docs | https://pypdf.readthedocs.io | PDF text extraction |
| `python-docx` docs | https://python-docx.readthedocs.io | Word document extraction |
| `httpx` async client | https://www.python-httpx.org | Async HTTP for batched embedding calls |

---

## Community & Learning

| Resource | URL | What it covers |
|----------|-----|----------------|
| Ollama Discord | https://discord.gg/ollama | Community support, model recommendations |
| r/LocalLLaMA | https://reddit.com/r/LocalLLaMA | GPU settings, quantization tips, new model releases |
| Andrej Karpathy — "Let's build GPT" | https://youtu.be/kCc8FmEb1nY | Deep intuition for how transformer LLMs work |
| Simon Willison's blog | https://simonwillison.net | Practical LLM engineering, tool updates |

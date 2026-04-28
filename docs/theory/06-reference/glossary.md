# Glossary

Quick definitions for every term used in the local RAG documentation. Cross-referenced to the doc where each concept is covered in depth.

---

| Term | Definition | See also |
|------|------------|----------|
| **ANN** | Approximate Nearest Neighbour — a search algorithm that finds vectors geometrically close to a query vector, trading a small accuracy loss for much faster search than exact search. | [ChromaDB](../02-ecosystem/chromadb.md) |
| **Apache 2.0** | Open-source license permitting commercial use, modification, and redistribution with attribution. Gemma 4 uses this license. | [Gemma 4 Models](../02-ecosystem/gemma-models.md) |
| **Agentic** | A pattern where a model can plan multi-step tasks, call tools, and use the results in subsequent reasoning steps. Gemma 4 has native function-calling support. | [Gemma 4 Models](../02-ecosystem/gemma-models.md) |
| **BPE** | Byte-Pair Encoding — a tokenization algorithm that merges frequent character pairs into sub-word tokens iteratively. Used by most modern LLM tokenizers. | [Tokens & Embeddings](../01-foundations/tokens-and-embeddings.md) |
| **Chunk** | A fixed-size fragment of text produced by splitting a document. Each chunk is independently embedded and stored in ChromaDB. | [Chunking Strategies](../01-foundations/chunking-strategies.md) |
| **Chunk overlap** | The number of tokens shared between adjacent chunks. Prevents information at chunk boundaries from being lost. | [Chunking Strategies](../01-foundations/chunking-strategies.md) |
| **ChromaDB** | An open-source embedded vector database that stores text chunks alongside their embedding vectors and supports ANN search. | [ChromaDB](../02-ecosystem/chromadb.md) |
| **Collection** | A named namespace inside ChromaDB that groups related documents and their embeddings. Analogous to a table in a relational DB. | [ChromaDB](../02-ecosystem/chromadb.md) |
| **Context window** | The maximum number of tokens a language model can process in a single inference call (input + output combined). Gemma 4 E2B/E4B: 128K tokens; 26B/31B: 256K tokens. | [Prompting & Temperature](../01-foundations/prompting-and-temperature.md) |
| **Cosine similarity** | A measure of the angle between two vectors. Range −1 to +1; higher is more similar. Default distance metric for ChromaDB text collections. | [Tokens & Embeddings](../01-foundations/tokens-and-embeddings.md) |
| **Dense retrieval** | Retrieval using dense embedding vectors and ANN search, as opposed to sparse keyword search (BM25/TF-IDF). | [What is RAG?](../00-overview/what-is-rag.md) |
| **Embedding** | A fixed-length vector of floating-point numbers that encodes the semantic meaning of a piece of text. Similar texts produce geometrically similar vectors. | [Tokens & Embeddings](../01-foundations/tokens-and-embeddings.md) |
| **Embedding model** | A model trained specifically to produce high-quality embeddings for retrieval tasks. In this app: `embeddinggemma`. Different from an instruction-following generative model. | [Gemma 4 Models](../02-ecosystem/gemma-models.md) |
| **Faithfulness** | An evaluation metric measuring whether every claim in a generated answer is supported by the retrieved context. Scored 1–5 in this app's eval harness. | [Evaluating RAG](../05-operations/evaluating-rag.md) |
| **Fine-tuning** | Training a pre-existing model on new data to update its weights. More expensive and less flexible than RAG for adding new knowledge. | [What is RAG?](../00-overview/what-is-rag.md) |
| **Gemma 4** | Google DeepMind's fourth-generation open-weight language model family, released April 2026. Multimodal, Apache 2.0, available via Ollama. | [Gemma 4 Models](../02-ecosystem/gemma-models.md) |
| **GGUF** | A binary file format for quantized LLM weights, used by llama.cpp and Ollama. Enables fast CPU and GPU inference without the full PyTorch stack. | [Gemma 4 Models](../02-ecosystem/gemma-models.md) |
| **Grounding** | Constraining a language model's output to facts present in a provided context, reducing hallucination. The core goal of RAG's system prompt. | [Prompting & Temperature](../01-foundations/prompting-and-temperature.md) |
| **Hallucination** | When a language model generates confident-sounding but factually incorrect or unsupported statements. RAG mitigates this by grounding answers in retrieved text. | [What is RAG?](../00-overview/what-is-rag.md) |
| **HNSW** | Hierarchical Navigable Small World — the graph-based ANN index algorithm used by ChromaDB for fast vector search. | [ChromaDB](../02-ecosystem/chromadb.md) |
| **Hugging Face (HF)** | A platform hosting open-source model weights, datasets, and tooling. Used in this app for tokenizer configs only. | [Hugging Face](../02-ecosystem/hugging-face.md) |
| **Inference** | Running a trained model to generate output from a given input. In this app, handled by Ollama calling Gemma 4. | [Ollama](../02-ecosystem/ollama.md) |
| **Ingestion** | The pipeline of loading, parsing, chunking, embedding, and storing documents into ChromaDB. | [Ingestion Pipeline](../04-build-the-app/02-ingestion-pipeline.md) |
| **LLM** | Large Language Model — a neural network trained on massive text corpora to predict and generate text. Gemma 4 is an LLM. | [What is RAG?](../00-overview/what-is-rag.md) |
| **Mean pooling** | Averaging the hidden-state vectors of all tokens in a sequence to produce a single sentence-level embedding vector. Used by `embeddinggemma`. | [Tokens & Embeddings](../01-foundations/tokens-and-embeddings.md) |
| **Metadata** | Structured key-value pairs stored alongside each chunk in ChromaDB (e.g., source filename, page number, chunk index). Used for filtering and citation. | [ChromaDB](../02-ecosystem/chromadb.md) |
| **MoE** | Mixture of Experts — an architecture where only a subset of the model's parameters are active per forward pass. Gemma 4's 26B variant is MoE: 26B total params, ~3.8B active. | [Gemma 4 Models](../02-ecosystem/gemma-models.md) |
| **Model card** | A README hosted on Hugging Face describing a model's architecture, training data, intended use, limitations, and license. | [Hugging Face](../02-ecosystem/hugging-face.md) |
| **embeddinggemma** | A 768-dimensional text embedding model by Nomic AI, available on Ollama. The primary embedding model used in this app. | [Gemma 4 Models](../02-ecosystem/gemma-models.md) |
| **Nucleus sampling** | See `top_p`. | [Prompting & Temperature](../01-foundations/prompting-and-temperature.md) |
| **Ollama** | A Go binary that downloads, manages, and serves GGUF models via a local REST API on port `11434`. | [Ollama](../02-ecosystem/ollama.md) |
| **Overlap** | See *Chunk overlap*. | |
| **Persistent client** | A `chromadb.PersistentClient` that writes all changes to disk immediately, so data survives process restarts. | [ChromaDB](../02-ecosystem/chromadb.md) |
| **Prompt** | The input text (and structured messages) passed to an LLM. In RAG, the prompt includes a system message with retrieved context, chat history, and the user's question. | [Prompting & Temperature](../01-foundations/prompting-and-temperature.md) |
| **Q4_K_M** | A 4-bit quantization variant with a mixed (K-quant + M-type) scheme. The default quantization used by Ollama's Gemma 4 pull. | [Performance Tuning](../05-operations/performance-tuning.md) |
| **Quantization** | Reducing model weight precision (e.g., from 16-bit floats to 4-bit integers) to shrink memory use and increase inference speed, with a small quality trade-off. | [Performance Tuning](../05-operations/performance-tuning.md) |
| **RAG** | Retrieval-Augmented Generation — a technique that augments an LLM's answer with text retrieved from an external knowledge store at query time. | [What is RAG?](../00-overview/what-is-rag.md) |
| **Recursive character splitter** | A text splitting algorithm that tries to split on paragraph breaks first, then sentences, then words, to produce chunks as semantically coherent as possible. | [Chunking Strategies](../01-foundations/chunking-strategies.md) |
| **Retrieval@k** | The fraction of test queries where the expected source document appears in the top-k retrieved chunks. | [Evaluating RAG](../05-operations/evaluating-rag.md) |
| **Similarity search** | Finding the vectors in a database that are geometrically closest to a query vector. ChromaDB uses HNSW for this. | [ChromaDB](../02-ecosystem/chromadb.md) |
| **Streamlit** | A Python framework for building interactive web UIs. In this app it provides the file uploader, Embed button, ChromaDB browser, and chat interface. | [Streamlit UI](../04-build-the-app/04-streamlit-ui.md) |
| **System prompt** | A special message prepended to the conversation that instructs the model how to behave. In RAG, the system prompt injects retrieved context and instructs the model to stay grounded. | [Prompting & Temperature](../01-foundations/prompting-and-temperature.md) |
| **Temperature** | A sampling parameter that controls output randomness. Low (0.0–0.2) → deterministic; high (> 1.0) → creative/random. | [Prompting & Temperature](../01-foundations/prompting-and-temperature.md) |
| **Token** | The basic unit of text that an LLM processes. One token ≈ 4 English characters. Not equal to one word. | [Tokens & Embeddings](../01-foundations/tokens-and-embeddings.md) |
| **Tokenizer** | An algorithm (e.g., BPE) that splits raw text into tokens and maps them to integer IDs from a fixed vocabulary. | [Tokens & Embeddings](../01-foundations/tokens-and-embeddings.md) |
| **top_k** | (1) In sampling: keep only the top-k highest-probability tokens before sampling. (2) In retrieval: the number of chunks returned by ChromaDB. | [Prompting & Temperature](../01-foundations/prompting-and-temperature.md) |
| **top_p** | Nucleus sampling: keep the smallest set of tokens whose cumulative probability ≥ `top_p`. Adapts the candidate set to the distribution shape. | [Prompting & Temperature](../01-foundations/prompting-and-temperature.md) |
| **Vector** | A list of numbers (floats) representing a point in high-dimensional space. Embeddings are vectors. | [Tokens & Embeddings](../01-foundations/tokens-and-embeddings.md) |
| **Vector database** | A database optimised for storing and searching high-dimensional vectors using ANN algorithms. ChromaDB is the vector database used in this app. | [ChromaDB](../02-ecosystem/chromadb.md) |
| **VRAM** | Video RAM — memory on a GPU. Larger VRAM lets you run bigger Gemma 4 variants or avoid quantization. | [Performance Tuning](../05-operations/performance-tuning.md) |

---
title: "Embeddings — Ariadne"
description: "Use any embedding model with Ariadne. Recommended: sentence-transformers all-MiniLM-L6-v2 for 384-dim vectors."
---


Ariadne is model-agnostic — any embedding model that produces a fixed-dimensional vector works. Here's how to choose, configure, and use embeddings.

## Automatic embedding (recommended)

Pass an `embedder` to `AriadneMemory` and it embeds text for you: `remember()`
embeds content and `recall()` embeds the query, so you get semantic + hybrid
search without managing vectors by hand.

```bash
pip install "arriadne[embeddings]"
```

```python
from arriadne import AriadneMemory
from arriadne.embeddings import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")  # 384-dim

mem = AriadneMemory(db_path="memory.db", embedding_dim=embedder.dim, embedder=embedder)

mem.remember("User prefers dark mode in all applications", importance=0.8)
results = mem.recall("what are the user's UI preferences?", k=5)  # embedded automatically
```

`embedder` accepts a `SentenceTransformerEmbedder`, any callable
`str -> list[float]` that exposes a `.dim` attribute, or a model-name string.
Ariadne checks `embedder.dim` against `embedding_dim` and raises if they differ.

The rest of this guide covers picking a model and the manual path (passing
vectors yourself), which is still fully supported.

## Recommended Models

| Model | Dimensions | Speed | Quality | Best For |
|-------|-----------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | ⚡ Fast | ★★★☆ | Default, general-purpose |
| `all-mpnet-base-v2` | 768 | 🔵 Medium | ★★★★ | Higher quality search |
| `BAAI/bge-small-en-v1.5` | 384 | ⚡ Fast | ★★★★ | Best speed/quality ratio |
| `BAAI/bge-base-en-v1.5` | 768 | 🔵 Medium | ★★★★★ | Maximum quality |
| `nomic-embed-text-v1` | 768 | 🔵 Medium | ★★★★★ | Matryoshka support |
| `mixedbread-ai/mxbai-embed-large-v1` | 1024 | 🟡 Slow | ★★★★★ | Maximum accuracy |

## Using with sentence-transformers

Install the optional dependency:

```bash
pip install sentence-transformers
```

### Basic Usage

```python
from sentence_transformers import SentenceTransformer
from arriadne import AriadneMemory

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Ariadne with matching dimension
mem = AriadneMemory(db_path="memory.db", embedding_dim=384)

# Generate embedding and store
text = "User prefers dark mode in all applications"
embedding = model.encode(text).tolist()

result = mem.remember(
    content=text,
    memory_type="semantic",
    importance=0.8,
    embedding=embedding,
)
```

### Search with Embeddings

```python
# Generate query embedding
query = "what are the user's UI preferences?"
query_embedding = model.encode(query).tolist()

# Hybrid search (vector + FTS)
results = mem.recall(
    query=query,
    embedding=query_embedding,
    k=5,
)
```

### Batch Operations

```python
texts = [
    "Python is a high-level language",
    "JavaScript runs in the browser",
    "Go is compiled and statically typed",
]

# Batch encode
embeddings = model.encode(texts)

# Store all
for text, emb in zip(texts, embeddings):
    mem.remember(
        content=text,
        embedding=emb.tolist(),
        importance=0.7,
    )
```

## Matryoshka Embeddings

Some models (like `nomic-embed-text-v1`) support **Matryoshka representation learning** — you can truncate the embedding to a lower dimension while retaining most of the information.

```python
from sentence_transformers import SentenceTransformer

# Load a Matryoshka model
model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)

# Full dimension (768)
full_embedding = model.encode("Hello world")

# Truncated dimensions — still useful!
emb_384 = full_embedding[:384]   # 50% of dimensions
emb_256 = full_embedding[:256]   # 33% of dimensions
emb_128 = full_embedding[:128]   # 17% of dimensions
```

### When to Use Matryoshka

| Dimension | Accuracy | Storage | Speed |
|-----------|----------|---------|-------|
| 768 (full) | 100% | 3.0 KB | Baseline |
| 384 | ~97% | 1.5 KB | 1.5x faster |
| 256 | ~93% | 1.0 KB | 2x faster |
| 128 | ~85% | 0.5 KB | 3x faster |

Use truncated dimensions when:
- Storage is constrained
- Search speed is critical
- You have >100K memories

```python
# Store with 384-dim Matryoshka truncation
config = AriadneConfig(db_path="memory.db", embedding_dim=384)
mem = AriadneMemory(config=config)

embedding = model.encode("some text")[:384].tolist()
mem.remember(content="some text", embedding=embedding)
```

## Quantization Options

For edge deployment or memory-constrained environments, quantize embeddings:

```python
import numpy as np

embedding = model.encode("text")  # float32

# Quantize to float16 (50% storage reduction)
embedding_f16 = embedding.astype(np.float16)

# Quantize to int8 (75% storage reduction)
embedding_int8 = np.clip(embedding * 127, -128, 127).astype(np.int8)

# Note: Ariadne stores embeddings as float32 BLOBs internally.
# Quantization is useful for external storage or transfer.
```

## Choosing the Right Dimension

| Dataset Size | Recommended Dimension | Model |
|-------------|----------------------|-------|
| < 1K memories | 384 | `all-MiniLM-L6-v2` |
| 1K–10K memories | 384 | `BAAI/bge-small-en-v1.5` |
| 10K–100K memories | 384–768 | `BAAI/bge-base-en-v1.5` |
| >100K memories | 768 | `BAAI/bge-base-en-v1.5` or `nomic-embed-text-v1` |

## Custom Embedding Functions

If you're not using sentence-transformers, provide any function that returns a list of floats:

```python
import numpy as np

def my_embedding_fn(text: str) -> list[float]:
    """Custom embedding function."""
    # Could call an API, use a different library, etc.
    # Just return a list of floats with the right dimension
    return np.random.randn(384).tolist()

# Use with Ariadne
embedding = my_embedding_fn("User prefers dark mode")
result = mem.remember(
    content="User prefers dark mode",
    embedding=embedding,
    importance=0.8,
)
```

## Configuration Reference

```python
from arriadne import AriadneConfig

config = AriadneConfig(
    embedding_dim=384,          # Must match your model's output dimension
    faiss_type="auto",          # Auto-select FlatIP/IVFFlat
    ivf_threshold=50_000,       # auto: upgrade to IVFFlat at this count
    ivf_nlist=128,              # Voronoi cells (effective nlist = min(ivf_nlist, sqrt(n)))
)
```

::: warning
The `embedding_dim` must match your model's output dimension exactly. A mismatch will cause FAISS index errors.
:::


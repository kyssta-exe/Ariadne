# Core Memory, Reranking & Async

The second tier of Ariadne's memory model, borrowed from the systems that
pioneered it: **core memory blocks** (Letta/MemGPT), **cross-encoder
reranking** (Mem0/Zep), **semantic deduplication**, **entity resolution**
(Graphiti), and an **async facade** for asyncio-first frameworks.

## Core memory blocks

Long-term memory answers "what do I know about X?". Core memory is different:
a small set of named blocks that are **always in context** — persona, user
profile, project state — and that the agent edits itself during a run.

Ariadne keeps them in their own table, deliberately outside the memory
lifecycle: no dedup, no decay, no eviction, no consolidation. A core block is
working state, not a fact to be retrieved.

```python
from arriadne import AriadneMemory

mem = AriadneMemory(db_path="memory.db")

mem.core_set("persona", "You are Ariadne, a careful, concise agent.")
mem.core_append("user_profile", "Name: Kyssta. ")
mem.core_append("user_profile", "Prefers direct answers. ")
mem.core_get("user_profile")     # → block dict with content
mem.core_blocks()                # → all blocks (ordered by name)
mem.core_delete("scratchpad")
```

### Bounds and rendering

`core_append` is bounded by `core_block_char_limit` (default 10,000 chars):
when an append would overflow, the **oldest** content is trimmed so recent
observations survive a runaway loop.

`core_pack()` renders the non-empty blocks as a compact section:

```python
mem.core_pack()
# Core memory:
# ### persona
# You are Ariadne, a careful, concise agent.
# ### user_profile
# Name: Kyssta. Prefers direct answers.
```

`context_pack(include_core=True)` prepends it under the token budget
(together with `include_sessions=True`, a full "who am I / what happened
recently / what matches this query" prompt block).

### Agent surfaces

- **MCP**: `ariadne_core_view`, `ariadne_core_append`, `ariadne_core_replace`
- **Hermes plugin**: same three tools; blocks are also prepended to *every*
  turn's prefetch (that is the point of core memory)
- **Dashboard**: `GET/PUT/DELETE /api/core-blocks`

## Reranking

Hybrid recall fuses BM25 and vector ranks with RRF — cheap and robust, but
both signals score the query and document *separately*. A cross-encoder reads
the pair jointly and orders the top-k much better. It is the standard second
stage in Mem0 and Zep; Ariadne makes it a per-query opt-in:

```python
mem = AriadneMemory(db_path="memory.db", reranker=CrossEncoderReranker())
# or just: mem.recall("...", rerank=True) — loads the default model lazily

results = mem.recall("how to deploy to production", k=5, rerank=True)
results[0]["score_parts"]
# {'fused': 0.0143, 'rerank': 8.93, ...} — explainability survives
```

The default model is `cross-encoder/ms-marco-MiniLM-L-6-v2`
(`AriadneConfig.rerank_model`). Requires the `embeddings` extra; when it is
not installed, `rerank=True` logs a warning and returns the fused order —
the dependency stays optional.

## Semantic deduplication

MinHash dedup is lexical: "I live in Paris" and "Paris is my home" share few
shingles. When an embedder is configured, `remember()` additionally checks
the nearest stored vector and treats cosine ≥ `semantic_dedup_threshold`
(default 0.92) as a duplicate — and reinforces the existing memory's trust,
because a restatement is mild confirmation:

```python
mem.remember("I live in Paris")          # created
mem.remember("Paris is my home")         # duplicate (semantic_duplicate=True)
```

Disable with `AriadneConfig(semantic_dedup=False)`.

## Entity resolution

Abbreviations fragment the knowledge graph ("PG" vs "postgres"). Merge the
alias into the canonical entity — memory links and both edge directions are
re-pointed, the alias row is removed:

```python
moved = mem.merge_entities("pg", "postgres")
```

## Async API

`AriadneMemory` is synchronous by design (everything is in-process), but
asyncio frameworks must not call blocking code on the event loop.
`AsyncAriadneMemory` mirrors the hot path as coroutines via
`asyncio.to_thread`; the underlying store's RLock serializes access, so one
instance safely backs many concurrent coroutines:

```python
import asyncio
from arriadne import AsyncAriadneMemory

async def main():
    async with AsyncAriadneMemory(db_path="memory.db") as mem:
        await mem.remember("Async agents need memory too")
        hits = await asyncio.gather(
            mem.recall("async memory", k=3),
            mem.search_episodes("async memory", k=3),
        )

asyncio.run(main())
```

Wrap an existing instance with `AsyncAriadneMemory.from_memory(sync_mem)`;
the synchronous object is always available as `.sync`.

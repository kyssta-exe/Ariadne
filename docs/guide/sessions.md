# Session Intelligence & Trust

Ariadne records raw conversation turns as immutable **episodes**, and treats
them as a first-class recall surface alongside distilled memories. On top of
that sit two mechanisms that keep long-running agents coherent across
sessions: deterministic **session digests** and dynamic **trust scoring**.

## Episodes vs. memories

| | Memories | Episodes |
|---|---|---|
| What | distilled facts ("auth bug was token expiry") | raw turns as they happened |
| Mutability | updated, superseded, decayed | immutable |
| Searched by | `recall()` (hybrid) | `search_episodes()` (BM25) |
| Purpose | answer future questions | recover detail extraction dropped |

`LLMMemoryManager.process_turn()` records each turn as an episode
automatically; `record_episode()` does it directly.

## Searching raw history

```python
from arriadne import AriadneMemory

mem = AriadneMemory(db_path="memory.db")
mem.process_turn("The auth bug was token expiry", "Fixed by refreshing tokens.")

hits = mem.search_episodes("auth bug token", k=5)
# → the actual turns, with role, session id, and timestamps

mem.list_sessions()          # → [{session_id, turns, first_event_at, last_event_at}]
```

The full-text index over episodes (`episodes_fts`) is created and backfilled
automatically when an older database is opened — no migration step.

## Session digests

`digest_session()` distills a session into one compact memory —
deterministically, with no LLM. Turns are scored by role weight (assistant
statements carry decisions) and by how much they mention the session's
recurring terms (files, errors, libraries); the top turns are kept in
chronological order and stored with `kind=session_digest` metadata plus
provenance links back to the selected episodes.

```python
result = mem.digest_session("session-42")
# {'status': 'created', 'memory_id': 7, 'episodes': 5, 'digest': '...'}

result = mem.digest_session("session-42")          # → status 'exists' (idempotent)
result = mem.digest_session("session-42", force=True)  # supersede + re-digest
```

## Cross-session continuity

`session_context()` renders the most recent digests as a compact block, and
`context_pack(include_sessions=True)` prepends it under the token budget:

```python
block = mem.session_context(namespace="default", max_sessions=3)
packed = mem.context_pack("deploy pipeline", token_budget=800, include_sessions=True)
```

The Hermes plugin injects this automatically on the first turn of each
session; the MCP server exposes it as `ariadne_session_context`.

## Trust scoring

Confidence is dynamic. A new write that **contradicts** a stored memory decays
that memory's confidence (default −0.1 per contradiction); explicit
reinforcement and curation winners gain it back. Because hybrid retrieval
re-weights by confidence, contested memories sink in ranking instead of being
silently deleted:

```python
mem.remember("The API port is 8080")
mem.remember("The API port is not 8080, it is 9090")   # old fact: 1.0 → 0.9

mem.reinforce(memory_id)          # confirmed useful → +0.1 (capped at 1.0)
```

Knobs: `trust_contradiction_penalty`, `trust_reinforce_delta`. Set the
penalty to `0.0` to disable decay entirely.

## The update policy (mem0-style)

When `LLMMemoryManager.process_turn()` writes an extracted memory, each
candidate is resolved against the closest stored memory with a deterministic
decision:

- **ADD** — nothing similar stored, or similar but additive
- **UPDATE** — contradicts the stored memory → `supersede()` (history preserved)
- **NOOP** — near-duplicate (lexical similarity ≥ 0.75)
- **DELETE** — only when an explicit decision is passed (LLM-judge pipelines)

```python
mgr = LLMMemoryManager(mem, caller=my_llm)
summary = mgr.process_turn(user, assistant)
summary["policy"]   # [{'content': ..., 'operation': 'UPDATE', 'target_id': 3, ...}]
```

KV facts (`subject.attribute = value`) bypass lexical matching and upsert
precisely through `set_fact()`.

## Entity-graph expansion

`expand()` widens a recall result through the entity graph — memories sharing
entities with the hits join at a decayed score, so associations never outrank
direct matches:

```python
results = mem.recall("user data", k=3)
expanded = mem.expand(results, hops=1, limit=10, decay=0.5)
```

## Configuration

```python
from arriadne import AriadneConfig

cfg = AriadneConfig.from_env()   # reads ARIADNE_* variables, strict validation
```

| Variable | Field |
|---|---|
| `ARIADNE_DB_PATH` | `db_path` |
| `ARIADNE_EMBEDDING_DIM` | `embedding_dim` |
| `ARIADNE_FAISS_TYPE` | `faiss_type` |
| `ARIADNE_DEDUP_THRESHOLD` | `dedup_threshold` |
| `ARIADNE_RETENTION_HALF_LIFE` | `retention_half_life` |
| `ARIADNE_MAINTENANCE_INTERVAL` | `maintenance_interval` |
| `ARIADNE_TRUST_CONTRADICTION_PENALTY` | `trust_contradiction_penalty` |
| `ARIADNE_TRUST_REINFORCE_DELTA` | `trust_reinforce_delta` |

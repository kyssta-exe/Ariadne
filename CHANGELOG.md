# Changelog

All notable changes to Ariadne will be documented in this file.

## [0.13.0] - 2026-08-22

The "session intelligence & trust" release, informed by a competitive pass over
[ctx-memory](https://github.com/GhadiSaab/ctx-memory) (session lifecycle),
[Mem0](https://docs.mem0.ai/core-concepts/memory-operations) (update policies),
and holographic/HRR-style memory systems (trust scoring).

### Added
- **Session intelligence** — agents can now search and digest raw history:
  - `search_episodes(query, k, namespace, session_id)`: BM25 full-text search over recorded episode turns (new `episodes_fts` FTS5 table, kept in sync by triggers and backfilled for existing databases on open). The "what did I try last session?" surface, complementing distilled-memory `recall`.
  - `list_sessions()`: session summaries with turn counts and time ranges.
  - `digest_session(session_id)`: deterministic, LLM-free session digests. Turns are scored by role weight and recurring-term salience; the digest is stored as a memory with `kind=session_digest` metadata and provenance links back to the selected episodes. Idempotent (re-digesting returns the stored digest; `force=True` supersedes it).
  - `session_context()`: compact "recent session context" block from stored digests; also `context_pack(include_sessions=True)` prepends it, and the Hermes plugin auto-injects it on the first turn of each session.
- **Trust scoring** — confidence is now dynamic instead of a static write-time guess:
  - `remember()` applies a small confidence penalty (`trust_contradiction_penalty`, default 0.1) to stored memories contradicted by a new write.
  - `reinforce(memory_id, delta)`: explicit confirmation raises trust; `MemoryCurator` automatically reinforces contradiction winners.
  - All flows through the new `AriadneDB.adjust_confidence()` primitive.
- **mem0-style update policy** — `LLMMemoryManager` now resolves writes against existing memories with a deterministic ADD / UPDATE / NOOP decision (`decide_update_policy()`, `apply_policy()`): near-duplicates are skipped, contradictions supersede (history preserved), novel facts are added. `DELETE` is available for explicit LLM-judge pipelines. `process_turn()` logs every decision in its summary; KV facts keep their precise subject/attribute upsert path.
- **Entity-graph expansion** — `expand(results, hops, limit, decay)` surfaces memories sharing entities with recall hits, scored as a decaying fraction of the best seed score so direct hits always outrank associations (backed by `AriadneDB.expand_by_entities()`).
- **Core memory blocks (Letta/MemGPT-style)** — named, always-in-context working-state blocks (persona, user profile, project state) with `core_set` / `core_get` / `core_append` / `core_delete` / `core_blocks` / `core_pack`. Stored in a dedicated `core_memory_blocks` table, deliberately outside the memory lifecycle (no dedup/decay/eviction). `context_pack(include_core=True)` prepends them; MCP exposes `ariadne_core_view` / `ariadne_core_append` / `ariadne_core_replace`; the Hermes plugin injects them into every turn; the dashboard gets `GET/PUT/DELETE /api/core-blocks`. Appends are bounded by `core_block_char_limit` (oldest content trimmed on overflow).
- **Cross-encoder reranking** — `recall(..., rerank=True)` and `ariadne_recall(rerank=true)` add a second retrieval stage (`arriadne.rerank.CrossEncoderReranker`, default `ms-marco-MiniLM-L-6-v2`, lazily loaded). Fused scores are preserved in `score_parts["fused"]` so explainability survives; gracefully degrades to fused order when sentence-transformers is not installed.
- **Semantic (paraphrase) deduplication** — when an embedder is configured, `remember()` checks the nearest stored vector and treats cosine ≥ `semantic_dedup_threshold` (default 0.92) as a duplicate, reinforcing the existing memory instead of storing a paraphrase (`semantic_duplicate` flag on the result). Catches what lexical MinHash cannot.
- **Async API** — `arriadne.async_api.AsyncAriadneMemory` (also exported from the package root) mirrors the hot path (remember/recall/search/digest/core blocks/context packs) as coroutines via `asyncio.to_thread`, so asyncio-first agent frameworks never block the event loop on SQLite/FAISS work.
- **Entity resolution** — `merge_entities(source, target)` re-points memory-entity links and both directions of graph edges onto the canonical entity and removes the alias, repairing graph fragmentation from abbreviation variants (Zep/Graphiti-style hygiene).
- Dashboard: `GET /api/sessions` (list or `?q=` episode search) and `POST /api/sessions/digest`.
- **Fast-as-possible lifecycle** — measured and fixed the two costs that hurt self-hosters:
  - **Restart: FAISS + dedup sidecar indexes.** The index is serialized to `<db>.faiss` / `<db>.dedup.pkl` on close and reloaded on open when a fingerprint matches the database (any drift — external edits, corruption, dimension change — silently falls back to the correct full rebuild). Warm restart at 20k memories: **13.6 s → ~0.83 s (16×)**.
  - **Write throughput:** SQLite page cache raised to a configurable 64 MB (`cache_mb`; the 2 MB default thrashes once embeddings + 19 indexes + FTS compete — measured 4.3 → 0.7 ms/write in isolation), WAL `synchronous=NORMAL` (`synchronous` knob), tiered auto-maintenance (`heavy_maintenance_factor`: consolidation/eviction every 10th cycle — its cost grows with store size and was capping bulk ingest at 23 ms/write @20k), MinHash permutations shared via a template copy, and the hybrid-fusion third row fetch eliminated (input lists already carry full rows). Bulk ingest @20k: **23.3 → 3.9 ms/write (5.9×)**; single writes 9.2 → 4.9 ms.
  - Search stays flat with size (hybrid ≈3 ms, full `recall()` ≈3.8 ms @20k). Reproduce with `scripts/perf_scale.py`.
- **Accuracy evaluation harness** — `benchmarks/accuracy_eval.py`: deterministic synthetic corpus (unique subjects, paraphrase questions, temporal supersessions, distractors, cross-namespace isolation). Measured: FTS-only **0.950 recall@5** (exact 1.000, temporal 1.000, zero namespace leaks); hybrid with real all-MiniLM-L6-v2 embeddings **0.948** (exact 1.000, paraphrase 0.895, temporal 1.000); hybrid path additionally validated with a dependency-free hashing embedder (0.825) so CI covers it without model downloads. Runs as a CI regression floor via `tests/test_accuracy_eval.py`.
- **Composable retrievers** — `arriadne.retrievers`: `Retriever`/`Transformer` protocols with `FTSRetriever`, `VectorRetriever`, `HybridRetriever` first stages and `RerankRetriever`/`ExpansionRetriever` decorators composable through `Pipeline` (the last P0 from the internal audit).
- **`ariadne doctor`** — self-hoster diagnostics: SQLite integrity check, FAISS-count vs DB drift, FTS coverage gaps, orphaned provenance rows, sidecar presence. Exit code 1 on failure; CI-friendly.
- `AriadneConfig.from_env()` — build configuration from `ARIADNE_*` environment variables (db path, embedding dim, faiss type, dedup threshold, retention half-life, maintenance interval, trust knobs), with strict validation instead of silent fallbacks.
- New config knobs: `trust_contradiction_penalty`, `trust_reinforce_delta`, `session_digest_max_turns`, `maintenance_interval`.
- CLI: `ariadne sessions [query]` (search raw history or list sessions) and `ariadne digest <session>` (create a digest).
- MCP server: three new tools — `ariadne_search_sessions`, `ariadne_digest_session`, `ariadne_session_context`.
- Hermes plugin: `ariadne_search_sessions`, `ariadne_digest_session`, `ariadne_session_context` tools (23 total); `sync_turn` now records raw-turn episodes in the project namespace for future digests.

### Fixed
- **LangGraph adapter updated to langgraph 1.x**: `BaseItem` was removed upstream, which silently disabled `AriadneStore` with modern langgraph. The adapter now implements the 1.x `BaseStore` protocol (including the new abstract `batch`/`abatch`), returns `Item`-shaped dicts, makes `put()` a true upsert (same-key writes supersede instead of accumulating duplicates), implements `list_namespaces()` from real data, and falls back to real recent-memory listing for queryless `search()`.
- **Contradiction detector upgraded**: subjects now match up to 4 words (was 2, so "the cache ttl is..." never parsed); clause splitting handles commas without preceding whitespace and no longer breaks decimals ("3.14", "1,000"); same-shape numeric value conflicts ("ttl is 30 seconds" vs "ttl is 60 seconds") are now detected as contradictions.
- Retrieval performance: hybrid/FTS/vector search no longer issue one `get_memory` query per candidate (batch `IN` fetches), namespace-filtered vector search widens its candidate window geometrically instead of always scanning the whole index, and `recall()` attaches provenance with 2 bulk queries instead of 3×k singletons. Hybrid search ≈30% faster and FTS ≈17% faster at 1k memories (see `scripts/perf_probe.py`).
- Single `remember()` writes now count toward auto-maintenance (previously only the bulk path did).
- Lint/type stability: ruff now runs with an explicit, pinned rule set (`E, W, F, I, UP, B, RUF` + documented ignores) so results stop drifting with ruff releases; the whole tree is clean under current ruff 0.16 and mypy strict with the integration extras installed (CI now installs `.[dev,integrations]` in the lint job for this).
- Version manifests were out of sync (0.10.1–0.12.1 across files); all now read 0.13.0.
- Benchmark harness (`benchmarks/run_benchmarks.py`) imported the Unix-only `resource` module unconditionally and crashed on Windows; it now degrades to reporting 0.0 peak RSS there.
- Removed the dead, never-mounted `dashboard/routes.py` router (the SPA shell route lives in `server.py`); `arriadne.dashboard` now exposes `create_app` lazily so importing the package does not require fastapi.

## [Unreleased]

### Added
- **Autonomous memory layer** — `LLMMemoryManager` turns raw conversation turns into structured memories, facts, and knowledge-graph relations via an LLM caller (with a dependency-free fallback). Includes `process_turn()`, `extract()`, and `set_fact()` (KV upsert with provenance).
- **Memory Curator** — `MemoryCurator` runs the retention/hygiene cycle: time-based decay of stale low-importance memories, contradiction detection + supersede, and consolidation orchestration. Ships as a discoverable `CuratorAddon`.
- **MCP server** — dependency-free JSON-RPC 2.0 stdio server exposing `ariadne_recall`, `ariadne_remember`, `ariadne_forget`, and `ariadne_stats` tools to any MCP host (Claude, Cursor, VS Code Continue, etc.).
- **Framework adapters** — `AriadneStore` (LangGraph `BaseStore`), `AriadneTools` (OpenAI Agents SDK `function_tool` wrapper), both import-guarded so core stays dependency-light.
- **CLI** — new `ariadne list` (recent memories with `--type`/`--namespace`/`--limit`) and `ariadne purge` (permanently delete soft-deleted rows, `--older` to keep recent rows recoverable).
- Optional dependency groups in `pyproject.toml`: `[langgraph]`, `[openai-agents]`, `[integrations]`.

### Fixed
- `AriadneStorage.add_episode` defaulted `event_at=None` into an `INSERT` against a `NOT NULL` column, crashing every `record_episode()` call that omitted an explicit timestamp. Now defaults to `now()`; explicit timestamps still flow through.
- Removed a dead duplicate `recall()` definition in `AriadneMemory` (the first was fully shadowed by the temporal/`as_of` version).
- Dashboard `GET /api/health-report` referenced `mem._dedup` and a non-existent `dedup_hits` counter (both `AttributeError`). Now aggregates `mem._dedup_by_namespace` sizes.
- Graph edges silently accumulated duplicates because `add_edge` used `INSERT OR IGNORE` without a unique constraint. Added `idx_edges_uniq` on `(source_id, target_id, edge_type)` and switched to `ON CONFLICT DO UPDATE SET weight` (latest weight wins).
- Strengthened two `test_edge_cases.py` tests that were weakened by stale comments about an old indentation bug; removed leftover dead references.
- Excluded `tests/test_plugin.py` from the default pytest run (it requires an external Hermes `agent` module not present in the repo).

### Changed
- `ariadne.__init__` now exports `LLMMemoryManager`, `ExtractionResult`, `ExtractedMemory`, `ExtractedRelation`, `MemoryCurator`, `CurateReport`.
- Ruff-clean across all new modules.

## [0.12.1] - 2026-06-30

### Fixed
- CI mypy configuration now targets Python 3.12 to match the GitHub Actions runtime and installed NumPy stubs.

## [0.12.0] - 2026-06-30

### Added
- First-class memory namespace and scope fields for local/project/session isolation.
- Namespace-aware exact duplicate detection, MinHash near-duplicate detection, FTS search, vector search, and hybrid search.
- Storage metadata fields for `user_id`, `agent_id`, `session_id`, and `project_id`.
- In-place SQLite migrations and indexes for namespace/scope metadata.
- Hermes plugin namespace support for `ariadne_remember`, `ariadne_recall`, session prefetch, sync-turn memories, and shared memory.

### Changed
- Export/import and bulk memory insertion now preserve namespace/scope metadata.
- Memory stats aggregate per-namespace dedup indexes.

## [0.10.0] - 2026-06-10

### Added
- **Backup & Restore**: `ariadne backup` and `ariadne restore` CLI commands for SQLite-level database backup with WAL checkpoint, safety backups, and verification
- **Dashboard Backup/Restore**: Settings page now has Backup Database and Restore Database buttons with `/api/backup` and `/api/restore` endpoints
- **Hermes Skill**: Complete skill documentation at `~/.hermes/skills/productivity/ariadne/SKILL.md` covering all 25 plugin tools, setup, backup/restore, and best practices
- **Backup/Restore Documentation**: Full guide at `docs/guide/backup-restore.md` covering CLI, Dashboard, Python API, cron scheduling, and migration

### Improved
- README expanded with detailed Hermes integration section (all 19 tools documented) and Backup & Restore section
- CLI help now shows backup and restore commands

## [0.1.0] - 2024-01-01

### Added
- Core memory storage with SQLite (WAL mode, FTS5 full-text search)
- Vector search via FAISS (IndexFlatIP, auto-upgrade to IVFFlat)
- Hybrid search with Reciprocal Rank Fusion (RRF)
- Knowledge graph with recursive CTE BFS traversal
- MinHash LSH deduplication
- Contradiction detection via negation pattern matching
- Ebbinghaus retention scoring and priority-based eviction
- Memory consolidation with Jaccard similarity grouping
- Hermes-compatible API (remember, recall, forget, update, graph, stats)
- CLI interface (init, add, search, stats, migrate)
- Mnemosyne JSON migration support
- Full type hints and docstrings
- Comprehensive test suite

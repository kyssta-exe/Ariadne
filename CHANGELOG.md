# Changelog

All notable changes to Ariadne will be documented in this file.

## [0.13.0] - 2026-09-05

### Fixed
- **Eviction no longer destroys memories implicitly.** `evict()` previously soft-deleted ~10% of the store on *every* maintenance pass (including auto-maintenance every 50 bulk writes) regardless of size. Eviction is now capacity-driven: it only removes the lowest-priority overflow when the store exceeds `config.max_memories` (or an explicit `max_memories` argument), capped at `eviction_budget` per run. With no capacity configured (the default), eviction is a no-op — data is never destroyed without asking. Use `MemoryCurator.decay()` or set `max_memories` to bound the store.
- **Supersession filtering in `recall()`** now checks the whole store for an active replacement (one indexed query), not just memories present in the current result window — a better-ranked newer fact now reliably hides the memory it replaced.
- **Curator contradiction resolution** no longer re-writes the winning content (which bounced off the dedup layer and left dangling supersession chains). It links the existing newer memory onto the older one via `AriadneDB.link_supersession()` and soft-deletes the older statement. A new `allow_assistant_overwrite_user=False` guard also prevents non-user-authored content from erasing user-stated facts.
- **`LLMMemoryManager.set_fact()`** finds the prior fact via exact structured metadata lookup (`AriadneDB.find_facts`, SQLite JSON1 with a LIKE fallback) instead of fuzzy keyword recall, and retires the prior value once the new one is written — exactly one active value per `subject.attribute`.

### Added
- **Claude Code integration** (`arriadne.integrations.claude_code`) — hook adapter that turns Claude Code into a memory-backed agent: `UserPromptSubmit` records the prompt as an episode and injects a token-budgeted block of relevant memories as `additionalContext`; `Stop` records the assistant reply and (with `--extract-with openai|anthropic`) runs autonomous fact/relation extraction over the turn. Every handler is fail-open (a broken store can never block a session). Install via `ariadne mcp --host claude-code`, which also prints the `.claude/settings.json` hooks snippet.
- **`ariadne mcp --host <host>`** — prints ready-to-merge MCP server registration JSON for Claude Code, Claude Desktop, Cursor, VS Code, and Zed.
- **`ariadne hook claude-code`** — runs the hook adapter as a Claude Code hook command (JSON event on stdin → JSON output on stdout, always exit 0).
- **Dashboard bearer-token auth** — `create_app(auth_token=...)` (or `ariadne dashboard --token`, or `ARIADNE_DASHBOARD_TOKEN` for the ASGI-lazy app) requires `Authorization: Bearer <token>` on every `/api/*` route, compared in constant time. `/health` and `/metrics` stay open for uptime checks and scrapes.
- **Dashboard `/metrics`** — Prometheus text-format endpoint rendered without dependencies: memory/entity/edge/consolidation/FAISS/db-size gauges plus request counters (auth rejections included) and cumulative request time. Disable with `enable_metrics=False`.
- **Storage** — `AriadneDB.get_latest_episode()` for hook adapters to pair an assistant reply with its user prompt.
- **`python-multipart` added to the `[dashboard]` extra** — the `/api/restore` upload route crashed at route registration without it, so a bare `pip install arriadne-memory[dashboard]` was broken.
- **`httpx` added to the `[dev]` extra** — required by FastAPI's `TestClient` for the new dashboard tests.
- **Retrieval quality knobs on `recall()` / `context_pack()`**: `mmr` (Maximal Marginal Relevance diversification, 0–1) so top-k covers distinct facets instead of k near-duplicates, and `recency_boost` (recency weighting scaled by the Ebbinghaus half-life, recorded in `score_parts` for explainability).
- **Env & file configuration** — `AriadneConfig.from_env()` (`ARIADNE_*` variables, e.g. `ARIADNE_DB_PATH`, `ARIADNE_MAX_MEMORIES`), `AriadneConfig.from_toml()`, `from_dict()`, and `to_dict()`. Field types are coerced from annotations (PEP 604 unions incl. `int | None` handled).
- **`ariadne doctor`** — read-only integrity report: SQLite `quick_check`, FAISS↔database vector sync, FTS coverage, orphaned edges/links/provenance, dangling supersession pointers, duplicate active hashes. Exit code 1 on failure.
- **`ariadne feedback`** — record `approve`/`reject`/`correct` (plus `relevant`/`irrelevant`) with optional note/actor and a sensible per-action confidence delta default.
- **Storage performance** — search paths (`fts_search`, `vector_search`, `search_vector_batch`, `hybrid_search`) fetch result rows in one bulk query instead of one query per hit (N+1 eliminated); `recall()` attaches sources, feedback, and supersession chains with three batched queries total.
- **LLM provider callers** — `memory_manager.openai_caller()` and `anthropic_caller()` factories (lazily imported; `base_url` accepts any OpenAI-compatible local endpoint, keeping the pipeline fully local).
- **`process_turn()` fact upserts** — extracted `subject.attribute = value` memories now route through `set_fact()`, so a changed value supersedes the old fact instead of leaving both active.

### Changed
- `AriadneDB.evict()` signature: `evict(max_memories: int | None = None)`.
- Version bumped to 0.13.0.

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

# Changelog

All notable changes to Ariadne will be documented in this file.

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

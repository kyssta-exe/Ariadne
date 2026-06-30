# Changelog

All notable changes to Ariadne will be documented in this file.

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

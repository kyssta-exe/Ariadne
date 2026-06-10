# Changelog

All notable changes to Ariadne will be documented in this file.

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

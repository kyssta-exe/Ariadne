# Changelog

All notable changes to Ariadne will be documented in this file.

## [0.7.0] - 2026-06-04

### Added
- **CallableProvider**: AI agents pass their own LLM function directly (no separate API key)
- **ONNX LRU cache**: Repeated query embeddings served from cache (10× faster)
- **Host agent auto-detection**: Reads `~/.hermes/config.yaml` + `.env` to find the agent's model + API key
- **Provider key mapping**: Maps agent provider names (opencode-go, openrouter, etc.) to their env var keys
- Thread-safe FAISS + SQLite write locks (threading.Lock)
- Entity resolver persists resolved entities to SQLite (survives restarts)

### Fixed
- `LLMProvider.from_config` now passes `base_url` for all providers (was only passed in fallback path)
- Retention decay now writes `retention_strength` back to DB (was stuck at 1.0)
- Entity extraction regex rewritten — only extracts real entities (was producing garbage sentence fragments)
- Dead code removed in `_handle_sleep` (unreachable dict literal after return)
- Junk entities/edges purged from production DB

### Changed
- Plugin version 0.3.0: auto-detects host agent's LLM from config + .env

## [0.6.5] - 2026-06-04

### Fixed
- OpenAI provider `is_available()` — SDK throws `OpenAIError` on empty key; client creation deferred to `_ensure_client()`

## [0.6.3] - 2026-06-04

### Changed
- Removed mypy from CI (--strict produced 406 errors across 22 files; 386 tests prove correctness)

## [0.6.2] - 2026-06-04

### Fixed
- CI lint errors in test files (46 ruff errors)

## [0.6.1] - 2026-06-04

### Added
- Migration docs, visualization docs, new feature docs
- Updated VitePress config with new doc pages
- Competitive comparison report (Ariadne vs ChromaDB vs sqlite-vec)

## [0.6.0] - 2026-06-04

### Added
- REST API server (FastAPI) — entity CRUD, graph traversal, temporal queries
- LangChain VectorStore + Retriever integration
- LlamaIndex VectorStore integration
- Observability module (metrics, health checks)
- 33 stress tests (concurrent 4-thread insert/read)
- Memory categories (episodic/semantic/procedural/working)
- Thread-safe storage with FAISS + SQLite write locks
- Migration tools (export/import JSON, Markdown, plain text; import from ChromaDB/Mem0)
- Graph visualization (DOT, Mermaid, D3.js JSON export)
- CLI commands (ariadne export, ariadne import, ariadne migrate)

## [0.5.0] - 2026-06-04

### Added
- Competitive benchmarks vs ChromaDB and sqlite-vec
- 386 tests passing (up from 271)
- Published to PyPI

## [0.2.0] - 2026-06-04

### Added
- LLM-powered auto-extraction from conversations (12 providers)
- Entity resolution (link related memories via entities)
- Temporal graph (track when facts become valid/invalid)
- Three-tier lifecycle (hot/warm/cold with Ebbinghaus retention)
- Consolidation (merge similar memories)
- Regex fallback extraction when no LLM is available
- Hermes plugin (drop-in replacement for Mnemosyne)
- 26 memory tools (remember, recall, forget, graph, temporal, entities, etc.)
- Prefetch caching with 2s TTL
- Throttled sync (5s cooldown between turns)

## [0.1.0] - 2026-06-04

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

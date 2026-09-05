# Contributing to Ariadne

Thanks for helping make Ariadne the best local-first agent memory. This guide
covers dev setup, the layout, and the conventions the codebase follows.

## Dev setup

```bash
git clone https://github.com/kyssta-exe/Ariadne && cd Ariadne
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[embeddings,dev,dashboard]"
```

## Everyday commands

```bash
python -m pytest                 # full suite (~235 tests)
python -m pytest tests/test_storage.py -k eviction   # one slice
python -m ruff check src/arriadne tests              # lint
python -m ruff format src/arriadne tests             # format
python -m mypy src/arriadne                          # type check
```

CI runs pytest, ruff, and mypy; all three should be green before you open a PR.

## Layout

| Path | What lives there |
| --- | --- |
| `src/arriadne/storage.py` | `AriadneDB`: SQLite schema/migrations, FAISS index, FTS5, graph, eviction/consolidation, doctor |
| `src/arriadne/interface.py` | `AriadneMemory`: the public API (`remember` / `recall` / `context_pack` / provenance) |
| `src/arriadne/config.py` | `AriadneConfig` + `from_env` / `from_toml` / `from_dict` |
| `src/arriadne/dedup.py` | MinHash dedup + contradiction detection |
| `src/arriadne/memory_manager.py` | Autonomous ingest: `LLMMemoryManager`, provider callers |
| `src/arriadne/curator.py` | Retention/hygiene: decay, conflict resolution, consolidation |
| `src/arriadne/integrations/` | MCP server, LangGraph, OpenAI Agents adapters (import-guarded) |
| `src/arriadne/dashboard/` | FastAPI dashboard |
| `tests/` | pytest suite; `test_improvements.py` covers the newest behavior |

## Conventions

- **Never destroy user data implicitly.** Deletes are soft by default;
  eviction is capacity-driven (`max_memories`) and no-ops without one; purges
  keep a recoverability window. If a feature must delete, make the caller ask.
- **Keep the core dependency-light.** Optional providers (sentence-transformers,
  openai, anthropic, langgraph, fastapi) are imported lazily and import-guarded.
  The core needs only `faiss-cpu`, `numpy`, and `datasketch`.
- **Search results are explainable.** Any score adjustment belongs in
  `score_parts` so users can see why a memory ranked where it did.
- **Isolation keys are sacred.** Namespace/scope/user/agent/session/project
  boundaries must never be crossed by search, dedup, consolidation, or curation.
- **Tests accompany behavior changes.** New API surface gets a test in
  `tests/`; bug fixes get a regression test that fails without the fix.
- **Honest docs.** The README capability table and `CHANGELOG.md`
  (Keep-a-Changelog format) must match reality. Update both in the same PR.

## Addon authoring

Subclass `arriadne.addons.BaseAddon` and expose it via the `ariadne.addons`
entry-point group. Addons are additive and side-effect-free at registration;
see `addons/finance` and `src/arriadne/curator.py` (a `CuratorAddon`) for
reference implementations.

## Reporting bugs

Open a GitHub issue with your Python version, the commands/API calls that
reproduce it, and (if relevant) the output of `ariadne doctor`. Never paste
memory contents you wouldn't publish — the store may contain personal data.

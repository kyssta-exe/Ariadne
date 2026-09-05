# Security Policy

Ariadne is a **local-first** memory store: by default everything lives in a
single SQLite file on your machine and nothing leaves it. This page describes
the security model and how to report vulnerabilities.

## Security model

- **No network by default.** The core library performs no network I/O. The
  optional dashboard binds to `127.0.0.1` unless explicitly told otherwise.
- **Data at rest.** Memories, embeddings, and provenance live unencrypted in
  the SQLite database (`arriadne.db`). Protect the file with filesystem
  permissions or full-disk encryption; application-level encryption at rest is
  a known gap (tracked in the roadmap).
- **SQL injection.** All queries use parameterized statements; user text is
  never interpolated into SQL. FTS5 query strings are escaped via `_fts_escape`.
- **Feedback / provenance integrity.** Soft deletes keep a recoverability
  window (`purge_retention_seconds`, default 7 days) before purging; eviction
  never runs without an explicit capacity.
- **When you wire an LLM.** `LLMMemoryManager` sends conversation text to the
  LLM caller you provide (including `openai_caller`/`anthropic_caller`). That
  is the only path where memory content can leave the machine — point
  `base_url` at a local server to keep it offline.

## Hardening the dashboard

The dashboard ships for local inspection. If you must expose it beyond
loopback, put it behind a reverse proxy with authentication and TLS, and
restrict who can reach the port. Bearer-token auth for the dashboard API is on
the roadmap; until then treat the dashboard as single-user, localhost tooling.

## Reporting a vulnerability

Please report privately via GitHub
[security advisories](https://github.com/kyssta-exe/Ariadne/security/advisories/new)
rather than a public issue. Include the version (`python -c "import arriadne;
print(arriadne.__version__)"`), affected component, and a minimal reproduction.
You can expect an initial response within 7 days.

## Supported versions

| Version | Supported |
| --- | --- |
| 0.13.x | ✅ |
| < 0.13 | ❌ (upgrade; migrations are in-place and automatic) |

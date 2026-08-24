"""Session intelligence: episode search, session digests, and continuity.

Covers the ctx-memory-inspired layer: searching raw session history, distilling
a session into a digest memory with provenance, and injecting previous-session
context into context packs.
"""

from __future__ import annotations

from pathlib import Path

from arriadne import AriadneConfig, AriadneMemory


def _config(tmp_path: Path, name: str = "memory.db", **kwargs) -> AriadneConfig:
    return AriadneConfig(db_path=tmp_path / name, embedding_dim=2, **kwargs)


def _record_session(mem: AriadneMemory, session_id: str, namespace: str = "default") -> None:
    sessions = {
        "s1": [
            ("user", "We decided to use SQLite for the storage engine"),
            ("assistant", "Noted: storage engine choice is SQLite. I will remember that."),
            ("user", "The auth bug in login.py was caused by token expiry"),
            ("assistant", "Fixed the auth bug by refreshing tokens before expiry."),
            ("user", "Anyway, how is the weather?"),
        ],
        "s2": [
            ("user", "The billing migration to Stripe is now finished"),
            ("assistant", "Great: billing now runs on Stripe, invoices are webhooks-driven."),
            ("user", "Marketing wants a landing page redesign next sprint"),
            ("assistant", "Planned: landing page redesign scheduled for next sprint."),
            ("user", "Lunch was excellent today"),
        ],
    }
    for role, content in sessions.get(
        session_id,
        [("user", f"generic turn for {session_id}")],
    ):
        mem.record_episode(content=content, role=role, namespace=namespace, session_id=session_id)


def test_search_episodes_finds_raw_turns(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        _record_session(mem, "s1")
        mem.record_episode(
            content="unrelated chatter about lunch",
            role="user",
            namespace="default",
            session_id="s2",
        )

        hits = mem.search_episodes("auth bug token", k=5)
        assert hits, "expected raw episode hits"
        assert any("token" in h["content"] for h in hits)
        assert all(h["session_id"] == "s1" for h in hits)

        scoped = mem.search_episodes("auth bug", k=5, session_id="s2")
        assert scoped == []


def test_search_episodes_namespace_isolation(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        mem.record_episode("deploy pipeline broke", role="user", namespace="alpha", session_id="a1")
        mem.record_episode("deploy pipeline broke", role="user", namespace="beta", session_id="b1")

        alpha = mem.search_episodes("deploy", k=5, namespace="alpha")
        assert len(alpha) == 1
        assert alpha[0]["namespace"] == "alpha"


def test_list_sessions_summarizes_sessions(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        _record_session(mem, "s1")
        _record_session(mem, "s2")
        sessions = mem.list_sessions()
        assert {s["session_id"] for s in sessions} == {"s1", "s2"}
        by_id = {s["session_id"]: s for s in sessions}
        assert by_id["s1"]["turns"] == 5


def test_digest_session_creates_memory_with_provenance(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        _record_session(mem, "s1")
        result = mem.digest_session("s1")

        assert result["status"] == "created"
        assert result["episodes"] == 5
        assert "Session s1 digest" in result["digest"]
        # The salient turns (storage engine, auth bug) must be kept; the
        # weather chatter must not dominate the digest.
        assert "SQLite" in result["digest"]
        assert "auth bug" in result["digest"]

        memory_id = result["memory_id"]
        stored = mem._db.get_memory(memory_id)
        assert stored is not None
        assert (stored.get("metadata") or {}).get("kind") == "session_digest"
        assert (stored.get("metadata") or {}).get("session_id") == "s1"

        # Provenance: the digest links back to the selected episodes.
        sources = mem._db.get_sources_for_memory(memory_id)
        assert sources, "digest should carry episode provenance"
        assert all(s["source"] == "session_digest" for s in sources)


def test_digest_session_is_idempotent_and_force_redigestes(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        _record_session(mem, "s1")
        first = mem.digest_session("s1")
        assert first["status"] == "created"

        again = mem.digest_session("s1")
        assert again["status"] == "exists"
        assert again["memory_id"] == first["memory_id"]

        forced = mem.digest_session("s1", force=True)
        assert forced["status"] == "created"
        assert forced["memory_id"] != first["memory_id"]
        # The old digest is superseded, so only one digest stays current.
        digests = mem._db.list_session_digests(namespace="default", limit=10)
        active = [d for d in digests if not d.get("is_deleted")]
        assert len(active) == 1
        old = mem._db.get_memory(first["memory_id"])
        assert old is None or old.get("is_deleted") or old.get("supersedes_id") is not None


def test_digest_session_empty_session(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        result = mem.digest_session("nonexistent")
        assert result["status"] == "empty"
        assert result["digest"] is None


def test_session_context_returns_recent_digests(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        assert mem.session_context() == ""
        _record_session(mem, "s1")
        _record_session(mem, "s2")
        mem.digest_session("s1")
        mem.digest_session("s2")

        context = mem.session_context()
        assert "Recent session context" in context
        assert "s1" in context and "s2" in context


def test_context_pack_include_sessions(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        mem.remember("Paris is the capital of France", importance=0.9)
        _record_session(mem, "s1")
        mem.digest_session("s1")

        plain = mem.context_pack("capital of France", token_budget=400)
        assert "Paris" in plain
        assert "Recent session context" not in plain

        with_sessions = mem.context_pack(
            "capital of France", token_budget=400, include_sessions=True
        )
        assert "Paris" in with_sessions
        assert "Recent session context" in with_sessions


def test_search_sessions_after_restart(tmp_path: Path) -> None:
    # The FTS index for episodes must survive restarts via the schema backfill.
    db_path = tmp_path / "memory.db"
    with AriadneMemory(config=AriadneConfig(db_path=db_path, embedding_dim=2)) as mem:
        _record_session(mem, "s1")

    with AriadneMemory(config=AriadneConfig(db_path=db_path, embedding_dim=2)) as mem:
        hits = mem.search_episodes("storage engine SQLite", k=3)
        assert hits, "episodes must remain searchable after reopen"


def test_process_turn_session_id_feeds_sessions_and_digest(tmp_path: Path) -> None:
    from arriadne import LLMMemoryManager

    with AriadneMemory(config=_config(tmp_path)) as mem:
        mgr = LLMMemoryManager(mem)
        mgr.process_turn(
            "The migration to Postgres finished",
            "Noted: Postgres migration complete.",
            session_id="work-7",
        )
        mgr.process_turn(
            "Remember the retry budget is 3 attempts",
            "Stored the retry budget of 3 attempts.",
            session_id="work-7",
        )

        sessions = mem.list_sessions()
        assert [s["session_id"] for s in sessions] == ["work-7"]
        assert sessions[0]["turns"] == 2

        result = mem.digest_session("work-7")
        assert result["status"] == "created"
        assert "Postgres" in result["digest"] or "retry" in result["digest"]

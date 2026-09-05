"""Tests for the v0.13 improvements: capacity eviction, env/TOML config,
MMR + recency ranking, supersession links, structured facts, doctor, and
batched provenance."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from arriadne.config import AriadneConfig
from arriadne.interface import AriadneMemory
from arriadne.storage import AriadneDB


@pytest.fixture
def db() -> AriadneDB:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    config = AriadneConfig(db_path=db_path, embedding_dim=8)
    with AriadneDB(config) as database:
        yield database
    for suffix in ["", "-wal", "-shm"]:
        p = Path(db_path + suffix)
        if p.exists():
            p.unlink()


@pytest.fixture
def mem() -> AriadneMemory:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    memory = AriadneMemory(db_path=db_path, embedding_dim=8)
    yield memory
    memory.close()
    for suffix in ["", "-wal", "-shm"]:
        p = Path(db_path + suffix)
        if p.exists():
            p.unlink()


# ── Config: env / TOML / dict ───────────────────────────────────────────


class TestConfigLoading:
    def test_from_env_overrides_fields(self) -> None:
        env = {
            "ARIADNE_DB_PATH": "/tmp/env-test.db",
            "ARIADNE_EMBEDDING_DIM": "768",
            "ARIADNE_MAX_MEMORIES": "5000",
            "ARIADNE_DEDUP_THRESHOLD": "0.9",
        }
        cfg = AriadneConfig.from_env(environ=env)
        assert str(cfg.db_path) == "/tmp/env-test.db"
        assert cfg.embedding_dim == 768
        assert cfg.max_memories == 5000
        assert cfg.dedup_threshold == 0.9

    def test_from_env_none_clears_capacity(self) -> None:
        cfg = AriadneConfig.from_env(environ={"ARIADNE_MAX_MEMORIES": "none"})
        assert cfg.max_memories is None

    def test_from_env_bad_weights_raises(self) -> None:
        with pytest.raises(ValueError):
            AriadneConfig.from_env(environ={"ARIADNE_PRIORITY_WEIGHTS": "not-json"})

    def test_from_env_unknown_vars_ignored(self) -> None:
        cfg = AriadneConfig.from_env(environ={"ARIADNE_NOT_A_FIELD": "1"})
        assert cfg.embedding_dim == 384

    def test_from_toml(self, tmp_path: Path) -> None:
        toml = tmp_path / "ariadne.toml"
        toml.write_text(
            'db_path = "toml-test.db"\n'
            "embedding_dim = 128\n"
            "[ariadne]\nignored_because_top_level_wins = true\n"
        )
        cfg = AriadneConfig.from_toml(toml)
        assert str(cfg.db_path) == "toml-test.db"
        assert cfg.embedding_dim == 128

    def test_from_toml_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            AriadneConfig.from_toml("/nonexistent/ariadne.toml")

    def test_to_dict_roundtrip(self) -> None:
        cfg = AriadneConfig(max_memories=42)
        data = cfg.to_dict()
        assert data["max_memories"] == 42
        again = AriadneConfig.from_dict(data)
        assert again.max_memories == 42

    def test_max_memories_validation(self) -> None:
        with pytest.raises(ValueError):
            AriadneConfig(max_memories=0)


# ── Storage: link_supersession / find_facts / doctor / batch ────────────


class TestStorageHelpers:
    def test_link_supersession(self, db: AriadneDB) -> None:
        old = db.add_memory("user name is Alice")["memory_id"]
        new = db.add_memory("user name is Bob")["memory_id"]
        assert db.link_supersession(new_id=new, old_id=old) is True
        assert db.get_memory(new)["supersedes_id"] == old
        # Chain is visible through the walk.
        chain = db.get_supersession_chain(new)
        assert [m["id"] for m in chain] == [new, old]

    def test_link_supersession_rejects_relink(self, db: AriadneDB) -> None:
        a = db.add_memory("first")["memory_id"]
        b = db.add_memory("second")["memory_id"]
        c = db.add_memory("third")["memory_id"]
        assert db.link_supersession(new_id=b, old_id=a) is True
        # b already supersedes a; c cannot additionally claim a? (c→a is fine),
        # but b cannot be relinked onto c.
        assert db.link_supersession(new_id=b, old_id=c) is False

    def test_find_facts_json1(self, db: AriadneDB) -> None:
        db.add_memory(
            "user name is Alice",
            metadata={"fact_subject": "user", "fact_attribute": "name", "fact_value": "Alice"},
        )
        db.add_memory(
            "user name is Bob",
            metadata={"fact_subject": "user", "fact_attribute": "name", "fact_value": "Bob"},
        )
        db.add_memory(
            "project language is Python",
            metadata={"fact_subject": "project", "fact_attribute": "language"},
        )
        facts = db.find_facts("user", "name")
        assert len(facts) == 2
        assert facts[0]["metadata"]["fact_value"] == "Bob"  # newest first
        assert db.find_facts("missing", "attr") == []

    def test_get_memories_batch(self, db: AriadneDB) -> None:
        ids = [db.add_memory(f"batch {i}")["memory_id"] for i in range(3)]
        batch = db.get_memories_batch([ids[0], ids[2], 99999])
        assert set(batch) == {ids[0], ids[2]}
        assert batch[ids[0]]["content"] == "batch 0"

    def test_batched_provenance(self, db: AriadneDB) -> None:
        a = db.add_memory("with sources")["memory_id"]
        b = db.add_memory("without sources")["memory_id"]
        db.add_source(memory_id=a, episode_id=None, source="user")
        db.add_feedback(memory_id=a, action="approve")
        sources = db.get_sources_for_memories([a, b])
        feedback = db.get_feedback_for_memories([a, b])
        assert len(sources[a]) == 1 and sources[b] == []
        assert feedback[a][0]["action"] == "approve" and feedback[b] == []

    def test_doctor_clean_report(self, db: AriadneDB) -> None:
        db.add_memory("healthy memory")
        report = db.doctor()
        assert report["ok"] is True
        names = {c["name"] for c in report["checks"]}
        assert {"sqlite_quick_check", "vector_index_sync", "fts_coverage"} <= names
        assert report["summary"]["fail"] == 0

    def test_doctor_detects_fts_gap(self, db: AriadneDB) -> None:
        mid = db.add_memory("orphaned from fts")["memory_id"]
        db.conn.execute("DELETE FROM memories_fts WHERE rowid = ?", (mid,))
        db.conn.commit()
        report = db.doctor()
        check = next(c for c in report["checks"] if c["name"] == "fts_coverage")
        assert check["status"] == "warn"


# ── Retrieval: MMR, recency boost, supersession filter ──────────────────


class TestRetrievalUpgrades:
    def test_mmr_prefers_diverse_results(self, mem: AriadneMemory) -> None:
        # Three near-identical memories plus one distinct, all matching "color".
        mem.remember("my favorite color is blue", importance=0.9)
        mem.remember("my favorite color is blue definitely", importance=0.85)
        mem.remember("my favorite color is blue for sure", importance=0.8)
        mem.remember("the sky color at dusk is orange", importance=0.7)

        plain = mem.recall("color", k=3)
        diverse = mem.recall("color", k=3, mmr=0.7)

        plain_contents = [r["content"] for r in plain]
        # Plain ranking returns the near-duplicates together.
        assert sum("favorite color is blue" in c for c in plain_contents) >= 2
        # MMR surfaces the distinct orange memory instead of a third blue.
        assert any("orange" in r["content"] for r in diverse)

    def test_recency_boost_prefers_fresh(self, mem: AriadneMemory) -> None:
        import time as _time

        mem.remember("deploy target is staging", importance=0.5)
        old_ts = _time.time() - 86400 * 365  # a year ago
        mem.remember("deploy target was production", importance=0.5, event_at=old_ts)

        boosted = mem.recall("deploy target", k=2, recency_boost=1.0)
        assert boosted[0]["content"].startswith("deploy target is staging")
        assert "recency" in boosted[0]["score_parts"]

    def test_superseded_hidden_even_if_ranked_higher(self, mem: AriadneMemory) -> None:
        import time as _time

        old_ts = _time.time() - 3600
        old = mem.remember("api key is abc123", event_at=old_ts)["memory_id"]
        new = mem.remember("api key is xyz789")["memory_id"]
        assert mem._db.link_supersession(new_id=new, old_id=old)

        results = mem.recall("api key", k=10)
        contents = [r["content"] for r in results]
        assert "api key is xyz789" in contents
        assert "api key is abc123" not in contents

    def test_recall_attaches_batched_provenance(self, mem: AriadneMemory) -> None:
        r = mem.remember("provenance test fact")
        mem._db.add_source(memory_id=r["memory_id"], episode_id=None, source="user")
        results = mem.recall("provenance test fact", k=5)
        assert results[0]["sources"][0]["source"] == "user"
        assert results[0]["feedback"] == []
        assert results[0]["supersession_chain"][0]["id"] == results[0]["id"]

    def test_context_pack_forwards_options(self, mem: AriadneMemory) -> None:
        mem.remember("pack me into context")
        packed = mem.context_pack("pack me", token_budget=200, mmr=0.0, recency_boost=0.5)
        assert "pack me into context" in packed


# ── Memory manager: structured facts ────────────────────────────────────


class TestFactUpserts:
    def test_set_fact_supersedes_prior_value(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        mem = AriadneMemory(db_path=db_path, embedding_dim=8)
        try:
            from arriadne.memory_manager import LLMMemoryManager

            mgr = LLMMemoryManager(mem)
            first = mgr.set_fact("user", "name", "Alice")
            assert first["status"] == "created"
            second = mgr.set_fact("user", "name", "Bob")
            assert second["status"] == "created"

            # Exactly one active fact remains, holding the newest value.
            facts = mem._db.find_facts("user", "name")
            assert len(facts) == 1
            assert facts[0]["metadata"]["fact_value"] == "Bob"
        finally:
            mem.close()
            for suffix in ["", "-wal", "-shm"]:
                Path(db_path + suffix).unlink(missing_ok=True)

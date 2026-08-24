"""Regression tests for high-impact memory correctness boundaries."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from arriadne import AriadneConfig, AriadneMemory
from arriadne.storage import AriadneDB


def _config(tmp_path: Path, name: str = "memory.db", **kwargs) -> AriadneConfig:
    return AriadneConfig(db_path=tmp_path / name, embedding_dim=2, **kwargs)


def test_soft_delete_removes_live_vector(tmp_path: Path) -> None:
    with AriadneDB(_config(tmp_path)) as db:
        nearest = db.add_memory("nearest", np.array([1.0, 0.0], dtype=np.float32))["memory_id"]
        active = db.add_memory("active", np.array([0.9, 0.1], dtype=np.float32))["memory_id"]

        assert db.delete_memory(nearest, hard=False) is True
        results = db.vector_search(np.array([1.0, 0.0], dtype=np.float32), k=1)

        assert [item["id"] for item in results] == [active]
        assert db._faiss_index is not None
        assert db._faiss_index.ntotal == 1


def test_wrong_dimension_does_not_leave_a_row(tmp_path: Path) -> None:
    with AriadneDB(_config(tmp_path)) as db:
        try:
            db.add_memory("invalid", np.array([1.0, 2.0, 3.0], dtype=np.float32))
        except (ValueError, AssertionError):
            pass

        count = db.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        assert count == 0


def test_content_update_clears_stale_embedding_without_embedder(tmp_path: Path) -> None:
    with AriadneDB(_config(tmp_path)) as db:
        memory_id = db.add_memory("old semantic content", np.array([1.0, 0.0], dtype=np.float32))[
            "memory_id"
        ]

        assert db.update_memory(memory_id, content="new semantic content") is True
        results = db.vector_search(np.array([1.0, 0.0], dtype=np.float32), k=5)

        assert all(item["id"] != memory_id for item in results)
        row = db.conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        assert row[0] is None


def test_batch_vector_search_can_filter_namespace(tmp_path: Path) -> None:
    with AriadneDB(_config(tmp_path)) as db:
        db.add_memory("alpha", np.array([1.0, 0.0], dtype=np.float32), namespace="alpha")
        db.add_memory("beta", np.array([1.0, 0.0], dtype=np.float32), namespace="beta")

        results = db.search_vector_batch(
            np.array([[1.0, 0.0]], dtype=np.float32), k=2, namespace="alpha"
        )

        assert [item["namespace"] for item in results[0]] == ["alpha"]


def test_filtered_recall_overfetches_until_eligible_result(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with AriadneMemory(config=config) as mem:
        for index in range(3):
            mem._db.add_memory(f"deploy production candidate {index}", memory_type="episodic")
        eligible = mem._db.add_memory("deploy production semantic answer", memory_type="semantic")[
            "memory_id"
        ]

        results = mem.recall("deploy production", k=1, type_filter="semantic", namespace="default")

        assert [item["id"] for item in results] == [eligible]


def test_consolidation_never_crosses_namespaces(tmp_path: Path) -> None:
    with AriadneDB(_config(tmp_path)) as db:
        db.add_memory("the same durable fact", namespace="alpha")
        db.add_memory("the same durable fact today", namespace="alpha")
        db.add_memory("the same durable fact", namespace="beta")
        db.add_memory("the same durable fact today", namespace="beta")

        assert db.consolidate() == 2
        active = db.conn.execute(
            "SELECT namespace, COUNT(*) FROM memories WHERE is_deleted = 0 GROUP BY namespace"
        ).fetchall()

        assert {row[0]: row[1] for row in active} == {"alpha": 1, "beta": 1}


def test_export_import_preserves_graph_edges(tmp_path: Path) -> None:
    source_config = _config(tmp_path, "source.db")
    destination_config = _config(tmp_path, "destination.db")
    with AriadneDB(source_config) as source:
        source.add_edge("A", "B", edge_type="depends_on", weight=0.7)
        exported = source.export_all()

    assert exported["edges"]
    with AriadneDB(destination_config) as destination:
        assert destination.import_all(exported) == 0
        edge = destination.conn.execute(
            """SELECT s.name, t.name, e.edge_type, e.weight
               FROM edges e
               JOIN entities s ON s.id = e.source_id
               JOIN entities t ON t.id = e.target_id"""
        ).fetchone()

        assert tuple(edge) == ("A", "B", "depends_on", 0.7)


def test_export_import_preserves_confidence(tmp_path: Path) -> None:
    with AriadneDB(_config(tmp_path, "source.db")) as source:
        source.add_memory("confidence round trip", confidence=0.25)
        exported = source.export_all()

    with AriadneDB(_config(tmp_path, "destination.db")) as destination:
        assert destination.import_all(exported) == 1
        row = destination.conn.execute(
            "SELECT confidence FROM memories WHERE content = ?",
            ("confidence round trip",),
        ).fetchone()
        assert row[0] == 0.25


def test_legacy_schema_migrates_before_dependent_indexes(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            memory_type TEXT NOT NULL DEFAULT 'semantic',
            importance REAL NOT NULL DEFAULT 0.5,
            embedding BLOB,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            accessed_at REAL NOT NULL,
            access_count INTEGER NOT NULL DEFAULT 0,
            retention_strength REAL NOT NULL DEFAULT 1.0,
            is_deleted INTEGER NOT NULL DEFAULT 0,
            deleted_at REAL,
            metadata TEXT
        )"""
    )
    conn.execute(
        """INSERT INTO memories
           (content, content_hash, created_at, updated_at, accessed_at)
           VALUES ('legacy note', 'legacy-hash', 1, 1, 1)"""
    )
    conn.commit()
    conn.close()

    with AriadneDB(_config(tmp_path, "legacy.db")) as db:
        assert db.fts_search("legacy note", k=1)[0]["content"] == "legacy note"
        columns = {row[1] for row in db.conn.execute("PRAGMA table_info(memories)")}
        assert {"namespace", "scope", "user_id", "agent_id", "session_id", "project_id"} <= columns


def test_keyword_confidence_weight_is_clamped(tmp_path: Path) -> None:
    with AriadneDB(_config(tmp_path)) as db:
        db.conn.execute(
            "UPDATE memories SET confidence = ? WHERE id = ?",
            (2.0, db.add_memory("clamped confidence memory")["memory_id"]),
        )
        result = db.fts_search("clamped confidence memory", k=1)[0]
        assert result["score_parts"]["confidence"] == 1.0


def test_hybrid_search_ranks_higher_confidence_first(tmp_path: Path) -> None:
    """Approved memories must outrank rejected ones at equal RRF relevance."""
    with AriadneDB(_config(tmp_path)) as db:
        # Both memories match the same query; distinct embeddings resolve
        # vector order, then confidence reweights.
        low_id = db.add_memory(
            "apples and oranges fruit fresh", np.array([1.0, 0.0], dtype=np.float32), confidence=0.1
        )["memory_id"]
        high_id = db.add_memory(
            "apples and oranges fruit ripe", np.array([0.0, 1.0], dtype=np.float32), confidence=1.0
        )["memory_id"]

        results = db.hybrid_search(
            "apples and oranges fruit", np.array([0.95, 0.05], dtype=np.float32), k=2
        )
        ids = [r["id"] for r in results]
        assert high_id in ids and low_id in ids
        assert next(r["id"] for r in results) == high_id, ids
        parts = results[0]["score_parts"]
        assert parts["confidence"] == 1.0
        assert parts["confidence_weight"] == 1.0


def test_keyword_recall_uses_confidence_weighting(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        low_id = mem._db.add_memory("database deployment guide", confidence=0.0)["memory_id"]
        high_id = mem._db.add_memory("database deployment guide with extra notes", confidence=1.0)[
            "memory_id"
        ]

        results = mem.recall("database deployment guide", k=2)

        assert [item["id"] for item in results] == [high_id, low_id]


def test_hybrid_single_source_still_applies_confidence(tmp_path: Path) -> None:
    with AriadneDB(_config(tmp_path)) as db:
        low_id = db.add_memory("database deployment guide", confidence=0.0)["memory_id"]
        high_id = db.add_memory("database deployment guide with extra notes", confidence=1.0)[
            "memory_id"
        ]

        results = db.hybrid_search(
            "database deployment guide", np.array([1.0, 0.0], dtype=np.float32), k=2
        )

        assert [item["id"] for item in results] == [high_id, low_id]
        assert results[0]["score_parts"]["confidence"] == 1.0


def test_hybrid_single_source_preserves_rank_component(tmp_path: Path) -> None:
    with AriadneDB(_config(tmp_path)) as db:
        db.add_memory("database deployment guide exact", confidence=1.0)
        db.add_memory("database deployment guide exact extra notes", confidence=1.0)

        results = db.hybrid_search("database deployment guide exact", k=2)

        assert len(results) == 2
        assert results[0]["score_parts"]["rrf"] > results[1]["score_parts"]["rrf"]


def test_context_pack_respects_token_budget(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        for i in range(10):
            mem.remember(f"packed memory entry {i} " + "x" * 60, importance=0.5)

        tiny = mem.context_pack("packed memory entry", token_budget=20)
        # Estimate = chars/4 + 8 overhead; a single entry is ~29 tokens,
        # so a 20-token budget must not produce anything.
        assert tiny == ""
        assert len(tiny.splitlines()) == 0
        assert mem.context_pack("packed memory entry", token_budget=0) == ""

        big = mem.context_pack("packed memory entry", token_budget=10000)
        assert "packed memory entry" in big
        # Deterministic order: most relevant memory first.
        lines = [ln for ln in big.splitlines() if ln.strip()]
        assert lines[0].startswith("- packed memory entry")


def test_context_pack_merges_explicit_namespaces(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        mem.remember("alpha namespace memory", namespace="alpha")
        mem.remember("beta namespace memory", namespace="beta")
        mem.remember("default namespace memory", namespace="default")

        packed = mem.context_pack(
            "namespace memory",
            token_budget=100,
            namespaces=["alpha", "beta"],
        )

        assert "alpha namespace memory" in packed
        assert "beta namespace memory" in packed
        assert "default namespace memory" not in packed


def test_context_pack_breaks_score_ties_deterministically(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        mem.recall = lambda query, **kwargs: [  # type: ignore[method-assign]
            {"id": 2, "content": "second tied memory", "score": 1.0},
            {"id": 1, "content": "first tied memory", "score": 1.0},
        ]

        assert mem.context_pack("query", token_budget=100) == (
            "- first tied memory\n- second tied memory"
        )


def test_context_pack_skips_oversized_result_and_keeps_fitting_results(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        mem.recall = lambda query, **kwargs: [  # type: ignore[method-assign]
            {"content": "x" * 1000, "score": 1.0},
            {"content": "small fitting memory", "score": 0.5},
        ]

        packed = mem.context_pack("query", token_budget=20)

        assert packed == "- small fitting memory"

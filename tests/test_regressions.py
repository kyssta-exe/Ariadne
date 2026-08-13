"""Regression tests for high-impact memory correctness boundaries."""

from __future__ import annotations

import json
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
        memory_id = db.add_memory(
            "old semantic content", np.array([1.0, 0.0], dtype=np.float32)
        )["memory_id"]

        assert db.update_memory(memory_id, content="new semantic content") is True
        results = db.vector_search(np.array([1.0, 0.0], dtype=np.float32), k=5)

        assert all(item["id"] != memory_id for item in results)
        row = db.conn.execute("SELECT embedding FROM memories WHERE id = ?", (memory_id,)).fetchone()
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
            mem._db.add_memory(
                f"deploy production candidate {index}", memory_type="episodic"
            )
        eligible = mem._db.add_memory(
            "deploy production semantic answer", memory_type="semantic"
        )["memory_id"]

        results = mem.recall(
            "deploy production", k=1, type_filter="semantic", namespace="default"
        )

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


def test_hybrid_search_ranks_higher_confidence_first(tmp_path: Path) -> None:
    """Approved memories must outrank rejected ones at equal RRF relevance."""
    with AriadneDB(_config(tmp_path)) as db:
        # Two memories with identical text so FTS ties; distinct embeddings
        # so the vector side resolves order, then confidence reweights.
        low_id = db.add_memory(
            "apples and oranges fruit fresh",
            np.array([1.0, 0.0], dtype=np.float32), confidence=0.1
        )["memory_id"]
        high_id = db.add_memory(
            "apples and oranges fruit ripe",
            np.array([0.0, 1.0], dtype=np.float32), confidence=1.0
        )["memory_id"]

        results = db.hybrid_search(
            "apples and oranges fruit", np.array([0.95, 0.05], dtype=np.float32), k=2
        )
        ids = [r["id"] for r in results]
        assert high_id in ids and low_id in ids
        assert [r["id"] for r in results][0] == high_id, ids
        parts = results[0]["score_parts"]
        assert parts["confidence"] == 1.0
        assert parts["confidence_weight"] == 1.0

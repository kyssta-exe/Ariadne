"""Startup persistence (FAISS sidecar) and tiered auto-maintenance."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from arriadne import AriadneConfig, AriadneMemory

DIM = 2


def _cfg(tmp_path: Path, name: str = "m.db", **kwargs) -> AriadneConfig:
    return AriadneConfig(db_path=tmp_path / name, embedding_dim=DIM, **kwargs)


def _populate(tmp_path: Path) -> Path:
    db_path = tmp_path / "m.db"
    with AriadneMemory(config=_cfg(tmp_path)) as mem:
        for i, axis in enumerate([(1, 0), (0, 1), (1, 1)]):
            mem.remember(
                f"memory {i}",
                embedding=np.array(axis, dtype=np.float32),
            )
    return db_path


def test_sidecar_written_and_reused(tmp_path: Path) -> None:
    db_path = _populate(tmp_path)
    sidecar = Path(str(db_path) + ".faiss")
    fp = Path(str(db_path) + ".faiss.fp")
    assert sidecar.exists(), "close() must persist the FAISS index"
    assert fp.exists(), "close() must persist the fingerprint"

    # Reopen: index loads from the sidecar (no rebuild) and stays correct.
    mem = AriadneMemory(config=_cfg(tmp_path))
    try:
        assert mem._db._faiss_index is not None
        assert mem._db._faiss_index.ntotal == 3
        hits = mem.recall("memory", k=3)
        assert len(hits) == 3
    finally:
        mem.close()


def test_sidecar_skipped_for_in_memory_db() -> None:
    mem = AriadneMemory(config=AriadneConfig(db_path=":memory:", embedding_dim=DIM))
    mem.remember("x", embedding=np.array([1, 0], dtype=np.float32))
    mem.close()  # must not attempt any file persistence
    assert mem._db._faiss_sidecar_path() is None


def test_stale_sidecar_falls_back_to_rebuild(tmp_path: Path) -> None:
    db_path = _populate(tmp_path)
    # External mutation: delete one vector-bearing row directly in SQLite.
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM memories WHERE id = (SELECT MIN(id) FROM memories)")
    conn.commit()
    conn.close()

    mem = AriadneMemory(config=_cfg(tmp_path))
    try:
        assert mem._db._faiss_index is not None
        assert mem._db._faiss_index.ntotal == 2, "stale sidecar must not be trusted"
        hits = mem.recall("memory", k=5)
        assert len(hits) == 2
    finally:
        mem.close()


def test_corrupt_sidecar_falls_back_to_rebuild(tmp_path: Path) -> None:
    db_path = _populate(tmp_path)
    Path(str(db_path) + ".faiss").write_bytes(b"not an index")

    mem = AriadneMemory(config=_cfg(tmp_path))
    try:
        assert mem._db._faiss_index is not None
        assert mem._db._faiss_index.ntotal == 3
    finally:
        mem.close()


def test_dimension_change_falls_back_to_rebuild(tmp_path: Path) -> None:
    db_path = _populate(tmp_path)  # dim 4 sidecar persisted
    mem = AriadneMemory(config=AriadneConfig(db_path=db_path, embedding_dim=8))
    try:
        assert mem._db._faiss_index is not None
        assert mem._db._faiss_index.d == 8
        assert mem._db._faiss_index.ntotal == 0  # vectors have dim 4, dropped
    finally:
        mem.close()


def test_search_identical_with_and_without_sidecar(tmp_path: Path) -> None:
    _populate(tmp_path)
    with_sidecar = AriadneMemory(config=_cfg(tmp_path))
    q = np.array([1.0, 0.0], dtype=np.float32)
    a = with_sidecar._db.vector_search(q, k=3)
    with_sidecar.close()

    Path(str(tmp_path / "m.db") + ".faiss").unlink()
    without_sidecar = AriadneMemory(config=_cfg(tmp_path))
    b = without_sidecar._db.vector_search(q, k=3)
    without_sidecar.close()

    assert [m["id"] for m in a] == [m["id"] for m in b]
    assert np.allclose([m["score"] for m in a], [m["score"] for m in b])


def test_maintenance_light_skips_heavy_steps(tmp_path: Path) -> None:
    with AriadneMemory(config=_cfg(tmp_path)) as mem:
        calls: list[str] = []

        def _spy_consolidate() -> int:
            calls.append("consolidate")
            return 0

        def _spy_evict() -> int:
            calls.append("evict")
            return 0

        mem.consolidate = _spy_consolidate  # type: ignore[method-assign]
        mem._db.evict = _spy_evict  # type: ignore[method-assign]

        mem.maintenance(light=True)
        assert calls == [], "light cycle must skip consolidate + evict"

        mem.maintenance()
        assert sorted(set(calls)) == ["consolidate", "evict"]


def test_auto_maintenance_tiers_by_cycle_count(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, maintenance_interval=2, heavy_maintenance_factor=3)
    with AriadneMemory(config=cfg) as mem:
        heavy_cycles: list[bool] = []
        original = mem.maintenance

        def _spy(*, light: bool = False) -> dict[str, int]:
            heavy_cycles.append(not light)
            return original(light=light)

        mem.maintenance = _spy  # type: ignore[method-assign]

        for i in range(12):  # 6 cycles: light, light, heavy, ...
            mem.remember(f"tiered {i}")
        # False = light, True = heavy: every 3rd cycle (factor=3) is heavy.
        assert heavy_cycles == [False, False, True, False, False, True]


def test_dedup_sidecar_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "m.db"
    with AriadneMemory(config=_cfg(tmp_path)) as mem:
        mem.remember("the deploy pipeline uses github actions")
        mem.remember("completely different topic about cooking pasta")
    assert Path(str(db_path) + ".dedup.pkl").exists()

    mem = AriadneMemory(config=_cfg(tmp_path))
    try:
        total = sum(d.size for d in mem._dedup_by_namespace.values())
        assert total == 2, "dedup index must be restored from the sidecar"
        # Near-duplicate detection still works after restore.
        result = mem.remember("the deploy pipeline uses GitHub Actions today")
        assert result["status"] == "duplicate"
    finally:
        mem.close()


def test_dedup_sidecar_stale_falls_back(tmp_path: Path) -> None:
    db_path = tmp_path / "m.db"
    with AriadneMemory(config=_cfg(tmp_path)) as mem:
        mem.remember("alpha content")
    # External mutation invalidates the fingerprint.
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO memories (content, content_hash, created_at, updated_at, accessed_at) "
        "VALUES ('beta content', 'hash-beta', 1, 1, 1)"
    )
    conn.commit()
    conn.close()

    mem = AriadneMemory(config=_cfg(tmp_path))
    try:
        # Rebuild path: both active memories are indexed and dedup works.
        total = sum(d.size for d in mem._dedup_by_namespace.values())
        assert total == 2
        assert mem.remember("alpha content")["status"] == "duplicate"
    finally:
        mem.close()

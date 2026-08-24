"""Tests for the second batch of fixes.

Covers: embedder auto-embedding (#2), ivf_flat staging (#3), dedup persistence
across restart (#4), retention strength growth (#5), bidirectional graph
traversal (#6), real consolidation (#8), reads no longer write (#9), access-log
pruning + purge of soft-deleted rows (#11), and thread safety (#12).
"""

from __future__ import annotations

import tempfile
import threading
import time
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from arriadne.config import AriadneConfig
from arriadne.interface import AriadneMemory
from arriadne.storage import AriadneDB


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    for suffix in ["", "-wal", "-shm", ".faiss"]:
        p = Path(path + suffix)
        if p.exists():
            p.unlink()


# --------------------------------------------------------------------------- #
# #2 — embedder makes semantic recall work out of the box
# --------------------------------------------------------------------------- #

class _FakeEmbedder:
    """Deterministic stand-in for a real model (no heavy dependency in tests).

    Maps hardware-ish text to one axis and everything else to another, so a
    query can match a stored memory that shares *no* keywords.
    """

    dim = 4
    _hw: ClassVar[set[str]] = {
        "vps", "server", "cores", "core", "ram", "specs", "spec", "cpu", "memory", "gb"
    }

    def __call__(self, text: str) -> list[float]:
        words = set(text.lower().replace(",", " ").split())
        return [1.0, 0.0, 0.0, 0.0] if (words & self._hw) else [0.0, 1.0, 0.0, 0.0]


def test_embedder_enables_semantic_recall(db_path):
    mem = AriadneMemory(db_path=db_path, embedding_dim=4, embedder=_FakeEmbedder())
    mem.remember("VPS has 4 cores, 8GB RAM", importance=0.8)
    mem.remember("My favourite colour is teal")
    # "server specs" shares no keywords with the stored hardware fact, but the
    # embedder maps both to the same axis — so semantic recall finds it.
    results = mem.recall("server specs", k=5)
    assert results, "expected a semantic match"
    assert "VPS" in results[0]["content"]
    mem.close()


def test_embedder_dim_mismatch_raises(db_path):
    with pytest.raises(ValueError):
        AriadneMemory(db_path=db_path, embedding_dim=8, embedder=_FakeEmbedder())


def test_no_embedder_still_keyword_searches(db_path):
    mem = AriadneMemory(db_path=db_path)
    mem.remember("deploy script lives in infra/deploy.sh")
    assert mem.recall("deploy script", k=5)
    mem.close()


# --------------------------------------------------------------------------- #
# #3 — faiss_type="ivf_flat" no longer crashes on first add
# --------------------------------------------------------------------------- #

def test_ivf_flat_does_not_crash_and_upgrades(db_path):
    cfg = AriadneConfig(db_path=db_path, embedding_dim=8, faiss_type="ivf_flat",
                        ivf_min_points=3, dedup_threshold=0.99)
    import faiss
    with AriadneDB(cfg) as db:
        # First insert used to raise "is_trained failed"; now it stages on Flat.
        r = db.add_memory("first", np.random.randn(8).astype(np.float32))
        assert r["status"] == "created"
        for i in range(6):
            db.add_memory(f"vec {i}", np.random.randn(8).astype(np.float32))
        # Past ivf_min_points it should have switched to a (trained) IVF index.
        assert isinstance(db._faiss_index, faiss.IndexIDMap2)
        assert isinstance(faiss.downcast_index(db._faiss_index.index), faiss.IndexIVFFlat)
        # And search still works.
        assert isinstance(db.vector_search(np.random.randn(8).astype(np.float32), k=3), list)


# --------------------------------------------------------------------------- #
# #4 — MinHash dedup index is rebuilt on open
# --------------------------------------------------------------------------- #

def test_dedup_index_survives_restart(db_path):
    cfg = AriadneConfig(db_path=db_path, embedding_dim=8, dedup_threshold=0.6)
    with AriadneMemory(config=cfg) as mem:
        mem.remember("the server has four cpu cores and eight gigabytes of memory")

    with AriadneMemory(config=cfg) as mem:
        # Index was rebuilt from the DB, not reset to empty.
        assert mem.stats()["dedup_index_size"] == 1
        near_dup = mem.remember("the server has four cpu cores and eight gb of memory")
        assert near_dup["status"] == "duplicate"
        assert near_dup.get("duplicate_of") is not None


# --------------------------------------------------------------------------- #
# #5 — retention strength actually grows on access and feeds the curve
# --------------------------------------------------------------------------- #

def test_retention_strength_grows_on_access(db_path):
    cfg = AriadneConfig(db_path=db_path, embedding_dim=8, retention_growth_factor=1.5)
    with AriadneDB(cfg) as db:
        mid = db.add_memory("grows with use")["memory_id"]
        assert db.get_memory(mid)["retention_strength"] == 1.0
        for _ in range(5):
            db.touch_memory(mid)
        rs = db.get_memory(mid)["retention_strength"]
        assert rs == pytest.approx(1.5 ** 5, rel=1e-4)


def test_retention_strength_is_capped(db_path):
    cfg = AriadneConfig(db_path=db_path, embedding_dim=8,
                        retention_growth_factor=2.0, retention_strength_cap=3.0)
    with AriadneDB(cfg) as db:
        mid = db.add_memory("cap me")["memory_id"]
        for _ in range(10):
            db.touch_memory(mid)
        assert db.get_memory(mid)["retention_strength"] == 3.0


def test_retention_strength_feeds_the_curve(db_path):
    cfg = AriadneConfig(db_path=db_path, embedding_dim=8)
    with AriadneDB(cfg) as db:
        old = time.time() - 86400 * 5  # 5 days ago
        weak = {"accessed_at": old, "importance": 0.5, "retention_strength": 1.0}
        strong = {"accessed_at": old, "importance": 0.5, "retention_strength": 50.0}
        # A higher accrued retention strength decays more slowly.
        assert db.compute_retention_strength(strong) > db.compute_retention_strength(weak)


# --------------------------------------------------------------------------- #
# #6 — graph traversal follows edges in both directions
# --------------------------------------------------------------------------- #

def test_graph_traversal_follows_incoming_edges(db_path):
    with AriadneMemory(db_path=db_path, embedding_dim=8) as mem:
        mem.add_edge("X", "A", "child")  # A only has an *incoming* edge
        mem.add_edge("X", "B", "child")
        nodes = mem.graph("A", hops=1)["nodes"]
        assert "X" in nodes  # used to return just ["A"]


def test_graph_traversal_bidirectional_chain(db_path):
    with AriadneMemory(db_path=db_path, embedding_dim=8) as mem:
        mem.add_edge("A", "B", "next")
        mem.add_edge("B", "C", "next")
        mem.add_edge("C", "D", "next")
        # From C, both directions within 1 hop: B (incoming) and D (outgoing).
        nodes = set(mem.graph("C", hops=1)["nodes"])
        assert {"B", "C", "D"} <= nodes
        assert "A" not in nodes  # 2 hops away


# --------------------------------------------------------------------------- #
# #8 — consolidate actually merges and retires originals
# --------------------------------------------------------------------------- #

def test_consolidate_merges_and_retires_originals(db_path):
    cfg = AriadneConfig(db_path=db_path, embedding_dim=8, consolidation_threshold=0.7)
    with AriadneDB(cfg) as db:
        a = db.add_memory("the cat sat on the mat")["memory_id"]
        b = db.add_memory("the cat sat on the mat today")["memory_id"]
        groups = db.consolidate()
        assert groups == 1
        # Originals are soft-deleted.
        assert db.get_memory(a)["is_deleted"] is True
        assert db.get_memory(b)["is_deleted"] is True
        # A consolidated memory exists and is linked to both originals.
        rows = db.conn.execute("SELECT COUNT(*) FROM consolidations").fetchone()[0]
        assert rows == 1
        links = db.conn.execute("SELECT COUNT(*) FROM memory_links "
                                "WHERE link_type='consolidated'").fetchone()[0]
        assert links == 2


# --------------------------------------------------------------------------- #
# #9 — reads no longer mutate; only recall records access (batched, once)
# --------------------------------------------------------------------------- #

def test_get_memory_is_pure_read(db_path):
    with AriadneDB(AriadneConfig(db_path=db_path, embedding_dim=8)) as db:
        mid = db.add_memory("read me", np.random.randn(8).astype(np.float32))["memory_id"]
        for _ in range(5):
            db.get_memory(mid)
        assert db.get_memory(mid)["access_count"] == 0
        # Vector search is also a pure read.
        db.vector_search(np.random.randn(8).astype(np.float32), k=3)
        assert db.get_memory(mid)["access_count"] == 0


def test_recall_records_access_once(db_path):
    with AriadneMemory(db_path=db_path, embedding_dim=8) as mem:
        mid = mem.remember("deployment runbook for the api")["memory_id"]
        mem.recall("deployment runbook", k=5)
        # Exactly one access recorded for the surfaced memory (no 2-3x inflation).
        assert mem._db.get_memory(mid)["access_count"] == 1


# --------------------------------------------------------------------------- #
# importance clamping to [0, 1]
# --------------------------------------------------------------------------- #

def test_importance_clamped_on_add_and_update(db_path):
    with AriadneDB(AriadneConfig(db_path=db_path, embedding_dim=8)) as db:
        low = db.add_memory("too low", importance=-5.0)["memory_id"]
        high = db.add_memory("too high", importance=9.0)["memory_id"]
        assert db.get_memory(low)["importance"] == 0.0
        assert db.get_memory(high)["importance"] == 1.0

        db.update_memory(low, importance=2.0)
        assert db.get_memory(low)["importance"] == 1.0
        db.update_memory(high, importance=-1.0)
        assert db.get_memory(high)["importance"] == 0.0


# --------------------------------------------------------------------------- #
# #11 — access log pruning and purge of soft-deleted rows
# --------------------------------------------------------------------------- #

def test_prune_access_log(db_path):
    with AriadneDB(AriadneConfig(db_path=db_path, embedding_dim=8)) as db:
        mid = db.add_memory("noisy")["memory_id"]
        for _ in range(10):
            db.touch_memory(mid)
        assert db.conn.execute("SELECT COUNT(*) FROM access_log").fetchone()[0] == 10
        db.prune_access_log(keep_per_memory=3)
        assert db.conn.execute("SELECT COUNT(*) FROM access_log").fetchone()[0] == 3


def test_purge_deleted(db_path):
    with AriadneDB(AriadneConfig(db_path=db_path, embedding_dim=8)) as db:
        keep = db.add_memory("keep me")["memory_id"]
        drop = db.add_memory("drop me")["memory_id"]
        db.delete_memory(drop, hard=False)
        purged = db.purge_deleted(older_than_seconds=0.0)
        assert purged == 1
        assert db.get_memory(drop) is None
        assert db.get_memory(keep) is not None
        assert db.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE id=?", (drop,)
        ).fetchone()[0] == 0


# --------------------------------------------------------------------------- #
# #12 — thread safety: concurrent writers/readers don't crash or corrupt
# --------------------------------------------------------------------------- #

def test_concurrent_storage_is_safe(db_path):
    """Storage layer: a single AriadneDB shared across threads must not crash
    (old code used a thread-bound connection + unlocked FAISS) and must not lose
    writes. Content is unique so only exact-hash dedup applies — deterministic.
    """
    db = AriadneDB(AriadneConfig(db_path=db_path, embedding_dim=8))
    db.open()
    errors: list[Exception] = []

    def worker(tid: int) -> None:
        try:
            for i in range(25):
                db.add_memory(f"t{tid} item {i} unique {tid}-{i}",
                              np.random.randn(8).astype(np.float32))
                db.vector_search(np.random.randn(8).astype(np.float32), k=3)
                db.fts_search("item", k=3)
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    try:
        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"concurrent access raised: {errors[:3]}"
        assert db.stats()["active_memories"] == 4 * 25
        assert db.stats()["faiss_vectors"] == 4 * 25
    finally:
        db.close()


def test_concurrent_memory_interface_smoke(db_path):
    """Interface layer (with the MinHash dedup index): concurrent remember/recall
    must not raise. Exact counts are non-deterministic here because dedup may
    merge similar content, so we only assert no errors and a sane state.
    """
    import uuid

    mem = AriadneMemory(db_path=db_path, embedding_dim=8)
    errors: list[Exception] = []

    def worker(tid: int) -> None:
        try:
            for _ in range(20):
                mem.remember(f"thread {tid} note {uuid.uuid4().hex}", importance=0.5)
                mem.recall(f"thread {tid}", k=3)
        except Exception as e:  # pragma: no cover - failure path
            errors.append(e)

    try:
        threads = [threading.Thread(target=worker, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"concurrent access raised: {errors[:3]}"
        assert mem.stats()["active_memories"] >= 1
    finally:
        mem.close()

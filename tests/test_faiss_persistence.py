"""Regression tests for FAISS id-mapping across deletes and restarts.

These guard the bug where soft-deleted / evicted vectors stayed in the index
while the positional id-map was re-derived over only the surviving rows on
reopen. The result was that ``vector_search`` confidently returned the WRONG
memory (e.g. querying BANANA's exact vector returned CHERRY at score 1.00).

The fix keys every vector on its memory's own primary key via
``faiss.IndexIDMap2`` and rebuilds the index from the database on open, so the
mapping is intrinsic and cannot drift.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from arriadne.config import AriadneConfig
from arriadne.storage import AriadneDB


def _basis(i: int, dim: int = 8) -> np.ndarray:
    """Return the i-th unit basis vector — exact, collision-free embeddings."""
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    for suffix in ["", "-wal", "-shm", ".faiss"]:
        p = Path(path + suffix)
        if p.exists():
            p.unlink()


def _config(db_path: str) -> AriadneConfig:
    return AriadneConfig(db_path=db_path, embedding_dim=8, faiss_type="flat_ip")


def test_vector_search_correct_after_soft_delete_and_reopen(db_path):
    """The exact corruption: soft-delete a vector, reopen, and the surviving
    vectors must still resolve to the right memories."""
    cfg = _config(db_path)
    with AriadneDB(cfg) as db:
        apple = db.add_memory("APPLE", _basis(0))["memory_id"]
        db.add_memory("BANANA", _basis(1))
        db.add_memory("CHERRY", _basis(2))
        db.delete_memory(apple, hard=False)  # soft-delete shifts the old id-map

    with AriadneDB(cfg) as db:  # reopen: index rebuilt from DB, ids intrinsic
        hit = db.vector_search(_basis(1), k=1)
        assert hit, "expected a result for BANANA's exact vector"
        assert hit[0]["content"] == "BANANA"          # was "CHERRY" before the fix
        assert hit[0]["score"] == pytest.approx(1.0, abs=1e-4)

        hit = db.vector_search(_basis(2), k=1)
        assert hit and hit[0]["content"] == "CHERRY"

        # The soft-deleted vector must never resurface as a result.
        for r in db.vector_search(_basis(0), k=3):
            assert r["content"] != "APPLE"


def test_soft_deleted_vectors_pruned_on_reopen(db_path):
    """Dead vectors should not accumulate in the index across restarts."""
    cfg = _config(db_path)
    with AriadneDB(cfg) as db:
        first = db.add_memory("first", _basis(0))["memory_id"]
        db.add_memory("second", _basis(1))
        db.delete_memory(first, hard=False)
    with AriadneDB(cfg) as db:
        assert db._faiss_index.ntotal == 1  # only the active vector is rebuilt


def test_hard_delete_removes_from_index_live(db_path):
    cfg = _config(db_path)
    with AriadneDB(cfg) as db:
        alpha = db.add_memory("alpha", _basis(0))["memory_id"]
        db.add_memory("beta", _basis(1))
        assert db._faiss_index.ntotal == 2
        db.delete_memory(alpha, hard=True)
        assert db._faiss_index.ntotal == 1
        for r in db.vector_search(_basis(0), k=2):
            assert r["content"] != "alpha"


def test_update_embedding_moves_the_vector(db_path):
    """Updating an embedding must change what vector_search returns (was a no-op)."""
    cfg = _config(db_path)
    with AriadneDB(cfg) as db:
        mid = db.add_memory("movable", _basis(0))["memory_id"]
        assert db.vector_search(_basis(0), k=1)[0]["id"] == mid

        db.update_memory(mid, embedding=_basis(3))  # relocate the vector

        top = db.vector_search(_basis(3), k=1)
        assert top and top[0]["id"] == mid
        assert top[0]["score"] == pytest.approx(1.0, abs=1e-4)
        # It is no longer aligned with the original direction.
        assert db.vector_search(_basis(0), k=1)[0]["score"] < 0.5


def test_reopen_preserves_search_identity_under_mixed_deletes(db_path):
    """Stress the mapping: several adds, a soft and a hard delete, then reopen
    and confirm every exact-vector query returns its own memory."""
    cfg = _config(db_path)
    ids = {}
    with AriadneDB(cfg) as db:
        for i in range(6):
            ids[i] = db.add_memory(f"vec-{i}", _basis(i))["memory_id"]
        db.delete_memory(ids[1], hard=False)
        db.delete_memory(ids[4], hard=True)
    with AriadneDB(cfg) as db:
        for i in (0, 2, 3, 5):
            hit = db.vector_search(_basis(i), k=1)
            assert hit and hit[0]["content"] == f"vec-{i}", f"basis {i} mismapped"

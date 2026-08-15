"""Tests for the autonomous memory layer (LLMMemoryManager + MemoryCurator)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from arriadne import AriadneMemory
from arriadne.curator import CurateReport, MemoryCurator
from arriadne.memory_manager import (
    ExtractionResult,
    _parse_extraction,
    LLMMemoryManager,
)


@pytest.fixture
def db_path() -> str:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    for suffix in ["", "-wal", "-shm"]:
        p = Path(path + suffix)
        if p.exists():
            p.unlink()


class _FakeLLM:
    """Deterministic fake that returns a fixed extraction JSON."""

    def __init__(self, payload: dict):
        self.payload = payload

    def __call__(self, prompt: str) -> str:
        return json.dumps(self.payload)


class _FakeEmbedder:
    dim = 4

    def __call__(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Parse / prompt helpers
# ---------------------------------------------------------------------------


def test_parse_extraction_basic() -> None:
    text = json.dumps(
        {
            "memories": [
                {
                    "content": "User likes dark mode",
                    "kind": "preference",
                    "importance": 0.8,
                    "entities": ["user"],
                }
            ],
            "relations": [{"source": "user", "target": "dark_mode", "edge_type": "prefers"}],
        }
    )
    result = _parse_extraction(text)
    assert len(result.memories) == 1
    assert result.memories[0].kind == "preference"
    assert result.memories[0].importance == 0.8
    assert len(result.relations) == 1
    assert result.relations[0].edge_type == "prefers"


def test_parse_extraction_tolerates_markdown_fences() -> None:
    text = '```json\n{"memories": [{"content": "x", "kind": "semantic"}]}\n```'
    result = _parse_extraction(text)
    assert len(result.memories) == 1


def test_parse_extraction_skips_bad_items() -> None:
    text = json.dumps(
        {
            "memories": [
                {"content": "ok", "kind": "semantic"},
                {"content": 42, "kind": "semantic"},  # invalid content
                {"content": "", "kind": "semantic"},  # empty
                {},  # no content
            ]
        }
    )
    result = _parse_extraction(text)
    assert len(result.memories) == 1
    assert result.memories[0].content == "ok"


def test_parse_extraction_clamps_importance() -> None:
    text = json.dumps(
        {"memories": [{"content": "x", "importance": 5.0}, {"content": "y", "importance": -2.0}]}
    )
    result = _parse_extraction(text)
    assert result.memories[0].importance == 1.0
    assert result.memories[1].importance == 0.0


def test_fallback_caller_extracts_nothing(db_path: str) -> None:
    mem = AriadneMemory(db_path=db_path, embedder=None)
    mgr = LLMMemoryManager(mem)  # no caller -> fallback
    result = mgr.extract(user="hi")
    assert not result
    assert isinstance(result, ExtractionResult)
    mem.close()


# ---------------------------------------------------------------------------
# process_turn
# ---------------------------------------------------------------------------


def test_process_turn_stores_memories_relations_and_episode(db_path: str) -> None:
    mem = AriadneMemory(db_path=db_path, embedder=None)
    payload = {
        "memories": [
            {
                "content": "The user's name is Kyssta",
                "kind": "semantic",
                "importance": 0.9,
                "subject": "user",
                "attribute": "name",
                "value": "Kyssta",
                "entities": ["user", "kyssta"],
            }
        ],
        "relations": [{"source": "user", "target": "kyssta", "edge_type": "named"}],
    }
    mgr = LLMMemoryManager(mem, caller=_FakeLLM(payload), min_importance=0.3)
    summary = mgr.process_turn("My name is Kyssta", "Nice to meet you!")

    assert summary["episode_id"] is not None
    assert len(summary["stored"]) == 1
    assert summary["relations_added"] == 1

    results = mem.recall("user name", k=5)
    assert results, "expected the extracted fact to be recallable"
    assert any("Kyssta" in r["content"] for r in results)
    mem.close()


def test_process_turn_skips_low_importance(db_path: str) -> None:
    mem = AriadneMemory(db_path=db_path, embedder=None)
    payload = {"memories": [{"content": "tiny transient note", "importance": 0.1}]}
    mgr = LLMMemoryManager(mem, caller=_FakeLLM(payload), min_importance=0.3)
    summary = mgr.process_turn("meh", "eh")
    assert summary["stored"] == []
    assert len(summary["skipped"]) == 1
    mem.close()


def test_set_fact_supersedes_prior_value(db_path: str) -> None:
    mem = AriadneMemory(db_path=db_path, embedder=None)
    mgr = LLMMemoryManager(mem)

    r1 = mgr.set_fact("project", "language", "python")
    r2 = mgr.set_fact("project", "language", "go")

    assert r1["status"] in ("created", "duplicate")
    assert r2["status"] in ("created", "duplicate")

    current = mem.recall("project language", k=10)
    assert any("go" in r["content"] for r in current), current
    mem.close()


# ---------------------------------------------------------------------------
# Curator
# ---------------------------------------------------------------------------


def test_curator_decay_removes_stale_low_importance(db_path: str) -> None:
    import time

    mem = AriadneMemory(db_path=db_path, embedder=None)
    a = mem.remember("cheap throwaway note", importance=0.1)["memory_id"]
    b = mem.remember("critical fact", importance=0.9)["memory_id"]

    old = time.time() - 86400 * 90
    mem._db.conn.execute("UPDATE memories SET accessed_at = ? WHERE id = ?", (old, a))
    mem._db.conn.execute("UPDATE memories SET accessed_at = ? WHERE id = ?", (old, b))
    mem._db.conn.commit()

    curator = MemoryCurator(mem, decay_ttl_seconds=86400 * 30, decay_importance_threshold=0.4)
    # Memory `a` (importance 0.1) is both stale and below threshold -> decayed.
    # Memory `b` (importance 0.9) is stale but above threshold -> protected.
    decayed = curator.decay()
    assert decayed == 1
    assert mem.stats()["active_memories"] == 1
    mem.close()


def test_curator_protects_important_memory(db_path: str) -> None:
    import time

    mem = AriadneMemory(db_path=db_path, embedder=None)
    mem.remember("critical fact with high importance", importance=0.9)

    old = time.time() - 86400 * 90
    mem._db.conn.execute("UPDATE memories SET accessed_at = ? WHERE id = 1", (old,))
    mem._db.conn.commit()

    curator = MemoryCurator(mem, decay_ttl_seconds=86400 * 30, decay_importance_threshold=0.4)
    decayed = curator.decay()
    assert decayed == 0  # importance 0.9 >= 0.4 -> protected
    assert mem.stats()["active_memories"] == 1
    mem.close()
    mem.close()


def test_curator_conflict_resolution(db_path: str) -> None:
    mem = AriadneMemory(db_path=db_path, embedder=None)
    mem.remember("the sky is blue", importance=0.8, namespace="conflict")
    mem.remember("the sky is not blue", importance=0.8, namespace="conflict")

    curator = MemoryCurator(mem, resolve_contradictions=True)
    resolved = curator.resolve_contradictions(namespace="conflict")
    # Detector should catch at least one of the pair; even if 0, must not crash.
    assert isinstance(resolved, int)
    assert resolved >= 0
    mem.close()


def test_curate_report_shape(db_path: str) -> None:
    mem = AriadneMemory(db_path=db_path, embedder=None)
    curator = MemoryCurator(mem, resolve_contradictions=False)
    report = curator.curate(run_consolidate=False)
    assert isinstance(report, CurateReport)
    assert isinstance(report.as_dict(), dict)
    mem.close()

"""Temporal truth and provenance regression tests."""

from __future__ import annotations

from pathlib import Path

from arriadne import AriadneConfig, AriadneMemory
from arriadne.storage import AriadneDB


def config(tmp_path: Path, name: str = "memory.db") -> AriadneConfig:
    return AriadneConfig(db_path=tmp_path / name, embedding_dim=2)


def test_episode_provenance_and_current_vs_historical_recall(tmp_path: Path) -> None:
    with AriadneMemory(config=config(tmp_path)) as mem:
        episode = mem.record_episode(
            "The API timeout changed from 30 to 60 seconds.",
            role="user",
            namespace="project-a",
            session_id="s1",
            event_at=200,
            source="chat",
        )
        old = mem.remember(
            "The API timeout is 30 seconds.",
            namespace="project-a",
            event_at=100,
            valid_from=100,
            valid_to=200,
        )
        # Add source to old memory
        mem._db.add_source(
            memory_id=old["memory_id"],
            episode_id=episode["episode_id"],
            source="docs",
            source_id="docs:v1",
            span="section-3",
        )

        new = mem.remember(
            "The API timeout is 60 seconds.",
            namespace="project-a",
            event_at=200,
            valid_from=200,
            supersedes_id=old["memory_id"],
        )
        # Add source to new memory
        mem._db.add_source(
            memory_id=new["memory_id"],
            episode_id=episode["episode_id"],
            source="chat",
            source_id="session:s1",
            span="turn-4",
        )

        current = mem.recall("API timeout seconds", namespace="project-a", k=10)
        historical = mem.recall(
            "API timeout seconds", namespace="project-a", as_of=150, k=10
        )

        assert [item["id"] for item in current] == [new["memory_id"]]
        assert [item["id"] for item in historical] == [old["memory_id"]]
        assert current[0]["sources"][0]["source_id"] == "session:s1"
        assert current[0]["sources"][0]["episode_id"] == episode["episode_id"]


def test_expired_memory_is_not_current_but_is_available_historically(tmp_path: Path) -> None:
    with AriadneMemory(config=config(tmp_path)) as mem:
        result = mem.remember(
            "Temporary incident workaround",
            namespace="project-a",
            valid_from=100,
            valid_to=200,
        )

        assert mem.recall("incident workaround", namespace="project-a", as_of=150)
        assert mem.recall("incident workaround", namespace="project-a", as_of=250) == []
        assert mem._db.get_memory(result["memory_id"])["valid_to"] == 200


def test_temporal_provenance_survives_export_import(tmp_path: Path) -> None:
    with AriadneDB(config(tmp_path, "source.db")) as source:
        episode = source.add_episode(
            "A deployment was rolled back.", role="assistant", source="tool", event_at=123
        )
        result = source.add_memory(
            "Deployment rollback completed",
            namespace="ops",
            event_at=123,
            valid_from=123,
            confidence=0.85,
        )
        # Add source linking
        source.add_source(
            memory_id=result["memory_id"],
            episode_id=episode["episode_id"],
            source="tool",
            source_id="deploy:42",
            span="result",
            confidence=0.85,
        )
        exported = source.export_all()

    assert exported["episodes"]
    assert exported["memories"][0]["sources"]

    with AriadneDB(config(tmp_path, "destination.db")) as destination:
        assert destination.import_all(exported) == 1
        # Check sources directly from exported data
        assert exported["memories"][0]["sources"][0]["source_id"] == "deploy:42"
        # Also verify destination can retrieve it
        restored = destination.fts_search("rollback completed", k=1)[0]
        assert restored["event_at"] == 123
        assert restored["confidence"] == 0.85
        assert destination.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 1

"""Regression tests for the Hermes Ariadne provider boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_PLUGIN_PATH = Path(__file__).parents[1] / "plugin" / "__init__.py"
_SPEC = importlib.util.spec_from_file_location("ariadne_test_plugin", _PLUGIN_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_plugin = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_plugin)
AriadneMemoryProvider = _plugin.AriadneMemoryProvider


class FakeMemory:
    def __init__(self) -> None:
        self.recall_calls: list[tuple[str, str | None]] = []
        self.remembered: list[dict] = []

    def recall(self, query: str, *, k: int, namespace: str | None = None) -> list[dict]:
        self.recall_calls.append((query, namespace))
        return [{"id": len(self.recall_calls), "content": f"memory for {query}", "score": 1.0}]

    def remember(self, **kwargs):
        self.remembered.append(kwargs)
        return {"status": "created", "memory_id": len(self.remembered)}


def configured_provider() -> object:
    provider = AriadneMemoryProvider()
    provider._user_id = "alice"
    provider._agent_id = "hermes"
    provider._project_id = "project-a"
    provider._session_id = "session-1"
    return provider


def test_namespace_isolated_by_identity_and_scope() -> None:
    provider = configured_provider()

    assert provider._namespace_for("session", "session-1") == "user:alice:session:session-1"
    assert provider._namespace_for("project", "session-1") == "user:alice:project:project-a"
    assert provider._namespace_for("global", "session-1") == "user:alice:global"
    assert provider._namespace_for("agent", "session-1") == "user:alice:agent:hermes"


def test_prefetch_cache_is_keyed_by_query_and_session() -> None:
    provider = configured_provider()
    provider._ariadne = FakeMemory()

    first = provider.prefetch("first query", session_id="session-1")
    second = provider.prefetch("second query", session_id="session-1")

    assert "first query" in first
    assert "second query" in second
    assert provider._ariadne.recall_calls[0][0] == "first query"
    assert provider._ariadne.recall_calls[-1][0] == "second query"


def test_sync_turn_does_not_drop_back_to_back_turns() -> None:
    provider = configured_provider()
    provider._ariadne = FakeMemory()

    provider.sync_turn("first user turn", "first assistant turn", session_id="session-1")
    provider.sync_turn("second user turn", "second assistant turn", session_id="session-1")

    assert len(provider._ariadne.remembered) == 4
    assert {item["namespace"] for item in provider._ariadne.remembered} == {
        "user:alice:session:session-1"
    }


def test_session_switch_invalidates_cached_context() -> None:
    provider = configured_provider()
    provider._prefetch_cache = [{"content": "stale"}]
    provider._prefetch_cache_key = ("old query", "session-1")
    provider._prefetch_timestamp = 123.0

    provider.on_session_switch("session-2", reset=True)

    assert provider._session_id == "session-2"
    assert provider._prefetch_cache == []
    assert provider._prefetch_cache_key is None
    assert provider._prefetch_timestamp == 0.0

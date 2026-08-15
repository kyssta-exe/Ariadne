"""Tests for the adapter integrations (MCP server, OpenAI Agents tools, LangGraph store)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from arriadne import AriadneMemory


@pytest.fixture
def db_path() -> str:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    for suffix in ["", "-wal", "-shm"]:
        p = Path(path + suffix)
        if p.exists():
            p.unlink()


@pytest.fixture
def memory(db_path: str) -> AriadneMemory:
    m = AriadneMemory(db_path=db_path, embedder=None)
    yield m
    m.close()


# ---------------------------------------------------------------------------
# MCP server (dependency-free protocol driver)
# ---------------------------------------------------------------------------


def _call(server, method: str, params: dict, req_id: int = 1) -> dict:
    from arriadne.integrations.mcp_server import AriadneMCPServer  # noqa: F401

    msg = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}
    resp = server.handle_request(msg)
    assert resp is not None, f"{method} returned no response"
    return resp


def test_mcp_initialize_and_tools_list(memory: AriadneMemory) -> None:
    from arriadne.integrations.mcp_server import AriadneMCPServer

    srv = AriadneMCPServer(memory)
    resp = _call(srv, "initialize", {}, req_id=1)
    assert resp["result"]["serverInfo"]["name"] == "ariadne-memory"
    assert "tools" in resp["result"]["capabilities"]

    tools = _call(srv, "tools/list", {}, req_id=2)
    names = {t["name"] for t in tools["result"]["tools"]}
    assert {"ariadne_recall", "ariadne_remember", "ariadne_forget", "ariadne_stats"} <= names


def test_mcp_recall_and_stats(memory: AriadneMemory) -> None:
    from arriadne.integrations.mcp_server import AriadneMCPServer

    memory.remember(content="Kyssta prefers dark mode", importance=0.9, namespace="test")

    srv = AriadneMCPServer(memory, default_namespace="test")

    rec = _call(
        srv,
        "tools/call",
        {
            "name": "ariadne_recall",
            "arguments": {"query": "dark mode", "namespace": "test", "k": 5},
        },
        req_id=3,
    )
    assert rec["result"]["isError"] is False
    body = json.loads(rec["result"]["content"][0]["text"])
    assert body["count"] >= 1
    assert any("dark mode" in r["content"] for r in body["results"])

    stats = _call(srv, "tools/call", {"name": "ariadne_stats", "arguments": {}}, req_id=4)
    stats_body = json.loads(stats["result"]["content"][0]["text"])
    assert stats_body["active_memories"] >= 1


def test_mcp_remember_and_forget(memory: AriadneMemory) -> None:
    from arriadne.integrations.mcp_server import AriadneMCPServer

    srv = AriadneMCPServer(memory, default_namespace="test")

    rem = _call(
        srv,
        "tools/call",
        {
            "name": "ariadne_remember",
            "arguments": {"content": "Remember this", "namespace": "test"},
        },
        req_id=5,
    )
    body = json.loads(rem["result"]["content"][0]["text"])
    assert body["status"] == "created"

    rec = _call(
        srv,
        "tools/call",
        {
            "name": "ariadne_recall",
            "arguments": {"query": "Remember this", "namespace": "test", "k": 1},
        },
        req_id=6,
    )
    rec_body = json.loads(rec["result"]["content"][0]["text"])
    assert rec_body["results"]

    mid = rec_body["results"][0]["id"]
    forget = _call(
        srv,
        "tools/call",
        {"name": "ariadne_forget", "arguments": {"memory_id": mid}},
        req_id=7,
    )
    forget_body = json.loads(forget["result"]["content"][0]["text"])
    assert forget_body["forgotten"] is True


def test_mcp_errors(memory: AriadneMemory) -> None:
    from arriadne.integrations.mcp_server import AriadneMCPServer

    srv = AriadneMCPServer(memory)

    # unknown method -> -32601
    bad = _call(srv, "bogus", {}, req_id=9)
    assert bad["error"]["code"] == -32601

    # notification (no id) -> no response
    assert srv.handle_request({"jsonrpc": "2.0", "method": "tools/list"}) is None

    # tools/call with an invalid name -> -32602
    bad_tool = _call(srv, "tools/call", {"name": "nope", "arguments": {}}, req_id=10)
    assert bad_tool["error"]["code"] == -32602


# ---------------------------------------------------------------------------
# OpenAI Agents SDK tools (dependency-free JSON methods)
# ---------------------------------------------------------------------------


def test_agents_tools_round_trip(memory: AriadneMemory) -> None:
    from arriadne.integrations.openai_agents import AriadneTools

    tools = AriadneTools(memory, default_namespace="test")

    # remember
    r = tools.remember(json.dumps({"content": "Project uses Python", "namespace": "test"}))
    assert json.loads(r)["status"] == "created"

    # recall
    r2 = tools.recall(json.dumps({"query": "project python", "namespace": "test", "k": 5}))
    body = json.loads(r2)
    assert body["count"] >= 1

    # stats
    r3 = tools.stats()
    assert json.loads(r3)["active_memories"] >= 1

    # forget
    mid = body["results"][0]["id"]
    r4 = tools.forget(json.dumps({"memory_id": mid, "hard": False}))
    assert json.loads(r4)["forgotten"] is True


def test_agents_tools_bad_input(memory: AriadneMemory) -> None:
    from arriadne.integrations.openai_agents import AriadneTools

    tools = AriadneTools(memory)
    # invalid JSON -> error result, not an exception
    err = json.loads(tools.recall("not json"))
    assert "error" in err
    # missing query
    err2 = json.loads(tools.recall('{"k": 3}'))
    assert "error" in err2


# ---------------------------------------------------------------------------
# LangGraph store adapter
# ---------------------------------------------------------------------------


def test_langgraph_store_requires_langgraph(memory: AriadneMemory) -> None:
    pytest.importorskip("langgraph", reason="only if the optional dep is present")
    from arriadne.integrations.langgraph import AriadneStore

    store = AriadneStore(memory)
    # Namespace-tuple comes back flattened; just ensure the object exists.
    assert store._NS_PREFIX == "langgraph"


def test_langgraph_module_imports_without_dependency() -> None:
    """The module must import even when langgraph is not installed."""
    import arriadne.integrations.langgraph as lg

    # AriadneStore is always present as a class (constructor guards the import).
    assert hasattr(lg, "AriadneStore")

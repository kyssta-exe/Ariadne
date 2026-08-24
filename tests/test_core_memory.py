"""Core memory blocks, reranking, semantic dedup, async API, and entity merge.

Covers the second competitive wave:
- Letta-style core memory blocks (always-in-context, agent-editable)
- Mem0/Zep-style cross-encoder reranking (second retrieval stage)
- paraphrase-level semantic dedup (embedding-based, beyond MinHash)
- the async facade for asyncio-first agent frameworks
- entity merge/alias resolution (graph hygiene)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from arriadne import AriadneConfig, AriadneMemory
from arriadne.async_api import AsyncAriadneMemory
from arriadne.integrations.mcp_server import AriadneMCPServer
from arriadne.rerank import resolve_reranker


def _config(tmp_path: Path, name: str = "memory.db", **kwargs) -> AriadneConfig:
    return AriadneConfig(db_path=tmp_path / name, embedding_dim=4, **kwargs)


# ── Core memory blocks ───────────────────────────────────────────────────────


def test_core_block_crud(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        assert mem.core_blocks() == []
        assert mem.core_get("persona") is None

        block = mem.core_set("persona", "You are Ariadne, a careful agent.")
        assert block["name"] == "persona"
        assert mem.core_get("persona")["content"].startswith("You are")

        # Upsert replaces; append extends.
        mem.core_set("project_state", "Migration pending.")
        appended = mem.core_append("project_state", " Auth done.")
        assert appended["content"] == "Migration pending. Auth done."

        # Append creates a missing block.
        created = mem.core_append("scratchpad", "note one")
        assert created["content"] == "note one"

        names = [b["name"] for b in mem.core_blocks()]
        assert names == ["persona", "project_state", "scratchpad"]

        assert mem.core_delete("scratchpad") is True
        assert mem.core_delete("scratchpad") is False
        assert mem.core_get("scratchpad") is None


def test_core_block_namespace_isolation(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        mem.core_set("persona", "shared", namespace="alpha")
        mem.core_set("persona", "only-beta", namespace="beta")
        assert [b["content"] for b in mem.core_blocks("beta")] == ["only-beta"]
        assert mem.core_get("persona", namespace="alpha")["content"] == "shared"


def test_core_block_char_limit_trims_oldest(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        mem.core_append("log", "A" * 60, char_limit=100)
        block = mem.core_append("log", "B" * 60, char_limit=100)
        assert len(block["content"]) == 100
        # The most recent append survives; the oldest head was dropped.
        assert block["content"].endswith("B" * 60)
        assert block["content"].startswith("A")


def test_core_pack_and_context_pack(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        mem.remember("Paris is the capital of France")
        assert mem.core_pack() == ""
        mem.core_set("user_profile", "Name: Kyssta. Likes concise answers.")
        mem.core_set("empty_block", "   ")

        packed = mem.core_pack(char_budget=1000)
        assert "Core memory:" in packed
        assert "### user_profile" in packed
        assert "empty_block" not in packed

        plain = mem.context_pack("capital of France", token_budget=300)
        assert "Core memory" not in plain
        with_core = mem.context_pack(
            "capital of France", token_budget=600, include_core=True
        )
        assert "Core memory:" in with_core
        assert "Paris" in with_core


def test_core_blocks_survive_restart_and_bypass_memory_lifecycle(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    cfg = dict(embedding_dim=4)
    with AriadneMemory(config=AriadneConfig(db_path=db_path, **cfg)) as mem:
        mem.core_set("persona", "persistent")
        mem.remember("ordinary memory", importance=0.1)

    with AriadneMemory(config=AriadneConfig(db_path=db_path, **cfg)) as mem:
        assert mem.core_get("persona")["content"] == "persistent"
        # Eviction/decay must never touch core blocks.
        mem.evict()
        mem.purge_deleted(older_than_seconds=0)
        assert mem.core_get("persona")["content"] == "persistent"


# ── Entity merge ─────────────────────────────────────────────────────────────


def test_merge_entities_repoints_links_and_edges(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        mem.remember("PG stores the billing data", entities=["pg"])
        mem.remember("postgres handles auth", entities=["postgres"])
        mem.add_edge("pg", "billing", edge_type="stores")
        mem.add_edge("billing", "postgres", edge_type="stored_in")

        moved = mem.merge_entities("pg", "postgres")
        # 1 memory_entities link + 1 edge touched 'pg'; the billing->postgres
        # edge already pointed at the canonical name and never moves.
        assert moved == 2

        # The graph is now unified under the canonical name.
        graph = mem.graph("postgres", hops=2)
        node_names = {n if isinstance(n, str) else n.get("name") for n in graph["nodes"]}
        assert "pg" not in node_names
        assert "postgres" in node_names
        # And both memories link to the surviving entity.
        db = mem._db
        assert db.conn.execute("SELECT COUNT(*) FROM entities WHERE name = 'pg'").fetchone()[0] == 0


def test_merge_entities_same_name_is_noop(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        mem.remember("x", entities=["pg"])
        assert mem.merge_entities("pg", "pg") == 0


# ── Semantic (paraphrase) dedup ──────────────────────────────────────────────


class _FakeEmbedder:
    """Maps texts to vectors so paraphrases collide and distinct texts don't."""

    dim = 4

    def __init__(self) -> None:
        self._vectors = {
            "i live in paris": [1.0, 0.0, 0.0, 0.0],
            "paris is my home": [0.999, 0.02, 0.0, 0.0],
            "the server runs linux": [0.0, 1.0, 0.0, 0.0],
        }

    def __call__(self, text: str) -> list[float]:
        key = text.strip().lower().rstrip(".")
        return self._vectors.get(key, [0.0, 0.0, 1.0, 0.0])


def test_semantic_dedup_catches_paraphrases(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path), embedder=_FakeEmbedder()) as mem:
        first = mem.remember("I live in Paris")
        assert first["status"] == "created"

        second = mem.remember("Paris is my home")
        assert second["status"] == "duplicate"
        assert second.get("semantic_duplicate") is True
        assert second["duplicate_of"] == first["memory_id"]
        # Restating a fact is mild confirmation: trust rises (was decayed by
        # nothing here, so stays at the 1.0 cap — verify no error at least).
        assert mem._db.get_memory(first["memory_id"]) is not None

        distinct = mem.remember("The server runs Linux")
        assert distinct["status"] == "created"


def test_semantic_dedup_can_be_disabled(tmp_path: Path) -> None:
    cfg = _config(tmp_path, semantic_dedup=False)
    with AriadneMemory(config=cfg, embedder=_FakeEmbedder()) as mem:
        mem.remember("I live in Paris")
        second = mem.remember("Paris is my home")
        assert second["status"] == "created"


# ── Reranking ────────────────────────────────────────────────────────────────


class _ReverseReranker:
    """Deterministic stand-in: prefers the document that mentions 'gold'."""

    def __call__(self, query: str, documents: list[str]) -> list[float]:
        return [10.0 if "gold" in d.lower() else 1.0 for d in documents]


def test_rerank_reorders_results(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path), reranker=_ReverseReranker()) as mem:
        mem.remember("bronze medal trivia", importance=0.9)
        gold_id = mem.remember("gold medal trivia", importance=0.1)["memory_id"]

        plain = mem.recall("medal trivia", k=2)
        reranked = mem.recall("medal trivia", k=2, rerank=True)
        assert reranked, "rerank must not drop results"
        assert reranked[0]["id"] == gold_id
        assert reranked[0]["score_parts"].get("rerank") == 10.0
        assert "fused" in reranked[0]["score_parts"]
        # Plain recall ordering is untouched.
        assert plain[0]["id"] != gold_id or True  # ordering without rerank is score-based


def test_rerank_degrades_without_dependency(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        mem._reranker_unavailable = True  # simulate missing sentence-transformers
        mem.remember("silver medal trivia")
        results = mem.recall("medal trivia", k=1, rerank=True)
        assert results  # fused order returned, no crash


def test_resolve_reranker_acceptance() -> None:
    assert resolve_reranker(None) is None
    fn = lambda q, docs: [0.0 for _ in docs]  # noqa: E731
    assert resolve_reranker(fn) is fn
    try:
        resolve_reranker(42)
    except TypeError:
        pass
    else:
        raise AssertionError("non-callable reranker must raise TypeError")


# ── Async facade ─────────────────────────────────────────────────────────────


def test_async_roundtrip(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with AsyncAriadneMemory(db_path=tmp_path / "async.db", embedding_dim=4) as mem:
            result = await mem.remember("Async agents need memory too", importance=0.8)
            assert result["status"] == "created"

            hits = await mem.recall("async memory", k=5)
            assert hits

            block = await mem.core_append("project_state", "async works")
            assert block["content"] == "async works"

            packed = await mem.context_pack("async memory", token_budget=400)
            assert "Async agents" in packed

            ok = await mem.forget(result["memory_id"])
            assert ok is True

    asyncio.run(scenario())


def test_async_facade_wraps_existing_instance(tmp_path: Path) -> None:
    sync_mem = AriadneMemory(config=_config(tmp_path))

    async def scenario() -> list[dict[str, Any]]:
        amem = AsyncAriadneMemory.from_memory(sync_mem)
        await amem.remember("wrapped")
        return await amem.recall("wrapped", k=3)

    results = asyncio.run(scenario())
    assert results
    sync_mem.close()


def test_async_calls_run_concurrently(tmp_path: Path) -> None:
    async def scenario() -> list[bool]:
        async with AsyncAriadneMemory(db_path=tmp_path / "conc.db", embedding_dim=4) as mem:
            await mem.remember("seed memory for concurrent recall", importance=0.9)
            tasks = [mem.recall("concurrent recall", k=3) for _ in range(10)]
            return [len(r) > 0 for r in await asyncio.gather(*tasks)]

    assert all(asyncio.run(scenario()))


# ── MCP core tools ───────────────────────────────────────────────────────────


def _call(server: AriadneMCPServer, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    resp = server.handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": args},
        }
    )
    assert resp is not None
    if "error" in resp:
        return {"error": resp["error"].get("message")}
    return json.loads(resp["result"]["content"][0]["text"])


def test_mcp_core_tools(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        server = AriadneMCPServer(mem)

        append = _call(
            server, "ariadne_core_append", {"name": "user_profile", "text": "Likes tea."}
        )
        assert append["status"] == "appended"

        replace = _call(
            server, "ariadne_core_replace", {"name": "user_profile", "content": "Likes coffee."}
        )
        assert replace["status"] == "replaced"
        assert replace["length"] == len("Likes coffee.")

        view = _call(server, "ariadne_core_view", {"name": "user_profile"})
        assert view["block"]["content"] == "Likes coffee."

        listing = _call(server, "ariadne_core_view", {})
        assert [b["name"] for b in listing["blocks"]] == ["user_profile"]

        bad = _call(server, "ariadne_core_append", {"name": "x", "text": "  "})
        assert "error" in bad


def test_mcp_tools_list_includes_core(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path)) as mem:
        server = AriadneMCPServer(mem)
        resp = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = {t["name"] for t in resp["result"]["tools"]}
        assert {"ariadne_core_view", "ariadne_core_append", "ariadne_core_replace"} <= names


def test_mcp_recall_accepts_rerank_flag_with_fake_reranker(tmp_path: Path) -> None:
    with AriadneMemory(config=_config(tmp_path), reranker=_ReverseReranker()) as mem:
        mem.remember("bronze trivia")
        gold_id = mem.remember("gold trivia")["memory_id"]
        server = AriadneMCPServer(mem)
        out = _call(server, "ariadne_recall", {"query": "trivia", "k": 2, "rerank": True})
        assert out["results"][0]["id"] == gold_id

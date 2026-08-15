"""Dependency-free Model Context Protocol (MCP) server for Ariadne.

Implements the minimum subset of the MCP JSON-RPC 2.0 protocol that any MCP
host (Claude Desktop, VS Code Continue, Cursor, etc.) needs to expose Ariadne
as a memory tool provider:

    * ``initialize``              — handshake + capability advertisement
    * ``tools/list``              — enumerate the tools below
    * ``tools/call``              — invoke a tool by name with arguments

This module deliberately avoids the official ``mcp`` Python SDK so Ariadne has
no transitive dependency on a heavy SDK for what is a compact protocol surface.
The wire format is identical though, so hosts that expect an SDK server will
accept this implementation.

Tools exposed
-------------

* ``ariadne_recall``   — semantic + keyword search over memory
* ``ariadne_remember`` — store a new fact / preference / event
* ``ariadne_forget``   — soft-delete a memory by id
* ``ariadne_stats``    — counters / store snapshot

Usage::

    python -m arriadne.integrations.mcp_server --db-path memory.db
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable

from .. import AriadneMemory
from ..embeddings import SentenceTransformerEmbedder

SERVER_NAME = "ariadne-memory"
SERVER_VERSION = "0.1.0"
PROTOCOL_VERSION = "2024-11-05"


@dataclass(frozen=True)
class ToolDef:
    """Static description of a tool the server advertises to MCP hosts."""

    name: str
    description: str
    input_schema: dict[str, Any]


# JSON-Schema descriptors for the memory tools.
_TOOL_SCHEMAS: dict[str, ToolDef] = {
    "ariadne_recall": ToolDef(
        name="ariadne_recall",
        description=(
            "Search the agent's persistent memory for facts, preferences, or "
            "events relevant to a query. Returns ranked memories with scores."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "k": {"type": "integer", "description": "Max results.", "default": 5},
                "namespace": {"type": "string", "description": "Memory namespace."},
                "memory_type": {"type": "string", "description": "Optional memory-type filter."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    "ariadne_remember": ToolDef(
        name="ariadne_remember",
        description=(
            "Persist a new memory (fact, preference, or event) for the current "
            "agent. Returns the new memory id and status."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The content to remember."},
                "memory_type": {
                    "type": "string",
                    "enum": ["semantic", "episodic", "procedural"],
                    "default": "semantic",
                },
                "namespace": {"type": "string"},
                "importance": {"type": "number", "default": 0.5, "minimum": 0.0, "maximum": 1.0},
                "entities": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                    "description": "Entity names to attach (feeds the graph).",
                },
            },
            "required": ["content"],
            "additionalProperties": False,
        },
    ),
    "ariadne_forget": ToolDef(
        name="ariadne_forget",
        description="Soft-delete a memory by id.",
        input_schema={
            "type": "object",
            "properties": {
                "memory_id": {"type": "integer", "description": "Memory id to delete."},
                "hard": {"type": "boolean", "default": False},
            },
            "required": ["memory_id"],
            "additionalProperties": False,
        },
    ),
    "ariadne_stats": ToolDef(
        name="ariadne_stats",
        description="Return counters and type distribution for the memory store.",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    ),
}


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class AriadneMCPServer:
    """In-process MCP server bound to a single ``AriadneMemory`` instance."""

    def __init__(
        self,
        memory: AriadneMemory,
        *,
        default_namespace: str = "default",
    ) -> None:
        self.memory = memory
        self.default_namespace = default_namespace

    # -- public dispatch --------------------------------------------------

    def handle_request(self, msg: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch one JSON-RPC request and return a response.

        Returns ``None`` for notifications (JSON-RPC messages without an
        ``id``) — they expect no response, per the spec.
        """
        method = msg.get("method")
        req_id = msg.get("id")
        params = msg.get("params") or {}

        # Notifications carry no id: nothing to reply to.
        if req_id is None:
            return None

        if method == "initialize":
            return self._ok(
                req_id,
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                    "capabilities": {"tools": {}},
                },
            )
        if method == "ping":
            return self._ok(req_id, {})
        if method == "tools/list":
            return self._ok(
                req_id,
                {
                    "tools": [
                        {
                            "name": t.name,
                            "description": t.description,
                            "inputSchema": t.input_schema,
                        }
                        for t in _TOOL_SCHEMAS.values()
                    ]
                },
            )
        if method == "tools/call":
            return self._call_tool(req_id, params)
        return self._err(req_id, -32601, f"Method not found: {method}")

    def serve_stdio(self, stream_in: Any = None, stream_out: Any = None) -> None:
        """Run the server on stdio (or injected streams for testing).

        Reads newline-delimited JSON-RPC messages from ``stream_in`` and writes
        responses to ``stream_out``, flushing after each so hosts get results
        promptly.
        """
        stream_in = stream_in or sys.stdin
        stream_out = stream_out or sys.stdout

        for line in stream_in:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                err = self._err(None, -32700, f"Parse error: {exc}")
                stream_out.write(json.dumps(err) + "\n")
                stream_out.flush()
                continue
            resp = self.handle_request(msg)
            if resp is not None:
                stream_out.write(json.dumps(resp) + "\n")
                stream_out.flush()

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _ok(req_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _err(req_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
        err: dict[str, Any] = {"code": code, "message": message}
        if data is not None:
            err["data"] = data
        return {"jsonrpc": "2.0", "id": req_id, "error": err}

    def _call_tool(self, req_id: Any, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(args, dict):
            args = {}
        if name not in _TOOL_SCHEMAS:
            return self._err(req_id, -32602, f"Unknown tool: {name}")
        try:
            payload = _TOOL_HANDLERS[name](self, args)
        except (ValueError, KeyError, TypeError) as exc:
            return self._err(req_id, -32603, f"Tool error: {exc}")
        except Exception as exc:  # pragma: no cover - defensive
            return self._err(req_id, -32603, f"Internal error: {exc}")
        return self._ok(
            req_id,
            {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(payload, default=str, ensure_ascii=False),
                    }
                ],
                "isError": False,
            },
        )


# ---------------------------------------------------------------------------
# Tool handlers — thin; defer all heavy lifting to AriadneMemory.
# ---------------------------------------------------------------------------


def _ns(args: dict[str, Any], server: AriadneMCPServer) -> str:
    return args.get("namespace") or server.default_namespace


def _handle_recall(server: AriadneMCPServer, args: dict[str, Any]) -> dict[str, Any]:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query is required and must be a non-empty string")
    results = server.memory.recall(
        query=query,
        k=int(args.get("k") or 5),
        namespace=_ns(args, server),
        type_filter=args.get("memory_type"),
    )
    return {
        "count": len(results),
        "results": [
            {
                "id": r.get("id"),
                "content": r.get("content"),
                "memory_type": r.get("memory_type"),
                "importance": r.get("importance"),
                "score": r.get("score"),
                "namespace": r.get("namespace", "default"),
            }
            for r in results
        ],
    }


def _handle_remember(server: AriadneMCPServer, args: dict[str, Any]) -> dict[str, Any]:
    content = args.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content is required and must be a non-empty string")
    entities = args.get("entities") or None
    result = server.memory.remember(
        content=content,
        memory_type=args.get("memory_type", "semantic"),
        namespace=_ns(args, server),
        importance=float(args.get("importance", 0.5)),
        entities=entities,
    )
    return {
        "memory_id": result.get("memory_id"),
        "status": result.get("status"),
        "duplicate_of": result.get("duplicate_of"),
    }


def _handle_forget(server: AriadneMCPServer, args: dict[str, Any]) -> dict[str, Any]:
    mid = args.get("memory_id")
    if not isinstance(mid, int):
        raise ValueError("memory_id is required and must be an integer")
    ok = server.memory.forget(memory_id=mid, hard=bool(args.get("hard", False)))
    return {"memory_id": mid, "forgotten": ok}


def _handle_stats(server: AriadneMCPServer, args: dict[str, Any]) -> dict[str, Any]:
    stats = server.memory.stats()
    return {
        "active_memories": stats.get("active_memories", 0),
        "total_memories": stats.get("total_memories", 0),
        "deleted_memories": stats.get("deleted_memories", 0),
        "by_type": stats.get("by_type", {}),
        "by_namespace": stats.get("by_namespace", {}),
        "total_entities": stats.get("total_entities", 0),
        "total_edges": stats.get("total_edges", 0),
        "faiss_vectors": stats.get("faiss_vectors", 0),
        "faiss_type": stats.get("faiss_type", "none"),
    }


_TOOL_HANDLERS: dict[str, Callable[[AriadneMCPServer, dict[str, Any]], dict[str, Any]]] = {
    "ariadne_recall": _handle_recall,
    "ariadne_remember": _handle_remember,
    "ariadne_forget": _handle_forget,
    "ariadne_stats": _handle_stats,
}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ariadne MCP server (stdio JSON-RPC). Exposes memory tools to MCP hosts."
    )
    parser.add_argument("--db-path", default="memory.db", help="Path to the Ariadne SQLite DB.")
    parser.add_argument("--namespace", default="default", help="Default memory namespace.")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Optional sentence-transformers model name for the embedder.",
    )
    args = parser.parse_args(argv)

    embedder = None
    if args.embedding_model:
        embedder = SentenceTransformerEmbedder(model_name=args.embedding_model)

    memory = AriadneMemory(db_path=args.db_path, embedder=embedder)
    server = AriadneMCPServer(memory, default_namespace=args.namespace)
    server.serve_stdio()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

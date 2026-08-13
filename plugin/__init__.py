"""Ariadne Memory Provider for Hermes.

Implements the Hermes MemoryProvider interface with Ariadne (FAISS + FTS5 +
Knowledge Graph) as the backend. Exposes ``ariadne_*`` memory tools.

Installation:
    1. Place this directory in ~/.hermes/plugins/ariadne/
    2. Run: hermes memory setup  (select 'ariadne')
    3. Or set in ~/.hermes/config.yaml:
           memory:
             provider: ariadne
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

# Hermes MemoryProvider ABC
import sys
_HERMES_AGENT = os.environ.get("HERMES_AGENT_HOME", "/usr/local/lib/hermes-agent")
if _HERMES_AGENT not in sys.path:
    sys.path.insert(0, _HERMES_AGENT)
from agent.memory_provider import MemoryProvider  # noqa: E402 — needs the sys.path insert above

logger = logging.getLogger(__name__)

# Throttle sync_turn() processing — skip writes more frequent than this interval
_SYNC_THROTTLE_SECONDS = 5.0
# Max memories to inject per turn to keep prompt context slim
_MAX_PREFETCH_RESULTS = 10


def register(ctx):
    """Register the Ariadne memory provider with Hermes."""
    provider = AriadneMemoryProvider()
    ctx.register_memory_provider(provider)


class AriadneMemoryProvider(MemoryProvider):
    """Ariadne-based memory provider for Hermes."""

    def __init__(self):
        super().__init__()
        self._ariadne = None
        self._shared = None
        self._session_id = ""
        self._user_id = "default"
        self._agent_id = "hermes"
        self._project_id = "default"
        self._scratchpad: Dict[str, str] = {}
        self._db_path = ""
        self._shared_db_path = ""
        self._hermes_home = ""
        self._prefetch_cache: List[Dict] = []
        self._prefetch_cache_key: tuple[str, str] | None = None
        self._prefetch_timestamp = 0.0

    # ── MemoryProvider interface ──────────────────────────────────────

    @property
    def name(self) -> str:
        return "ariadne"

    def is_available(self) -> bool:
        """Check if Ariadne is installed and configured."""
        import importlib.util
        return importlib.util.find_spec("arriadne") is not None

    def initialize(self, session_id: str, **kwargs) -> None:
        """Open/initialize the Ariadne database."""
        hermes_home = kwargs.get("hermes_home", os.path.expanduser("~/.hermes"))
        self._hermes_home = hermes_home
        self._session_id = session_id
        self._user_id = str(kwargs.get("user_id") or kwargs.get("user_id_alt") or "default")
        self._agent_id = str(kwargs.get("agent_identity") or "hermes")
        self._project_id = str(
            kwargs.get("project_id") or kwargs.get("agent_workspace") or "default"
        )
        self._prefetch_cache = []
        self._prefetch_cache_key = None
        self._prefetch_timestamp = 0.0

        # Primary DB
        db_dir = os.path.join(hermes_home, "ariadne")
        os.makedirs(db_dir, exist_ok=True)
        self._db_path = os.path.join(db_dir, "memory.db")

        # Shared surface DB
        shared_dir = os.path.join(db_dir, "shared")
        os.makedirs(shared_dir, exist_ok=True)
        self._shared_db_path = os.path.join(shared_dir, "memory.db")

        try:
            from arriadne import AriadneMemory, AriadneConfig
            
            config = AriadneConfig(
                db_path=self._db_path,
                embedding_dim=384,
                faiss_type="auto",
                dedup_threshold=0.5,
                dedup_num_perm=128,
            )
            self._ariadne = AriadneMemory(config=config)

            # Upgrade the legacy literal namespace without rewriting the
            # database: the scoped recall fallback keeps those rows readable.
            # New writes are always identity-scoped.
            shared_config = AriadneConfig(
                db_path=self._shared_db_path,
                embedding_dim=384,
                faiss_type="auto",
                dedup_threshold=0.5,
            )
            self._shared = AriadneMemory(config=shared_config)

            logger.info("Ariadne initialized: %s (%d active memories)",
                        self._db_path,
                        self._ariadne.stats().get("active_memories", 0))

        except Exception as e:
            logger.error("Ariadne init failed: %s", e)
            self._ariadne = None
            self._shared = None

    @staticmethod
    def _namespace_part(value: object) -> str:
        """Make an identity safe to embed in a colon-delimited namespace."""
        return str(value or "default").strip().replace(":", "%3A") or "default"

    def _namespace_for(self, scope: str = "session", session_id: str = "") -> str:
        """Return an identity-scoped namespace for a logical memory scope.

        The old provider used the literal namespace ``session`` for every
        user/session. That made the scope labels decorative and made cached
        recall cross-session. Keep the logical aliases in the tool API while
        storing namespaced data under the active Hermes identity.
        """
        requested = str(scope or "session").strip()
        lowered = requested.lower()
        user = self._namespace_part(self._user_id)

        if lowered in {"global", "user"}:
            return f"user:{user}:global"
        if lowered == "project":
            return f"user:{user}:project:{self._namespace_part(self._project_id)}"
        if lowered == "agent":
            return f"user:{user}:agent:{self._namespace_part(self._agent_id)}"
        if lowered == "session":
            current = session_id or self._session_id or "default"
            return f"user:{user}:session:{self._namespace_part(current)}"

        # Explicit namespaces remain supported, but cannot escape the active
        # user boundary. This also makes custom namespaces deterministic.
        prefix = f"user:{user}:"
        if requested.startswith(prefix):
            return requested
        return f"{prefix}custom:{self._namespace_part(requested)}"

    def _scoped_namespaces(self, session_id: str = "") -> list[str]:
        """Return safe recall namespaces, including pre-identity legacy data.

        Legacy databases written by pre-identity plugin releases stored memory
        under the literal namespaces ``default``/``session``, owned by the
        default identity. Only the default user reads those legacy rows; a
        distinct (named) identity must not see another identity's legacy data
        when sharing one database.
        """
        names = [
            self._namespace_for("session", session_id),
            self._namespace_for("project", session_id),
            self._namespace_for("agent", session_id),
            self._namespace_for("global", session_id),
        ]
        # The pre-identity plugin defaulted user_id to "default" and wrote to
        # these literals. They are legacy, not the active identity's rows.
        if str(self._user_id or "default").strip() == "default":
            names.extend(["default", "session"])
        return list(dict.fromkeys(names))

    def _scoped_recall(
        self, query: str, *, k: int, session_id: str = "", scope: str | None = None,
        namespace: str | None = None,
    ) -> list[Dict]:
        """Recall from one logical scope or all identity-safe scopes."""
        if not self._ariadne or not query.strip() or k <= 0:
            return []
        if namespace:
            namespaces = [self._namespace_for(namespace, session_id)]
        elif scope:
            namespaces = [self._namespace_for(scope, session_id)]
        else:
            namespaces = self._scoped_namespaces(session_id)

        by_id: dict[object, Dict] = {}
        for current in namespaces:
            try:
                for result in self._ariadne.recall(query, k=k, namespace=current):
                    memory_id = result.get("id")
                    previous = by_id.get(memory_id)
                    if previous is None or result.get("score", 0.0) > previous.get("score", 0.0):
                        by_id[memory_id] = result
            except Exception as exc:
                logger.debug("Ariadne scoped recall failed for %s: %s", current, exc)
        return sorted(by_id.values(), key=lambda item: item.get("score", 0.0), reverse=True)[:k]

    def system_prompt_block(self) -> str:
        """Return static system prompt block for Ariadne."""
        return ""  # No static block; memories are injected per-turn via prefetch()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant memories before each turn."""
        if not self._ariadne or not query.strip():
            return ""

        key = (query.strip(), session_id or self._session_id)
        # Cache by both query and session. A time-only cache returned the
        # previous turn's memories for a different query/session.
        if (
            self._prefetch_cache_key == key
            and (time.time() - self._prefetch_timestamp) < 2.0
        ):
            return self._format_recall_block(self._prefetch_cache)

        try:
            results = self._scoped_recall(
                query, k=_MAX_PREFETCH_RESULTS, session_id=session_id
            )
            self._prefetch_cache = results
            self._prefetch_cache_key = key
            self._prefetch_timestamp = time.time()
            if results:
                return self._format_recall_block(results)
        except Exception as e:
            logger.debug("Ariadne prefetch failed: %s", e)

        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Pre-warm the prefetch cache for next turn."""
        if not self._ariadne or not query.strip():
            return
        try:
            self._prefetch_cache = self._scoped_recall(
                query, k=_MAX_PREFETCH_RESULTS, session_id=session_id
            )
            self._prefetch_cache_key = (query.strip(), session_id or self._session_id)
            self._prefetch_timestamp = time.time()
        except Exception:
            pass

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs: Any,
    ) -> None:
        """Rebind session-scoped state when Hermes rotates the session id."""
        if not new_session_id:
            return
        self._session_id = new_session_id
        self._prefetch_cache = []
        self._prefetch_cache_key = None
        self._prefetch_timestamp = 0.0

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages: Optional[List[Dict]] = None) -> None:
        """Persist completed turn to memory."""
        if not self._ariadne:
            return

        effective_session = session_id or self._session_id or "default"
        namespace = self._namespace_for("session", effective_session)

        # Store user message
        if user_content.strip():
            try:
                self._ariadne.remember(
                    content=f"[USER] {user_content[:500]}",
                    memory_type="episodic",
                    importance=0.3,
                    metadata={
                        "source": "sync_turn",
                        "session_id": effective_session,
                        "user_id": self._user_id,
                        "agent_id": self._agent_id,
                    },
                    namespace=namespace,
                    scope="session",
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    project_id=self._project_id,
                    session_id=effective_session,
                )
            except Exception:
                pass

        # Store assistant response summary
        if assistant_content.strip():
            try:
                self._ariadne.remember(
                    content=f"[ASSISTANT] {assistant_content[:500]}",
                    memory_type="episodic",
                    importance=0.2,
                    metadata={
                        "source": "sync_turn",
                        "session_id": effective_session,
                        "user_id": self._user_id,
                        "agent_id": self._agent_id,
                    },
                    namespace=namespace,
                    scope="session",
                    user_id=self._user_id,
                    agent_id=self._agent_id,
                    project_id=self._project_id,
                    session_id=effective_session,
                )
            except Exception:
                pass

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "db_path", "description": "Path to Ariadne database (default: ~/.hermes/ariadne/memory.db)", "default": "~/.hermes/ariadne/memory.db", "required": False},
            {"key": "embedding_dim", "description": "Embedding dimension (must match model, default: 384)", "default": 384, "required": False},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """No-op — Ariadne uses defaults from initialize()."""

    # ── Tool schemas ───────────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return all memory tool schemas."""
        return [
            {
                "name": "ariadne_remember",
                "description": "Store a durable memory in Ariadne. Use for ANY fact, preference, identity, insight, or context that should persist across sessions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The memory content to store."},
                        "importance": {"type": "number", "description": "Importance 0.0-1.0. Default 0.5.", "default": 0.5},
                        "source": {"type": "string", "description": "Source tag: preference, fact, insight, identity, task, etc.", "default": "user"},
                        "scope": {"type": "string", "description": "'session' (default) or 'global'.", "default": "session"},
                        "namespace": {"type": "string", "description": "Isolation namespace. Defaults to scope, e.g. session/project/global.", "default": "session"},
                        "memory_type": {"type": "string", "description": "semantic, episodic, or procedural", "default": "semantic"},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "ariadne_recall",
                "description": "Search Ariadne for relevant memories. Hybrid ranking: FTS5 text + FAISS vector.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language query."},
                        "limit": {"type": "integer", "description": "Max results. Default 5.", "default": 5},
                        "namespace": {"type": "string", "description": "Namespace to search. Defaults to 'session'.", "default": "session"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "ariadne_context_pack",
                "description": "Pack relevant memories into a compact token-budget-aware context block.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language query."},
                        "token_budget": {"type": "integer", "description": "Maximum estimated tokens. Default 2000.", "default": 2000},
                        "namespace": {"type": "string", "description": "Namespace to search. Defaults to all active identity-safe scopes."},
                        "scope": {"type": "string", "description": "Logical scope filter: session, project, agent, or global."},
                        "include_scores": {"type": "boolean", "description": "Include relevance scores in each line.", "default": False},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "ariadne_stats",
                "description": "Return Ariadne memory statistics.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "ariadne_forget",
                "description": "Permanently delete a memory by ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "integer", "description": "ID of the memory to delete"},
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "ariadne_update",
                "description": "Update the content or importance of an existing memory by ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "integer", "description": "ID of the memory to update"},
                        "content": {"type": "string", "description": "New content (optional)"},
                        "importance": {"type": "number", "description": "New importance 0.0-1.0 (optional)"},
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "ariadne_invalidate",
                "description": "Soft-delete a memory (mark as superseded).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "integer", "description": "Memory ID to invalidate."},
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "ariadne_export",
                "description": "Export all Ariadne memories to a JSON file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "output_path": {"type": "string", "description": "File path for export JSON"},
                    },
                    "required": ["output_path"],
                },
            },
            {
                "name": "ariadne_import",
                "description": "Import memories from a JSON file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_path": {"type": "string", "description": "JSON file to import"},
                    },
                    "required": ["input_path"],
                },
            },
            {
                "name": "ariadne_diagnose",
                "description": "Run diagnostics on the Ariadne installation.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "ariadne_graph_query",
                "description": "Traverse the memory graph from a seed entity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "Entity name to start from."},
                        "hops": {"type": "integer", "description": "Max traversal depth (default 2)", "default": 2},
                    },
                    "required": ["entity"],
                },
            },
            {
                "name": "ariadne_graph_link",
                "description": "Declare a relationship between two entities.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Source entity"},
                        "target": {"type": "string", "description": "Target entity"},
                        "relationship": {"type": "string", "description": "Relationship label (e.g. 'uses', 'depends_on')", "default": "related"},
                        "weight": {"type": "number", "description": "Edge weight 0.0-1.0", "default": 0.5},
                    },
                    "required": ["source", "target", "relationship"],
                },
            },
            {
                "name": "ariadne_sleep",
                "description": "Run the Ariadne memory consolidation cycle. Compresses old working memories into summaries.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dry_run": {"type": "boolean", "description": "Report without writing changes.", "default": False},
                    },
                },
            },
            {
                "name": "ariadne_scratchpad_write",
                "description": "Write a temporary note to the scratchpad.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "ariadne_scratchpad_read",
                "description": "Read the scratchpad entries.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "ariadne_scratchpad_clear",
                "description": "Clear all scratchpad entries.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "ariadne_shared_remember",
                "description": "Store a memory in the shared surface DB (cross-agent).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Memory content"},
                        "kind": {"type": "string", "description": "meta | preference | correction | identity", "default": "meta"},
                        "importance": {"type": "number", "description": "Importance 0.0-1.0", "default": 0.8},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "ariadne_shared_recall",
                "description": "Search the shared surface DB.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "ariadne_shared_forget",
                "description": "Delete a shared surface memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "integer", "description": "Memory ID to delete"},
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "ariadne_shared_stats",
                "description": "Return shared surface DB stats.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    # ── Tool dispatch ──────────────────────────────────────────────────

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatch tool calls to the appropriate handler."""
        if not self._ariadne:
            return json.dumps({"error": "Ariadne not initialized"})

        handlers = {
            "ariadne_remember": self._handle_remember,
            "ariadne_recall": self._handle_recall,
            "ariadne_context_pack": self._handle_context_pack,
            "ariadne_stats": self._handle_stats,
            "ariadne_forget": self._handle_forget,
            "ariadne_update": self._handle_update,
            "ariadne_invalidate": self._handle_invalidate,
            "ariadne_export": self._handle_export,
            "ariadne_import": self._handle_import,
            "ariadne_diagnose": self._handle_diagnose,
            "ariadne_graph_query": self._handle_graph_query,
            "ariadne_graph_link": self._handle_graph_link,
            "ariadne_sleep": self._handle_sleep,
            "ariadne_scratchpad_write": self._handle_scratchpad_write,
            "ariadne_scratchpad_read": self._handle_scratchpad_read,
            "ariadne_scratchpad_clear": self._handle_scratchpad_clear,
            "ariadne_shared_remember": self._handle_shared_remember,
            "ariadne_shared_recall": self._handle_shared_recall,
            "ariadne_shared_forget": self._handle_shared_forget,
            "ariadne_shared_stats": self._handle_shared_stats,
        }

        handler = handlers.get(tool_name)
        if handler:
            try:
                return handler(args)
            except Exception as e:
                logger.error("Ariadne tool %s failed: %s", tool_name, e)
                return json.dumps({"error": str(e)})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # ── Tool handlers ─────────────────────────────────────────────────

    def _handle_remember(self, args: Dict) -> str:
        scope = args.get("scope", "session")
        namespace = self._namespace_for(args.get("namespace") or scope, self._session_id)
        metadata = {
            "source": args.get("source", "user"),
            "scope": scope,
            "session_id": self._session_id,
            "user_id": self._user_id,
            "agent_id": self._agent_id,
            "project_id": self._project_id,
        }
        result = self._ariadne.remember(
            content=args["content"],
            memory_type=args.get("memory_type", "semantic"),
            importance=args.get("importance", 0.5),
            metadata=metadata,
            namespace=namespace,
            scope=scope,
            user_id=self._user_id,
            agent_id=self._agent_id,
            project_id=self._project_id,
            session_id=self._session_id or None,
        )
        return json.dumps(result)

    def _handle_recall(self, args: Dict) -> str:
        results = self._scoped_recall(
            args["query"],
            k=args.get("limit", 5),
            session_id=self._session_id,
            namespace=args.get("namespace"),
            scope=args.get("scope"),
        )
        return json.dumps(self._simplify_results(results))

    def _handle_context_pack(self, args: Dict) -> str:
        namespace = args.get("namespace") or args.get("scope")
        namespaces = (
            [self._namespace_for(str(namespace), self._session_id)]
            if namespace
            else self._scoped_namespaces(self._session_id)
        )
        packed = self._ariadne.context_pack(
            args["query"],
            token_budget=args.get("token_budget", 2000),
            include_scores=args.get("include_scores", False),
            namespaces=namespaces,
        )
        return json.dumps({"query": args["query"], "context": packed})

    def _handle_stats(self, args: Dict) -> str:
        stats = self._ariadne.stats()
        return json.dumps(stats)

    def _handle_forget(self, args: Dict) -> str:
        mid = args["memory_id"]
        success = self._ariadne.forget(memory_id=mid, hard=True)
        return json.dumps({"deleted": success, "memory_id": mid})

    def _handle_update(self, args: Dict) -> str:
        success = self._ariadne.update(
            memory_id=args["memory_id"],
            content=args.get("content"),
            importance=args.get("importance"),
        )
        return json.dumps({"updated": success})

    def _handle_invalidate(self, args: Dict) -> str:
        mid = args["memory_id"]
        success = self._ariadne.forget(memory_id=mid, hard=False)
        return json.dumps({"invalidated": success, "memory_id": mid})

    def _handle_export(self, args: Dict) -> str:
        import json as _json
        # export_json() dumps every active memory; recall("") matches nothing
        # in FTS, so the old approach silently exported an empty list.
        data = self._ariadne.export_json()
        data["stats"] = self._ariadne.stats()
        with open(args["output_path"], "w") as f:
            _json.dump(data, f, indent=2, default=str)
        return json.dumps(
            {"exported": len(data.get("memories", [])), "path": args["output_path"]}
        )

    def _handle_import(self, args: Dict) -> str:
        import json as _json
        with open(args["input_path"]) as f:
            data = _json.load(f)
        memories = data.get("memories", data if isinstance(data, list) else [])
        imported = 0
        for m in memories:
            if isinstance(m, dict) and m.get("content"):
                result = self._ariadne.remember(
                    content=m["content"],
                    memory_type=m.get("type", "semantic"),
                    importance=m.get("importance", 0.5),
                    metadata=m.get("metadata"),
                    namespace=m.get("namespace", "default"),
                    scope=m.get("scope", "session"),
                    user_id=m.get("user_id"),
                    agent_id=m.get("agent_id"),
                    session_id=m.get("session_id"),
                    project_id=m.get("project_id"),
                )
                if result.get("status") == "created":
                    imported += 1
        return json.dumps({"imported": imported})

    def _handle_diagnose(self, args: Dict) -> str:
        try:
            stats = self._ariadne.stats()
            return json.dumps({
                "status": "healthy",
                "provider": "ariadne",
                "db_path": self._db_path,
                "active_memories": stats.get("active_memories", 0),
                "faiss_type": stats.get("faiss_type", "unknown"),
                "entities": stats.get("total_entities", 0),
                "edges": stats.get("total_edges", 0),
                "db_size_kb": stats.get("db_size_bytes", 0) / 1024,
            })
        except Exception as e:
            return json.dumps({"status": "unhealthy", "error": str(e)})

    def _handle_graph_query(self, args: Dict) -> str:
        result = self._ariadne.graph(
            entity=args["entity"],
            hops=args.get("hops", 2),
        )
        return json.dumps(result)

    def _handle_graph_link(self, args: Dict) -> str:
        self._ariadne.add_edge(
            source=args["source"],
            target=args["target"],
            edge_type=args.get("relationship", "related"),
            weight=args.get("weight", 0.5),
        )
        return json.dumps({"status": "created"})

    def _handle_sleep(self, args: Dict) -> str:
        dry_run = args.get("dry_run", False)
        if dry_run:
            stats = self._ariadne.stats()
            return json.dumps({
                "active_memories": stats["active_memories"],
                "message": "Dry run — no changes made",
            })
        consolidated = self._ariadne.consolidate()
        evicted = self._ariadne.evict()
        return json.dumps({
            "consolidated_groups": consolidated,
            "evicted": evicted,
        })

    def _handle_scratchpad_write(self, args: Dict) -> str:
        import hashlib
        key = hashlib.md5(args["content"].encode()).hexdigest()[:8]
        self._scratchpad[key] = args["content"]
        return json.dumps({"key": key, "stored": True})

    def _handle_scratchpad_read(self, args: Dict) -> str:
        return json.dumps(list(self._scratchpad.values()))

    def _handle_scratchpad_clear(self, args: Dict) -> str:
        self._scratchpad.clear()
        return json.dumps({"cleared": True})

    # ── Shared surface ────────────────────────────────────────────────

    def _handle_shared_remember(self, args: Dict) -> str:
        if not self._shared:
            return json.dumps({"error": "shared surface not available"})
        result = self._shared.remember(
            content=args["content"],
            memory_type="semantic",
            importance=args.get("importance", 0.8),
            metadata={"kind": args.get("kind", "meta"), "source": "shared"},
            namespace="shared",
            scope="global",
        )
        return json.dumps(result)

    def _handle_shared_recall(self, args: Dict) -> str:
        if not self._shared:
            return json.dumps([])
        results = self._shared.recall(
            query=args["query"],
            k=args.get("limit", 5),
            namespace="shared",
        )
        return json.dumps(self._simplify_results(results))

    def _handle_shared_forget(self, args: Dict) -> str:
        if not self._shared:
            return json.dumps({"error": "shared surface not available"})
        mid = args["memory_id"]
        success = self._shared.forget(memory_id=mid, hard=True)
        return json.dumps({"deleted": success, "memory_id": mid})

    def _handle_shared_stats(self, args: Dict) -> str:
        if not self._shared:
            return json.dumps({"error": "shared surface not available"})
        stats = self._shared.stats()
        return json.dumps(stats)

    # ── Helpers ───────────────────────────────────────────────────────

    def _simplify_results(self, results: List[Dict]) -> List[Dict]:
        """Strip large fields from recall results for clean JSON output."""
        out = []
        for r in results:
            item = {
                "id": r.get("id"),
                "content": r.get("content", "")[:300],
                "memory_type": r.get("memory_type", ""),
                "namespace": r.get("namespace", "default"),
                "scope": r.get("scope", "session"),
                "importance": r.get("importance", 0.5),
                "score": r.get("score", 0),
                "score_parts": r.get("score_parts", {}),
                "confidence": r.get("confidence", 1.0),
                "search_type": r.get("search_type", ""),
            }
            out.append(item)
        return out

    def _format_recall_block(self, results: List[Dict]) -> str:
        """Format recall results as a memory context block."""
        if not results:
            return ""
        lines = ["", "══════════════════════════════════════════════"]
        lines.append("ARIADNE MEMORY (context for this turn)")
        lines.append("══════════════════════════════════════════════")
        for i, r in enumerate(results[:_MAX_PREFETCH_RESULTS], 1):
            content = r.get("content", "")[:200].replace("\n", " ")
            lines.append(f"{i}. [{r.get('memory_type', '?')}] {content}")
        lines.append("")
        return "\n".join(lines)

    def shutdown(self) -> None:
        """Close databases."""
        try:
            if self._ariadne:
                self._ariadne.close()
        except Exception:
            logger.debug("Error closing Ariadne DB on shutdown", exc_info=True)
        try:
            if self._shared:
                self._shared.close()
        except Exception:
            logger.debug("Error closing shared Ariadne DB on shutdown", exc_info=True)

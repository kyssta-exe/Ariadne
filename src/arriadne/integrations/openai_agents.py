"""OpenAI Agents SDK integration: expose Ariadne as agent-callable tools.

The OpenAI Agents SDK (``openai-agents``) lets an agent invoke *tools* defined
with the ``@function_tool`` decorator. This module wraps an :class:`AriadneMemory`
in a set of self-contained tools so an agent can read and write its own memory
during a run — without the caller wiring up the prompts by hand.

Design goals:

* **Optional dependency.** Importing this module never requires
  ``openai-agents``. The tool *logic* lives in plain methods that accept a
  JSON string and return a JSON string, so it is usable with *any* agent
  framework (LangChain, custom loops, etc.).
* **SDK-aware wrapping.** When ``openai-agents`` is installed, call
  :meth:`AriadneTools.to_openai_agents_tools` to get fully-typed SDK tool
  objects ready to pass to an ``Agent``.

Tools exposed: ``recall``, ``remember``, ``forget``, ``stats``.
"""

from __future__ import annotations

import json
from typing import Any

from .. import AriadneMemory

function_tool: Any = None
try:  # pragma: no cover - exercised only when openai-agents is installed
    from agents import function_tool as _function_tool

    function_tool = _function_tool
    _AGENTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AGENTS_AVAILABLE = False


class AriadneTools:
    """Tool provider wrapping an :class:`AriadneMemory` for agent use.

    Each method accepts a JSON-string argument and returns a JSON-string
    result, matching the OpenAI Agents SDK convention for ``function_tool``
    callables. Instantiate with a live memory and an optional default
    namespace.
    """

    def __init__(self, memory: AriadneMemory, default_namespace: str = "default") -> None:
        self.memory = memory
        self.default_namespace = default_namespace

    def _ns(self, args: dict[str, Any]) -> str:
        return args.get("namespace") or self.default_namespace

    # -- core tool methods (JSON in / JSON out) ---------------------------

    def recall(self, arguments: str) -> str:
        """Search memory for facts/events relevant to the query.

        Args (JSON): query (str), k (int, default 5), namespace (str, optional),
        memory_type (str, optional).
        """
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"invalid JSON: {exc}"})
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return json.dumps({"error": "query is required"})
        results = self.memory.recall(
            query=query,
            k=int(args.get("k") or 5),
            namespace=self._ns(args),
            type_filter=args.get("memory_type"),
        )
        return json.dumps(
            {
                "count": len(results),
                "results": [
                    {
                        "id": r.get("id"),
                        "content": r.get("content"),
                        "memory_type": r.get("memory_type"),
                        "importance": r.get("importance"),
                        "score": r.get("score"),
                    }
                    for r in results
                ],
            },
            ensure_ascii=False,
        )

    def remember(self, arguments: str) -> str:
        """Store a new memory (fact, preference, or event).

        Args (JSON): content (str), memory_type (str, default semantic),
        namespace (str, optional), importance (float, default 0.5),
        entities (list[str], optional).
        """
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"invalid JSON: {exc}"})
        content = args.get("content")
        if not isinstance(content, str) or not content.strip():
            return json.dumps({"error": "content is required"})
        entities = args.get("entities") or None
        result = self.memory.remember(
            content=content,
            memory_type=args.get("memory_type", "semantic"),
            namespace=self._ns(args),
            importance=float(args.get("importance", 0.5)),
            entities=entities,
        )
        return json.dumps(result, ensure_ascii=False)

    def forget(self, arguments: str) -> str:
        """Soft-delete a memory by id.

        Args (JSON): memory_id (int), hard (bool, default false).
        """
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            return json.dumps({"error": f"invalid JSON: {exc}"})
        mid = args.get("memory_id")
        if not isinstance(mid, int):
            return json.dumps({"error": "memory_id (int) is required"})
        ok = self.memory.forget(memory_id=mid, hard=bool(args.get("hard", False)))
        return json.dumps({"memory_id": mid, "forgotten": ok})

    def stats(self, arguments: str = "") -> str:
        """Return counters and type distribution for the memory store."""
        s = self.memory.stats()
        return json.dumps(
            {
                "active_memories": s.get("active_memories", 0),
                "total_memories": s.get("total_memories", 0),
                "by_type": s.get("by_type", {}),
                "by_namespace": s.get("by_namespace", {}),
                "total_entities": s.get("total_entities", 0),
                "total_edges": s.get("total_edges", 0),
            },
            ensure_ascii=False,
        )

    # -- SDK-aware wrapping ------------------------------------------------

    def to_openai_agents_tools(self) -> list[Any]:
        """Return OpenAI Agents SDK ``function_tool`` objects.

        Raises:
            ImportError: If ``openai-agents`` is not installed.
        """
        if not _AGENTS_AVAILABLE or function_tool is None:
            raise ImportError(
                "to_openai_agents_tools requires the 'openai-agents' package. "
                "Install it with: pip install openai-agents"
            )

        @function_tool
        def ariadne_recall(arguments: str) -> str:
            return self.recall(arguments)

        @function_tool
        def ariadne_remember(arguments: str) -> str:
            return self.remember(arguments)

        @function_tool
        def ariadne_forget(arguments: str) -> str:
            return self.forget(arguments)

        @function_tool
        def ariadne_stats(arguments: str = "") -> str:
            return self.stats(arguments)

        return [ariadne_recall, ariadne_remember, ariadne_forget, ariadne_stats]


__all__ = ["AriadneTools"]

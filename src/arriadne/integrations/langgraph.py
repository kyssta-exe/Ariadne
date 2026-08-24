"""LangGraph integration: expose Ariadne as a ``BaseStore`` for long-term memory.

LangGraph's persistent memory (``langgraph.store``) is a key/value "BaseStore"
that checkpoint objects store memories in. When you build a LangGraph agent you
can pass a store so the graph can read/write durable memory between runs.

This adapter bridges Ariadne onto that interface: every LangGraph ``Item``
becomes an Ariadne memory, and vice versa. Namespace tuples from LangGraph map
onto Ariadne's ``namespace`` string so multi-tenant / per-thread isolation is
preserved.

Compatible with both the langgraph 0.x store protocol (``BaseItem`` class,
positional ``query``) and the 1.x protocol (``Item`` TypedDict, keyword-only
arguments): items are returned as plain dicts shaped like ``Item`` (key,
namespace, value, created_at, updated_at).

The official ``langgraph`` package is an *optional* dependency: importing this
module never requires it. Only instantiating :class:`AriadneStore` does.

Example::

    from arriadne.integrations.langgraph import AriadneStore

    store = AriadneStore.from_memory(mem)   # mem is an AriadneMemory
    # pass ``store=store`` to a LangGraph graph / checkpointer
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .. import AriadneMemory

# `langgraph.store.base` may not be installed; import it lazily so the module
# can always be imported, and only fail when the store is actually constructed.
try:  # pragma: no cover - exercised only when langgraph is installed
    from langgraph.store.base import (
        BaseStore,
        GetOp,
        ListNamespacesOp,
        PutOp,
        SearchOp,
    )

    _LANGRAPH_AVAILABLE = True
    _BaseStore: Any = BaseStore
    _PutOp: Any = PutOp
    _GetOp: Any = GetOp
    _SearchOp: Any = SearchOp
    _ListNamespacesOp: Any = ListNamespacesOp
except ImportError:  # pragma: no cover
    _LANGRAPH_AVAILABLE = False
    _BaseStore = object
    _PutOp = _GetOp = _SearchOp = _ListNamespacesOp = ()


def _ns_str(namespace: tuple[str, ...]) -> str:
    """Flatten a LangGraph namespace tuple into an Ariadne namespace string."""
    if not namespace:
        return "default"
    return "::".join(namespace)


def _item_to_text(value: dict[str, Any]) -> str:
    """Best-effort render of a non-string value as memory content."""
    if not value:
        return ""
    if len(value) == 1:
        only = next(iter(value.values()))
        return str(only)
    return str(value)


def _value_text(value: dict[str, Any]) -> str:
    content = value.get("content") or value.get("text")
    if isinstance(content, str) and content.strip():
        return content
    return _item_to_text(value)


def _matches_filter(value: dict[str, Any], filter: dict[str, Any] | None) -> bool:
    if filter is None:
        return True
    return all(value.get(k) == v for k, v in filter.items())


class AriadneStore(_BaseStore):  # type: ignore[misc]
    """Adapter turning an :class:`AriadneMemory` into a LangGraph ``BaseStore``.

    LangGraph calls these methods during graph execution (``put`` /
    ``search`` / ``get`` / ``delete``). Content stored through LangGraph is
    embedded by the provided embedder (if any) so hybrid Ariadne search over
    LangGraph memories works out of the box.

    Implementation notes:

    * Namespaces are flattened to Ariadne's colon-joined namespace field, and
      are written under a reserved ``langgraph`` scope so they never collide
      with memories written directly through Ariadne.
    * Items are returned as dicts shaped like langgraph's ``Item`` TypedDict:
      ``{key, namespace, value, created_at, updated_at}`` — accepted by both
      0.x and 1.x consumers.
    """

    # Prefix applied to namespaces so LangGraph KV rows are identifiable and
    # isolated from raw Ariadne memories.
    _NS_PREFIX = "langgraph"

    def __init__(self, memory: AriadneMemory) -> None:
        if not _LANGRAPH_AVAILABLE:
            raise ImportError(
                "AriadneStore requires the 'langgraph' package. "
                "Install it with: pip install langgraph"
            )
        self.memory = memory

    @classmethod
    def from_memory(cls, memory: AriadneMemory) -> AriadneStore:
        """Construct from an existing :class:`AriadneMemory`."""
        return cls(memory)

    # -- helpers ----------------------------------------------------------

    def _ns(self, namespace: tuple[str, ...]) -> str:
        return f"{self._NS_PREFIX}::{_ns_str(namespace)}"

    def _item(self, memory: dict[str, Any]) -> dict[str, Any]:
        """Wrap a single Ariadne memory dict as an Item-shaped dict."""
        value = {
            "content": memory.get("content", ""),
            **(memory.get("metadata") or {}),
        }
        # tags are consumed into metadata; keep the value dict self-contained.
        tags = memory.get("tags") or []
        if tags:
            value["tags"] = list(tags)
        created = memory.get("created_at") or 0.0
        updated = memory.get("updated_at") or created
        return {
            "key": str(memory.get("id")),
            "namespace": self._item_namespace(memory),
            "value": value,
            "created_at": datetime.fromtimestamp(float(created), tz=timezone.utc),
            "updated_at": datetime.fromtimestamp(float(updated), tz=timezone.utc),
        }

    @staticmethod
    def _item_namespace(memory: dict[str, Any]) -> tuple[str, ...]:
        ns = str(memory.get("namespace") or "default")
        parts = ns.split("::")
        if parts and parts[0] == "langgraph":
            parts = parts[1:]
        return tuple(parts) or ("default",)

    # -- write ------------------------------------------------------------

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Any = None,
        *,
        ttl: Any = None,
    ) -> None:
        """Write a memory under ``key``.

        The LangGraph ``key`` is recorded in metadata (``_lg_key``) so the
        adapter can later retrieve the same record by (namespace, key), since
        Ariadne stores no external key identifier of its own. An existing row
        with the same key in the same namespace is superseded so ``put``
        behaves as an upsert.
        """
        content = _value_text(value)
        metadata = dict(value)
        # Strip tags out of the stored value (Ariadne indexes them via `tags`).
        tags = metadata.pop("tags", None) if isinstance(metadata.get("tags"), list) else None
        metadata["_lg_key"] = key

        ns = self._ns(namespace)
        prior_id = self._find_key_id(ns, key)
        self.memory.remember(
            content=content,
            metadata=metadata,
            namespace=ns,
            tags=list(tags) if tags else None,
            supersedes_id=prior_id,
        )

    async def aput(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Any = None,
        *,
        ttl: Any = None,
    ) -> None:
        """Async variant mirroring LangGraph's ``aput``."""
        self.put(namespace, key, value, index, ttl=ttl)

    def _find_key_id(self, ns: str, key: str) -> int | None:
        """Find the active memory id stored under a LangGraph (namespace, key)."""
        for row in self.memory._db.recent_memories(limit=200, namespace=ns):
            meta = row.get("metadata") or {}
            if meta.get("_lg_key") == key:
                return int(row["id"])
        return None

    # -- read -------------------------------------------------------------

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: Any = None,
    ) -> dict[str, Any] | None:
        """Return the item stored at (namespace, key), or ``None``.

        Ariadne has no native (namespace, key) index, so the recent rows of the
        namespace are scanned for the marker metadata ``_lg_key``, falling
        back to an FTS lookup on the key itself.
        """
        ns = self._ns(namespace)
        direct = self._find_key_id(ns, key)
        if direct is not None:
            row = self.memory._db.get_memory(direct)
            if row is not None and not row.get("is_deleted"):
                return self._item(row)
        results = self.memory.recall(key, k=50, namespace=ns)
        for r in results:
            meta = r.get("metadata") or {}
            if meta.get("_lg_key") == key:
                return self._item(r)
        return None

    async def aget(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: Any = None,
    ) -> dict[str, Any] | None:
        """Async ``aget``."""
        return self.get(namespace, key)

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        query: str | None = None,
        *,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        max_distance: float | None = None,
        refresh_ttl: Any = None,
        **_extra: Any,
    ) -> list[dict[str, Any]]:
        """Search memories scoped to a LangGraph namespace prefix.

        Accepts both calling conventions: langgraph 1.x passes ``query``
        keyword-only, 0.x positionally.
        """
        q = query or ""
        results = (
            self.memory.recall(q, k=limit + offset, namespace=self._ns(namespace_prefix))
            if q
            else self._recent(limit, offset, namespace_prefix)
        )
        items = [self._item(r) for r in results]
        items = items[offset : offset + limit]
        if filter:
            items = [i for i in items if _matches_filter(i["value"], filter)]
        return items

    def _recent(self, limit: int, offset: int, namespace: tuple[str, ...]) -> list[dict[str, Any]]:
        """Fallback listing for when search has no query (newest first)."""
        return self.memory._db.recent_memories(limit=limit + offset, namespace=self._ns(namespace))

    async def asearch(
        self,
        namespace_prefix: tuple[str, ...],
        query: str | None = None,
        *,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        max_distance: float | None = None,
        refresh_ttl: Any = None,
        **_extra: Any,
    ) -> list[dict[str, Any]]:
        """Async ``asearch``."""
        return self.search(
            namespace_prefix,
            query,
            filter=filter,
            limit=limit,
            offset=offset,
            max_distance=max_distance,
            refresh_ttl=refresh_ttl,
        )

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete the memory at (namespace, key)."""
        found = self.get(namespace, key)
        if found is not None:
            self.memory.forget(int(found["key"]), hard=False)

    async def adelete(self, namespace: tuple[str, ...], key: str) -> None:
        """Async ``adelete``."""
        self.delete(namespace, key)

    def list_namespaces(
        self,
        prefix: tuple[str, ...] | None = None,
        *,
        suffix: Any = None,
        max_depth: Any = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """Return the distinct LangGraph namespaces currently in use.

        Ariadne namespaces under the reserved ``langgraph::`` prefix are split
        back into tuples; an optional prefix tuple filters hierarchically.
        """
        rows = self.memory._db.conn.execute(
            """SELECT DISTINCT namespace FROM memories
               WHERE namespace LIKE 'langgraph::%' AND is_deleted = 0
               ORDER BY namespace LIMIT ? OFFSET ?""",
            (limit + offset, offset),
        ).fetchall()
        namespaces: list[tuple[str, ...]] = []
        for row in rows:
            parts = str(row[0]).split("::")
            # Drop the reserved prefix; the rest are the caller's tuple parts.
            ns = tuple(parts[1:])
            if prefix is not None:
                if ns[: len(prefix)] != tuple(prefix):
                    continue
            if max_depth is not None and len(ns) > int(max_depth):
                ns = ns[: int(max_depth)]
            namespaces.append(ns)
        return namespaces

    async def alist_namespaces(
        self,
        prefix: tuple[str, ...] | None = None,
        *,
        suffix: Any = None,
        max_depth: Any = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """Async ``alist_namespaces``."""
        return self.list_namespaces(prefix, limit=limit, offset=offset)

    # -- batch (abstract in langgraph >= 1.0) ------------------------------

    def batch(self, ops: Any) -> list[Any]:
        """Execute a sequence of store operations, returning their results."""
        results: list[Any] = []
        for op in ops:
            if isinstance(op, _PutOp):
                self.put(op.namespace, op.key, op.value, op.index, ttl=op.ttl)
                results.append(None)
            elif isinstance(op, _GetOp):
                results.append(self.get(op.namespace, op.key))
            elif isinstance(op, _SearchOp):
                results.append(
                    self.search(
                        op.namespace_prefix,
                        query=op.query,
                        filter=op.filter,
                        limit=op.limit,
                        offset=op.offset,
                    )
                )
            elif isinstance(op, _ListNamespacesOp):
                results.append(
                    self.list_namespaces(
                        max_depth=op.max_depth, limit=op.limit, offset=op.offset
                    )
                )
            else:
                raise TypeError(f"Unsupported batch operation: {type(op).__name__}")
        return results

    async def abatch(self, ops: Any) -> list[Any]:
        """Async ``abatch`` — Ariadne is fully in-process, so it mirrors batch."""
        return self.batch(ops)

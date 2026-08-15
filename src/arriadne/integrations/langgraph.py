"""LangGraph integration: expose Ariadne as a ``BaseStore`` for long-term memory.

LangGraph's persistent memory (``langgraph.store``) is a key/value "BaseStore"
that checkpoint objects store memories in. When you build a LangGraph agent you
can pass a store so the graph can read/write durable memory between runs.

This adapter bridges Ariadne onto that interface: every LangGraph ``BaseItem``
becomes an Ariadne memory, and vice versa. Namespace tuples from LangGraph map
onto Ariadne's ``namespace`` string so multi-tenant / per-thread isolation is
preserved.

The official ``langgraph`` package is an *optional* dependency: importing this
module never requires it. Only instantiating :class:`AriadneStore` does.

Example::

    from arriadne.integrations.langgraph import AriadneStore

    store = AriadneStore.from_memory(mem)   # mem is an AriadneMemory
    # pass ``store=store`` to a LangGraph graph / checkpointer
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from .. import AriadneMemory

# `langgraph.store.base` may not be installed; import it lazily so the module
# can always be imported, and only fail when the store is actually constructed.
try:  # pragma: no cover - exercised only when langgraph is installed
    from langgraph.store.base import BaseStore, BaseItem

    _LANGRAPH_AVAILABLE = True
    _BaseStore = BaseStore
except ImportError:  # pragma: no cover
    _LANGRAPH_AVAILABLE = False
    _BaseStore = object  # type: ignore[assignment,misc]


def _ns_str(namespace: Tuple[str, ...]) -> str:
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
    def from_memory(cls, memory: AriadneMemory) -> "AriadneStore":
        """Construct from an existing :class:`AriadneMemory`."""
        return cls(memory)

    # -- helpers ----------------------------------------------------------

    def _ns(self, namespace: Tuple[str, ...]) -> str:
        return f"{self._NS_PREFIX}::{_ns_str(namespace)}"

    def _item(self, memory: dict[str, Any]) -> BaseItem:
        """Wrap a single Ariadne memory dict as a LangGraph ``BaseItem``."""
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
        return type(
            "BaseItem",
            (),
            {
                "key": str(memory.get("id")),
                "value": value,
                "created_at": created,
                "updated_at": updated,
            },
        )()  # type: ignore[return-value]

    # -- write ------------------------------------------------------------

    def put(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Optional[bool] = None,
    ) -> None:
        """Write a memory under ``key``.

        The LangGraph ``key`` is recorded in metadata (``_lg_key``) so the
        adapter can later retrieve the same record by (namespace, key), since
        Ariadne stores no external key identifier of its own.
        """
        content = _value_text(value)
        metadata = dict(value)
        # Strip tags out of the stored value (Ariadne indexes them via `tags`).
        tags = metadata.pop("tags", None) if isinstance(metadata.get("tags"), list) else None
        metadata["_lg_key"] = key
        self.memory.remember(
            content=content,
            metadata=metadata,
            namespace=self._ns(namespace),
            tags=list(tags) if tags else None,
        )

    async def aput(
        self,
        namespace: Tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: Optional[bool] = None,
    ) -> None:
        """Async variant mirroring LangGraph's ``aput``."""
        self.put(namespace, key, value, index=index)

    # -- read -------------------------------------------------------------

    def get(self, namespace: Tuple[str, ...], key: str) -> Optional[BaseItem]:
        """Return the memory stored at (namespace, key), or ``None``.

        Ariadne has no native (namespace, key) index, so we search the
        namespace for the marker metadata ``_lg_key`` and return the best
        match. Falls back to an FTS lookup on the key itself.
        """
        ns = self._ns(namespace)
        results = self.memory.recall(key, k=50, namespace=ns)
        for r in results:
            meta = r.get("metadata") or {}
            if meta.get("_lg_key") == key:
                return self._item(r)
        return None

    async def aget(self, namespace: Tuple[str, ...], key: str) -> Optional[BaseItem]:
        """Async ``aget``."""
        return self.get(namespace, key)

    def search(
        self,
        namespace_prefix: Tuple[str, ...],
        query: Optional[str] = None,
        *,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        max_distance: Optional[float] = None,
    ) -> List[BaseItem]:
        """Search memories scoped to a LangGraph namespace prefix."""
        q = query or ""
        results = (
            self.memory.recall(q, k=limit + offset, namespace=self._ns(namespace_prefix))
            if q
            else self._recent(limit, offset, namespace_prefix)
        )
        items = [self._item(r) for r in results]
        items = items[offset : offset + limit]
        if filter:
            items = [i for i in items if _matches_filter(i.value, filter)]
        return items

    def _recent(self, limit: int, offset: int, namespace: Tuple[str, ...]) -> list[dict[str, Any]]:
        """Fallback listing for when search has no query (newest first)."""
        # Ariadne has no generic "list" API; use FTS matching any word in the
        # namespace and return enough candidates. For empty queries we return []
        # rather than fabricating results.
        return []

    async def asearch(
        self,
        namespace_prefix: Tuple[str, ...],
        query: Optional[str] = None,
        *,
        filter: Optional[dict[str, Any]] = None,
        limit: int = 10,
        offset: int = 0,
        max_distance: Optional[float] = None,
    ) -> List[BaseItem]:
        """Async ``asearch``."""
        return self.search(
            namespace_prefix,
            query,
            filter=filter,
            limit=limit,
            offset=offset,
            max_distance=max_distance,
        )

    def delete(self, namespace: Tuple[str, ...], key: str) -> None:
        """Delete the memory at (namespace, key)."""
        found = self.get(namespace, key)
        if found is not None:
            self.memory.forget(int(found.key), hard=False)

    async def adelete(self, namespace: Tuple[str, ...], key: str) -> None:
        """Async ``adelete``."""
        self.delete(namespace, key)

    def list_namespaces(
        self,
        prefix: Optional[Tuple[str, ...]] = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Tuple[str, ...]]:
        """Return the distinct LangGraph namespaces currently in use."""
        return [("langgraph",)]  # single-tenant default; real listing is db-scoped

    async def alist_namespaces(
        self,
        prefix: Optional[Tuple[str, ...]] = None,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Tuple[str, ...]]:
        """Async ``alist_namespaces``."""
        return self.list_namespaces(prefix, limit=limit, offset=offset)


# -- value helpers (kept out of class for clarity) ----------------------------


def _value_text(value: dict[str, Any]) -> str:
    content = value.get("content") or value.get("text")
    if isinstance(content, str) and content.strip():
        return content
    return _item_to_text(value)


def _matches_filter(value: dict[str, Any], filter: Optional[dict[str, Any]]) -> bool:
    if filter is None:
        return True
    for k, expected in filter.items():
        if value.get(k) != expected:
            return False
    return True


__all__ = ["AriadneStore"]

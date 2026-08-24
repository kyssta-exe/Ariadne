"""Async facade over :class:`~arriadne.interface.AriadneMemory`.

Ariadne's core is synchronous by design — SQLite, FAISS, and MinHash are all
in-process, so there is no I/O to await. But modern agent frameworks
(LangGraph, OpenAI Agents, MCP hosts) are ``async``-first, and calling
blocking code directly on the event loop stalls every concurrent task.

This facade mirrors the frequently used surface as coroutines by delegating
to ``asyncio.to_thread``, so the event loop stays responsive while the
(small, local) operations run on worker threads. The underlying
:class:`AriadneMemory` is thread-safe — a single RLock serializes SQLite +
FAISS access — so one shared instance can back many concurrent coroutines.

Example::

    import asyncio
    from arriadne.async_api import AsyncAriadneMemory

    async def main():
        async with AsyncAriadneMemory(db_path="memory.db") as mem:
            await mem.remember("Paris is the capital of France")
            results = await mem.recall("capital of France", k=5)
        # or wrap an existing sync instance:
        # async_mem = AsyncAriadneMemory.from_memory(sync_mem)

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
from types import TracebackType
from typing import Any

from arriadne.interface import AriadneMemory

__all__ = ["AsyncAriadneMemory"]


class AsyncAriadneMemory:
    """Coroutine-friendly wrapper around a thread-safe :class:`AriadneMemory`.

    Args:
        memory: An existing ``AriadneMemory`` to wrap. When omitted, one is
            constructed from the remaining keyword arguments (same signature
            as ``AriadneMemory``).

    The synchronous instance is available as ``.sync`` for the rare
    operation not mirrored here.
    """

    def __init__(
        self,
        memory: AriadneMemory | None = None,
        **kwargs: Any,
    ) -> None:
        self.sync = memory if memory is not None else AriadneMemory(**kwargs)

    @classmethod
    def from_memory(cls, memory: AriadneMemory) -> AsyncAriadneMemory:
        """Wrap an existing synchronous :class:`AriadneMemory`."""
        return cls(memory)

    # -- lifecycle ----------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying store."""
        await asyncio.to_thread(self.sync.close)

    async def __aenter__(self) -> AsyncAriadneMemory:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()

    # -- write path ---------------------------------------------------------

    async def remember(self, content: str, **kwargs: Any) -> dict[str, Any]:
        """Store a memory. See ``AriadneMemory.remember``."""
        return await asyncio.to_thread(self.sync.remember, content, **kwargs)

    async def remember_many(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Store many memories in one transaction. See ``AriadneMemory.remember_many``."""
        return await asyncio.to_thread(self.sync.remember_many, items)

    async def forget(self, memory_id: int, hard: bool = False) -> bool:
        """Forget (delete) a memory. See ``AriadneMemory.forget``."""
        return await asyncio.to_thread(self.sync.forget, memory_id, hard=hard)

    async def update(self, memory_id: int, **kwargs: Any) -> bool:
        """Update a memory. See ``AriadneMemory.update``."""
        return await asyncio.to_thread(self.sync.update, memory_id, **kwargs)

    async def reinforce(self, memory_id: int, delta: float | None = None) -> float | None:
        """Confirm a memory (trust scoring). See ``AriadneMemory.reinforce``."""
        return await asyncio.to_thread(self.sync.reinforce, memory_id, delta)

    # -- read path ----------------------------------------------------------

    async def recall(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Recall memories. See ``AriadneMemory.recall`` (supports ``rerank=``)."""
        return await asyncio.to_thread(self.sync.recall, query, **kwargs)

    async def search_episodes(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        """Search raw session history. See ``AriadneMemory.search_episodes``."""
        return await asyncio.to_thread(self.sync.search_episodes, query, **kwargs)

    async def context_pack(self, query: str, **kwargs: Any) -> str:
        """Assemble a token-budgeted context block. See ``AriadneMemory.context_pack``."""
        return await asyncio.to_thread(self.sync.context_pack, query, **kwargs)

    async def session_context(self, **kwargs: Any) -> str:
        """Recent-session continuity block. See ``AriadneMemory.session_context``."""
        return await asyncio.to_thread(self.sync.session_context, **kwargs)

    # -- sessions -----------------------------------------------------------

    async def digest_session(self, session_id: str, **kwargs: Any) -> dict[str, Any]:
        """Distill a session into a digest memory. See ``AriadneMemory.digest_session``."""
        return await asyncio.to_thread(self.sync.digest_session, session_id, **kwargs)

    # -- core memory blocks ---------------------------------------------------

    async def core_set(self, name: str, content: str, namespace: str = "default") -> dict[str, Any]:
        """Create/replace a core memory block. See ``AriadneMemory.core_set``."""
        return await asyncio.to_thread(self.sync.core_set, name, content, namespace)

    async def core_append(self, name: str, text: str, namespace: str = "default") -> dict[str, Any]:
        """Append to a core memory block. See ``AriadneMemory.core_append``."""
        return await asyncio.to_thread(self.sync.core_append, name, text, namespace)

    async def core_blocks(self, namespace: str = "default") -> list[dict[str, Any]]:
        """List core memory blocks. See ``AriadneMemory.core_blocks``."""
        return await asyncio.to_thread(self.sync.core_blocks, namespace)

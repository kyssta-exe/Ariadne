"""Composable retrieval pipelines for Ariadne.

``AriadneMemory.recall`` is a fixed, optimized pipeline (hybrid search →
filters → confidence weighting). It is the right default — but power users
want to *compose* retrieval: swap the first stage, prepend a reranker, append
graph expansion, chain arbitrary stages. This module provides the protocol
and the standard building blocks, mirroring retriever interfaces popularized
by LangChain/LlamaIndex so custom stages feel familiar.

Two roles, one shared output shape (``list[memory-dict]`` with ``score``):

- **First stages** implement ``retrieve(query, k, **filters)``:
  :class:`FTSRetriever`, :class:`VectorRetriever`, :class:`HybridRetriever`.
- **Decorators** implement ``apply(results, query)`` (and the convenience
  ``retrieve`` that runs their base first): :class:`RerankRetriever`,
  :class:`ExpansionRetriever`.
- :class:`Pipeline` composes any first stage with decorators left-to-right.

Example::

    from arriadne.retrievers import (
        HybridRetriever, RerankRetriever, ExpansionRetriever, Pipeline,
    )
    from arriadne.rerank import CrossEncoderReranker

    pipe = Pipeline(
        HybridRetriever(mem),
        RerankRetriever(reranker=CrossEncoderReranker()),
        ExpansionRetriever(mem, hops=1),
    )
    results = pipe.retrieve("how to deploy", k=5)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from arriadne.interface import AriadneMemory

__all__ = [
    "ExpansionRetriever",
    "FTSRetriever",
    "HybridRetriever",
    "Pipeline",
    "RerankRetriever",
    "Retriever",
    "Transformer",
    "VectorRetriever",
]


@runtime_checkable
class Retriever(Protocol):
    """A first stage: maps a query to ranked memory dicts."""

    def retrieve(self, query: str, k: int = 10, **filters: Any) -> list[dict[str, Any]]: ...


@runtime_checkable
class Transformer(Protocol):
    """A decorator stage: transforms another stage's output."""

    def apply(self, results: list[dict[str, Any]], query: str) -> list[dict[str, Any]]: ...


class _BaseMemoryRetriever:
    """Shared plumbing for first stages bound to an ``AriadneMemory``."""

    def __init__(self, memory: AriadneMemory) -> None:
        self.memory = memory


class FTSRetriever(_BaseMemoryRetriever):
    """Keyword-only first stage (BM25 via FTS5)."""

    def retrieve(self, query: str, k: int = 10, **filters: Any) -> list[dict[str, Any]]:
        namespace = filters.pop("namespace", None)
        return self.memory._db.fts_search(query, k=k, namespace=namespace)


class VectorRetriever(_BaseMemoryRetriever):
    """Vector-only first stage; requires a query embedder on the memory."""

    def retrieve(self, query: str, k: int = 10, **filters: Any) -> list[dict[str, Any]]:
        if self.memory._embedder is None:
            return []
        import numpy as np

        embedding = np.asarray(self.memory._embedder(query), dtype=np.float32)
        namespace = filters.pop("namespace", None)
        return self.memory._db.vector_search(embedding, k=k, namespace=namespace)


class HybridRetriever(_BaseMemoryRetriever):
    """The default Ariadne pipeline as a composable first stage.

    Args:
        reranker: Optional reranker applied to the final set (equivalent to
            configuring the memory and calling ``recall(rerank=True)``).
    """

    def __init__(self, memory: AriadneMemory, reranker: Any = None) -> None:
        super().__init__(memory)
        if reranker is not None:
            memory._reranker = reranker
            memory._reranker_unavailable = False

    def retrieve(self, query: str, k: int = 10, **filters: Any) -> list[dict[str, Any]]:
        rerank = self.memory._reranker is not None
        return self.memory.recall(query, k=k, rerank=rerank, **filters)


class RerankRetriever:
    """Decorator: re-score any stage's output with a cross-encoder.

    Works standalone via ``retrieve`` (runs no base — reranks whatever a
    :class:`Pipeline` feeds it) and exposes ``apply`` for pipeline use.
    """

    def __init__(self, reranker: Any) -> None:
        self.reranker = reranker

    def apply(self, results: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        if len(results) > 1 and self.reranker is not None:
            scores = self.reranker(query, [r.get("content") or "" for r in results])
            for r, s in zip(results, scores, strict=False):
                parts = r.get("score_parts") or {}
                parts["fused"] = r.get("score")
                parts["rerank"] = float(s)
                r["score_parts"] = parts
                r["score"] = float(s)
            results = sorted(
                results, key=lambda item: (-item.get("score", 0.0), item.get("id", 0))
            )
        return results

    def retrieve(self, query: str, k: int = 10, **filters: Any) -> list[dict[str, Any]]:
        # Convenience when used as the only stage is meaningless without a
        # base; pipelines and explicit apply() are the intended usage.
        raise NotImplementedError("RerankRetriever is a decorator: use Pipeline or apply()")


class ExpansionRetriever:
    """Decorator: widen any stage's output through the entity graph."""

    def __init__(self, memory: AriadneMemory, *, hops: int = 1, limit: int = 10,
                 decay: float = 0.5) -> None:
        self.memory = memory
        self.hops = hops
        self.limit = limit
        self.decay = decay

    def apply(self, results: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        return self.memory.expand(
            results, hops=self.hops, limit=self.limit, decay=self.decay
        )

    def retrieve(self, query: str, k: int = 10, **filters: Any) -> list[dict[str, Any]]:
        return self.apply(self.memory.recall(query, k=k, **filters), query)


class Pipeline:
    """Compose a first stage with decorators, left-to-right.

    Example: ``Pipeline(HybridRetriever(mem), RerankRetriever(r), ExpansionRetriever(mem))``
    runs hybrid recall, reranks it, then expands the reranked seeds.
    """

    def __init__(self, first: Retriever, *transformers: Any) -> None:
        if not isinstance(first, Retriever):
            raise TypeError(f"first stage must implement retrieve(): {type(first).__name__}")
        self.first = first
        self.transformers = list(transformers)

    def retrieve(self, query: str, k: int = 10, **filters: Any) -> list[dict[str, Any]]:
        results = self.first.retrieve(query, k=k, **filters)
        for stage in self.transformers:
            applier = getattr(stage, "apply", None)
            if not callable(applier):
                raise TypeError(
                    f"pipeline stage {type(stage).__name__} implements neither "
                    "apply(results, query) nor a known decorator contract"
                )
            results = applier(results, query)
        return results

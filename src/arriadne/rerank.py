"""Optional cross-encoder reranking for Ariadne.

Hybrid recall fuses BM25 and vector rankings with Reciprocal Rank Fusion —
cheap and robust, but both signals are *retrieval-grade*: they score a query
against a document representation, not the pair jointly. A cross-encoder reads
``(query, document)`` together and is markedly better at ordering the top-k.
This module adds that second stage behind an optional dependency, mirroring
the reranking stage Mem0 and Zep expose.

Install the optional dependency::

    pip install "arriadne[embeddings]"     # sentence-transformers

then::

    from arriadne import AriadneMemory
    from arriadne.rerank import CrossEncoderReranker

    mem = AriadneMemory(db_path="memory.db", reranker=CrossEncoderReranker())
    results = mem.recall("how to deploy", k=5, rerank=True)

``recall(..., rerank=True)`` also works without a pre-configured reranker: the
default model from ``AriadneConfig.rerank_model`` is loaded lazily on first
use. When sentence-transformers is not installed the rerank request is
skipped with a logged warning and the fused ordering is returned unchanged,
so the dependency stays truly optional.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@runtime_checkable
class Reranker(Protocol):
    """Anything that scores (query, document) pairs jointly.

    A reranker is ``callable(query, documents) -> list[float]`` returning one
    relevance score per document, higher = more relevant.
    """

    def __call__(self, query: str, documents: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """Reranker backed by a sentence-transformers ``CrossEncoder``.

    Args:
        model_name: Any CrossEncoder-compatible model id. The default
            (``ms-marco-MiniLM-L-6-v2``) is small and CPU-friendly.
        device: Optional device string (e.g. ``"cpu"``, ``"cuda"``).

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANK_MODEL,
        *,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "CrossEncoderReranker requires the 'embeddings' extra. "
                "Install it with:  pip install \"arriadne[embeddings]\""
            ) from exc

        self.model_name = model_name
        self._model = CrossEncoder(model_name, device=device)
        logger.info("Loaded reranker model %s", model_name)

    def __call__(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = list(zip([query] * len(documents), documents, strict=True))
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


def resolve_reranker(reranker: Any) -> Reranker | None:
    """Coerce a user-supplied value into a Reranker (or None).

    Accepts a :class:`Reranker`, a plain callable, a model-name string (which
    loads a :class:`CrossEncoderReranker`), or None.
    """
    if reranker is None:
        return None
    if isinstance(reranker, str):
        return CrossEncoderReranker(reranker)
    if callable(reranker):
        return reranker  # type: ignore[no-any-return]
    raise TypeError(f"Unsupported reranker: {reranker!r}")

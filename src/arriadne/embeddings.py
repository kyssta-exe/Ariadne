"""Optional embedding helpers for Ariadne.

Ariadne's core stays dependency-light: it stores and searches whatever vectors
you give it. But "semantic" recall only works if *something* turns text into
vectors. Wiring an embedder here means ``remember``/``recall`` embed content and
queries automatically, so the documented quick-start actually returns results
instead of silently falling back to keyword-only search.

Install the optional dependency::

    pip install "arriadne[embeddings]"

then::

    from arriadne import AriadneMemory
    from arriadne.embeddings import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")  # 384-dim
    mem = AriadneMemory(db_path="memory.db", embedding_dim=embedder.dim,
                        embedder=embedder)
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class Embedder(Protocol):
    """Anything callable that turns text into a fixed-length vector.

    An embedder is ``callable(str) -> list[float]`` and exposes its output
    dimension via ``dim`` so callers can validate it against the configured
    ``embedding_dim``.
    """

    dim: int

    def __call__(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Embedder backed by a ``sentence-transformers`` model (loaded lazily).

    Args:
        model_name: Any sentence-transformers model id. Defaults to the small,
            fast ``all-MiniLM-L6-v2`` (384 dimensions).
        device: Optional device string (e.g. ``"cpu"``, ``"cuda"``).
        normalize: If True, ask the model to L2-normalize embeddings. Ariadne
            also normalizes internally, so this is mostly a no-op safety net.

    Raises:
        ImportError: If ``sentence-transformers`` is not installed.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        *,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "SentenceTransformerEmbedder requires the 'embeddings' extra. "
                "Install it with:  pip install \"arriadne[embeddings]\""
            ) from exc

        self.model_name = model_name
        self._normalize = normalize
        self._model = SentenceTransformer(model_name, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        logger.info("Loaded embedding model %s (dim=%d)", model_name, self.dim)

    def __call__(self, text: str) -> list[float]:
        vec = self._model.encode(
            text,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )
        return [float(x) for x in vec.tolist()]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Encode many texts at once (faster than calling per-text)."""
        arr = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            convert_to_numpy=True,
        )
        return [[float(x) for x in row] for row in arr.tolist()]


def resolve_embedder(embedder: Any) -> Embedder | None:
    """Coerce a user-supplied value into an Embedder (or None).

    Accepts an :class:`Embedder`, a plain callable (``str -> sequence``), or a
    model-name string (which loads a :class:`SentenceTransformerEmbedder`).
    """
    if embedder is None:
        return None
    if isinstance(embedder, str):
        return SentenceTransformerEmbedder(embedder)
    if callable(embedder):
        return embedder  # type: ignore[return-value]
    raise TypeError(f"Unsupported embedder: {embedder!r}")

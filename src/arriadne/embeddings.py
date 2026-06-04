"""Built-in embedding engine for Ariadne.

Zero-config embeddings using ONNX Runtime + quantized models.
Auto-downloads model on first use, caches locally.

Supports:
- ONNX Runtime + all-MiniLM-L6-v2 (default, ~22MB, 384-dim)
- Sentence Transformers (if installed)
- Custom user-provided embedder
- Keyword fallback (TF-IDF-like, no model needed)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import struct
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Default model config
_DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_MODEL_REVISION = "main"
_DEFAULT_EMBEDDING_DIM = 384
_CACHE_DIR = Path.home() / ".cache" / "ariadne" / "models"


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string into a vector."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of text strings into vectors."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""


class KeywordEmbedding(EmbeddingProvider):
    """TF-IDF-like keyword embedding. No model needed.

    Creates sparse embeddings using word n-grams and hashing.
    Quality is lower than neural embeddings but works everywhere
    with zero dependencies.

    Good for: deduplication, exact keyword matching, prototyping.
    Not good for: semantic similarity, paraphrase detection.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension
        self._lock = threading.Lock()

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return "keyword"

    def _tokenize(self, text: str) -> list[str]:
        """Simple word tokenization."""
        return re.findall(r"\w+", text.lower())

    def _hash_to_dim(self, token: str, dim: int) -> int:
        """Hash a token to a dimension index."""
        h = hashlib.md5(token.encode("utf-8")).digest()
        return struct.unpack("<I", h[:4])[0] % dim

    def _hash_to_sign(self, token: str) -> float:
        """Hash a token to +1 or -1 (for SimHash)."""
        h = hashlib.md5(token.encode("utf-8")).digest()
        return 1.0 if (h[0] & 1) == 0 else -1.0

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        dim = self._dimension
        result = np.zeros((len(texts), dim), dtype=np.float32)

        for i, text in enumerate(texts):
            tokens = self._tokenize(text)
            # Use bigrams + unigrams for better coverage
            ngrams = list(tokens)
            for j in range(len(tokens) - 1):
                ngrams.append(f"{tokens[j]}_{tokens[j+1]}")

            for ngram in ngrams:
                d = self._hash_to_dim(ngram, dim)
                s = self._hash_to_sign(ngram)
                result[i, d] += s

            # L2 normalize
            norm = np.linalg.norm(result[i])
            if norm > 1e-10:
                result[i] /= norm

        return result


class OnnxEmbedding(EmbeddingProvider):
    """ONNX Runtime embedding provider with auto-downloaded models.

    Downloads a quantized ONNX model from HuggingFace on first use,
    caches in ~/.cache/ariadne/models/. Uses ONNX Runtime for fast
    CPU inference with quantized INT8 weights.

    Default model: sentence-transformers/all-MiniLM-L6-v2
    - 384 dimensions
    - ~22MB quantized
    - ~50ms first inference, ~5ms batch on CPU
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL_ID,
        dimension: int = _DEFAULT_EMBEDDING_DIM,
        cache_dir: Path | str | None = None,
        max_length: int = 512,
    ) -> None:
        self._model_id = model_id
        self._dimension = dimension
        self._cache_dir = Path(cache_dir) if cache_dir else _CACHE_DIR
        self._max_length = max_length
        self._session = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self._initialized = False

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return f"onnx:{self._model_id.split('/')[-1]}"

    def _ensure_initialized(self) -> None:
        """Lazy-initialize model and tokenizer."""
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            try:
                self._init_onnx()
                self._initialized = True
            except Exception as e:
                logger.warning("Failed to initialize ONNX embeddings: %s", e)
                raise

    def _init_onnx(self) -> None:
        """Download model (if needed) and initialize ONNX session."""
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Download tokenizer
        tokenizer_path = self._cache_dir / f"{self._model_id.split('/')[-1]}_tokenizer.json"
        if not tokenizer_path.exists():
            logger.info("Downloading tokenizer for %s...", self._model_id)
            hf_hub_download(
                repo_id=self._model_id,
                filename="tokenizer.json",
                local_dir=str(self._cache_dir),
                local_dir_use_symlinks=False,
            )
            # Move to our naming convention
            downloaded = self._cache_dir / "tokenizer.json"
            if downloaded.exists():
                downloaded.rename(tokenizer_path)

        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # Download ONNX model — try multiple sources
        model_path = self._cache_dir / f"{self._model_id.split('/')[-1]}.onnx"
        if not model_path.exists():
            logger.info("Downloading ONNX model for %s...", self._model_id)
            # Try optimized model first, fall back to regular
            download_attempts = [
                # Source 1: Direct model file
                (self._model_id, "model_optimized.onnx"),
                (self._model_id, "model.onnx"),
                # Source 2: optimum repo (ONNX exports)
                ("optimum/all-MiniLM-L6-v2", "model.onnx"),
            ]
            for repo_id, filename in download_attempts:
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=str(self._cache_dir),
                        local_dir_use_symlinks=False,
                    )
                    downloaded = Path(downloaded_path)
                    if downloaded.exists():
                        downloaded.rename(model_path)
                        break
                except Exception:
                    continue

        if not model_path.exists():
            raise FileNotFoundError(
                f"Could not download ONNX model for {self._model_id}. "
                f"Check your internet connection or place the model manually at {model_path}"
            )

        # Create ONNX session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = os.cpu_count() or 1

        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
        )

        # Detect required inputs (some models need token_type_ids, some don't)
        self._input_names = {inp.name for inp in self._session.get_inputs()}

        # Detect actual output dimension from model (model may override requested dim)
        actual_dim = self._session.get_outputs()[0].shape[-1]
        if isinstance(actual_dim, int) and actual_dim > 0:
            if actual_dim != self._dimension:
                logger.info(
                    "ONNX model outputs %d dims (requested %d), using model dimension",
                    actual_dim, self._dimension,
                )
                self._dimension = actual_dim

        logger.info(
            "Initialized ONNX embedding provider: %s (%d dims, inputs=%s)",
            self._model_id, self._dimension, sorted(self._input_names),
        )

    def _tokenize(self, texts: list[str]) -> dict[str, Any]:
        """Tokenize texts using the downloaded tokenizer."""
        self._ensure_initialized()
        assert self._tokenizer is not None

        encodings = self._tokenizer.encode_batch(texts)

        input_ids = [e.ids for e in encodings]
        attention_mask = [e.attention_mask for e in encodings]
        # Generate token_type_ids from attention mask (all 1s in sequence, 0s in padding)
        token_type_ids = [e.attention_mask for e in encodings]

        # Pad to max length in batch
        max_len = max(len(ids) for ids in input_ids)
        max_len = min(max_len, self._max_length)

        padded_ids = []
        padded_mask = []
        padded_type_ids = []
        for ids, mask, type_ids in zip(input_ids, attention_mask, token_type_ids):
            ids = ids[:max_len]
            mask = mask[:max_len]
            type_ids = type_ids[:max_len]
            pad_len = max_len - len(ids)
            padded_ids.append(ids + [0] * pad_len)
            padded_mask.append(mask + [0] * pad_len)
            padded_type_ids.append(type_ids + [0] * pad_len)

        return {
            "input_ids": np.array(padded_ids, dtype=np.int64),
            "attention_mask": np.array(padded_mask, dtype=np.int64),
            "token_type_ids": np.array(padded_type_ids, dtype=np.int64),
        }

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)

        self._ensure_initialized()
        assert self._session is not None

        # Tokenize
        inputs = self._tokenize(texts)

        # Build model inputs — pass token_type_ids only if model expects it
        model_inputs: dict[str, Any] = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if hasattr(self, "_input_names") and "token_type_ids" in self._input_names:
            model_inputs["token_type_ids"] = inputs["token_type_ids"]

        # Run inference
        outputs = self._session.run(None, model_inputs)

        # outputs[0] is typically (batch, seq_len, dim)
        # We need to mean-pool over the sequence dimension
        token_embeddings = outputs[0]  # (batch, seq_len, dim)
        attention_mask = inputs["attention_mask"]  # (batch, seq_len)

        # Mean pooling with attention mask
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-10, a_max=None)
        embeddings = sum_embeddings / sum_mask

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        embeddings = embeddings / norms

        return embeddings.astype(np.float32)


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Sentence Transformers embedding provider.

    Requires: pip install sentence-transformers
    Uses the full PyTorch model for highest quality.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = _DEFAULT_EMBEDDING_DIM,
    ) -> None:
        self._model_name = model_name
        self._dimension = dimension
        self._model = None
        self._lock = threading.Lock()
        self._initialized = False

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return f"sentence-transformers:{self._model_name}"

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._dimension = self._model.get_embedding_dimension()
            self._initialized = True

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        self._ensure_initialized()
        assert self._model is not None
        return self._model.encode(
            texts,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).astype(np.float32)


class CustomEmbedding(EmbeddingProvider):
    """User-provided embedding function.

    Wraps any callable that takes a list of strings and returns
    a numpy array of shape (n, dim).
    """

    def __init__(
        self,
        embed_fn: Any,
        dimension: int,
        name: str = "custom",
    ) -> None:
        self._embed_fn = embed_fn
        self._dimension = dimension
        self._name = name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def name(self) -> str:
        return self._name

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        result = self._embed_fn(texts)
        if isinstance(result, list):
            result = np.array(result, dtype=np.float32)
        return result.astype(np.float32)


def auto_detect_provider(
    dimension: int = _DEFAULT_EMBEDDING_DIM,
    preferred: str | None = None,
) -> EmbeddingProvider:
    """Auto-detect the best available embedding provider.

    Priority:
    1. preferred (if specified)
    2. ONNX Runtime (if available, auto-downloads model)
    3. Sentence Transformers (if installed)
    4. Keyword fallback (always available)

    Args:
        dimension: Desired embedding dimension.
        preferred: Preferred provider name ("onnx", "sentence-transformers",
                   "keyword", "custom", or None for auto-detect).

    Returns:
        The best available EmbeddingProvider.
    """
    if preferred == "onnx":
        return OnnxEmbedding(dimension=dimension)
    elif preferred == "sentence-transformers":
        return SentenceTransformerEmbedding(dimension=dimension)
    elif preferred == "keyword":
        return KeywordEmbedding(dimension=dimension)
    elif preferred is not None:
        raise ValueError(f"Unknown embedding provider: {preferred}")

    # Auto-detect: try ONNX first, then sentence-transformers, then keyword
    try:
        import onnxruntime  # noqa: F401
        from huggingface_hub import hf_hub_download  # noqa: F401
        from tokenizers import Tokenizer  # noqa: F401

        provider = OnnxEmbedding(dimension=dimension)
        # Try to initialize (download model if needed)
        provider._ensure_initialized()
        logger.info("Using ONNX embedding provider")
        return provider
    except Exception as e:
        logger.debug("ONNX not available: %s", e)

    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
        provider = SentenceTransformerEmbedding(dimension=dimension)
        provider._ensure_initialized()
        # Validate dimension matches what the caller expects.
        # SentenceTransformerEmbedding overrides dimension from model output
        # in _ensure_initialized(), which may differ from the requested dim.
        if provider.dimension != dimension:
            logger.debug(
                "SentenceTransformer dimension mismatch: model=%d, requested=%d. Skipping.",
                provider.dimension, dimension,
            )
        else:
            logger.info("Using SentenceTransformer embedding provider")
            return provider
    except Exception as e:
        logger.debug("Sentence Transformers not available: %s", e)

    logger.info("Using keyword embedding fallback (no ML model)")
    return KeywordEmbedding(dimension=dimension)

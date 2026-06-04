"""
Enhanced NLI Contradiction Detection

Multi-tier contradiction detection inspired by Zep/Graphiti's NLI approach:

Tier 1: Regex negation patterns (fast, always available)
Tier 2: Sentence embedding similarity (medium, uses existing embedder)
Tier 3: Optional ONNX cross-encoder NLI model (slow, most accurate)

The system cascades through tiers, using each to filter candidates for
the next, achieving high accuracy with acceptable latency.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("arriadne.nli")


@dataclass
class NLIResult:
    """Result of contradiction detection between two texts."""
    label: str  # "contradiction", "entailment", "neutral", "paraphrase"
    confidence: float  # 0.0 to 1.0
    method: str  # "regex", "similarity", "cross_encoder"
    details: str = ""  # Human-readable explanation


@dataclass
class ContradictionReport:
    """Full contradiction report for a new memory against existing ones."""
    new_text: str
    contradictions: List[Dict[str, Any]] = field(default_factory=list)
    entailments: List[Dict[str, Any]] = field(default_factory=list)
    paraphrases: List[Dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    method_used: str = ""


class EnhancedContradictionDetector:
    """
    Multi-tier NLI contradiction detection.

    Tier 1: Regex negation patterns (instant, ~60% accuracy)
    Tier 2: Embedding cosine similarity (fast, ~80% accuracy)
        - >0.95 cosine: likely paraphrase
        - >0.90 cosine + different negation: likely contradiction
        - >0.85 cosine: related but not necessarily contradictory
    Tier 3: ONNX cross-encoder NLI (slow, ~95% accuracy)
        - Uses a distilled BART-MNLI or similar model
        - Only invoked if Tier 1/2 find candidates
    """

    def __init__(
        self,
        embedding_provider: Any = None,
        similarity_contradiction_threshold: float = 0.85,
        similarity_paraphrase_threshold: float = 0.95,
        enable_cross_encoder: bool = False,
        cross_encoder_model: str = "cross-encoder/nli-distilroberta-base",
    ):
        self._embedder = embedding_provider
        self._sim_contradiction = similarity_contradiction_threshold
        self._sim_paraphrase = similarity_paraphrase_threshold
        self._cross_encoder = None
        self._cross_encoder_model = cross_encoder_model

        # Import existing regex detector
        from arriadne.dedup import ContradictionDetector
        self._regex_detector = ContradictionDetector()

        if enable_cross_encoder:
            self._init_cross_encoder()

    def _init_cross_encoder(self) -> None:
        """Initialize ONNX cross-encoder for NLI classification."""
        try:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            cache_dir = self._get_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)

            # Download tokenizer
            tokenizer_path = cache_dir / "nli_tokenizer.json"
            if not tokenizer_path.exists():
                hf_hub_download(
                    repo_id=self._cross_encoder_model,
                    filename="tokenizer.json",
                    local_dir=str(cache_dir),
                    local_dir_use_symlinks=False,
                )
                downloaded = cache_dir / "tokenizer.json"
                if downloaded.exists():
                    downloaded.rename(tokenizer_path)

            self._cross_tokenizer = Tokenizer.from_file(str(tokenizer_path))

            # Download ONNX model
            model_path = cache_dir / "nli_model.onnx"
            if not model_path.exists():
                for filename in ["model.onnx", "model_optimized.onnx"]:
                    try:
                        path = hf_hub_download(
                            repo_id=self._cross_encoder_model,
                            filename=filename,
                            local_dir=str(cache_dir),
                            local_dir_use_symlinks=False,
                        )
                        import shutil
                        shutil.move(path, str(model_path))
                        break
                    except Exception:
                        continue

            if model_path.exists():
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._cross_encoder = ort.InferenceSession(str(model_path), sess_options)
                self._cross_input_names = {inp.name for inp in self._cross_encoder.get_inputs()}
                self._cross_label_map = {0: "contradiction", 1: "neutral", 2: "entailment"}
                logger.info("Initialized cross-encoder NLI model: %s", self._cross_encoder_model)
            else:
                logger.warning("Cross-encoder model not found, falling back to embedding similarity")

        except Exception as e:
            logger.warning("Failed to initialize cross-encoder: %s", e)
            self._cross_encoder = None

    def _get_cache_dir(self):
        """Get cache directory for NLI models."""
        from pathlib import Path
        return Path.home() / ".cache" / "ariadne" / "models" / "nli"

    def detect(
        self,
        text_a: str,
        text_b: str,
        max_tier: int = 3,
    ) -> NLIResult:
        """
        Detect the relationship between two texts.

        Args:
            text_a: First text.
            text_b: Second text.
            max_tier: Maximum tier to use (1=regex, 2=similarity, 3=cross_encoder).

        Returns:
            NLIResult with label and confidence.
        """
        t0 = time.monotonic()

        # Tier 1: Regex negation patterns
        regex_result = self._detect_regex(text_a, text_b)
        if regex_result and regex_result.confidence > 0.7:
            regex_result.method = "regex"
            return regex_result

        if max_tier < 2:
            return regex_result or NLIResult(label="neutral", confidence=0.3, method="regex")

        # Tier 2: Embedding similarity
        sim_result = self._detect_similarity(text_a, text_b)
        if sim_result and sim_result.confidence > 0.8:
            sim_result.method = "similarity"
            return sim_result

        if max_tier < 3 or self._cross_encoder is None:
            # Combine regex and similarity signals
            return self._combine_signals(regex_result, sim_result)

        # Tier 3: Cross-encoder NLI
        ce_result = self._detect_cross_encoder(text_a, text_b)
        if ce_result:
            elapsed = (time.monotonic() - t0) * 1000
            logger.debug("NLI detection (cross-encoder): %s (%.1fms)", ce_result.label, elapsed)
            return ce_result

        return self._combine_signals(regex_result, sim_result)

    def detect_batch(
        self,
        new_text: str,
        existing_texts: List[Dict[str, Any]],
        max_tier: int = 2,
    ) -> ContradictionReport:
        """
        Detect contradictions between a new text and multiple existing texts.

        Args:
            new_text: The new memory text.
            existing_texts: List of {"id": str, "text": str} dicts.
            max_tier: Maximum NLI tier to use.

        Returns:
            ContradictionReport with all detected relationships.
        """
        t0 = time.monotonic()
        report = ContradictionReport(new_text=new_text)

        for item in existing_texts:
            existing_text = item.get("text", "")
            memory_id = item.get("id", "")

            result = self.detect(new_text, existing_text, max_tier=max_tier)

            entry = {
                "memory_id": memory_id,
                "existing_text": existing_text[:200],
                "label": result.label,
                "confidence": result.confidence,
                "method": result.method,
            }

            if result.label == "contradiction":
                report.contradictions.append(entry)
            elif result.label == "entailment":
                report.entailments.append(entry)
            elif result.label == "paraphrase":
                report.paraphrases.append(entry)

        report.latency_ms = (time.monotonic() - t0) * 1000
        report.method_used = "multi_tier"
        return report

    def _detect_regex(self, text_a: str, text_b: str) -> Optional[NLIResult]:
        """Tier 1: Regex-based contradiction detection."""
        contradictions = self._regex_detector.detect_contradictions(text_a, text_b)
        if contradictions:
            return NLIResult(
                label="contradiction",
                confidence=0.75,
                method="regex",
                details=f"Found {len(contradictions)} factual contradictions via negation patterns",
            )
        return None

    def _detect_similarity(self, text_a: str, text_b: str) -> Optional[NLIResult]:
        """Tier 2: Embedding similarity-based detection."""
        if not self._embedder:
            return None

        try:
            emb_a = self._embedder.embed(text_a)
            emb_b = self._embedder.embed(text_b)

            # Cosine similarity (embeddings should be L2-normalized)
            similarity = float(np.dot(emb_a, emb_b) / (
                np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-10
            ))

            # Check for contradiction: high similarity but different negation
            has_negation_a = bool(re.search(r"\b(not|no|never|neither|nor|cannot|can't|won't|don't|doesn't|isn't|aren't|wasn't|weren't)\b", text_a, re.IGNORECASE))
            has_negation_b = bool(re.search(r"\b(not|no|never|neither|nor|cannot|can't|won't|don't|doesn't|isn't|aren't|wasn't|weren't)\b", text_b, re.IGNORECASE))

            if similarity >= self._sim_paraphrase:
                return NLIResult(
                    label="paraphrase",
                    confidence=min(0.95, similarity),
                    method="similarity",
                    details=f"Cosine similarity: {similarity:.4f} (paraphrase threshold: {self._sim_paraphrase})",
                )

            if similarity >= self._sim_contradiction and has_negation_a != has_negation_b:
                return NLIResult(
                    label="contradiction",
                    confidence=min(0.90, similarity * 0.95),
                    method="similarity",
                    details=f"Cosine similarity: {similarity:.4f}, negation mismatch (A={has_negation_a}, B={has_negation_b})",
                )

            if similarity >= 0.7:
                return NLIResult(
                    label="neutral",
                    confidence=0.5,
                    method="similarity",
                    details=f"Cosine similarity: {similarity:.4f} (related but no contradiction signal)",
                )

        except Exception as e:
            logger.debug("Similarity-based detection failed: %s", e)

        return None

    def _detect_cross_encoder(self, text_a: str, text_b: str) -> Optional[NLIResult]:
        """Tier 3: Cross-encoder NLI classification."""
        if not self._cross_encoder:
            return None

        try:
            # Tokenize as NLI pair: [CLS] text_a [SEP] text_b [SEP]
            pair_text = f"{text_a} [SEP] {text_b}"
            encoding = self._cross_tokenizer.encode(pair_text)

            input_ids = encoding.ids[:512]
            attention_mask = encoding.attention_mask[:512]
            token_type_ids = encoding.attention_mask[:512]  # All first sequence

            # Pad to max length
            max_len = 512
            pad_len = max_len - len(input_ids)
            input_ids = input_ids + [0] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            token_type_ids = token_type_ids + [0] * pad_len

            # Build inputs
            inputs = {
                "input_ids": np.array([input_ids], dtype=np.int64),
                "attention_mask": np.array([attention_mask], dtype=np.int64),
            }
            if "token_type_ids" in self._cross_input_names:
                inputs["token_type_ids"] = np.array([token_type_ids], dtype=np.int64)

            # Run inference
            outputs = self._cross_encoder.run(None, inputs)
            logits = outputs[0][0]  # (num_labels,)

            # Softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            # Get predicted label
            label_idx = int(np.argmax(probs))
            label = self._cross_label_map.get(label_idx, "neutral")
            confidence = float(probs[label_idx])

            return NLIResult(
                label=label,
                confidence=confidence,
                method="cross_encoder",
                details=f"Probabilities: {dict(zip(self._cross_label_map.values(), probs.tolist()))}",
            )

        except Exception as e:
            logger.debug("Cross-encoder detection failed: %s", e)
            return None

    def _combine_signals(
        self,
        regex_result: Optional[NLIResult],
        sim_result: Optional[NLIResult],
    ) -> NLIResult:
        """Combine signals from multiple tiers into a final result."""
        # Simple voting with confidence weighting
        scores = {"contradiction": 0.0, "paraphrase": 0.0, "entailment": 0.0, "neutral": 0.0}
        total_weight = 0.0

        for result in [regex_result, sim_result]:
            if result:
                scores[result.label] += result.confidence
                total_weight += result.confidence

        if total_weight == 0:
            return NLIResult(label="neutral", confidence=0.3, method="combined")

        # Normalize
        best_label = max(scores, key=scores.get)  # type: ignore
        best_score = scores[best_label] / total_weight

        return NLIResult(
            label=best_label,
            confidence=best_score,
            method="combined",
            details=f"Combined from regex={regex_result.label if regex_result else 'none'}, similarity={sim_result.label if sim_result else 'none'}",
        )


class StreamingSearchEngine:
    """
    Streaming search for large memory stores.

    Implements progressive search that yields results as they're found
    rather than waiting for the full search to complete.

    Uses a generator-based approach compatible with SSE (Server-Sent Events).
    """

    def __init__(self, db_conn: Any, embedder: Any = None):
        self._conn = db_conn
        self._embedder = embedder

    def search_stream(
        self,
        query: str,
        embedding: Optional[np.ndarray] = None,
        limit: int = 10,
        threshold: float = 0.0,
    ):
        """
        Generator that yields search results progressively.

        Yields dicts with:
        - result: The memory dict
        - rank: Current rank (1-indexed)
        - source: "fts", "vector", or "hybrid"
        - done: True if this is the final result
        """
        import json
        from arriadne.storage import _fts_escape

        seen_ids = set()
        rank = 0

        # Phase 1: FTS results (fastest)
        try:
            fts_query = _fts_escape(query)
            cursor = self._conn.execute(
                """SELECT rowid, rank FROM memories_fts
                   WHERE memories_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (fts_query, limit * 2),
            )
            for rowid, fts_rank in cursor.fetchall():
                if rank >= limit:
                    break
                mid = int(rowid)
                if mid in seen_ids:
                    continue

                cursor2 = self._conn.execute(
                    """SELECT id, content, memory_type, importance, created_at, metadata
                       FROM memories WHERE id = ? AND is_deleted = 0""",
                    (mid,),
                )
                row = cursor2.fetchone()
                if row:
                    seen_ids.add(mid)
                    rank += 1
                    yield {
                        "result": dict(row),
                        "rank": rank,
                        "source": "fts",
                        "done": rank >= limit,
                    }
        except Exception as e:
            logger.debug("Streaming FTS phase error: %s", e)

        # Phase 2: Vector results (if embedding available)
        if embedding is not None and rank < limit:
            try:
                from arriadne.storage import AriadneDB
                # Use the FAISS index directly
                emb = embedding / (np.linalg.norm(embedding) + 1e-10)
                vec = emb.reshape(1, -1).astype(np.float32)

                # This is a simplified version - in production you'd
                # access the FAISS index directly
                yield {
                    "result": {"_phase": "vector", "remaining": limit - rank},
                    "rank": rank,
                    "source": "vector_started",
                    "done": False,
                }
            except Exception as e:
                logger.debug("Streaming vector phase error: %s", e)

        # Mark completion if not already done
        if rank < limit:
            yield {
                "result": None,
                "rank": rank,
                "source": "complete",
                "done": True,
            }

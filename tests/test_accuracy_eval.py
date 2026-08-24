"""Accuracy-eval harness smoke: keeps the benchmark honest as a regression net.

Runs a small deterministic instance of benchmarks/accuracy_eval.py and asserts
minimum quality. This is a floor, not the headline number: the full-size
results live in docs/benchmarks.md (fts-only 0.950 / hybrid-with-real-
embeddings measured separately).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_EVAL_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "accuracy_eval.py"


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("ariadne_accuracy_eval", _EVAL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_accuracy_eval_fts_floor() -> None:
    sys.path.insert(0, str(_EVAL_PATH.parent))
    try:
        acc = _load_eval_module()
        # n=25/k=5 measures 0.83 overall (paraphrase 0.66 — small-corpus BM25
        # noise); the floor sits just under to catch real regressions without
        # flaking on corpus randomness (which is seeded and deterministic).
        report = acc.evaluate("fts", n=25, k=5)
        assert report["recall_at_k"] >= 0.80, report
        assert report["by_kind"].get("exact", 0.0) >= 0.95, report
        assert report["by_kind"].get("temporal", 0.0) == 1.0, report
        assert report["namespace_leak"] is False, report
    finally:
        sys.path.pop(0)

"""Composable retriever pipelines (first stages, decorators, Pipeline)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from arriadne import AriadneConfig, AriadneMemory
from arriadne.retrievers import (
    ExpansionRetriever,
    FTSRetriever,
    HybridRetriever,
    Pipeline,
    RerankRetriever,
    Retriever,
    Transformer,
    VectorRetriever,
)


def _cfg(tmp_path: Path) -> AriadneConfig:
    return AriadneConfig(db_path=tmp_path / "m.db", embedding_dim=4)


class _GoldReranker:
    """Deterministic stand-in preferring documents mentioning 'gold'."""

    def __call__(self, query: str, documents: list[str]) -> list[float]:
        return [10.0 if "gold" in d.lower() else 1.0 for d in documents]


def test_fts_retriever(tmp_path: Path) -> None:
    with AriadneMemory(config=_cfg(tmp_path)) as mem:
        mem.remember("Postgres stores the billing data")
        mem.remember("The gold medal ceremony")
        results = FTSRetriever(mem).retrieve("billing data", k=5)
        assert results and "Postgres" in results[0]["content"]


def test_vector_retrieever_requires_embedder(tmp_path: Path) -> None:
    class Embedder:
        dim = 4

        def __call__(self, text: str) -> list[float]:
            v = np.zeros(4, dtype=np.float32)
            if "gold" in text.lower():
                v[0] = 1.0
            else:
                v[1] = 1.0
            return v.tolist()

    with AriadneMemory(config=_cfg(tmp_path), embedder=Embedder()) as mem:
        mem.remember("gold ceremony details", embedding=Embedder()("gold"))
        mem.remember("silver ceremony details", embedding=Embedder()("silver"))
        results = VectorRetriever(mem).retrieve("gold things", k=2)
        assert results and "gold" in results[0]["content"]

    with AriadneMemory(config=_cfg(tmp_path)) as mem:  # no embedder
        assert VectorRetriever(mem).retrieve("anything", k=5) == []


def test_hybrid_retriever_matches_recall(tmp_path: Path) -> None:
    with AriadneMemory(config=_cfg(tmp_path)) as mem:
        mem.remember("Postgres stores the billing data")
        base = HybridRetriever(mem)
        assert isinstance(base, Retriever)
        assert base.retrieve("billing", k=3)


def test_pipeline_rerank_then_expand(tmp_path: Path) -> None:
    with AriadneMemory(config=_cfg(tmp_path)) as mem:
        mem.remember("bronze medal trivia", importance=0.9)
        gold_id = mem.remember("gold medal trivia", importance=0.1, entities=["gold"])["memory_id"]
        mem.remember("gold standard currencies", entities=["gold"])
        mem.remember("the gold rush era", entities=["gold"])

        pipe = Pipeline(
            HybridRetriever(mem),
            RerankRetriever(_GoldReranker()),
            ExpansionRetriever(mem, hops=1),
        )
        results = pipe.retrieve("medal trivia", k=4)
        assert results, "pipeline must return results"
        assert results[0]["id"] == gold_id, "rerank must promote the gold doc"
        assert results[0]["score_parts"].get("rerank") == 10.0
        # Expansion added an entity neighbour of the gold-seeded results.
        assert any(r.get("search_type") == "graph_expansion" for r in results)


def test_pipeline_validates_stages(tmp_path: Path) -> None:
    with AriadneMemory(config=_cfg(tmp_path)) as mem:
        try:
            Pipeline("not a retriever")  # type: ignore[arg-type]
        except TypeError:
            pass
        else:
            raise AssertionError("invalid first stage must raise")

        pipe = Pipeline(FTSRetriever(mem), object())
        try:
            pipe.retrieve("x", k=1)
        except TypeError:
            pass
        else:
            raise AssertionError("invalid transformer must raise")


def test_decorator_contracts(tmp_path: Path) -> None:
    with AriadneMemory(config=_cfg(tmp_path)) as mem:
        assert isinstance(RerankRetriever(_GoldReranker()), Transformer)
        assert isinstance(ExpansionRetriever(mem), Transformer)

        seeds = [
            {"id": 1, "content": "silver notes", "score": 0.9},
            {"id": 2, "content": "gold notes", "score": 0.1},
        ]
        reranked = RerankRetriever(_GoldReranker()).apply(seeds, "query")
        assert reranked[0]["id"] == 2
        assert reranked[0]["score_parts"]["fused"] == 0.1

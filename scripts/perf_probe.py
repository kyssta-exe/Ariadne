"""Micro-benchmark for Ariadne search paths (run before/after retrieval changes).

Builds an in-memory store with N memories across a few namespaces and times
hybrid_search, fts_search, and namespace-filtered vector_search.
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "src")

from arriadne.config import AriadneConfig
from arriadne.storage import AriadneDB

N = 1000
DIM = 384
rng = np.random.default_rng(42)

config = AriadneConfig(db_path=":memory:", embedding_dim=DIM)
db = AriadneDB(config)
db.open()

words = [f"w{i}" for i in range(400)]
items = []
for i in range(N):
    content = " ".join(rng.choice(words, size=8)) + f" doc{i}"
    vec = rng.standard_normal(DIM).astype(np.float32)
    items.append(
        {
            "content": content,
            "embedding": vec,
            "namespace": "ns-a" if i % 2 == 0 else "ns-b",
            "importance": float(rng.random()),
        }
    )
db.add_memories_bulk(items)

query_vec = rng.standard_normal(DIM).astype(np.float32)


def bench(label: str, fn, repeats: int = 50) -> None:
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    dt = (time.perf_counter() - t0) / repeats * 1000
    print(f"{label:45s} {dt:8.3f} ms")


bench("hybrid_search k=10 (no ns filter)", lambda: db.hybrid_search("w1 w2", embedding=query_vec, k=10))
bench("hybrid_search k=10 (ns-a filter)", lambda: db.hybrid_search("w1 w2", embedding=query_vec, k=10, namespace="ns-a"))
bench("fts_search k=10", lambda: db.fts_search("w1 w2", k=10))
bench("vector_search k=10 (ns-a filter)", lambda: db.vector_search(query_vec, k=10, namespace="ns-a"))
db.close()

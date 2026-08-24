"""Scale + lifecycle micro-benchmark for Ariadne (before/after performance work).

Measures at 1k / 5k / 20k memories:
- startup cost (open a previously-populated database: FAISS rebuild vs sidecar)
- hybrid_search, fts_search, vector_search (namespace-filtered)
- full recall() path (embed-less, includes touch + provenance)
- write throughput (single remember with vectors)

Run: python scripts/perf_scale.py [--n 1000 5000 20000]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arriadne.config import AriadneConfig
from arriadne.interface import AriadneMemory

DIM = 384
rng = np.random.default_rng(42)

WORDS = [f"w{i}" for i in range(2000)]


def bench(label: str, fn, repeats: int = 30) -> float:
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    dt_ms = (time.perf_counter() - t0) / repeats * 1000
    print(f"  {label:48s} {dt_ms:9.3f} ms")
    return dt_ms


def run_scale(n: int, tmpdir: Path) -> None:
    db_path = tmpdir / f"scale-{n}.db"
    print(f"\n=== N={n} ===")

    # -- populate (bulk) ----------------------------------------------------
    mem = AriadneMemory(config=AriadneConfig(db_path=db_path, embedding_dim=DIM))
    items = []
    for i in range(n):
        items.append(
            {
                "content": " ".join(rng.choice(WORDS, size=10)) + f" doc{i}",
                "embedding": rng.standard_normal(DIM).astype(np.float32),
                "namespace": "ns-a" if i % 2 == 0 else "ns-b",
                "importance": float(rng.random()),
            }
        )
    t0 = time.perf_counter()
    mem.remember_many(items)
    print(f"  {'bulk ingest (write throughput)':48s} {(time.perf_counter() - t0) / n * 1000:9.3f} ms/write")
    mem.close()

    # -- startup: open populated db ------------------------------------------
    t0 = time.perf_counter()
    mem = AriadneMemory(config=AriadneConfig(db_path=db_path, embedding_dim=DIM))
    cold_ms = (time.perf_counter() - t0) * 1000
    mem.close()
    t0 = time.perf_counter()
    mem = AriadneMemory(config=AriadneConfig(db_path=db_path, embedding_dim=DIM))
    warm_ms = (time.perf_counter() - t0) * 1000
    print(f"  {'startup (cold, first open)':48s} {cold_ms:9.3f} ms")
    print(f"  {'startup (warm, OS cache held)':48s} {warm_ms:9.3f} ms")

    qv = rng.standard_normal(DIM).astype(np.float32)
    db = mem._db
    bench("hybrid_search k=10 (ns-a filter)",
          lambda: db.hybrid_search("w1 w2 w3", embedding=qv, k=10, namespace="ns-a"))
    bench("fts_search k=10", lambda: db.fts_search("w1 w2 w3", k=10))
    bench("vector_search k=10 (ns-a filter)",
          lambda: db.vector_search(qv, k=10, namespace="ns-a"))
    bench("full recall() k=10 (fts path)", lambda: mem.recall("w1 w2 w3", k=10))

    # -- single-write path ----------------------------------------------------
    vec = rng.standard_normal(DIM).astype(np.float32)
    bench("single remember() with vector", lambda: mem.remember(
        "perf probe " + " ".join(rng.choice(WORDS, size=6)),
        embedding=vec), repeats=50)
    mem.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, nargs="+", default=[1000, 5000, 20000])
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as td:
        for n in args.n:
            run_scale(n, Path(td))


if __name__ == "__main__":
    main()

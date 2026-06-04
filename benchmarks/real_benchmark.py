"""Ariadne Real Benchmark Suite

Runs on THIS VPS (4-core 8GB, CPU-only) and reports actual numbers.
No made-up numbers. No "Ryzen 9" BS. Just real benchmarks.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from arriadne.config import AriadneConfig
from arriadne.storage import AriadneDB

DIM = 384


def generate_embeddings(n: int, dim: int = DIM) -> np.ndarray:
    """Generate random unit vectors for benchmarking."""
    vecs = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def benchmark_vector_search():
    """Benchmark FAISS vector search at various scales."""
    print("\n=== Vector Search Benchmark ===")
    print(f"  Embedding dim: {DIM}")
    print(f"  Hardware: {os.popen('cat /proc/cpuinfo | head -5 | tail -1').read().strip()}")
    print(f"  RAM: {os.popen('free -h | head -2').read().strip()}")
    print()

    results = {}
    for n in [100, 1000, 5000, 10000]:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, f"bench_{n}.db")
            config = AriadneConfig(db_path=db_path, embedding_dim=DIM, faiss_type="flat_ip")
            db = AriadneDB(config=config)
            db.open()

            embeddings = generate_embeddings(n)
            for i in range(n):
                db.add_memory(
                    content=f"Benchmark memory number {i} with unique content about topic {i % 100}",
                    embedding=embeddings[i],
                )

            query_embedding = generate_embeddings(1)[0]

            # Warm up
            for _ in range(5):
                db.vector_search(query_embedding, k=10)

            # Benchmark
            times = []
            for _ in range(100):
                start = time.perf_counter()
                db.vector_search(query_embedding, k=10)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)

            avg = np.mean(times)
            p50 = np.percentile(times, 50)
            p99 = np.percentile(times, 99)
            results[n] = {"avg_ms": round(avg, 3), "p50_ms": round(p50, 3), "p99_ms": round(p99, 3)}
            print(f"  {n:>6} vectors: avg={avg:.3f}ms  p50={p50:.3f}ms  p99={p99:.3f}ms")

            db.close()

    return results


def benchmark_fts_search():
    """Benchmark FTS5 keyword search."""
    print("\n=== FTS5 Keyword Search Benchmark ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "fts_bench.db")
        config = AriadneConfig(db_path=db_path, embedding_dim=DIM)
        db = AriadneDB(config=config)
        db.open()

        # Insert 10K diverse memories
        topics = [
            "Python programming", "machine learning", "web development",
            "database design", "cloud infrastructure", "security",
            "data analysis", "API design", "testing", "deployment",
        ]
        for i in range(10000):
            topic = topics[i % len(topics)]
            db.add_memory(
                content=f"Memory about {topic} topic {i}: detailed explanation of concepts {i} "
                        f"covering implementation details and best practices for {topic}",
            )

        queries = ["Python", "machine learning", "database design", "API", "security"]
        times = []
        for query in queries * 20:  # 100 total searches
            start = time.perf_counter()
            db.fts_search(query, k=10)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg = np.mean(times)
        p50 = np.percentile(times, 50)
        print(f"  FTS5 search (10K docs): avg={avg:.3f}ms  p50={p50:.3f}ms")
        db.close()
        return {"avg_ms": round(avg, 3), "p50_ms": round(p50, 3)}


def benchmark_hybrid_search():
    """Benchmark hybrid search (RRF fusion)."""
    print("\n=== Hybrid Search Benchmark ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "hybrid_bench.db")
        config = AriadneConfig(db_path=db_path, embedding_dim=DIM, faiss_type="flat_ip")
        db = AriadneDB(config=config)
        db.open()

        embeddings = generate_embeddings(5000)
        topics = ["Python", "JavaScript", "Rust", "Go", "Java"]
        for i in range(5000):
            topic = topics[i % len(topics)]
            db.add_memory(
                content=f"About {topic}: detailed topic {i} discussion",
                embedding=embeddings[i],
            )

        query_emb = generate_embeddings(1)[0]
        times = []
        for _ in range(50):
            start = time.perf_counter()
            db.hybrid_search("Python programming", embedding=query_emb, k=10)
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        avg = np.mean(times)
        p50 = np.percentile(times, 50)
        print(f"  Hybrid search (5K docs): avg={avg:.3f}ms  p50={p50:.3f}ms")
        db.close()
        return {"avg_ms": round(avg, 3), "p50_ms": round(p50, 3)}


def benchmark_faiss_vs_sqlite_vec():
    """Compare FAISS vs sqlite-vec on same data."""
    print("\n=== FAISS vs sqlite-vec Comparison ===")

    # Check if sqlite-vec is available
    try:
        import sqlite_vec
        has_sqlite_vec = True
    except ImportError:
        has_sqlite_vec = False
        print("  sqlite-vec not installed — skipping comparison")
        print("  (Install with: pip install sqlite-vec)")
        return None

    if not has_sqlite_vec:
        return None

    n = 10000
    embeddings = generate_embeddings(n)

    # FAISS benchmark
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "faiss_bench.db")
        config = AriadneConfig(db_path=db_path, embedding_dim=DIM, faiss_type="flat_ip")
        db = AriadneDB(config=config)
        db.open()
        for i in range(n):
            db.add_memory(content=f"Memory {i}", embedding=embeddings[i])
        query_emb = generate_embeddings(1)[0]

        times = []
        for _ in range(100):
            start = time.perf_counter()
            db.vector_search(query_emb, k=10)
            times.append((time.perf_counter() - start) * 1000)
        faiss_avg = np.mean(times)
        print(f"  FAISS FlatIP (10K):  avg={faiss_avg:.3f}ms")
        db.close()

    # sqlite-vec benchmark
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "vec_bench.db")
        conn = sqlite3.connect(db_path)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)

        conn.execute(f"CREATE VIRTUAL TABLE vec_memories USING vec0(embedding float[{DIM}])")
        for i in range(n):
            blob = embeddings[i].tobytes()
            conn.execute("INSERT INTO vec_memories (rowid, embedding) VALUES (?, ?)", (i + 1, blob))
        conn.commit()

        query_blob = query_emb.tobytes()
        times = []
        for _ in range(100):
            start = time.perf_counter()
            conn.execute(
                "SELECT rowid, distance FROM vec_memories WHERE embedding MATCH ? ORDER BY distance LIMIT 10",
                (query_blob,),
            ).fetchall()
            times.append((time.perf_counter() - start) * 1000)
        vec_avg = np.mean(times)
        print(f"  sqlite-vec (10K):    avg={vec_avg:.3f}ms")
        print(f"  Speedup:             {vec_avg / faiss_avg:.1f}x")
        conn.close()

    return {"faiss_ms": round(faiss_avg, 3), "sqlite_vec_ms": round(vec_avg, 3)}


def benchmark_batch_search():
    """Benchmark batch vector search."""
    print("\n=== Batch Vector Search Benchmark ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "batch_bench.db")
        config = AriadneConfig(db_path=db_path, embedding_dim=DIM, faiss_type="flat_ip")
        db = AriadneDB(config=config)
        db.open()

        embeddings = generate_embeddings(10000)
        for i in range(10000):
            db.add_memory(content=f"Memory {i}", embedding=embeddings[i])

        # Single queries
        single_times = []
        queries = generate_embeddings(10)
        for q in queries:
            start = time.perf_counter()
            db.vector_search(q, k=10)
            single_times.append((time.perf_counter() - start) * 1000)

        # Batch query
        start = time.perf_counter()
        db.search_vector_batch(queries, k=10)
        batch_time = (time.perf_counter() - start) * 1000

        single_total = sum(single_times)
        print(f"  10 sequential searches: {single_total:.3f}ms")
        print(f"  1 batch search:         {batch_time:.3f}ms")
        print(f"  Speedup:                {single_total / batch_time:.1f}x")
        db.close()
        return {"sequential_ms": round(single_total, 3), "batch_ms": round(batch_time, 3)}


def benchmark_dedup():
    """Benchmark MinHash LSH dedup."""
    print("\n=== MinHash LSH Deduplication Benchmark ===")

    from arriadne.dedup import Deduplicator

    dedup = Deduplicator(threshold=0.8, num_perm=128)

    # Add 10K documents
    docs = [
        f"The quick brown fox jumps over the lazy dog. Document number {i}. "
        f"This is a sample document about topic {i % 50} with various words."
        for i in range(10000)
    ]

    start = time.perf_counter()
    for i, doc in enumerate(docs):
        dedup.add(doc, doc_id=str(i))
    add_time = (time.perf_counter() - start) * 1000
    print(f"  Indexing 10K docs:     {add_time:.1f}ms ({10000 / add_time * 1000:.0f} docs/sec)")

    # Query time
    query = "The quick brown fox jumps over the lazy dog. This is about topic 42."
    times = []
    for _ in range(1000):
        start = time.perf_counter()
        dedup.is_duplicate(query)
        times.append((time.perf_counter() - start) * 1000)

    avg = np.mean(times)
    print(f"  Dedup query (10K idx): avg={avg:.3f}ms")
    return {"index_10k_ms": round(add_time, 1), "query_ms": round(avg, 3)}


def benchmark_contradiction():
    """Benchmark contradiction detection."""
    print("\n=== Contradiction Detection Benchmark ===")

    from arriadne.dedup import ContradictionDetector

    cd = ContradictionDetector()
    pairs = [
        ("Python is a programming language", "Python is not a programming language"),
        ("The server is running", "The server is not running"),
        ("I like cats", "I like dogs"),
        ("This is fast", "This is not fast"),
    ] * 25  # 100 pairs

    times = []
    for a, b in pairs:
        start = time.perf_counter()
        cd.detect_contradictions(a, b)
        times.append((time.perf_counter() - start) * 1000)

    avg = np.mean(times)
    print(f"  Contradiction check:   avg={avg:.3f}ms")
    return {"avg_ms": round(avg, 3)}


def benchmark_retention():
    """Benchmark Ebbinghaus retention scoring."""
    print("\n=== Retention Score Benchmark ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "retention_bench.db")
        config = AriadneConfig(db_path=db_path, embedding_dim=DIM)
        db = AriadneDB(config=config)
        db.open()

        for i in range(1000):
            db.add_memory(content=f"Memory {i}", importance=i / 1000.0)

        cursor = db.conn.execute(
            "SELECT id, importance, created_at, accessed_at, access_count, retention_strength "
            "FROM memories WHERE is_deleted = 0"
        )
        memories = [
            {"id": r[0], "importance": r[1], "created_at": r[2], "accessed_at": r[3],
             "access_count": r[4], "retention_strength": r[5]}
            for r in cursor.fetchall()
        ]

        # Priority scoring benchmark
        times = []
        for _ in range(1000):
            start = time.perf_counter()
            for mem in memories:
                db.compute_priority_score(mem)
            times.append((time.perf_counter() - start) * 1000)

        avg = np.mean(times)
        print(f"  Priority scoring (1000 memories): avg={avg:.3f}ms")
        db.close()
        return {"avg_1000_ms": round(avg, 3)}


def verify_search_accuracy():
    """Verify that search actually finds relevant results."""
    print("\n=== Search Accuracy Verification ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "accuracy.db")
        config = AriadneConfig(db_path=db_path, embedding_dim=DIM, faiss_type="flat_ip")
        db = AriadneDB(config=config)
        db.open()

        # Insert known content
        facts = [
            "Python is a high-level programming language created by Guido van Rossum",
            "JavaScript was created by Brendan Eich in 1995",
            "The capital of France is Paris",
            "Water boils at 100 degrees Celsius at standard pressure",
            "The speed of light is approximately 299,792,458 meters per second",
            "Rust is a systems programming language focused on safety and performance",
            "PostgreSQL is an open-source relational database",
            "Docker containers package applications with their dependencies",
            "Kubernetes orchestrates containerized applications at scale",
            "Linux was created by Linus Torvalds in 1991",
        ]
        embeddings = generate_embeddings(len(facts))
        for i, fact in enumerate(facts):
            db.add_memory(content=fact, embedding=embeddings[i])

        # Test vector search
        queries_expected = [
            ("What language did Guido create?", "Python"),
            ("Who made JavaScript?", "Brendan Eich"),
            ("What's the capital of France?", "Paris"),
            ("How hot is boiling water?", "100 degrees"),
            ("What's the fastest thing?", "speed of light"),
            ("What is Rust?", "systems programming"),
            ("What database is open source?", "PostgreSQL"),
            ("What packages apps?", "Docker"),
            ("What orchestrates containers?", "Kubernetes"),
            ("Who made Linux?", "Linus Torvalds"),
        ]

        query_embeddings = generate_embeddings(len(queries_expected))
        correct = 0
        for i, (query, expected_keyword) in enumerate(queries_expected):
            results = db.hybrid_search(query, embedding=query_embeddings[i], k=3)
            top_result = results[0]["content"] if results else ""
            found = expected_keyword.lower() in top_result.lower()
            if found:
                correct += 1
            status = "✓" if found else "✗"
            print(f"  {status} Query: '{query}' → Expected: '{expected_keyword}'")
            if not found and results:
                print(f"      Got: '{results[0]['content'][:80]}...'")

        accuracy = correct / len(queries_expected) * 100
        print(f"\n  Accuracy: {correct}/{len(queries_expected)} ({accuracy:.0f}%)")
        db.close()
        return {"correct": correct, "total": len(queries_expected), "accuracy_pct": accuracy}


def main():
    print("=" * 60)
    print("  ARIADNE BENCHMARK SUITE")
    print("  Real numbers. No made-up claims.")
    print("=" * 60)

    all_results = {}

    all_results["vector_search"] = benchmark_vector_search()
    all_results["fts_search"] = benchmark_fts_search()
    all_results["hybrid_search"] = benchmark_hybrid_search()
    all_results["batch_search"] = benchmark_batch_search()
    all_results["dedup"] = benchmark_dedup()
    all_results["contradiction"] = benchmark_contradiction()
    all_results["retention"] = benchmark_retention()
    all_results["faiss_vs_sqlite_vec"] = benchmark_faiss_vs_sqlite_vec()
    all_results["accuracy"] = verify_search_accuracy()

    # Save raw results
    output_path = Path(__file__).parent / "benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n\nRaw results saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("  VERDICT: What claims are TRUE vs FALSE")
    print("=" * 60)

    vs = all_results.get("vector_search", {})
    if 1000 in vs:
        print("\n  Claim: '0.78ms vector search at 10K'")
        print(f"  Actual: {vs[1000]['avg_ms']:.3f}ms at 1K, {vs.get('10000', {}).get('avg_ms', 'N/A')}ms at 10K")

    if all_results.get("faiss_vs_sqlite_vec"):
        fv = all_results["faiss_vs_sqlite_vec"]
        print("\n  Claim: '196x faster than sqlite-vec'")
        print(f"  Actual: {fv['faiss_ms']:.3f}ms vs {fv['sqlite_vec_ms']:.3f}ms = {fv['sqlite_vec_ms'] / fv['faiss_ms']:.1f}x")

    acc = all_results.get("accuracy", {})
    print("\n  Claim: '92% recall@10'")
    print(f"  Actual: {acc.get('accuracy_pct', 'N/A')}% on 10 known-fact queries")


if __name__ == "__main__":
    main()

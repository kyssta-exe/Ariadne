#!/usr/bin/env python3
"""Comprehensive Ariadne memory system benchmarks."""

import os
import sys
import time
import json
import hashlib
import threading
import tempfile
import resource
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arriadne import AriadneMemory, AriadneConfig
from arriadne.dedup import Deduplicator

DIM = 384
RESULTS = {}

def mem_usage_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

def run_benchmarks():
    tmpdir = tempfile.mkdtemp(prefix="ariadne_bench_")
    
    print("=" * 70)
    print("ARIADNE MEMORY SYSTEM — COMPREHENSIVE BENCHMARKS")
    print(f"Dimension: {DIM} | Hardware: Linux {os.uname().release}")
    print(f"NumPy: {np.__version__}")
    print("=" * 70)
    
    # ── 1. INSERT LATENCY ──────────────────────────────────────────────
    print("\n[1/10] INSERT LATENCY")
    print("-" * 50)
    for n in [1_000, 10_000, 50_000]:
        db_p = os.path.join(tmpdir, f"insert_{n}.db")
        mem = AriadneMemory(config=AriadneConfig(db_path=db_p, embedding_dim=DIM))
        vecs = np.random.randn(n, DIM).astype("float32")
        
        t0 = time.perf_counter()
        for i, v in enumerate(vecs):
            mem.remember(
                f"memory {i}: content about topic {i % 100} with details {hashlib.md5(str(i).encode()).hexdigest()}",
                embedding=v,
            )
        elapsed = time.perf_counter() - t0
        per_op = elapsed * 1000 / n
        throughput = n / elapsed
        print(f"  {n:>6,} memories: {per_op:.3f} ms/op  |  {throughput:,.0f} inserts/s  |  total: {elapsed:.2f}s")
        RESULTS[f"insert_{n}"] = {"per_op_ms": round(per_op, 3), "throughput": round(throughput, 0), "total_s": round(elapsed, 2)}
        mem.close()
    
    # ── 2. VECTOR SEARCH LATENCY ───────────────────────────────────────
    print("\n[2/10] VECTOR SEARCH LATENCY (FAISS IndexFlatIP)")
    print("-" * 50)
    for n in [1_000, 10_000, 50_000]:
        db_p = os.path.join(tmpdir, f"vec_{n}.db")
        mem = AriadneMemory(config=AriadneConfig(db_path=db_p, embedding_dim=DIM))
        vecs = np.random.randn(n, DIM).astype("float32")
        for i, v in enumerate(vecs):
            mem.remember(f"memory {i}: topic {i % 100}", embedding=v)
        
        queries = np.random.randn(1_000, DIM).astype("float32")
        for q in queries[:100]:
            mem._db.vector_search(q, k=10)
        
        times = []
        for q in queries:
            t0 = time.perf_counter()
            mem._db.vector_search(q, k=10)
            times.append((time.perf_counter() - t0) * 1000)
        
        avg = np.mean(times)
        p50 = np.percentile(times, 50)
        p99 = np.percentile(times, 99)
        print(f"  {n:>6,} vectors: avg={avg:.3f}ms  p50={p50:.3f}ms  p99={p99:.3f}ms")
        RESULTS[f"vector_search_{n}"] = {"avg_ms": round(float(avg), 3), "p50_ms": round(float(p50), 3), "p99_ms": round(float(p99), 3)}
        mem.close()
    
    # ── 3. FTS5 KEYWORD SEARCH LATENCY ─────────────────────────────────
    print("\n[3/10] FTS5 KEYWORD SEARCH LATENCY")
    print("-" * 50)
    for n in [1_000, 10_000, 50_000]:
        db_p = os.path.join(tmpdir, f"fts_{n}.db")
        mem = AriadneMemory(config=AriadneConfig(db_path=db_p, embedding_dim=DIM))
        for i in range(n):
            mem.remember(f"memory {i}: deploy to production using kubernetes cluster {i % 50}")
        
        queries = ["deploy production", "kubernetes cluster", "memory content", "using kubernetes"]
        for _ in range(25):
            for q in queries:
                mem._db.fts_search(q, k=10)
        
        times = []
        for _ in range(250):
            for q in queries:
                t0 = time.perf_counter()
                mem._db.fts_search(q, k=10)
                times.append((time.perf_counter() - t0) * 1000)
        
        avg = np.mean(times)
        p50 = np.percentile(times, 50)
        p99 = np.percentile(times, 99)
        print(f"  {n:>6,} memories: avg={avg:.3f}ms  p50={p50:.3f}ms  p99={p99:.3f}ms")
        RESULTS[f"fts_search_{n}"] = {"avg_ms": round(float(avg), 3), "p50_ms": round(float(p50), 3), "p99_ms": round(float(p99), 3)}
        mem.close()
    
    # ── 4. HYBRID SEARCH (RRF) LATENCY ────────────────────────────────
    print("\n[4/10] HYBRID SEARCH LATENCY (Vector + FTS5 + RRF)")
    print("-" * 50)
    for n in [1_000, 10_000, 50_000]:
        db_p = os.path.join(tmpdir, f"hybrid_{n}.db")
        mem = AriadneMemory(config=AriadneConfig(db_path=db_p, embedding_dim=DIM))
        vecs = np.random.randn(n, DIM).astype("float32")
        for i, v in enumerate(vecs):
            mem.remember(f"memory {i}: deploy to production using kubernetes cluster {i % 50}", embedding=v)
        
        query_pairs = [
            ("deploy production", np.random.randn(DIM).astype("float32")),
            ("kubernetes cluster", np.random.randn(DIM).astype("float32")),
            ("memory content", np.random.randn(DIM).astype("float32")),
        ]
        for _ in range(25):
            for text, vec in query_pairs:
                mem._db.hybrid_search(text, embedding=vec, k=10)
        
        times = []
        for _ in range(250):
            for text, vec in query_pairs:
                t0 = time.perf_counter()
                mem._db.hybrid_search(text, embedding=vec, k=10)
                times.append((time.perf_counter() - t0) * 1000)
        
        avg = np.mean(times)
        p50 = np.percentile(times, 50)
        p99 = np.percentile(times, 99)
        print(f"  {n:>6,} memories: avg={avg:.3f}ms  p50={p50:.3f}ms  p99={p99:.3f}ms")
        RESULTS[f"hybrid_search_{n}"] = {"avg_ms": round(float(avg), 3), "p50_ms": round(float(p50), 3), "p99_ms": round(float(p99), 3)}
        mem.close()
    
    # ── 5. FULL RECALL() LATENCY ───────────────────────────────────────
    print("\n[5/10] FULL recall() LATENCY (hybrid + access logging)")
    print("-" * 50)
    for n in [1_000, 10_000, 50_000]:
        db_p = os.path.join(tmpdir, f"recall_{n}.db")
        mem = AriadneMemory(config=AriadneConfig(db_path=db_p, embedding_dim=DIM))
        vecs = np.random.randn(n, DIM).astype("float32")
        for i, v in enumerate(vecs):
            mem.remember(f"memory {i}: deploy to production using kubernetes cluster {i % 50}", embedding=v)
        
        queries = np.random.randn(1_000, DIM).astype("float32")
        for q in queries[:100]:
            mem.recall("deploy production", embedding=q, k=10)
        
        times = []
        for q in queries:
            t0 = time.perf_counter()
            mem.recall("deploy production", embedding=q, k=10)
            times.append((time.perf_counter() - t0) * 1000)
        
        avg = np.mean(times)
        p50 = np.percentile(times, 50)
        p99 = np.percentile(times, 99)
        print(f"  {n:>6,} memories: avg={avg:.3f}ms  p50={p50:.3f}ms  p99={p99:.3f}ms")
        RESULTS[f"recall_{n}"] = {"avg_ms": round(float(avg), 3), "p50_ms": round(float(p50), 3), "p99_ms": round(float(p99), 3)}
        mem.close()
    
    # ── 6. KNOWLEDGE GRAPH ─────────────────────────────────────────────
    print("\n[6/10] KNOWLEDGE GRAPH (add edges + multi-hop traversal)")
    print("-" * 50)
    db_p = os.path.join(tmpdir, "graph.db")
    mem = AriadneMemory(config=AriadneConfig(db_path=db_p, embedding_dim=DIM))
    
    edges = [
        ("WebApp", "API", "depends_on"),
        ("API", "Database", "depends_on"),
        ("API", "Auth", "depends_on"),
        ("WebApp", "Cache", "depends_on"),
        ("API", "Logger", "sends_to"),
        ("API", "Metrics", "sends_to"),
        ("API", "Queue", "publishes_to"),
        ("Queue", "Storage", "writes_to"),
        ("CDN", "Storage", "reads_from"),
        ("Metrics", "Logger", "related"),
    ]
    
    t0 = time.perf_counter()
    for src, tgt, etype in edges:
        mem.add_edge(src, tgt, edge_type=etype)
    graph_write_ms = (time.perf_counter() - t0) * 1000
    print(f"  Graph build ({len(edges)} edges): {graph_write_ms:.2f}ms total")
    
    # Multi-hop traversal
    hop_times = []
    last_result = {}
    for _ in range(500):
        t0 = time.perf_counter()
        result = mem.graph("WebApp", hops=3)
        hop_times.append((time.perf_counter() - t0) * 1000)
        last_result = result
    
    avg = np.mean(hop_times)
    p99 = np.percentile(hop_times, 99)
    n_connected = len(last_result.get("entities", [])) if isinstance(last_result, dict) else len(last_result)
    print(f"  Multi-hop traversal (hops=3): avg={avg:.3f}ms  p99={p99:.3f}ms")
    print(f"  Connected entities returned: {n_connected}")
    RESULTS["graph"] = {
        "write_ms": round(graph_write_ms, 2),
        "traverse_avg_ms": round(float(avg), 3),
        "traverse_p99_ms": round(float(p99), 3),
        "connected_entities": n_connected,
    }
    mem.close()
    
    # ── 7. DEDUPLICATION (MinHash) ─────────────────────────────────────
    print("\n[7/10] DEDUPLICATION (MinHash LSH)")
    print("-" * 50)
    dedup = Deduplicator(threshold=0.8, num_perm=128)
    
    # Bulk insert
    docs = [f"memory about topic {i % 50} with content for dedup testing" for i in range(1_000)]
    t0 = time.perf_counter()
    for m in docs:
        dedup.add(m)
    insert_ms = (time.perf_counter() - t0) * 1000 / len(docs)
    
    # Near-duplicate pairs (original, near-duplicate)
    near_dupes = [
        ("The quick brown fox jumps over the lazy dog in the park", "A quick brown fox leaps over the lazy dog in the park"),
        ("Deploy to production using kubernetes with rolling updates", "Deploy to production using kubernetes via rolling updates"),
        ("Machine learning model training requires GPU acceleration", "Machine learning model training needs GPU acceleration"),
        ("The database connection pool has been exhausted completely", "The database connection pool is completely exhausted"),
        ("Python 3.12 includes significant performance improvements", "Python 3.12 brings significant performance improvements"),
    ]
    # Also add non-duplicates
    non_dupes = [
        "Completely unrelated text about cooking recipes",
        "The stock market crashed yesterday unexpectedly",
        "Quantum computing breakthrough in error correction",
    ]
    
    # Add originals first
    for orig, _ in near_dupes:
        dedup.add(orig)
    for nd in non_dupes:
        dedup.add(nd)
    
    dupe_times = []
    detected = 0
    for orig, near in near_dupes:
        t0 = time.perf_counter()
        is_dup, score = dedup.is_duplicate(near)
        dupe_times.append((time.perf_counter() - t0) * 1000)
        if is_dup:
            detected += 1
    
    non_dup_detected = 0
    for nd in non_dupes:
        is_dup, score = dedup.is_duplicate(nd)
        if is_dup:
            non_dup_detected += 1
    
    dupe_avg = np.mean(dupe_times)
    print(f"  Insert 1K docs: {insert_ms:.4f} ms/doc")
    print(f"  Near-duplicate check: avg={dupe_avg:.4f} ms/check")
    print(f"  True positive rate: {detected}/{len(near_dupes)} ({detected/len(near_dupes)*100:.0f}%)")
    print(f"  False positive rate: {non_dup_detected}/{len(non_dupes)} ({non_dup_detected/len(non_dupes)*100:.0f}%)")
    RESULTS["dedup"] = {
        "insert_ms_per_doc": round(insert_ms, 4),
        "check_ms": round(float(dupe_avg), 4),
        "true_positive_rate": f"{detected}/{len(near_dupes)}",
        "false_positive_rate": f"{non_dup_detected}/{len(non_dupes)}",
    }
    
    # ── 8. CONCURRENT READ/WRITE ───────────────────────────────────────
    print("\n[8/10] CONCURRENT READ/WRITE THROUGHPUT")
    print("-" * 50)
    db_p = os.path.join(tmpdir, "concurrent.db")
    mem = AriadneMemory(config=AriadneConfig(db_path=db_p, embedding_dim=DIM))
    
    vecs = np.random.randn(5_000, DIM).astype("float32")
    for i, v in enumerate(vecs):
        mem.remember(f"memory {i}: concurrent test data", embedding=v)
    
    read_count = [0]
    write_count = [0]
    errors = [0]
    stop = threading.Event()
    
    def reader():
        q = np.random.randn(DIM).astype("float32")
        while not stop.is_set():
            try:
                mem.recall("concurrent", embedding=q, k=5)
                read_count[0] += 1
            except Exception:
                errors[0] += 1
    
    def writer():
        i = 5000
        while not stop.is_set():
            try:
                v = np.random.randn(DIM).astype("float32")
                mem.remember(f"new memory {i}: concurrent write", embedding=v)
                write_count[0] += 1
                i += 1
            except Exception:
                errors[0] += 1
    
    threads = [threading.Thread(target=reader) for _ in range(4)]
    threads += [threading.Thread(target=writer) for _ in range(2)]
    
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(5)
    stop.set()
    for t in threads:
        t.join()
    elapsed = time.perf_counter() - t0
    
    rps = read_count[0] / elapsed
    wps = write_count[0] / elapsed
    print(f"  4 readers + 2 writers over {elapsed:.1f}s:")
    print(f"  Reads:  {read_count[0]:,} ops ({rps:.0f} reads/s)")
    print(f"  Writes: {write_count[0]:,} ops ({wps:.0f} writes/s)")
    print(f"  Errors: {errors[0]}")
    RESULTS["concurrent"] = {
        "reads": read_count[0],
        "writes": write_count[0],
        "reads_per_sec": round(rps, 0),
        "writes_per_sec": round(wps, 0),
        "errors": errors[0],
    }
    mem.close()
    
    # ── 9. MEMORY FOOTPRINT ────────────────────────────────────────────
    print("\n[9/10] MEMORY FOOTPRINT")
    print("-" * 50)
    for n in [1_000, 10_000]:
        db_p = os.path.join(tmpdir, f"foot_{n}.db")
        mem = AriadneMemory(config=AriadneConfig(db_path=db_p, embedding_dim=DIM))
        
        vecs = np.random.randn(n, DIM).astype("float32")
        for i, v in enumerate(vecs):
            mem.remember(f"memory {i}: footprint test with realistic content length for a real agent memory", embedding=v)
        
        db_size = os.path.getsize(db_p) / (1024 * 1024)
        per_mem_kb = (db_size * 1024) / n
        print(f"  {n:>6,} memories: DB={db_size:.1f}MB  ~{per_mem_kb:.1f}KB/memory")
        RESULTS[f"footprint_{n}"] = {"db_mb": round(db_size, 1), "per_memory_kb": round(per_mem_kb, 1)}
        mem.close()
    
    # ── 10. COLD START TIME ────────────────────────────────────────────
    print("\n[10/10] COLD START TIME (DB open + FAISS rebuild)")
    print("-" * 50)
    for n in [1_000, 10_000, 50_000]:
        db_p = os.path.join(tmpdir, f"cold_{n}.db")
        mem = AriadneMemory(config=AriadneConfig(db_path=db_p, embedding_dim=DIM))
        vecs = np.random.randn(n, DIM).astype("float32")
        for i, v in enumerate(vecs):
            mem.remember(f"memory {i}: cold start test", embedding=v)
        mem.close()
        
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            mem2 = AriadneMemory(config=AriadneConfig(db_path=db_p, embedding_dim=DIM))
            elapsed_ms = (time.perf_counter() - t0) * 1000
            times.append(elapsed_ms)
            mem2.close()
        
        avg = np.mean(times)
        p99 = np.percentile(times, 99)
        print(f"  {n:>6,} vectors: avg={avg:.1f}ms  p99={p99:.1f}ms")
        RESULTS[f"cold_start_{n}"] = {"avg_ms": round(float(avg), 1), "p99_ms": round(float(p99), 1)}
    
    # ── SUMMARY ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RAW RESULTS (JSON)")
    print("=" * 70)
    print(json.dumps(RESULTS, indent=2))
    
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    run_benchmarks()

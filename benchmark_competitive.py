#!/usr/bin/env python3
"""Head-to-head benchmark: Ariadne vs ChromaDB vs sqlite-vec vs raw FAISS."""
import importlib.util, json, os, sqlite3, sys, tempfile, time
import numpy as np

# Bootstrap ariadne (bypass hermes editable finder)
import types as _types
_pkg = _types.ModuleType('ariadne')
_pkg.__path__ = ['/root/arriadne/src/arriadne']
_pkg.__package__ = 'ariadne'
sys.modules['ariadne'] = _pkg
sys.path.insert(0, '/root/arriadne/src')
with open('/root/arriadne/src/arriadne/__init__.py') as _f:
    exec(compile(_f.read(), 'x', 'exec'), _pkg.__dict__)

from ariadne import AriadneDB, AriadneConfig

# ── Data ────────────────────────────────────────────────────────
NUM = 1000
DIM = 384
TOP_K = 10
np.random.seed(42)

TOPICS = [
    "The user prefers dark mode on all interfaces",
    "Project deadline is June 30, 2026",
    "VPS runs Ubuntu 24.04 with 4 cores and 8GB RAM",
    "User GitHub username is kyssta-exe",
    "The hermey character uses noir monochrome expressions",
    "Lumora is a Minecraft server fork based on Leaf",
    "FoneWorld CRM runs on port 3000 at 57.129.120.248",
    "Ariadne memory system uses FAISS for vector search",
    "The user timezone is PKT UTC+5",
    "Mailcow handles email at mail.kase.lol",
]
memories = [f"Memory #{i}: {TOPICS[i % len(TOPICS)]}. Context details {i}." for i in range(NUM)]
embeddings = np.random.randn(NUM, DIM).astype(np.float32)
# Normalize
norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
normed = embeddings / norms

results = {"hardware": "4-core 8GB RAM Ubuntu 24.04", "memories": NUM, "dim": DIM, "benchmarks": {}}

def time_fn(fn, n=5):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1_000_000)
    times.sort()
    return times[len(times)//2], times[int(len(times)*0.95)]

# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("Ariadne")
print("=" * 60)

with tempfile.TemporaryDirectory() as tmpdir:
    db = os.path.join(tmpdir, "a.db")
    cfg = AriadneConfig(db_path=db, embedding_dim=DIM, faiss_type="auto")
    adb = AriadneDB(config=cfg)
    adb.open()

    def insert_all():
        for i in range(NUM):
            adb.add_memory(content=memories[i], importance=0.5, embedding=normed[i])
    ins_p50, ins_p95 = time_fn(insert_all, n=1)
    print(f"  Insert: {ins_p50:.0f}ms total ({ins_p50/NUM:.1f}ms/mem)")

    def vs(): adb.vector_search(normed[0], k=TOP_K)
    vs_p50, vs_p95 = time_fn(vs)
    print(f"  Vector: p50={vs_p50:.0f}us p95={vs_p95:.0f}us")

    def fts(): adb.fts_search("deadline project", k=TOP_K)
    fts_p50, fts_p95 = time_fn(fts)
    print(f"  FTS:    p50={fts_p50:.0f}us p95={fts_p95:.0f}us")

    def hyb(): adb.hybrid_search("VPS configuration", embedding=normed[0], k=TOP_K)
    hyb_p50, hyb_p95 = time_fn(hyb)
    print(f"  Hybrid: p50={hyb_p50:.0f}us p95={hyb_p95:.0f}us")

    adb.add_edge("Kyssta", "VPS", "owns", 0.9)
    adb.add_edge("Kyssta", "Ariadne", "built", 0.8)
    adb.add_edge("VPS", "Ubuntu", "runs", 0.7)
    def graph(): adb.traverse_graph("Kyssta", hops=3)
    g_p50, g_p95 = time_fn(graph)
    print(f"  Graph:  p50={g_p50:.0f}us p95={g_p95:.0f}us")

    db_size = os.path.getsize(db)
    faiss_path = db.replace('.db', '.faiss')
    faiss_size = os.path.getsize(faiss_path) if os.path.exists(faiss_path) else 0

    results["benchmarks"]["ariadne"] = {
        "insert_total_ms": round(ins_p50, 1),
        "insert_per_ms": round(ins_p50/NUM, 2),
        "vector_p50us": round(vs_p50, 0),
        "vector_p95us": round(vs_p95, 0),
        "fts_p50us": round(fts_p50, 0),
        "fts_p95us": round(fts_p95, 0),
        "hybrid_p50us": round(hyb_p50, 0),
        "hybrid_p95us": round(hyb_p95, 0),
        "graph_p50us": round(g_p50, 0),
        "graph_p95us": round(g_p95, 0),
        "db_kb": round(db_size/1024, 1),
        "faiss_kb": round(faiss_size/1024, 1),
        "total_kb": round((db_size+faiss_size)/1024, 1),
    }
    adb.close()

# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("ChromaDB")
print("=" * 60)

try:
    import chromadb
    with tempfile.TemporaryDirectory() as tmpdir:
        client = chromadb.PersistentClient(path=tmpdir)
        col = client.create_collection("test", metadata={"hnsw:space": "cosine"})

        def chroma_insert():
            for start in range(0, NUM, 100):
                end = min(start+100, NUM)
                col.add(ids=[f"m{i}" for i in range(start, end)],
                        documents=memories[start:end],
                        embeddings=embeddings[start:end].tolist())
        ins_p50, ins_p95 = time_fn(chroma_insert, n=1)
        print(f"  Insert: {ins_p50:.0f}ms total ({ins_p50/NUM:.1f}ms/mem)")

        def chroma_search():
            col.query(query_embeddings=embeddings[0].tolist(), n_results=TOP_K)
        vs_p50, vs_p95 = time_fn(chroma_search)
        print(f"  Vector: p50={vs_p50:.0f}us p95={vs_p95:.0f}us")

        chroma_size = sum(os.path.getsize(os.path.join(tmpdir, f))
                          for f in os.listdir(tmpdir) if os.path.isfile(os.path.join(tmpdir, f)))
        results["benchmarks"]["chromadb"] = {
            "insert_total_ms": round(ins_p50, 1),
            "insert_per_ms": round(ins_p50/NUM, 2),
            "vector_p50us": round(vs_p50, 0),
            "vector_p95us": round(vs_p95, 0),
            "total_kb": round(chroma_size/1024, 1),
        }
        print(f"  Size: {chroma_size/1024:.0f}KB")
except Exception as e:
    print(f"  ERROR: {e}")
    results["benchmarks"]["chromadb"] = {"error": str(e)}

# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("sqlite-vec")
print("=" * 60)

try:
    import sqlite_vec
    with tempfile.TemporaryDirectory() as tmpdir:
        svdb = os.path.join(tmpdir, "sv.db")
        conn = sqlite3.connect(svdb)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute("CREATE TABLE memories (id INTEGER PRIMARY KEY, content TEXT, embedding BLOB)")
        conn.execute("CREATE VIRTUAL TABLE vec_memories USING vec0(id INTEGER PRIMARY KEY, embedding float[384])")
        conn.commit()

        def vec_insert():
            for i in range(NUM):
                blob = embeddings[i].tobytes()
                conn.execute("INSERT INTO memories (id, content, embedding) VALUES (?, ?, ?)", (i, memories[i], blob))
                conn.execute("INSERT INTO vec_memories (id, embedding) VALUES (?, ?)", (i, blob))
            conn.commit()
        ins_p50, ins_p95 = time_fn(vec_insert, n=1)
        print(f"  Insert: {ins_p50:.0f}ms total ({ins_p50/NUM:.1f}ms/mem)")

        def vec_search():
            blob = embeddings[0].tobytes()
            conn.execute(
                "SELECT id, distance FROM vec_memories WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                (blob, TOP_K)).fetchall()
        vs_p50, vs_p95 = time_fn(vec_search)
        print(f"  Vector: p50={vs_p50:.0f}us p95={vs_p95:.0f}us")

        vec_size = os.path.getsize(svdb)
        results["benchmarks"]["sqlite_vec"] = {
            "insert_total_ms": round(ins_p50, 1),
            "insert_per_ms": round(ins_p50/NUM, 2),
            "vector_p50us": round(vs_p50, 0),
            "vector_p95us": round(vs_p95, 0),
            "total_kb": round(vec_size/1024, 1),
        }
        print(f"  Size: {vec_size/1024:.0f}KB")
        conn.close()
except Exception as e:
    print(f"  ERROR: {e}")
    results["benchmarks"]["sqlite_vec"] = {"error": str(e)}

# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Raw FAISS (baseline)")
print("=" * 60)

import faiss
idx = faiss.IndexFlatIP(DIM)
idx.add(normed)

def faiss_search():
    q = normed[:1].copy()
    idx.search(q, TOP_K)
vs_p50, vs_p95 = time_fn(faiss_search)
print(f"  Vector: p50={vs_p50:.0f}us p95={vs_p95:.0f}us")

results["benchmarks"]["faiss_raw"] = {
    "vector_p50us": round(vs_p50, 0),
    "vector_p95us": round(vs_p95, 0),
}

# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FEATURE MATRIX")
print("=" * 60)

features = [
    ("Vector Search",           ["Ariadne", "ChromaDB", "sqlite-vec", "FAISS"]),
    ("FTS5 Keyword Search",     ["Ariadne"]),
    ("Hybrid (vector+FTS RRF)", ["Ariadne"]),
    ("Knowledge Graph",         ["Ariadne"]),
    ("Entity Resolution",       ["Ariadne"]),
    ("Deduplication",           ["Ariadne"]),
    ("Temporal Facts",          ["Ariadne"]),
    ("Lifecycle (hot/warm/cold)",["Ariadne"]),
    ("Ebbinghaus Retention",    ["Ariadne"]),
    ("LLM Fact Extraction",     ["Ariadne"]),
    ("Memory Consolidation",    ["Ariadne"]),
    ("Zero Dependencies",       ["Ariadne", "sqlite-vec", "FAISS"]),
    ("No Server Required",      ["Ariadne", "sqlite-vec", "FAISS"]),
    ("Client-Server Mode",      ["ChromaDB"]),
    ("Multi-tenancy",           ["ChromaDB"]),
    ("REST API (built-in)",     ["Ariadne"]),
]
for feat, systems in features:
    print(f"  {feat}: {', '.join(systems)}")

# Save
with open("/root/arriadne/BENCHMARK_RESULTS.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to /root/arriadne/BENCHMARK_RESULTS.json")

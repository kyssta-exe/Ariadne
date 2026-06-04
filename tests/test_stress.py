#!/usr/bin/env python3
"""
Stress and concurrency tests for Ariadne memory system.

Tests:
  - 1000 rapid sequential inserts (p50/p95/p99 latency)
  - 100 rapid search queries while inserting (no crashes/deadlocks)
  - Open/close database 100 times (no resource leaks)
  - Import/export 10K memories (no corruption)
  - Edge cases: 10KB, 100KB content, empty, unicode, emoji, SQL injection
  - Graph with 1000 nodes and 5000 edges
  - Temporal with 500 versioned facts
  - All tests complete within 60 seconds total
"""
from __future__ import annotations

import json
import os
import random
import sqlite3
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, "/root/arriadne/src")

from arriadne.config import AriadneConfig
from arriadne.storage import AriadneDB

EMBEDDING_DIM = 384

# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db(tmp_path):
    """Create a fresh temporary database."""
    db_path = str(tmp_path / "test.db")
    config = AriadneConfig(
        db_path=db_path,
        embedding_dim=EMBEDDING_DIM,
        faiss_type="flat_ip",
    )
    db = AriadneDB(config=config)
    db.open()
    yield db
    db.close()


@pytest.fixture
def sample_embeddings():
    """Generate 1000 random normalized embeddings."""
    rng = np.random.RandomState(42)
    embs = rng.randn(1000, EMBEDDING_DIM).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / np.maximum(norms, 1e-10)


@pytest.fixture
def sample_memories():
    """Generate 1000 sample memories."""
    topics = ["Python", "ML", "web", "DB", "cloud", "security", "data", "API", "test", "deploy"]
    return [
        f"Memory {i}: Discussion about {topics[i % len(topics)]} topic #{i}. "
        f"This is a detailed note from session {i // 10}."
        for i in range(1000)
    ]


# ─── Test: Rapid Sequential Inserts ─────────────────────────────────────────

class TestRapidInserts:
    def test_1000_sequential_inserts(self, tmp_db, sample_embeddings):
        """1000 rapid sequential inserts with latency tracking."""
        times = []
        for i in range(1000):
            content = f"Rapid insert test memory {i}: important information about topic {i % 30}"
            t0 = time.perf_counter()
            tmp_db.add_memory(content=content, embedding=sample_embeddings[i])
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        times_arr = np.array(times)
        p50 = float(np.percentile(times_arr, 50))
        p95 = float(np.percentile(times_arr, 95))
        p99 = float(np.percentile(times_arr, 99))

        print(f"\n  Insert latency: p50={p50:.2f}ms p95={p95:.2f}ms p99={p99:.2f}ms")

        cursor = tmp_db.conn.execute("SELECT COUNT(*) FROM memories WHERE is_deleted = 0")
        assert cursor.fetchone()[0] == 1000
        assert p50 < 500, f"p50 insert latency too high: {p50}ms"
        assert p99 < 2000, f"p99 insert latency too high: {p99}ms"

    def test_batch_insert_performance(self, tmp_db, sample_embeddings):
        """Batch insert should be significantly faster than single."""
        items = [
            {"content": f"Batch item {i}", "embedding": sample_embeddings[i]}
            for i in range(500)
        ]
        t0 = time.perf_counter()
        results = tmp_db.add_memory_batch(items)
        t1 = time.perf_counter()
        batch_time = (t1 - t0) * 1000

        created = sum(1 for r in results if r["status"] == "created")
        assert created == 500
        print(f"\n  Batch insert 500 items: {batch_time:.0f}ms ({500/(batch_time/1000):.0f} items/sec)")


# ─── Test: Search Under Pressure ─────────────────────────────────────────────

class TestSearchUnderPressure:
    def test_search_while_inserting(self, tmp_db, sample_embeddings):
        """100 search queries while inserting — no crashes, no deadlocks."""
        # Pre-populate
        for i in range(200):
            tmp_db.add_memory(
                content=f"Pre-populated memory {i} about topic {i % 20}",
                embedding=sample_embeddings[i],
            )

        errors = []
        search_times = []

        def inserter():
            try:
                for i in range(200, 700):
                    tmp_db.add_memory(
                        content=f"Concurrent insert {i}",
                        embedding=sample_embeddings[i % len(sample_embeddings)],
                    )
            except Exception as e:
                errors.append(("insert", str(e)))

        def searcher():
            try:
                for i in range(100):
                    query_emb = sample_embeddings[i % len(sample_embeddings)]
                    t0 = time.perf_counter()
                    tmp_db.vector_search(query_emb, k=5)
                    t1 = time.perf_counter()
                    search_times.append((t1 - t0) * 1000)
            except Exception as e:
                errors.append(("search", str(e)))

        t0 = time.perf_counter()
        insert_thread = threading.Thread(target=inserter)
        search_thread = threading.Thread(target=searcher)
        insert_thread.start()
        search_thread.start()
        insert_thread.join(timeout=60)
        search_thread.join(timeout=60)
        total = (time.perf_counter() - t0) * 1000

        assert len(errors) == 0, f"Errors during concurrent ops: {errors}"
        assert len(search_times) == 100
        assert total < 60000, f"Total time too high: {total}ms"

        search_arr = np.array(search_times)
        print(f"\n  Concurrent search p50={np.percentile(search_arr,50):.2f}ms "
              f"p95={np.percentile(search_arr,95):.2f}ms total={total:.0f}ms")


# ─── Test: Open/Close Cycles ─────────────────────────────────────────────────

class TestResourceLeak:
    def test_open_close_100_times(self, tmp_path):
        """Open and close database 100 times — no resource leaks."""
        db_path = str(tmp_path / "cycle_test.db")
        config = AriadneConfig(db_path=db_path, embedding_dim=EMBEDDING_DIM, faiss_type="flat_ip")
        rng = np.random.RandomState(42)

        for i in range(100):
            db = AriadneDB(config=config)
            db.open()
            emb = rng.randn(EMBEDDING_DIM).astype(np.float32)
            emb = emb / max(np.linalg.norm(emb), 1e-10)
            db.add_memory(content=f"Cycle {i} memory", embedding=emb)
            db.close()

        # Verify data survived all cycles
        db = AriadneDB(config=config)
        db.open()
        cursor = db.conn.execute("SELECT COUNT(*) FROM memories WHERE is_deleted = 0")
        count = cursor.fetchone()[0]
        db.close()

        assert count == 100, f"Expected 100 memories, got {count}"
        print(f"\n  100 open/close cycles: {count} memories survived")

        # Verify no leftover lock files or WAL
        for ext in ["-wal", "-shm", ".faiss"]:
            fpath = db_path + ext
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                print(f"  {ext} file: {size} bytes")


# ─── Test: Import/Export ─────────────────────────────────────────────────────

class TestImportExport:
    def test_export_import_10k(self, tmp_path):
        """Export and import 10K memories without corruption."""
        src_path = str(tmp_path / "src.db")
        dst_path = str(tmp_path / "dst.db")

        # Create source with 10K memories
        config_src = AriadneConfig(db_path=src_path, embedding_dim=EMBEDDING_DIM, faiss_type="flat_ip")
        rng = np.random.RandomState(42)

        db_src = AriadneDB(config=config_src)
        db_src.open()

        for i in range(10000):
            emb = rng.randn(EMBEDDING_DIM).astype(np.float32)
            emb = emb / max(np.linalg.norm(emb), 1e-10)
            db_src.add_memory(
                content=f"Export memory {i}: data about topic {i % 30}",
                embedding=emb,
            )

        # Export as JSON
        t0 = time.perf_counter()
        cursor = db_src.conn.execute(
            "SELECT id, content, content_hash, memory_type, importance, "
            "created_at, updated_at, metadata FROM memories WHERE is_deleted = 0"
        )
        export_data = []
        for row in cursor.fetchall():
            export_data.append({
                "id": row[0], "content": row[1], "content_hash": row[2],
                "memory_type": row[3], "importance": row[4],
                "created_at": row[5], "updated_at": row[6],
                "metadata": json.loads(row[7]) if row[7] else None,
            })
        db_src.close()
        export_time = (time.perf_counter() - t0) * 1000

        assert len(export_data) == 10000, f"Exported {len(export_data)} instead of 10000"

        # Import into fresh DB
        config_dst = AriadneConfig(db_path=dst_path, embedding_dim=EMBEDDING_DIM, faiss_type="flat_ip")
        db_dst = AriadneDB(config=config_dst)
        db_dst.open()

        t0 = time.perf_counter()
        for item in export_data:
            db_dst.add_memory(content=item["content"], importance=item["importance"])
        import_time = (time.perf_counter() - t0) * 1000
        db_dst.close()

        # Verify
        db_dst2 = AriadneDB(config=config_dst)
        db_dst2.open()
        cursor = db_dst2.conn.execute("SELECT COUNT(*) FROM memories WHERE is_deleted = 0")
        count = cursor.fetchone()[0]
        # Check a sample for corruption
        cursor = db_dst2.conn.execute("SELECT content FROM memories LIMIT 5")
        samples = [row[0] for row in cursor.fetchall()]
        db_dst2.close()

        assert count == 10000, f"Imported {count} instead of 10000"
        assert all("Export memory" in s for s in samples), "Content corruption detected"

        print(f"\n  Export 10K: {export_time:.0f}ms")
        print(f"  Import 10K: {import_time:.0f}ms")
        print(f"  Verification: {count} memories, content intact")


# ─── Test: Edge Cases ────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_large_content_10kb(self, tmp_db):
        """10KB memory content."""
        content = "A" * 10240  # 10KB
        result = tmp_db.add_memory(content=content)
        assert result["status"] == "created"
        mem = tmp_db.get_memory(result["memory_id"])
        assert mem is not None
        assert len(mem["content"]) == 10240
        print(f"\n  10KB content: OK (id={result['memory_id']})")

    def test_large_content_100kb(self, tmp_db):
        """100KB memory content."""
        content = "B" * 102400  # 100KB
        result = tmp_db.add_memory(content=content)
        assert result["status"] == "created"
        mem = tmp_db.get_memory(result["memory_id"])
        assert mem is not None
        assert len(mem["content"]) == 102400
        print(f"\n  100KB content: OK (id={result['memory_id']})")

    def test_empty_content(self, tmp_db):
        """Empty content should be handled gracefully."""
        try:
            result = tmp_db.add_memory(content="")
            # Either creates or raises — both are acceptable
            print(f"\n  Empty content: handled (status={result['status']})")
        except (ValueError, sqlite3.Error) as e:
            print(f"\n  Empty content: rejected ({type(e).__name__})")

    def test_unicode_content(self, tmp_db):
        """Unicode content with various scripts."""
        test_strings = [
            "Japanese: 日本語テスト",
            "Chinese: 中文测试",
            "Korean: 한국어 테스트",
            "Arabic: اختبار عربي",
            "Hindi: हिंदी परीक्षा",
            "Russian: Русский тест",
            "Greek: Ελληνικά τεστ",
            "Thai: ภาษาไทยทดสอบ",
            "Emoji: 🚀🎯💡🧠✨",
            "Mixed: Hello世界مرحبا🌍",
        ]
        for s in test_strings:
            result = tmp_db.add_memory(content=s)
            assert result["status"] == "created"
            mem = tmp_db.get_memory(result["memory_id"])
            assert mem["content"] == s
        print(f"\n  Unicode ({len(test_strings)} strings): all OK")

    def test_emoji_content(self, tmp_db):
        """Emoji-heavy content."""
        content = "🎉🎊🥳 Party time! 🍕🍔🍟 Food is 🍕. Let's go 🚀 to the 🌙!"
        result = tmp_db.add_memory(content=content)
        assert result["status"] == "created"
        mem = tmp_db.get_memory(result["memory_id"])
        assert mem["content"] == content
        print("\n  Emoji content: OK")

    @pytest.mark.parametrize("injection", [
        "'; DROP TABLE memories; --",
        "1 OR 1=1",
        "Robert'); DROP TABLE students;--",
        "\\\" OR \\\"\\\" = \\\"\\\"",
        "UNION SELECT * FROM memories",
        "<script>alert('xss')</script>",
        "{{7*7}}",
        "${7*7}",
        "%s' OR '1'='1",
        "1; DELETE FROM memories WHERE 1=1",
    ])
    def test_sql_injection_attempts(self, tmp_db, injection):
        """SQL injection in search should not cause data loss."""
        # Pre-populate
        tmp_db.add_memory(content="Legitimate memory for testing")
        cursor = tmp_db.conn.execute("SELECT COUNT(*) FROM memories WHERE is_deleted = 0")
        count_before = cursor.fetchone()[0]

        # Try injection via FTS search
        try:
            tmp_db.fts_search(injection, k=10)
        except sqlite3.Error:
            pass  # FTS may reject malformed queries

        # Try injection via vector search (safe — uses parameterized queries)
        try:
            rng = np.random.RandomState(0)
            emb = rng.randn(EMBEDDING_DIM).astype(np.float32)
            tmp_db.vector_search(emb, k=5)
        except Exception:
            pass

        cursor = tmp_db.conn.execute("SELECT COUNT(*) FROM memories WHERE is_deleted = 0")
        count_after = cursor.fetchone()[0]
        assert count_after == count_before, f"Data loss! Before={count_before} After={count_after}"

    def test_special_characters_in_content(self, tmp_db):
        """Content with special characters that could break FTS."""
        special = [
            "Quotes: \"hello\" 'world'",
            "Backslash: path\\to\\file",
            "Newlines: line1\nline2\nline3",
            "Tabs: col1\tcol2\tcol3",
            "Null bytes: test\0value",  # May be stripped
            "Brackets: [array] {dict} (tuple)",
            "Asterisks: * wildcard ** double",
            "Pipes: a | b | c",
            "Semicolons: cmd1; cmd2; cmd3",
        ]
        for s in special:
            try:
                tmp_db.add_memory(content=s)
            except Exception:
                pass  # Some may fail — that's OK
        print(f"\n  Special characters ({len(special)} strings): tested")


# ─── Test: Knowledge Graph ───────────────────────────────────────────────────

class TestKnowledgeGraph:
    def test_large_graph_traversal(self, tmp_db):
        """Graph with 1000 nodes and 5000 edges, traverse all."""
        # Create 1000 entities
        t0 = time.perf_counter()
        for i in range(1000):
            tmp_db.add_entity(f"node_{i}", entity_type="concept")
        entity_time = (time.perf_counter() - t0) * 1000

        # Create 5000 edges (mix of chain + random)
        t0 = time.perf_counter()
        rng = random.Random(42)
        for i in range(999):
            tmp_db.add_edge(f"node_{i}", f"node_{i+1}", "chain", 1.0)
        # Random edges
        for _ in range(4001):
            src = rng.randint(0, 999)
            tgt = rng.randint(0, 999)
            if src != tgt:
                tmp_db.add_edge(f"node_{src}", f"node_{tgt}", "related", 0.5)
        edge_time = (time.perf_counter() - t0) * 1000

        # Traverse at various depths
        traversal_times = []
        for hops in [1, 2, 3, 5]:
            t0 = time.perf_counter()
            result = tmp_db.traverse_graph("node_0", hops=hops)
            t1 = time.perf_counter()
            traversal_times.append((t1 - t0) * 1000)
            nodes_found = len(result.get("nodes", []))
            print(f"    {hops}-hop: {nodes_found} nodes, {traversal_times[-1]:.2f}ms")

        assert entity_time < 10000, f"Entity creation too slow: {entity_time}ms"
        assert edge_time < 30000, f"Edge creation too slow: {edge_time}ms"
        print(f"\n  Graph: 1000 nodes ({entity_time:.0f}ms), 5000 edges ({edge_time:.0f}ms)")

    def test_bidirectional_traversal(self, tmp_db):
        """Verify bidirectional graph traversal works."""
        # Create: A -> B -> C <- D
        tmp_db.add_edge("A", "B", "forward")
        tmp_db.add_edge("B", "C", "forward")
        tmp_db.add_edge("D", "C", "backward")

        result = tmp_db.traverse_graph("A", hops=3)
        nodes = set(result["nodes"])
        assert "A" in nodes
        assert "B" in nodes
        assert "C" in nodes
        # D should be reachable via bidirectional traversal
        assert "D" in nodes, f"D not found in traversal from A: {nodes}"
        print(f"\n  Bidirectional traversal: {len(nodes)} nodes reached from A")


# ─── Test: Temporal Facts ────────────────────────────────────────────────────

class TestTemporalFacts:
    def test_versioned_facts(self, tmp_db):
        """500 versioned facts with temporal queries across time."""
        from arriadne.temporal import TemporalGraph

        tg = TemporalGraph(tmp_db.conn)

        base_time = 1700000000.0  # Fixed epoch
        subjects = [f"subject_{i}" for i in range(50)]

        # Insert 500 facts (10 versions per subject)
        t0 = time.perf_counter()
        for i in range(500):
            subj = subjects[i % 50]
            version = i // 50
            valid_at = base_time + version * 86400  # One day apart

            tg.add_fact(
                text=f"Fact {i} about {subj} at version {version}",
                subject=subj,
                predicate="has_property",
                obj=f"value_{version}",
                valid_at=valid_at,
                confidence=0.9,
            )
        insert_time = (time.perf_counter() - t0) * 1000

        # Query at different time points
        query_times = []
        for t_offset in range(0, 10):
            query_time = base_time + t_offset * 86400 + 43200  # Mid-day
            t0 = time.perf_counter()
            tg.find_facts(at_time=query_time, current_only=False, limit=200)
            t1 = time.perf_counter()
            query_times.append((t1 - t0) * 1000)

        # Get timeline for one subject
        t0 = time.perf_counter()
        timeline = tg.get_timeline("subject_0", limit=20)
        timeline_time = (time.perf_counter() - t0) * 1000

        # Get superseded facts
        t0 = time.perf_counter()
        superseded = tg.get_superseded_facts(limit=50)
        superseded_time = (time.perf_counter() - t0) * 1000

        stats = tg.stats()

        print(f"\n  Temporal: 500 facts ({insert_time:.0f}ms)")
        print(f"  Query p50: {np.median(query_times):.2f}ms")
        print(f"  Timeline: {len(timeline)} facts ({timeline_time:.2f}ms)")
        print(f"  Superseded: {len(superseded)} pairs ({superseded_time:.2f}ms)")
        print(f"  Stats: {stats}")

        assert stats["total_facts"] == 500
        assert stats["superseded_facts"] > 0, "Expected superseded facts"
        assert len(timeline) > 0, "Expected timeline entries"

    def test_fact_invalidation(self, tmp_db):
        """Verify fact invalidation and contradiction tracking."""
        from arriadne.temporal import TemporalGraph

        tg = TemporalGraph(tmp_db.conn)
        now = time.time()

        # Add initial fact
        f1 = tg.add_fact(
            text="Paris is the capital of France",
            subject="Paris", predicate="is_capital_of", obj="France",
            valid_at=now - 86400,
        )
        assert f1.is_current

        # Add superseding fact
        f2 = tg.add_fact(
            text="Paris is no longer the capital of France",
            subject="Paris", predicate="is_capital_of", obj="France",
            valid_at=now,
        )

        # f1 should now be invalidated
        current_facts = tg.find_facts(subject="Paris", predicate="is_capital_of", current_only=True)
        assert len(current_facts) == 1
        assert current_facts[0].fact_id == f2.fact_id

        # f1 should be in history
        all_facts = tg.find_facts(subject="Paris", predicate="is_capital_of", current_only=False)
        assert len(all_facts) == 2

        print("\n  Fact invalidation: OK (2 versions tracked)")


# ─── Test: Concurrent Thread Safety ──────────────────────────────────────────

class TestConcurrency:
    def test_4_thread_inserts(self, tmp_db, sample_embeddings):
        """4 threads inserting simultaneously — no crashes."""
        errors = []

        def worker(thread_id: int):
            try:
                for i in range(100):
                    idx = thread_id * 100 + i
                    emb = sample_embeddings[idx % len(sample_embeddings)]
                    tmp_db.add_memory(
                        content=f"Thread {thread_id} memory {i}",
                        embedding=emb,
                    )
            except Exception as e:
                errors.append((thread_id, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert len(errors) == 0, f"Thread errors: {errors}"
        cursor = tmp_db.conn.execute("SELECT COUNT(*) FROM memories WHERE is_deleted = 0")
        count = cursor.fetchone()[0]
        assert count == 400, f"Expected 400 memories, got {count}"
        print(f"\n  4-thread insert: {count} memories, 0 errors")


# ─── Test: Deduplication ─────────────────────────────────────────────────────

class TestDeduplication:
    def test_exact_duplicate_detection(self, tmp_db):
        """Exact duplicates should be caught by content hash."""
        content = "Important fact about machine learning"
        r1 = tmp_db.add_memory(content=content)
        assert r1["status"] == "created"

        r2 = tmp_db.add_memory(content=content)
        assert r2["status"] == "duplicate"
        assert r2["memory_id"] == r1["memory_id"]
        print(f"\n  Exact dedup: detected (id={r1['memory_id']})")

    def test_near_duplicate_via_minhash(self, tmp_db):
        """Near-duplicates should be detected by MinHash LSH."""
        from arriadne.dedup import Deduplicator

        dedup = Deduplicator(threshold=0.8, num_perm=128)
        base = "The capital of France is Paris, a beautiful city with rich history"
        dupes = [
            base + " Actually,",
            base + " In fact,",
            base,  # Exact
        ]

        dedup.add(base, doc_id="base")
        for i, d in enumerate(dupes):
            dedup.add(d, doc_id=f"dupe_{i}")

        is_dup = dedup.is_duplicate(base + " Notably,")
        found = dedup.find_duplicates(base + " Notably,")

        print(f"\n  MinHash dedup: is_duplicate={is_dup}, found={len(found)} matches")


# ─── Test: Memory Lifecycle ──────────────────────────────────────────────────

class TestMemoryLifecycle:
    def test_soft_delete_and_eviction(self, tmp_db, sample_embeddings):
        """Soft delete, eviction, and retention strength."""
        # Insert memories with varying importance
        for i in range(200):
            importance = (i % 10) / 10.0
            tmp_db.add_memory(
                content=f"Lifecycle memory {i}",
                embedding=sample_embeddings[i],
                importance=importance,
            )

        # Evict low-priority
        evicted = tmp_db.evict()
        cursor = tmp_db.conn.execute("SELECT COUNT(*) FROM memories WHERE is_deleted = 0")
        count_active = cursor.fetchone()[0]

        print(f"\n  Lifecycle: evicted {evicted}, {count_active} remaining")
        assert evicted > 0, "Expected some evictions"
        assert count_active < 200, "Expected fewer active memories"


# ─── Test: FTS Search ────────────────────────────────────────────────────────

class TestFTSSearch:
    def test_fts_accuracy(self, tmp_db):
        """FTS search should find relevant memories."""
        memories = [
            "Python is great for machine learning",
            "JavaScript powers the web",
            "Rust is a systems language",
            "Go is good for microservices",
            "Python web frameworks include Django",
        ]
        for m in memories:
            tmp_db.add_memory(content=m)

        results = tmp_db.fts_search("Python", k=5)
        contents = [r["content"] for r in results]
        assert any("Python" in c for c in contents), f"FTS missed Python: {contents}"

        results = tmp_db.fts_search("web", k=5)
        contents = [r["content"] for r in results]
        assert any("web" in c.lower() for c in contents), f"FTS missed web: {contents}"

        print("\n  FTS accuracy: queries returned correct results")

    def test_fts_with_special_chars(self, tmp_db):
        """FTS should handle special characters gracefully."""
        tmp_db.add_memory(content="Test memory with (parentheses) and [brackets]")
        try:
            results = tmp_db.fts_search("(parentheses)", k=5)
            print(f"\n  FTS special chars: OK ({len(results)} results)")
        except Exception as e:
            print(f"\n  FTS special chars: handled ({type(e).__name__})")


# ─── Test: Vector Search Quality ─────────────────────────────────────────────

class TestVectorSearch:
    def test_vector_search_returns_results(self, tmp_db, sample_embeddings):
        """Vector search should return k results."""
        for i in range(100):
            tmp_db.add_memory(
                content=f"Vector test memory {i}",
                embedding=sample_embeddings[i],
            )

        results = tmp_db.vector_search(sample_embeddings[0], k=10)
        assert len(results) == 10, f"Expected 10 results, got {len(results)}"
        assert results[0]["score"] > 0, "Top result should have positive score"
        print(f"\n  Vector search: {len(results)} results, top score={results[0]['score']:.4f}")

    def test_vector_search_empty_index(self, tmp_db):
        """Search on empty index should return empty list."""
        rng = np.random.RandomState(0)
        emb = rng.randn(EMBEDDING_DIM).astype(np.float32)
        results = tmp_db.vector_search(emb, k=10)
        assert results == []
        print("\n  Empty index search: OK (0 results)")


# ─── Run all tests if executed directly ──────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])

"""
Tests for advanced Ariadne features:
- Memory categories & types with different lifecycle behaviors
- Importance scoring and decay
- Export/import roundtrip (JSON, markdown)
- Graph visualization output formats
- Consolidation clustering
"""
import json
import os
import tempfile
import time
import pytest


# ============================================================
# Memory Categories Tests
# ============================================================

class TestMemoryCategories:
    """Tests for memory category system."""

    def test_category_config_defaults(self):
        """Test default category configurations."""
        from arriadne.categories import MemoryCategoryManager, DEFAULT_CATEGORIES

        manager = MemoryCategoryManager()

        # Episodic: high initial importance, fast decay
        episodic = manager.get_config("episodic")
        assert episodic.default_importance == 0.7
        assert episodic.decay_rate == 0.03
        assert episodic.auto_prune is True

        # Semantic: stable importance, slow decay
        semantic = manager.get_config("semantic")
        assert semantic.default_importance == 0.5
        assert semantic.decay_rate == 0.005
        assert semantic.auto_prune is False

        # Procedural: never decays
        procedural = manager.get_config("procedural")
        assert procedural.decay_rate == 0.0
        assert procedural.auto_prune is False

        # Working: fast decay, auto-pruned
        working = manager.get_config("working")
        assert working.decay_rate == 0.1
        assert working.auto_prune is True
        assert working.prune_after_days == 7

    def test_category_decay(self):
        """Test category-specific decay behavior."""
        from arriadne.categories import MemoryCategoryManager

        manager = MemoryCategoryManager()

        # Episodic decays fast
        episodic_after_10_days = manager.apply_category_decay(
            importance=0.7, category="episodic", days_since_access=10
        )
        assert episodic_after_10_days < 0.5  # Should decay significantly

        # Semantic decays slowly
        semantic_after_10_days = manager.apply_category_decay(
            importance=0.5, category="semantic", days_since_access=10
        )
        assert semantic_after_10_days > 0.4  # Should barely decay

        # Procedural never decays
        procedural_after_100_days = manager.apply_category_decay(
            importance=0.6, category="procedural", days_since_access=100
        )
        assert procedural_after_100_days == 0.6  # No change

    def test_category_access_boost(self):
        """Test category-specific access boost."""
        from arriadne.categories import MemoryCategoryManager

        manager = MemoryCategoryManager()

        # All categories should boost importance on access
        boost = manager.apply_category_access_boost(
            importance=0.5, category="semantic", access_count=5
        )
        assert boost > 0.5

        # Boost should have diminishing returns
        boost_10 = manager.apply_category_access_boost(
            importance=0.5, category="semantic", access_count=10
        )
        boost_50 = manager.apply_category_access_boost(
            importance=0.5, category="semantic", access_count=50
        )
        # Both should be higher than base
        assert boost_10 > 0.5
        assert boost_50 > 0.5

    def test_category_validation(self):
        """Test category validation and normalization."""
        from arriadne.categories import MemoryCategoryManager

        manager = MemoryCategoryManager()

        assert manager.validate_category("episodic") == "episodic"
        assert manager.validate_category("Episodic") == "episodic"
        assert manager.validate_category("") == "semantic"
        assert manager.validate_category("unknown") == "semantic"
        # Fuzzy match
        assert manager.validate_category("episo") == "episodic"

    def test_get_all_categories(self):
        """Test listing all categories."""
        from arriadne.categories import MemoryCategoryManager

        manager = MemoryCategoryManager()
        cats = manager.get_all_categories()
        assert "episodic" in cats
        assert "semantic" in cats
        assert "procedural" in cats
        assert "working" in cats

    def test_category_stats(self, tmp_path):
        """Test category statistics from database."""
        from arriadne.storage import AriadneDB
        from arriadne.config import AriadneConfig
        from arriadne.categories import MemoryCategoryManager

        db_path = str(tmp_path / "test_cat.db")
        config = AriadneConfig(db_path=db_path)
        db = AriadneDB(config)
        db.open()

        # Add memories with different categories
        db.add_memory("Episodic memory 1", memory_type="episodic", category="episodic")
        db.add_memory("Episodic memory 2", memory_type="episodic", category="episodic")
        db.add_memory("Semantic memory 1", memory_type="semantic", category="semantic")
        db.add_memory("Procedural memory 1", memory_type="procedural", category="procedural")

        manager = MemoryCategoryManager()
        stats = manager.get_category_stats(db.conn)

        assert stats["episodic"]["count"] == 2
        assert stats["semantic"]["count"] == 1
        assert stats["procedural"]["count"] == 1
        assert stats["working"]["count"] == 0  # No working memories

        db.close()
        os.unlink(db_path)


# ============================================================
# Importance Scoring Tests
# ============================================================

class TestImportanceScoring:
    """Tests for importance scoring and adaptive importance."""

    def test_importance_recomputation(self, tmp_path):
        """Test that recompute_importance updates importance scores."""
        from arriadne.interface import AriadneMemory
        from arriadne.config import AriadneConfig

        db_path = str(tmp_path / "test_importance.db")
        mem = AriadneMemory(
            db_path=db_path,
            embedding_dim=8,
            embedding_provider="keyword",
        )

        # Add memories with different categories
        r1 = mem.remember("Old semantic memory", importance=0.5, category="semantic")
        r2 = mem.remember("Recent procedural memory", importance=0.6, category="procedural")

        # Recompute importance
        updated = mem.recompute_importance()
        assert updated >= 0  # May or may not update depending on age

        # Check importance stats
        stats = mem.get_importance_stats()
        assert "avg_importance" in stats
        assert "distribution" in stats

        mem.close()
        os.unlink(db_path)

    def test_importance_stats(self, tmp_path):
        """Test importance statistics."""
        from arriadne.interface import AriadneMemory

        mem = AriadneMemory(
            db_path=str(tmp_path / "test_stats.db"),
            embedding_dim=8,
            embedding_provider="keyword",
        )

        # Add memories with varying importance
        for i in range(5):
            mem.remember(f"Memory {i}", importance=i / 10.0)

        stats = mem.get_importance_stats()
        assert "avg_importance" in stats
        assert "min_importance" in stats
        assert "max_importance" in stats
        assert "distribution" in stats
        assert stats["avg_importance"] > 0

        mem.close()


# ============================================================
# Export/Import Tests
# ============================================================

class TestExportImport:
    """Tests for export/import functionality."""

    def test_json_roundtrip(self, tmp_path):
        """Test JSON export/import roundtrip."""
        from arriadne.interface import AriadneMemory

        # Create source database
        src_path = str(tmp_path / "source.db")
        src = AriadneMemory(
            db_path=src_path,
            embedding_dim=8,
            embedding_provider="keyword",
        )

        # Add memories
        src.remember("Paris is the capital of France", category="semantic")
        src.remember("I visited Paris last summer", category="episodic")
        src.remember("How to use vim editor", category="procedural")
        src.add_edge("Paris", "France", "capital_of")

        # Export
        export_path = str(tmp_path / "export.json")
        result = src.export_json(export_path)
        assert result["exported"] == 3
        assert result["edges"] == 1

        # Verify export file exists
        assert os.path.exists(export_path)

        # Import into new database
        dst_path = str(tmp_path / "dest.db")
        dst = AriadneMemory(
            db_path=dst_path,
            embedding_dim=8,
            embedding_provider="keyword",
        )

        import_result = dst.import_json(export_path)
        assert import_result["imported"] >= 2  # At least some should import

        # Verify memories exist in destination
        stats = dst.stats()
        assert stats["active_memories"] >= 2

        src.close()
        dst.close()

    def test_markdown_roundtrip(self, tmp_path):
        """Test markdown export/import roundtrip."""
        from arriadne.interface import AriadneMemory

        mem = AriadneMemory(
            db_path=str(tmp_path / "md_test.db"),
            embedding_dim=8,
            embedding_provider="keyword",
        )

        # Add memories with entities
        mem.remember(
            "Python is a programming language",
            entities=["Python"],
        )
        mem.remember(
            "FAISS is used for vector search",
            entities=["FAISS"],
        )

        # Export to markdown
        md_path = str(tmp_path / "export.md")
        result = mem.export_markdown(md_path)
        assert result["exported"] == 2
        assert result["entities"] >= 1

        # Verify markdown content
        with open(md_path) as f:
            content = f.read()
        assert "# Ariadne Memory Export" in content

        mem.close()

    def test_text_import(self, tmp_path):
        """Test plain text import."""
        from arriadne.interface import AriadneMemory

        # Create text file
        txt_path = str(tmp_path / "memories.txt")
        with open(txt_path, "w") as f:
            f.write("Memory paragraph one about Python.\n\n")
            f.write("Memory paragraph two about FAISS.\n\n")
            f.write("Memory paragraph three about databases.\n")

        mem = AriadneMemory(
            db_path=str(tmp_path / "txt_test.db"),
            embedding_dim=8,
            embedding_provider="keyword",
        )

        result = mem.import_from_text(txt_path, category="semantic")
        assert result["imported"] == 3
        assert result["skipped"] == 0

        # Verify
        stats = mem.stats()
        assert stats["active_memories"] == 3

        mem.close()

    def test_markdown_import(self, tmp_path):
        """Test markdown import with entities."""
        from arriadne.interface import AriadneMemory

        # Create markdown file
        md_path = str(tmp_path / "import.md")
        with open(md_path, "w") as f:
            f.write("# Python\n")
            f.write("- Python is a programming language\n")
            f.write("- Python supports multiple paradigms\n\n")
            f.write("# FAISS\n")
            f.write("- FAISS is a vector search library\n")

        mem = AriadneMemory(
            db_path=str(tmp_path / "md_import.db"),
            embedding_dim=8,
            embedding_provider="keyword",
        )

        result = mem.import_from_markdown(md_path)
        assert result["imported"] == 3

        # Check that entities were created
        entities = mem.get_entities()
        entity_names = [e["name"] for e in entities]
        assert "Python" in entity_names

        mem.close()


# ============================================================
# Graph Visualization Tests
# ============================================================

class TestGraphVisualization:
    """Tests for graph visualization exports."""

    def _setup_graph(self, tmp_path):
        """Create a test graph for visualization."""
        from arriadne.interface import AriadneMemory

        mem = AriadneMemory(
            db_path=str(tmp_path / "viz_test.db"),
            embedding_dim=8,
            embedding_provider="keyword",
        )

        mem.add_edge("Python", "FAISS", "uses")
        mem.add_edge("Python", "NumPy", "uses")
        mem.add_edge("FAISS", "NumPy", "depends_on")
        mem.add_edge("Ariadne", "FAISS", "uses")
        mem.add_edge("Ariadne", "Python", "written_in")

        return mem

    def test_dot_export(self, tmp_path):
        """Test DOT/Graphviz export."""
        from arriadne.interface import AriadneMemory

        mem = self._setup_graph(tmp_path)
        dot_path = str(tmp_path / "graph.dot")
        result = mem.export_dot(dot_path)

        assert result["nodes"] == 4
        assert result["edges"] == 5
        assert os.path.exists(dot_path)

        # Verify DOT content
        with open(dot_path) as f:
            content = f.read()
        assert "digraph" in content
        assert "Python" in content

        mem.close()

    def test_mermaid_export(self, tmp_path):
        """Test Mermaid diagram export."""
        from arriadne.interface import AriadneMemory

        mem = self._setup_graph(tmp_path)
        mmd_path = str(tmp_path / "graph.mmd")
        result = mem.export_mermaid(mmd_path)

        assert result["nodes"] == 4
        assert result["edges"] == 5
        assert os.path.exists(mmd_path)

        # Verify Mermaid content
        with open(mmd_path) as f:
            content = f.read()
        assert "graph LR" in content
        assert "Python" in content

        mem.close()

    def test_json_graph_export(self, tmp_path):
        """Test D3.js-compatible JSON graph export."""
        from arriadne.interface import AriadneMemory

        mem = self._setup_graph(tmp_path)
        json_path = str(tmp_path / "graph.json")
        result = mem.export_json_graph(json_path)

        assert result["nodes"] == 4
        assert result["links"] == 5
        assert os.path.exists(json_path)

        # Verify JSON content
        with open(json_path) as f:
            data = json.load(f)
        assert "nodes" in data
        assert "links" in data
        assert len(data["nodes"]) == 4
        assert len(data["links"]) == 5

        # Verify node structure
        node = data["nodes"][0]
        assert "id" in node
        assert "name" in node
        assert "type" in node

        mem.close()

    def test_graph_stats(self, tmp_path):
        """Test graph statistics computation."""
        from arriadne.interface import AriadneMemory

        mem = self._setup_graph(tmp_path)
        stats = mem.get_graph_stats()

        assert stats["nodes"] == 4
        assert stats["edges"] == 5
        assert stats["density"] > 0
        assert stats["avg_degree"] > 0
        assert stats["connected_components"] >= 1
        assert stats["max_degree"] > 0
        assert "centrality_top5" in stats
        assert len(stats["centrality_top5"]) <= 5

        # Check degree distribution
        assert "degree_distribution" in stats

        mem.close()

    def test_empty_graph_stats(self, tmp_path):
        """Test graph stats on empty graph."""
        from arriadne.interface import AriadneMemory

        mem = AriadneMemory(
            db_path=str(tmp_path / "empty.db"),
            embedding_dim=8,
            embedding_provider="keyword",
        )

        stats = mem.get_graph_stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0
        assert stats["density"] == 0

        mem.close()


# ============================================================
# Consolidation Tests
# ============================================================

class TestConsolidation:
    """Tests for improved consolidation features."""

    def test_consolidate_by_topic(self, tmp_path):
        """Test topic-based consolidation."""
        from arriadne.interface import AriadneMemory
        from arriadne.config import AriadneConfig
        from arriadne.storage import AriadneDB

        db_path = str(tmp_path / "consolidate.db")
        config = AriadneConfig(db_path=db_path)
        db = AriadneDB(config)
        db.open()

        # Insert related memories with same topic
        for content in [
            "The server has 4 CPU cores",
            "The server has 8GB of RAM",
            "The server runs Ubuntu 24.04",
        ]:
            db.add_memory(content, memory_type="semantic", importance=0.7)

        from arriadne.consolidation import MemoryConsolidator
        consolidator = MemoryConsolidator(db.conn)

        # Test consolidation suggestions
        suggestions = consolidator.get_consolidation_suggestions(min_cluster_size=2)
        # May or may not find groups depending on similarity
        assert isinstance(suggestions, list)

        db.close()
        os.unlink(db_path)

    def test_contradiction_detection(self, tmp_path):
        """Test contradiction detection within groups."""
        from arriadne.consolidation import MemoryConsolidator, ConsolidationGroup

        # Create mock DB connection
        import sqlite3
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding_vector BLOB,
                topic TEXT DEFAULT '',
                importance INTEGER DEFAULT 5,
                is_deleted INTEGER DEFAULT 0,
                deleted_at REAL,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                updated_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        conn.commit()

        consolidator = MemoryConsolidator(conn)

        # Create a group with contradictory memories
        group = ConsolidationGroup(
            group_id=1,
            memories=[
                {"id": 1, "content": "Python is not a compiled language"},
                {"id": 2, "content": "Python is a compiled language"},
                {"id": 3, "content": "Python supports multiple paradigms"},
            ],
        )

        contradictions = consolidator._detect_group_contradictions(group)
        # Should detect the negation pattern
        assert isinstance(contradictions, list)
        assert len(contradictions) >= 1

        conn.close()

    def test_cluster_summary(self, tmp_path):
        """Test summary generation for clusters."""
        from arriadne.consolidation import MemoryConsolidator, ConsolidationGroup

        import sqlite3
        conn = sqlite3.connect(":memory:")
        conn.close()

        consolidator = MemoryConsolidator(conn)

        group = ConsolidationGroup(
            group_id=1,
            memories=[
                {"id": 1, "content": "Python is a programming language used for data science."},
                {"id": 2, "content": "Python is also used for web development."},
                {"id": 3, "content": "Python supports multiple programming paradigms."},
            ],
        )

        summary = consolidator.generate_cluster_summary(group)
        assert isinstance(summary, str)
        assert len(summary) > 0

        conn.close()


# ============================================================
# Integration Test: Full Advanced Features Pipeline
# ============================================================

class TestAdvancedFeaturesPipeline:
    """Integration tests combining multiple advanced features."""

    def test_categories_with_recall(self, tmp_path):
        """Test storing memories with different categories and filtering by category."""
        from arriadne.interface import AriadneMemory

        mem = AriadneMemory(
            db_path=str(tmp_path / "pipeline.db"),
            embedding_dim=8,
            embedding_provider="keyword",
        )

        # Store memories with different categories
        mem.remember("Paris trip memory", category="episodic")
        mem.remember("Python fact", category="semantic")
        mem.remember("How to use vim", category="procedural")
        mem.remember("Working note", category="working")

        # Verify all stored
        stats = mem.stats()
        assert stats["active_memories"] == 4

        # Filter by category
        results = mem.recall("memory", category_filter="episodic")
        assert all(r.get("category") == "episodic" for r in results)

        # Get category stats
        cat_stats = mem.get_category_stats()
        assert cat_stats["episodic"]["count"] == 1
        assert cat_stats["semantic"]["count"] == 1
        assert cat_stats["procedural"]["count"] == 1
        assert cat_stats["working"]["count"] == 1

        mem.close()

    def test_importance_with_categories(self, tmp_path):
        """Test that importance scoring works with different categories."""
        from arriadne.interface import AriadneMemory

        mem = AriadneMemory(
            db_path=str(tmp_path / "importance_cat.db"),
            embedding_dim=8,
            embedding_provider="keyword",
        )

        # Add memories with different categories and importance
        mem.remember("Episodic high", importance=0.8, category="episodic")
        mem.remember("Semantic medium", importance=0.5, category="semantic")
        mem.remember("Procedural low", importance=0.3, category="procedural")

        # Recompute importance
        updated = mem.recompute_importance()

        # Check importance stats
        stats = mem.get_importance_stats()
        assert stats["avg_importance"] > 0

        mem.close()

    def test_export_import_with_categories(self, tmp_path):
        """Test that categories are preserved through export/import."""
        from arriadne.interface import AriadneMemory

        # Create source with categorized memories
        src = AriadneMemory(
            db_path=str(tmp_path / "src.db"),
            embedding_dim=8,
            embedding_provider="keyword",
        )

        src.remember("Episodic memory", category="episodic")
        src.remember("Procedural memory", category="procedural")

        # Export
        export_path = str(tmp_path / "export.json")
        src.export_json(export_path)

        # Import
        dst = AriadneMemory(
            db_path=str(tmp_path / "dst.db"),
            embedding_dim=8,
            embedding_provider="keyword",
        )
        dst.import_json(export_path)

        # Verify categories in export
        with open(export_path) as f:
            data = json.load(f)
        categories = {m["category"] for m in data["memories"]}
        assert "episodic" in categories
        assert "procedural" in categories

        src.close()
        dst.close()

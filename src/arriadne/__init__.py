"""
Ariadne — Fast local memory for AI agents.

Zero-config memory system with FAISS vector search, FTS5 keyword search,
hybrid retrieval (RRF), knowledge graph, entity resolution, LLM-powered
extraction, temporal awareness, and conversation memory.

Usage:
    from arriadne import AriadneMemory

    mem = AriadneMemory("my_memory.db")
    mem.remember("Paris is the capital of France")
    results = mem.recall("capital of France")

    # With LLM-powered extraction:
    from arriadne import LLMProvider
    mem = AriadneMemory("agent.db", llm_config={"provider": "openai", "model": "gpt-4o-mini"})
    extracted = mem.extract_from_conversation([
        {"role": "user", "content": "I love Paris"},
        {"role": "assistant", "content": "Paris is beautiful!"},
    ], auto_store=True)

    # REST API:
    from arriadne.server import create_app
    import uvicorn
    app = create_app(db_path="memory.db")
    uvicorn.run(app, port=8899)
"""

__version__ = "0.1.4"

from arriadne.interface import AriadneMemory
from arriadne.config import AriadneConfig
from arriadne.storage import AriadneDB
from arriadne.embeddings import EmbeddingProvider, auto_detect_provider
from arriadne.dedup import Deduplicator, ContradictionDetector
from arriadne.conversation import ConversationTracker, AgentTools

# New modules (optional imports — may fail if deps missing)
try:
    from arriadne.llm import LLMProvider, LLMMessage, LLMResponse
except ImportError:
    LLMProvider = None

try:
    from arriadne.extraction import MemoryExtractor, ExtractedMemory
except ImportError:
    MemoryExtractor = None

try:
    from arriadne.entity_resolution import EntityResolver, EntityExtractor, Entity
except ImportError:
    EntityResolver = None

try:
    from arriadne.temporal import TemporalGraph, TemporalFact
except ImportError:
    TemporalGraph = None

try:
    from arriadne.consolidation import MemoryConsolidator
except ImportError:
    MemoryConsolidator = None

try:
    from arriadne.lifecycle import MemoryLifecycle
except ImportError:
    MemoryLifecycle = None

try:
    from arriadne.server import create_app
except ImportError:
    create_app = None

__all__ = [
    "AriadneMemory",
    "AriadneConfig",
    "AriadneDB",
    "EmbeddingProvider",
    "auto_detect_provider",
    "Deduplicator",
    "ContradictionDetector",
    "ConversationTracker",
    "AgentTools",
    "LLMProvider",
    "LLMMessage",
    "LLMResponse",
    "MemoryExtractor",
    "ExtractedMemory",
    "EntityResolver",
    "EntityExtractor",
    "Entity",
    "TemporalGraph",
    "TemporalFact",
    "MemoryConsolidator",
    "MemoryLifecycle",
    "create_app",
]

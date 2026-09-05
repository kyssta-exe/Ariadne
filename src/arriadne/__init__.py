"""Ariadne - Production-ready memory system with vector search and graph traversal."""

from __future__ import annotations

__version__ = "0.13.0"
__author__ = "Mantes"

from arriadne.addons import (
    AddonRegistry,
    APIRoute,
    BaseAddon,
    CLICommand,
    EntityType,
    ExtractionError,
    ExtractorBase,
    GraphRelationship,
    SearchFilter,
)
from arriadne.config import AriadneConfig
from arriadne.curator import CurateReport, MemoryCurator
from arriadne.dedup import ContradictionDetector, Deduplicator
from arriadne.embeddings import Embedder, SentenceTransformerEmbedder
from arriadne.interface import AriadneMemory
from arriadne.memory_manager import (
    ExtractedMemory,
    ExtractedRelation,
    ExtractionResult,
    LLMMemoryManager,
)
from arriadne.storage import AriadneDB

__all__ = [
    "APIRoute",
    "AddonRegistry",
    "AriadneConfig",
    "AriadneDB",
    "AriadneMemory",
    "BaseAddon",
    "CLICommand",
    "ContradictionDetector",
    "CurateReport",
    "Deduplicator",
    "Embedder",
    "EntityType",
    "ExtractedMemory",
    "ExtractedRelation",
    "ExtractionError",
    "ExtractionResult",
    "ExtractorBase",
    "GraphRelationship",
    "LLMMemoryManager",
    "MemoryCurator",
    "SearchFilter",
    "SentenceTransformerEmbedder",
]

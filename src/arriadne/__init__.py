"""Ariadne - Production-ready memory system with vector search and graph traversal."""

from __future__ import annotations

__version__ = "0.12.0"
__author__ = "Mantes"

from arriadne.config import AriadneConfig
from arriadne.interface import AriadneMemory
from arriadne.storage import AriadneDB
from arriadne.dedup import Deduplicator, ContradictionDetector
from arriadne.embeddings import Embedder, SentenceTransformerEmbedder
from arriadne.addons import BaseAddon, ExtractorBase, EntityType
from arriadne.addons import (
    AddonRegistry,
    ExtractionError,
    CLICommand,
    APIRoute,
    SearchFilter,
    GraphRelationship,
)

__all__ = [
    "AriadneConfig",
    "AriadneDB",
    "AriadneMemory",
    "Deduplicator",
    "ContradictionDetector",
    "Embedder",
    "SentenceTransformerEmbedder",
    "BaseAddon",
    "ExtractorBase",
    "EntityType",
    "AddonRegistry",
    "ExtractionError",
    "CLICommand",
    "APIRoute",
    "SearchFilter",
    "GraphRelationship",
]

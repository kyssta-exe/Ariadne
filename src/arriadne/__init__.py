"""Ariadne — Fast, local memory system for AI agents.

Zero-config. Auto-embeds. No cloud, no daemon, no API keys.
"""

from __future__ import annotations

__version__ = "0.2.0"
__author__ = "Mantes"

from arriadne.config import AriadneConfig
from arriadne.interface import AriadneMemory
from arriadne.storage import AriadneDB
from arriadne.dedup import Deduplicator, ContradictionDetector
from arriadne.embeddings import (
    EmbeddingProvider,
    OnnxEmbedding,
    KeywordEmbedding,
    auto_detect_provider,
)
from arriadne.conversation import AgentTools, ConversationTracker

__all__ = [
    "AriadneConfig",
    "AriadneDB",
    "AriadneMemory",
    "Deduplicator",
    "ContradictionDetector",
    "EmbeddingProvider",
    "OnnxEmbedding",
    "KeywordEmbedding",
    "auto_detect_provider",
    "AgentTools",
    "ConversationTracker",
]

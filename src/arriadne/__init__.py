"""Ariadne - Production-ready memory system with vector search and graph traversal."""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Mantes"

from arriadne.config import AriadneConfig
from arriadne.interface import AriadneMemory
from arriadne.storage import AriadneDB
from arriadne.dedup import Deduplicator, ContradictionDetector

__all__ = [
    "AriadneConfig",
    "AriadneDB",
    "AriadneMemory",
    "Deduplicator",
    "ContradictionDetector",
]

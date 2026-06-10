"""Ariadne Finance — finance research add-on for Ariadne memory system."""

from __future__ import annotations

__version__ = "0.10.0"

# Lazy import to avoid circular import during package initialization
def __getattr__(name: str):
    if name == "Addon":
        from arriadne_finance.addon import Addon
        return Addon
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Addon"]

"""Ariadne Memory Provider for Hermes — v2 (0.3.0)

Drop-in replacement for Mnemosyne. Implements the MemoryProvider interface
with Ariadne (FAISS + FTS5 + Knowledge Graph + LLM Extraction + Lifecycle) as the backend.

New in v2:
    - LLM-powered auto-extraction from every conversation turn
    - Entity resolution (link related memories via entities)
    - Temporal graph (track when facts become valid/invalid)
    - Three-tier lifecycle (hot/warm/cold with Ebbinghaus retention)
    - Consolidation (merge similar memories)
    - Regex fallback extraction when no LLM is available
    - Graceful degradation if optional deps missing

Installation:
    1. Place this directory in ~/.hermes/plugins/ariadne/
    2. Run: hermes memory setup  (select 'ariadne')
    3. Or set in ~/.hermes/config.yaml:
           memory:
             provider: ariadne
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# Hermes MemoryProvider ABC
import sys
_HERMES_AGENT = "/usr/local/lib/hermes-agent"
if _HERMES_AGENT not in sys.path:
    sys.path.insert(0, _HERMES_AGENT)
from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

# ── Throttle & limits ─────────────────────────────────────────────
_SYNC_THROTTLE_SECONDS = 5.0
_MAX_PREFETCH_RESULTS = 10
_CONSOLIDATION_INTERVAL = 600          # 10 min between auto-consolidation
_LIFECYCLE_INTERVAL = 3600             # 1 hr between lifecycle runs
_MAX_EXTRACT_PER_TURN = 10            # max facts extracted per turn


def register(ctx):
    """Register the Ariadne memory provider with Hermes."""
    provider = AriadneMemoryProvider()
    ctx.register_memory_provider(provider)


# ── Regex fallback extractor ──────────────────────────────────────
# When no LLM is available, use simple pattern-based extraction.
# Catches: "X is Y", "X has Y", "X uses Y", "X prefers Y", etc.
_PATTERNS = [
    re.compile(r"(?:^|\.)\s*(?:I|We)\s+(?:am|are|use|prefer|want|need|like|hate|love)\s+(.{10,120})\.", re.I),
    re.compile(r"(?:^|\.)\s*(\w[\w\s]{3,40})\s+(?:is|are|was|were|has|have|runs|uses|serves|hosts|contains)\s+(.{10,120})\.", re.I),
]


def _regex_extract(text: str) -> List[Dict[str, Any]]:
    """Extract facts from text using regex patterns (no LLM)."""
    facts = []
    for pat in _PATTERNS:
        for m in pat.finditer(text):
            # Collect all groups as a single fact
            parts = [g.strip() for g in m.groups() if g]
            if len(parts) < 2:
                continue
            fact_text = f"{parts[0]} is/are {parts[-1]}" if len(parts) == 2 else " ".join(parts)
            fact_text = fact_text[:200]
            if len(fact_text) < 15:
                continue
            facts.append({
                "text": fact_text,
                "attributed_to": "user",
                "topic": "general",
                "importance": 5,
                "entities": [],
            })
    return facts[:_MAX_EXTRACT_PER_TURN]


class AriadneMemoryProvider(MemoryProvider):
    """Ariadne-based memory provider for Hermes — v2 with LLM extraction and lifecycle."""

    def __init__(self):
        super().__init__()
        self._ariadne = None
        self._shared = None
        self._session_id = ""
        self._last_sync = 0.0
        self._last_consolidation = 0.0
        self._last_lifecycle = 0.0
        self._scratchpad: Dict[str, str] = {}
        self._db_path = ""
        self._shared_db_path = ""
        self._hermes_home = ""
        self._prefetch_cache: List[Dict] = []
        self._prefetch_timestamp = 0.0
        self._turn_messages: List[Dict[str, str]] = []  # buffered turns for extraction
        self._extraction_enabled = True
        self._agent_context = "primary"

    # ── MemoryProvider interface ──────────────────────────────────────

    @property
    def name(self) -> str:
        return "ariadne"

    def is_available(self) -> bool:
        """Check if Ariadne is installed and configured."""
        try:
            from arriadne import AriadneMemory
            return True
        except ImportError:
            return False

    def initialize(self, session_id: str, **kwargs) -> None:
        """Open/initialize the Ariadne database."""
        hermes_home = kwargs.get("hermes_home", os.path.expanduser("~/.hermes"))
        self._hermes_home = hermes_home
        self._session_id = session_id
        self._agent_context = kwargs.get("agent_context", "primary")

        # Primary DB
        db_dir = os.path.join(hermes_home, "ariadne")
        os.makedirs(db_dir, exist_ok=True)
        self._db_path = os.path.join(db_dir, "memory.db")

        # Shared surface DB
        shared_dir = os.path.join(db_dir, "shared")
        os.makedirs(shared_dir, exist_ok=True)
        self._shared_db_path = os.path.join(shared_dir, "memory.db")

        try:
            from arriadne import AriadneMemory, AriadneConfig

            # Auto-detect LLM provider from environment
            llm_config = self._detect_llm_config()

            config = AriadneConfig(
                db_path=self._db_path,
                embedding_dim=384,
                faiss_type="auto",
                dedup_threshold=0.5,
                dedup_num_perm=128,
            )
            self._ariadne = AriadneMemory(config=config, llm_config=llm_config)

            shared_config = AriadneConfig(
                db_path=self._shared_db_path,
                embedding_dim=384,
                faiss_type="auto",
                dedup_threshold=0.5,
            )
            self._shared = AriadneMemory(config=shared_config)

            # Run initial lifecycle to set tiers
            try:
                self._ariadne.run_lifecycle()
            except Exception:
                pass

            stats = self._ariadne.stats()
            logger.info(
                "Ariadne v2 initialized: %s (%d active memories, embedding=%s, llm=%s)",
                self._db_path,
                stats.get("active_memories", 0),
                stats.get("embedding_provider", "?"),
                "available" if self._ariadne._llm_provider else "none",
            )

        except Exception as e:
            logger.error("Ariadne init failed: %s", e)
            self._ariadne = None
            self._shared = None

    def _detect_llm_config(self) -> Optional[Dict[str, Any]]:
        """Auto-detect LLM configuration from environment variables."""
        # Check for common LLM env vars
        if os.environ.get("OPENAI_API_KEY"):
            return {"provider": "openai", "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini")}
        if os.environ.get("ANTHROPIC_API_KEY"):
            return {"provider": "anthropic", "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")}
        if os.environ.get("OLLAMA_BASE_URL") or os.environ.get("OLLAMA_HOST"):
            return {"provider": "ollama", "model": os.environ.get("OLLAMA_MODEL", "llama3")}
        return None

    def system_prompt_block(self) -> str:
        """Return static system prompt block for Ariadne."""
        if not self._ariadne:
            return ""
        stats = self._ariadne.stats()
        active = stats.get("active_memories", 0)
        if active == 0:
            return ""
        return (
            f"\n[ARIADNE MEMORY: {active} memories stored. "
            f"Embedding: {stats.get('embedding_provider', '?')}. "
            f"LLM extraction: {'on' if self._ariadne._llm_provider else 'regex fallback'}]\n"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall relevant memories before each turn."""
        if not self._ariadne or not query.strip():
            return ""

        # Use cached results if recent enough (< 2s)
        if self._prefetch_cache and (time.time() - self._prefetch_timestamp) < 2.0:
            return self._format_recall_block(self._prefetch_cache)

        try:
            results = self._ariadne.recall(query, k=_MAX_PREFETCH_RESULTS)
            if results:
                self._prefetch_cache = results
                self._prefetch_timestamp = time.time()

                # Record access for lifecycle promotion
                for r in results:
                    try:
                        self._ariadne._get_lifecycle().record_access(str(r["id"]))
                    except Exception:
                        pass

                return self._format_recall_block(results)
        except Exception as e:
            logger.debug("Ariadne prefetch failed: %s", e)

        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Pre-warm the prefetch cache for next turn."""
        if not self._ariadne or not query.strip():
            return
        try:
            self._prefetch_cache = self._ariadne.recall(query, k=_MAX_PREFETCH_RESULTS)
            self._prefetch_timestamp = time.time()
        except Exception:
            pass

    def sync_turn(self, user_content: str, assistant_content: str, *,
                  session_id: str = "", messages: Optional[List[Dict]] = None) -> None:
        """Persist completed turn to memory with LLM-powered extraction."""
        if not self._ariadne:
            return

        # Throttle to avoid excessive writes
        now = time.time()
        if now - self._last_sync < _SYNC_THROTTLE_SECONDS:
            return
        self._last_sync = now

        # Skip extraction for non-primary contexts (cron, flush, etc.)
        if self._agent_context != "primary":
            return

        # --- Phase 1: Auto-extract facts from conversation ---
        extraction_results = self._extract_from_turn(user_content, assistant_content)

        # --- Phase 2: Store user message as episodic memory (compact) ---
        if user_content.strip():
            try:
                self._ariadne.remember(
                    content=f"[USER] {user_content[:500]}",
                    memory_type="episodic",
                    importance=0.3,
                    metadata={"source": "sync_turn", "session_id": self._session_id},
                )
            except Exception:
                pass

        # --- Phase 3: Store assistant response summary ---
        if assistant_content.strip():
            try:
                self._ariadne.remember(
                    content=f"[ASSISTANT] {assistant_content[:500]}",
                    memory_type="episodic",
                    importance=0.2,
                    metadata={"source": "sync_turn", "session_id": self._session_id},
                )
            except Exception:
                pass

        # --- Phase 4: Periodic consolidation ---
        if now - self._last_consolidation > _CONSOLIDATION_INTERVAL:
            self._last_consolidation = now
            self._maybe_consolidate()

        # --- Phase 5: Periodic lifecycle ---
        if now - self._last_lifecycle > _LIFECYCLE_INTERVAL:
            self._last_lifecycle = now
            self._maybe_run_lifecycle()

    def _extract_from_turn(self, user_content: str, assistant_content: str) -> List[Dict]:
        """Extract facts from a conversation turn."""
        if not self._ariadne:
            return []

        results = []

        # Try LLM-powered extraction first
        if self._ariadne._llm_provider:
            try:
                messages = []
                if user_content.strip():
                    messages.append({"role": "user", "content": user_content})
                if assistant_content.strip():
                    messages.append({"role": "assistant", "content": assistant_content})

                if messages:
                    extracted = self._ariadne.extract_from_conversation(
                        messages, auto_store=True,
                        observation_date=datetime.now(timezone.utc).isoformat(),
                    )
                    results = [{"text": e.text, "topic": e.topic} for e in extracted]
                    if results:
                        logger.info("Extracted %d facts via LLM from turn", len(results))
                        return results
            except Exception as e:
                logger.debug("LLM extraction failed, falling back to regex: %s", e)

        # Fallback: regex extraction
        for content in [user_content, assistant_content]:
            if not content.strip():
                continue
            try:
                facts = _regex_extract(content)
                for fact in facts:
                    result = self._ariadne.remember(
                        content=fact["text"],
                        memory_type="semantic",
                        importance=fact["importance"] / 10.0,
                        metadata={
                            "source": "regex_extraction",
                            "attributed_to": fact.get("attributed_to", "user"),
                            "topic": fact.get("topic", "general"),
                            "session_id": self._session_id,
                        },
                    )
                    if result.get("status") == "created":
                        results.append({"text": fact["text"], "topic": fact.get("topic", "general")})
            except Exception:
                pass

        if results:
            logger.info("Extracted %d facts via regex from turn", len(results))

        return results

    def _maybe_consolidate(self):
        """Run consolidation if there are enough memories."""
        if not self._ariadne:
            return
        try:
            stats = self._ariadne.stats()
            if stats.get("active_memories", 0) < 10:
                return  # Not enough memories to consolidate

            result = self._ariadne.consolidate_with_llm(dry_run=False)
            if result.get("memories_before", 0) > result.get("memories_after", 0):
                logger.info(
                    "Consolidation: %d -> %d memories (%.0f%% compression)",
                    result["memories_before"],
                    result["memories_after"],
                    (1 - result.get("compression_ratio", 1.0)) * 100,
                )
        except Exception as e:
            logger.debug("Consolidation failed: %s", e)

    def _maybe_run_lifecycle(self):
        """Run lifecycle to demote/prune old memories."""
        if not self._ariadne:
            return
        try:
            result = self._ariadne.run_lifecycle()
            stats = result.get("stats")
            if stats:
                logger.debug(
                    "Lifecycle: hot=%d warm=%d cold=%d pruned=%d",
                    stats.hot_count, stats.warm_count, stats.cold_count,
                    result.get("due_for_pruning", 0),
                )
        except Exception as e:
            logger.debug("Lifecycle failed: %s", e)

    def get_config_schema(self) -> List[Dict[str, Any]]:
        return [
            {"key": "db_path", "description": "Path to Ariadne database (default: ~/.hermes/ariadne/memory.db)", "default": "~/.hermes/ariadne/memory.db", "required": False},
            {"key": "embedding_dim", "description": "Embedding dimension (must match model, default: 384)", "default": 384, "required": False},
            {"key": "extraction_enabled", "description": "Enable LLM-powered auto-extraction from conversations", "default": True, "required": False},
        ]

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """No-op — Ariadne uses defaults from initialize()."""

    # ── Tool schemas ───────────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return all memory tool schemas including new v2 tools."""
        return [
            # ── Core tools (backward-compatible names) ──
            {
                "name": "mnemosyne_remember",
                "description": "Store a durable memory in Ariadne. Use for ANY fact, preference, identity, insight, or context that should persist across sessions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "The memory content to store."},
                        "importance": {"type": "number", "description": "Importance 0.0-1.0. Default 0.5.", "default": 0.5},
                        "source": {"type": "string", "description": "Source tag: preference, fact, insight, identity, task, etc.", "default": "user"},
                        "scope": {"type": "string", "description": "'session' (default) or 'global'.", "default": "session"},
                        "memory_type": {"type": "string", "description": "semantic, episodic, or procedural", "default": "semantic"},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "mnemosyne_recall",
                "description": "Search Ariadne for relevant memories. Hybrid ranking: FTS5 text + FAISS vector.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language query."},
                        "limit": {"type": "integer", "description": "Max results. Default 5.", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "mnemosyne_stats",
                "description": "Return Ariadne memory statistics (v2: includes lifecycle tier counts, embedding info, extraction stats).",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "mnemosyne_forget",
                "description": "Permanently delete a memory by ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "integer", "description": "ID of the memory to delete"},
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "mnemosyne_update",
                "description": "Update the content or importance of an existing memory by ID.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "integer", "description": "ID of the memory to update"},
                        "content": {"type": "string", "description": "New content (optional)"},
                        "importance": {"type": "number", "description": "New importance 0.0-1.0 (optional)"},
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "mnemosyne_invalidate",
                "description": "Soft-delete a memory (mark as superseded).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "integer", "description": "Memory ID to invalidate."},
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "mnemosyne_export",
                "description": "Export all Ariadne memories to a JSON file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "output_path": {"type": "string", "description": "File path for export JSON"},
                    },
                    "required": ["output_path"],
                },
            },
            {
                "name": "mnemosyne_import",
                "description": "Import memories from a JSON file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "input_path": {"type": "string", "description": "JSON file to import"},
                    },
                    "required": ["input_path"],
                },
            },
            {
                "name": "mnemosyne_diagnose",
                "description": "Run diagnostics on the Ariadne installation.",
                "parameters": {"type": "object", "properties": {}},
            },
            # ── Graph tools ──
            {
                "name": "mnemosyne_graph_query",
                "description": "Traverse the memory graph from a seed entity.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity": {"type": "string", "description": "Entity name to start from."},
                        "hops": {"type": "integer", "description": "Max traversal depth (default 2)", "default": 2},
                    },
                    "required": ["entity"],
                },
            },
            {
                "name": "mnemosyne_graph_link",
                "description": "Declare a relationship between two entities.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "description": "Source entity"},
                        "target": {"type": "string", "description": "Target entity"},
                        "relationship": {"type": "string", "description": "Relationship label (e.g. 'uses', 'depends_on')", "default": "related"},
                        "weight": {"type": "number", "description": "Edge weight 0.0-1.0", "default": 0.5},
                    },
                    "required": ["source", "target", "relationship"],
                },
            },
            # ── Lifecycle & consolidation ──
            {
                "name": "mnemosyne_sleep",
                "description": "Run the Ariadne memory consolidation and lifecycle cycle.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dry_run": {"type": "boolean", "description": "Report without writing changes.", "default": False},
                    },
                },
            },
            # ── Scratchpad ──
            {
                "name": "mnemosyne_scratchpad_write",
                "description": "Write a temporary note to the scratchpad.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to write"},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "mnemosyne_scratchpad_read",
                "description": "Read the scratchpad entries.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "mnemosyne_scratchpad_clear",
                "description": "Clear all scratchpad entries.",
                "parameters": {"type": "object", "properties": {}},
            },
            # ── Shared surface ──
            {
                "name": "mnemosyne_shared_remember",
                "description": "Store a memory in the shared surface DB (cross-agent).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Memory content"},
                        "kind": {"type": "string", "description": "meta | preference | correction | identity", "default": "meta"},
                        "importance": {"type": "number", "description": "Importance 0.0-1.0", "default": 0.8},
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "mnemosyne_shared_recall",
                "description": "Search the shared surface DB.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results", "default": 5},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "mnemosyne_shared_forget",
                "description": "Delete a shared surface memory.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "integer", "description": "Memory ID to delete"},
                    },
                    "required": ["memory_id"],
                },
            },
            {
                "name": "mnemosyne_shared_stats",
                "description": "Return shared surface DB stats.",
                "parameters": {"type": "object", "properties": {}},
            },
            # ══════ NEW v2 TOOLS ══════
            {
                "name": "mnemosyne_temporal_search",
                "description": "Query the temporal knowledge graph. Find what was true about an entity at a given time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subject": {"type": "string", "description": "Entity/subject to query (optional — empty for all)."},
                        "limit": {"type": "integer", "description": "Max results. Default 20.", "default": 20},
                    },
                },
            },
            {
                "name": "mnemosyne_temporal_add",
                "description": "Add a temporal fact to the knowledge graph with validity tracking.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Fact statement."},
                        "subject": {"type": "string", "description": "Main entity."},
                        "predicate": {"type": "string", "description": "Relationship or property (e.g. 'lives_in', 'uses', 'version')."},
                        "object": {"type": "string", "description": "Target entity or value."},
                        "memory_id": {"type": "integer", "description": "Linked memory ID (optional)."},
                    },
                    "required": ["text", "subject", "predicate", "object"],
                },
            },
            {
                "name": "mnemosyne_consolidate",
                "description": "Manually run memory consolidation. Merges similar/duplicate memories into fewer, richer ones.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "method": {"type": "string", "description": "Grouping method: similarity, topic, or temporal", "default": "similarity"},
                        "dry_run": {"type": "boolean", "description": "Report without writing changes.", "default": False},
                    },
                },
            },
            {
                "name": "mnemosyne_lifecycle_status",
                "description": "Show the three-tier lifecycle status: hot/warm/cold distribution, retention scores, and pruning candidates.",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "name": "mnemosyne_entities",
                "description": "List resolved entities in the knowledge graph.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_type": {"type": "string", "description": "Filter by entity type (optional)."},
                        "limit": {"type": "integer", "description": "Max results. Default 20.", "default": 20},
                    },
                },
            },
            {
                "name": "mnemosyne_prune",
                "description": "Identify or prune cold, forgotten memories based on Ebbinghaus retention curves.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "min_age_days": {"type": "integer", "description": "Minimum age in days to consider. Default 90.", "default": 90},
                        "dry_run": {"type": "boolean", "description": "Report candidates without deleting.", "default": True},
                    },
                },
            },
        ]

    # ── Tool dispatch ──────────────────────────────────────────────────

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Dispatch tool calls to the appropriate handler."""
        if not self._ariadne:
            return json.dumps({"error": "Ariadne not initialized"})

        handlers = {
            "mnemosyne_remember": self._handle_remember,
            "mnemosyne_recall": self._handle_recall,
            "mnemosyne_stats": self._handle_stats,
            "mnemosyne_forget": self._handle_forget,
            "mnemosyne_update": self._handle_update,
            "mnemosyne_invalidate": self._handle_invalidate,
            "mnemosyne_export": self._handle_export,
            "mnemosyne_import": self._handle_import,
            "mnemosyne_diagnose": self._handle_diagnose,
            "mnemosyne_graph_query": self._handle_graph_query,
            "mnemosyne_graph_link": self._handle_graph_link,
            "mnemosyne_sleep": self._handle_sleep,
            "mnemosyne_scratchpad_write": self._handle_scratchpad_write,
            "mnemosyne_scratchpad_read": self._handle_scratchpad_read,
            "mnemosyne_scratchpad_clear": self._handle_scratchpad_clear,
            "mnemosyne_shared_remember": self._handle_shared_remember,
            "mnemosyne_shared_recall": self._handle_shared_recall,
            "mnemosyne_shared_forget": self._handle_shared_forget,
            "mnemosyne_shared_stats": self._handle_shared_stats,
            # v2 tools
            "mnemosyne_temporal_search": self._handle_temporal_search,
            "mnemosyne_temporal_add": self._handle_temporal_add,
            "mnemosyne_consolidate": self._handle_consolidate,
            "mnemosyne_lifecycle_status": self._handle_lifecycle_status,
            "mnemosyne_entities": self._handle_entities,
            "mnemosyne_prune": self._handle_prune,
        }

        handler = handlers.get(tool_name)
        if handler:
            try:
                return handler(args)
            except Exception as e:
                logger.error("Ariadne tool %s failed: %s", tool_name, e)
                return json.dumps({"error": str(e)})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # ── Tool handlers ─────────────────────────────────────────────────

    def _handle_remember(self, args: Dict) -> str:
        result = self._ariadne.remember(
            content=args["content"],
            memory_type=args.get("memory_type", "semantic"),
            importance=args.get("importance", 0.5),
            metadata={"source": args.get("source", "user"), "scope": args.get("scope", "session")},
        )
        return json.dumps(result)

    def _handle_recall(self, args: Dict) -> str:
        results = self._ariadne.recall(
            query=args["query"],
            k=args.get("limit", 5),
        )
        # Record access for lifecycle
        for r in results:
            try:
                self._ariadne._get_lifecycle().record_access(str(r["id"]))
            except Exception:
                pass
        return json.dumps(self._simplify_results(results))

    def _handle_stats(self, args: Dict) -> str:
        stats = self._ariadne.stats()
        # Enrich with v2 info
        try:
            lifecycle = self._ariadne._get_lifecycle()
            sample = {"importance": 5, "access_count": 0, "created_at": time.time()}
            stats["ebbinghaus_retention"] = round(lifecycle.get_retention_score(sample), 3)
        except Exception:
            pass
        stats["llm_extraction"] = bool(self._ariadne._llm_provider)
        stats["consolidation_last"] = self._last_consolidation
        stats["lifecycle_last"] = self._last_lifecycle
        return json.dumps(stats)

    def _handle_forget(self, args: Dict) -> str:
        mid = args["memory_id"]
        success = self._ariadne.forget(memory_id=mid, hard=True)
        return json.dumps({"deleted": success, "memory_id": mid})

    def _handle_update(self, args: Dict) -> str:
        success = self._ariadne.update(
            memory_id=args["memory_id"],
            content=args.get("content"),
            importance=args.get("importance"),
        )
        return json.dumps({"updated": success})

    def _handle_invalidate(self, args: Dict) -> str:
        mid = args["memory_id"]
        success = self._ariadne.forget(memory_id=mid, hard=False)
        return json.dumps({"invalidated": success, "memory_id": mid})

    def _validate_path(self, path: str) -> str:
        """Validate that a path is safe (no traversal outside allowed dirs)."""
        import os as _os
        expanded = _os.path.expanduser(path)
        resolved = _os.path.realpath(expanded)
        # Allow paths under hermes_home or current directory
        allowed_roots = [
            _os.path.realpath(self._hermes_home),
            _os.path.realpath("."),
            _os.path.realpath("/tmp"),
        ]
        if not any(resolved.startswith(root) for root in allowed_roots):
            raise ValueError(f"Path not allowed: {path} (resolved to {resolved})")
        return resolved

    def _handle_export(self, args: Dict) -> str:
        import json as _json
        output_path = self._validate_path(args["output_path"])
        stats = self._ariadne.stats()
        try:
            all_results = self._ariadne.recall("", k=10000)
        except Exception:
            all_results = []
        data = {"stats": stats, "memories": self._simplify_results(all_results)}
        with open(output_path, "w") as f:
            _json.dump(data, f, indent=2, default=str)
        return json.dumps({"exported": len(all_results), "path": output_path})

    def _handle_import(self, args: Dict) -> str:
        import json as _json
        input_path = self._validate_path(args["input_path"])
        with open(input_path) as f:
            data = _json.load(f)
        memories = data.get("memories", data if isinstance(data, list) else [])
        imported = 0
        for m in memories:
            if isinstance(m, dict) and m.get("content"):
                result = self._ariadne.remember(
                    content=m["content"],
                    memory_type=m.get("type", "semantic"),
                    importance=m.get("importance", 0.5),
                )
                if result.get("status") == "created":
                    imported += 1
        return json.dumps({"imported": imported})

    def _handle_diagnose(self, args: Dict) -> str:
        try:
            stats = self._ariadne.stats()
            diag = {
                "status": "healthy",
                "provider": "ariadne",
                "version": "2.0.0",
                "db_path": self._db_path,
                "active_memories": stats.get("active_memories", 0),
                "faiss_type": stats.get("faiss_type", "unknown"),
                "entities": stats.get("total_entities", 0),
                "edges": stats.get("total_edges", 0),
                "db_size_kb": stats.get("db_size_bytes", 0) / 1024,
                "embedding_provider": stats.get("embedding_provider", "?"),
                "embedding_dimension": stats.get("embedding_dimension", 0),
                "dedup_index_size": stats.get("dedup_index_size", 0),
                "llm_extraction": bool(self._ariadne._llm_provider),
                "llm_model": self._ariadne._llm_provider.name if self._ariadne._llm_provider else "none",
                "auto_consolidation": True,
                "auto_lifecycle": True,
                "session_id": self._session_id,
            }
            return json.dumps(diag)
        except Exception as e:
            return json.dumps({"status": "unhealthy", "error": str(e)})

    def _handle_graph_query(self, args: Dict) -> str:
        result = self._ariadne.graph(
            entity=args["entity"],
            hops=args.get("hops", 2),
        )
        return json.dumps(result)

    def _handle_graph_link(self, args: Dict) -> str:
        self._ariadne.add_edge(
            source=args["source"],
            target=args["target"],
            edge_type=args.get("relationship", "related"),
            weight=args.get("weight", 0.5),
        )
        return json.dumps({"status": "created"})

    def _handle_sleep(self, args: Dict) -> str:
        dry_run = args.get("dry_run", False)
        if dry_run:
            stats = self._ariadne.stats()
            return json.dumps({
                "active_memories": stats["active_memories"],
                "message": "Dry run — no changes made",
            })

        # Run each step independently — partial results still useful
        results = {}
        try:
            consolidation_result = self._ariadne.consolidate_with_llm(dry_run=False)
            results["consolidation"] = {
                "groups_processed": consolidation_result.get("groups_processed", 0),
                "memories_before": consolidation_result.get("memories_before", 0),
                "memories_after": consolidation_result.get("memories_after", 0),
                "compression_ratio": consolidation_result.get("compression_ratio", 1.0),
            }
        except Exception as e:
            results["consolidation"] = {"error": str(e)}

        try:
            results["evicted"] = self._ariadne.evict()
        except Exception as e:
            results["evicted"] = {"error": str(e)}

        try:
            lifecycle_result = self._ariadne.run_lifecycle()
            results["lifecycle"] = {
                "hot": lifecycle_result.get("stats", {}).hot_count if lifecycle_result.get("stats") else 0,
                "warm": lifecycle_result.get("stats", {}).warm_count if lifecycle_result.get("stats") else 0,
                "cold": lifecycle_result.get("stats", {}).cold_count if lifecycle_result.get("stats") else 0,
                "due_for_pruning": lifecycle_result.get("due_for_pruning", 0),
            }
        except Exception as e:
            results["lifecycle"] = {"error": str(e)}

        return json.dumps(results)
            "consolidation": {
                "groups_processed": consolidation_result.get("groups_processed", 0),
                "memories_before": consolidation_result.get("memories_before", 0),
                "memories_after": consolidation_result.get("memories_after", 0),
                "compression_ratio": consolidation_result.get("compression_ratio", 1.0),
            },
            "evicted": evicted,
            "lifecycle": {
                "hot": lifecycle_result.get("stats", {}).hot_count if lifecycle_result.get("stats") else 0,
                "warm": lifecycle_result.get("stats", {}).warm_count if lifecycle_result.get("stats") else 0,
                "cold": lifecycle_result.get("stats", {}).cold_count if lifecycle_result.get("stats") else 0,
                "due_for_pruning": lifecycle_result.get("due_for_pruning", 0),
            },
        })

    def _handle_scratchpad_write(self, args: Dict) -> str:
        import hashlib
        key = hashlib.sha256(args["content"].encode()).hexdigest()[:12]
        # If collision, append counter
        orig_key = key
        counter = 1
        while key in self._scratchpad and self._scratchpad.get(key) != args["content"]:
            key = f"{orig_key}_{counter}"
            counter += 1
        self._scratchpad[key] = args["content"]
        return json.dumps({"key": key, "stored": True})

    def _handle_scratchpad_read(self, args: Dict) -> str:
        return json.dumps(list(self._scratchpad.values()))

    def _handle_scratchpad_clear(self, args: Dict) -> str:
        self._scratchpad.clear()
        return json.dumps({"cleared": True})

    # ── v2 tool handlers ─────────────────────────────────────────────

    def _handle_temporal_search(self, args: Dict) -> str:
        """Query the temporal knowledge graph."""
        try:
            results = self._ariadne.query_temporal(
                subject=args.get("subject"),
                limit=args.get("limit", 20),
            )
            return json.dumps(results)
        except Exception as e:
            return json.dumps({"error": str(e), "results": []})

    def _handle_temporal_add(self, args: Dict) -> str:
        """Add a temporal fact."""
        try:
            result = self._ariadne.add_temporal_fact(
                text=args["text"],
                subject=args["subject"],
                predicate=args["predicate"],
                obj=args["object"],
                memory_id=str(args.get("memory_id", "")) or None,
            )
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _handle_consolidate(self, args: Dict) -> str:
        """Run memory consolidation."""
        method = args.get("method", "similarity")
        dry_run = args.get("dry_run", False)

        try:
            result = self._ariadne.consolidate_with_llm(method=method, dry_run=dry_run)
            return json.dumps(result)
        except Exception as e:
            # Fallback to basic consolidation
            try:
                evicted = self._ariadne.evict()
                return json.dumps({"evicted": evicted, "method": "basic_fallback", "error": str(e)})
            except Exception as e2:
                return json.dumps({"error": str(e2)})

    def _handle_lifecycle_status(self, args: Dict) -> str:
        """Show lifecycle status with tier distribution and retention scores."""
        try:
            lifecycle = self._ariadne._get_lifecycle()
            result = lifecycle.run_lifecycle()

            stats = result.get("stats")
            status = {
                "hot_count": stats.hot_count if stats else 0,
                "warm_count": stats.warm_count if stats else 0,
                "cold_count": stats.cold_count if stats else 0,
                "total_count": stats.total_count if stats else 0,
                "demoted_to_warm": result.get("demoted_to_warm", 0),
                "demoted_to_cold": result.get("demoted_to_cold", 0),
                "due_for_pruning": result.get("due_for_pruning", 0),
                "avg_retention": result.get("avg_retention", {}),
                "latency_ms": result.get("latency_ms", 0),
            }
            return json.dumps(status)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def _handle_entities(self, args: Dict) -> str:
        """List resolved entities."""
        try:
            entities = self._ariadne.get_entities(
                entity_type=args.get("entity_type"),
                limit=args.get("limit", 20),
            )
            return json.dumps(entities)
        except Exception as e:
            return json.dumps({"error": str(e), "entities": []})

    def _handle_prune(self, args: Dict) -> str:
        """Identify or prune cold, forgotten memories."""
        try:
            lifecycle = self._ariadne._get_lifecycle()
            result = lifecycle.prune_cold_memories(
                min_age_days=args.get("min_age_days", 90),
                dry_run=args.get("dry_run", True),
            )
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── Shared surface ────────────────────────────────────────────────

    def _handle_shared_remember(self, args: Dict) -> str:
        if not self._shared:
            return json.dumps({"error": "shared surface not available"})
        result = self._shared.remember(
            content=args["content"],
            memory_type="semantic",
            importance=args.get("importance", 0.8),
        )
        return json.dumps(result)

    def _handle_shared_recall(self, args: Dict) -> str:
        if not self._shared:
            return json.dumps([])
        results = self._shared.recall(
            query=args["query"],
            k=args.get("limit", 5),
        )
        return json.dumps(self._simplify_results(results))

    def _handle_shared_forget(self, args: Dict) -> str:
        if not self._shared:
            return json.dumps({"error": "shared surface not available"})
        mid = args["memory_id"]
        success = self._shared.forget(memory_id=mid, hard=True)
        return json.dumps({"deleted": success, "memory_id": mid})

    def _handle_shared_stats(self, args: Dict) -> str:
        if not self._shared:
            return json.dumps({"error": "shared surface not available"})
        stats = self._shared.stats()
        return json.dumps(stats)

    # ── Optional hooks ───────────────────────────────────────────────

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Per-turn hook: count turns, check for periodic maintenance."""
        # Run lifecycle every 100 turns even if time hasn't elapsed
        if turn_number > 0 and turn_number % 100 == 0:
            self._maybe_run_lifecycle()

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """End-of-session: extract final facts and consolidate."""
        if not self._ariadne or self._agent_context != "primary":
            return

        # Extract from the last portion of the conversation
        if messages:
            try:
                recent = messages[-10:]  # Last 10 messages
                self._ariadne.extract_from_conversation(
                    recent, auto_store=True,
                    observation_date=datetime.now(timezone.utc).isoformat(),
                )
            except Exception as e:
                logger.debug("Session-end extraction failed: %s", e)

        # Run final consolidation
        try:
            self._ariadne.consolidate_with_llm(dry_run=False)
        except Exception:
            pass

        # Run lifecycle
        try:
            self._ariadne.run_lifecycle()
        except Exception:
            pass

    def on_session_switch(self, new_session_id: str, **kwargs) -> None:
        """Handle session ID rotation."""
        old_session = self._session_id
        reset = kwargs.get("reset", False)
        self._session_id = new_session_id
        # Only clear caches on full reset, not on /resume or /branch
        if reset or not kwargs.get("rewound", False):
            self._turn_messages.clear()
            self._prefetch_cache.clear()
        logger.debug("Session switched: %s -> %s (reset=%s)", old_session, new_session_id, reset)

    def on_delegation(self, task: str, result: str, **kwargs) -> None:
        """Store delegation observation (parent-side)."""
        if not self._ariadne:
            return
        try:
            self._ariadne.remember(
                content=f"[DELEGATION] Task: {task[:200]} | Result: {result[:200]}",
                memory_type="episodic",
                importance=0.4,
                metadata={"source": "delegation"},
            )
        except Exception:
            pass

    # ── Helpers ───────────────────────────────────────────────────────

    def _simplify_results(self, results: List[Dict]) -> List[Dict]:
        """Strip large fields from recall results for clean JSON output."""
        out = []
        for r in results:
            item = {
                "id": r.get("id"),
                "content": r.get("content", "")[:300],
                "memory_type": r.get("memory_type", ""),
                "importance": r.get("importance", 0.5),
                "score": r.get("score", 0),
                "search_type": r.get("search_type", ""),
                "created_at": r.get("created_at", 0),
            }
            # Include entities if present
            if r.get("entities"):
                item["entities"] = r["entities"]
            out.append(item)
        return out

    def _format_recall_block(self, results: List[Dict]) -> str:
        """Format recall results as a memory context block."""
        if not results:
            return ""
        lines = ["", "══════════════════════════════════════════════"]
        lines.append("ARIADNE MEMORY (context for this turn)")
        lines.append("══════════════════════════════════════════════")
        for i, r in enumerate(results[:_MAX_PREFETCH_RESULTS], 1):
            content = r.get("content", "")[:200].replace("\n", " ")
            score = r.get("score", 0)
            lines.append(f"{i}. [{r.get('memory_type', '?')}] {content} (score: {score:.2f})")
        lines.append("")
        return "\n".join(lines)

    def shutdown(self) -> None:
        """Close databases."""
        try:
            if self._ariadne:
                self._ariadne.close()
        except Exception:
            pass
        try:
            if self._shared:
                self._shared.close()
        except Exception:
            pass

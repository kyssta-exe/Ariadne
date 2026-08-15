"""Programmatic audit: verify new-module API usage against the real interfaces.

Walks each integration / manager method and asserts the AriadneMemory / AriadneDB
methods they call actually exist with compatible signatures. Run with:
    python scripts/audit_api.py
"""

import inspect
import tempfile

from arriadne import AriadneMemory
from arriadne.curator import MemoryCurator, CuratorAddon
from arriadne.memory_manager import LLMMemoryManager  # noqa: F401 (smoke-import)

fail = []


def check(cond, msg):
    if not cond:
        fail.append(msg)
        print("FAIL:", msg)
    else:
        print("ok:", msg)


# --- AriadneMemory public surface used by new code ---
m = AriadneMemory(db_path=tempfile.mktemp(suffix=".db"), embedder=None)
mem_methods = set(dir(m))

requires_mem = [
    "remember",
    "recall",
    "forget",
    "record_episode",
    "consolidate",
    "stats",
    "add_edge",
    "supersede",
    "close",
]
for name in requires_mem:
    check(name in mem_methods, f"AriadneMemory.{name} exists")

# --- signatures ---
sig_remember = inspect.signature(m.remember)
check("event_at" in sig_remember.parameters, "remember(event_at=...) accepted")
check("valid_from" in sig_remember.parameters, "remember(valid_from=...) accepted")
check("supersedes_id" in sig_remember.parameters, "remember(supersedes_id=...) accepted")
check("metadata" in sig_remember.parameters, "remember(metadata=...) accepted")
check("entities" in sig_remember.parameters, "remember(entities=...) accepted")

sig_recall = inspect.signature(m.recall)
check("namespace" in sig_recall.parameters, "recall(namespace=...) accepted")
check("type_filter" in sig_recall.parameters, "recall(type_filter=...) accepted")
check("as_of" in sig_recall.parameters, "recall(as_of=...) accepted")

sig_record = inspect.signature(m.record_episode)
check("namespace" in sig_record.parameters, "record_episode(namespace=...) accepted")
check("role" in sig_record.parameters, "record_episode(role=...) accepted")
check("event_at" in sig_record.parameters, "record_episode(event_at=...) accepted")

sig_supersede = inspect.signature(m.supersede)
check("old_memory_id" in sig_supersede.parameters, "supersede(old_memory_id=...) accepted")

sig_forget = inspect.signature(m.forget)
check("hard" in sig_forget.parameters, "forget(hard=...) accepted")

# stats should include by_namespace (we added it)
stats = m.stats()
check("by_namespace" in stats, "stats() includes by_namespace")
check(isinstance(stats["by_namespace"], dict), "by_namespace is a dict")

# add_edge signature
sig_add_edge = inspect.signature(m.add_edge)
check(
    {"source", "target", "edge_type", "weight"} <= set(sig_add_edge.parameters),
    "add_edge(source,target,edge_type,weight) accepted",
)

m.close()

# --- curator: decay/resolve/curate signatures ---
c_curate = inspect.signature(MemoryCurator.curate)
check("run_consolidate" in c_curate.parameters, "MemoryCurator.curate(run_consolidate=...)")
c_resolve = inspect.signature(MemoryCurator.resolve_contradictions)
check("namespace" in c_resolve.parameters, "resolve_contradictions(namespace=...)")

# CuratorAddon satisfies BaseAddon contract
addon = CuratorAddon()
check(addon.name == "ariadne-curator", "CuratorAddon.name")
check(callable(addon.get_cli_commands), "CuratorAddon.get_cli_commands")
cmds = addon.get_cli_commands()
check(len(cmds) >= 1 and cmds[0].name == "curate", "CuratorAddon exposes curate CLI command")

# --- MCP server: tool handlers exist ---
from arriadne.integrations.mcp_server import (  # noqa: E402
    AriadneMCPServer,
    _TOOL_SCHEMAS,
    _TOOL_HANDLERS,
    _ns,
)

check(set(_TOOL_SCHEMAS) == set(_TOOL_HANDLERS), "MCP tools and handlers have matching names")
check(
    {"ariadne_recall", "ariadne_remember", "ariadne_forget", "ariadne_stats"} <= set(_TOOL_SCHEMAS),
    "MCP exposes the four memory tools",
)
check(callable(_ns), "_ns helper callable")
check(
    hasattr(AriadneMCPServer, "handle_request") and hasattr(AriadneMCPServer, "serve_stdio"),
    "MCP server has handle_request and serve_stdio",
)

# --- OpenAI Agents tools ---
from arriadne.integrations.openai_agents import AriadneTools  # noqa: E402

check(
    set(AriadneTools.__dict__)
    >= {"recall", "remember", "forget", "stats", "to_openai_agents_tools"},
    "AriadneTools exposes recall/remember/forget/stats/to_openai_agents_tools",
)

# --- LangGraph adapter importable, constructor guards ---
import arriadne.integrations.langgraph as lg  # noqa: E402

check(hasattr(lg, "AriadneStore"), "langgraph module has AriadneStore")
check(lg._LANGRAPH_AVAILABLE is False, "langgraph not installed in this env (expected)")

print("\n=== RESULT ===")
if fail:
    print(f"{len(fail)} audit failure(s)")
    raise SystemExit(1)
print("ALL API AUDIT CHECKS PASSED")

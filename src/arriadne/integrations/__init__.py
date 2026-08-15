"""Framework integrations for Ariadne (MCP, LangGraph, OpenAI Agents).

Each integration lives in its own module and is import-guarded so the core
package stays dependency-light: importing ``arriadne.integrations`` never
requires ``mcp``, ``langgraph``, or ``openai-agents``. Only calling a specific
adapter pulls in its optional dependency.
"""

from __future__ import annotations

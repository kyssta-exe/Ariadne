"""Claude Code integration: memory hooks + MCP host configuration.

Claude Code supports *hooks* — shell commands that receive a JSON event on
stdin at well-known points in the agent loop (user prompt submitted, assistant
finished, session start/end). Wiring Ariadne in gives Claude Code persistent,
local-first memory without any server:

* ``UserPromptSubmit`` — records the prompt as an immutable episode and
  injects a packed block of relevant memories as ``additionalContext`` so the
  model sees what it already knows before answering.
* ``Stop`` — records the assistant's reply as an episode, closing the turn.
  With an optional LLM caller it can also run autonomous extraction on the
  turn (facts, relations) via :class:`~arriadne.memory_manager.LLMMemoryManager`.

Every handler is fail-open: any error results in exit code 0 with no output
rather than breaking the user's session. Install with the snippet from
:func:`install_snippet` (or ``ariadne mcp --host claude-code`` for the MCP
server registration).

Usage (standalone)::

    echo '{"hook_event_name": "UserPromptSubmit", "prompt": "hi"}' \\
        | ariadne hook claude-code --db-path ~/.ariadne/memory.db
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from .. import AriadneMemory

logger = logging.getLogger(__name__)

# Role tags used for hook-recorded episodes, so the Stop handler can find the
# user prompt that opened the turn.
_USER_PROMPT_ROLE = "claude_code_user_prompt"
_ASSISTANT_ROLE = "claude_code_assistant"

# Claude Code hook events this adapter consumes.
SUPPORTED_EVENTS = (
    "UserPromptSubmit",
    "Stop",
    "SessionStart",
    "SessionEnd",
    "SubagentStop",
    "PreToolUse",
    "PostToolUse",
    "Notification",
)


def parse_hook_event(raw: str | bytes | None) -> dict[str, Any]:
    """Parse a Claude Code hook event from raw stdin bytes.

    Tolerant by design: malformed, empty, or non-object input yields ``{}``
    instead of raising — hooks must never crash the host session.
    """
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _episode_text(event: dict[str, Any]) -> str:
    """Best-effort text of the user prompt from a UserPromptSubmit event."""
    prompt = event.get("prompt")
    return prompt.strip() if isinstance(prompt, str) else ""


def _session(event: dict[str, Any]) -> str | None:
    session_id = event.get("session_id")
    return str(session_id) if session_id else None


def handle_user_prompt_submit(
    memory: AriadneMemory,
    event: dict[str, Any],
    *,
    k: int = 5,
    namespace: str = "default",
    token_budget: int = 1200,
    record: bool = True,
    recency_boost: float = 0.5,
) -> dict[str, Any]:
    """Handle ``UserPromptSubmit``: record the prompt, inject known context.

    Records the user prompt as an immutable episode (provenance for later
    extraction), then packs the most relevant memories under a token budget
    and returns them as Claude Code ``additionalContext``. When nothing
    relevant is stored, returns an empty dict — no context is injected.
    """
    prompt = _episode_text(event)
    if not prompt:
        return {}

    if record:
        try:
            memory.record_episode(
                content=prompt,
                role=_USER_PROMPT_ROLE,
                source="claude-code",
                namespace=namespace,
                session_id=_session(event),
            )
        except Exception as exc:  # fail-open: provenance is best-effort
            logger.warning("Failed to record prompt episode: %s", exc)

    try:
        packed = memory.context_pack(
            prompt,
            token_budget=token_budget,
            k=k,
            namespace=namespace,
            recency_boost=recency_boost,
        )
    except Exception as exc:
        logger.warning("Context pack failed: %s", exc)
        packed = ""

    if not packed.strip():
        return {}

    return {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": (
                "Relevant memories from previous sessions "
                f"(source: Ariadne local memory):\n{packed}"
            ),
        }
    }


def handle_stop(
    memory: AriadneMemory,
    event: dict[str, Any],
    *,
    namespace: str = "default",
    caller: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    """Handle ``Stop``: record the assistant reply, optionally extract facts.

    Records the assistant's final message as an episode. When ``caller`` (an
    LLM ``prompt -> str`` callable) is provided, pairs the reply with the most
    recent recorded user prompt for the same session and runs autonomous
    memory extraction over the turn. Returns a summary dict (never hook
    output — Stop handlers inject nothing).
    """
    reply = ""
    last = event.get("last_message")
    if isinstance(last, dict):
        reply = str(last.get("content", "") or "").strip()
    elif isinstance(last, str):
        reply = last.strip()

    session = _session(event)
    if reply:
        try:
            memory.record_episode(
                content=reply,
                role=_ASSISTANT_ROLE,
                source="claude-code",
                namespace=namespace,
                session_id=session,
            )
        except Exception as exc:  # fail-open
            logger.warning("Failed to record assistant episode: %s", exc)

    if caller is None or not reply:
        return {"recorded_reply": bool(reply), "extracted": False}

    user_prompt = ""
    try:
        latest = memory._db.get_latest_episode(role=_USER_PROMPT_ROLE, session_id=session)
        if latest is not None:
            user_prompt = str(latest.get("content") or "")
    except Exception as exc:
        logger.warning("Could not fetch prompt episode: %s", exc)

    if not user_prompt:
        return {"recorded_reply": bool(reply), "extracted": False}

    try:
        from ..memory_manager import LLMCaller, LLMMemoryManager

        manager = LLMMemoryManager(
            memory, caller=cast("LLMCaller | None", caller), default_namespace=namespace
        )
        summary = manager.process_turn(
            user=user_prompt,
            assistant=reply,
            namespace=namespace,
            record_episode=False,  # both halves already recorded by the hooks
        )
        return {"recorded_reply": True, "extracted": True, "summary": summary}
    except Exception as exc:  # fail-open: extraction must never break the session
        logger.warning("Autonomous extraction failed: %s", exc)
        return {"recorded_reply": True, "extracted": False}


def handle_event(
    memory: AriadneMemory,
    event: dict[str, Any],
    *,
    namespace: str = "default",
    caller: Callable[[str], str] | None = None,
    recall_k: int = 5,
    token_budget: int = 1200,
) -> dict[str, Any]:
    """Dispatch one parsed hook event to its handler.

    Unknown or missing event names return ``{}`` (a no-op), so new Claude
    Code hook types degrade gracefully.
    """
    name = str(event.get("hook_event_name", ""))
    if name == "UserPromptSubmit":
        return handle_user_prompt_submit(
            memory, event, k=recall_k, namespace=namespace, token_budget=token_budget
        )
    if name == "Stop":
        return handle_stop(memory, event, namespace=namespace, caller=caller)
    return {}


# ---------------------------------------------------------------------------
# Standalone hook runner (the `ariadne hook claude-code` entry point)
# ---------------------------------------------------------------------------


def _resolve_memory(db_path: str | None, namespace: str) -> AriadneMemory:
    """Open the memory store for a hook run (env config aware)."""
    from ..config import AriadneConfig

    path = db_path or os.environ.get("ARIADNE_DB_PATH") or "arriadne.db"
    config = AriadneConfig.from_env(base=AriadneConfig(db_path=Path(path)))
    return AriadneMemory(config=config)


def _resolve_caller(caller_name: str | None) -> Callable[[str], str] | None:
    """Build the extraction LLM caller requested on the command line.

    ``--extract-with openai`` / ``--extract-with anthropic`` construct the
    provider caller lazily; anything else (including unset) disables
    autonomous extraction.
    """
    if not caller_name:
        return None
    from .. import memory_manager

    factories: dict[str, Callable[[], Callable[[str], str]]] = {
        "openai": memory_manager.openai_caller,
        "anthropic": memory_manager.anthropic_caller,
    }
    factory = factories.get(caller_name)
    return factory() if factory is not None else None


def run_hook(argv: list[str] | None = None, *, stdin: str | bytes | None = None) -> int:
    """Read one hook event from stdin and handle it. Always exits 0.

    Fail-open contract: any unexpected error logs a warning and exits 0 with
    no stdout, so a broken memory store can never block a Claude Code session.
    """
    parser = argparse.ArgumentParser(prog="ariadne hook claude-code")
    parser.add_argument("--db-path", default=None, help="Memory database path")
    parser.add_argument(
        "--namespace", default="default", help="Memory namespace (default: default)"
    )
    parser.add_argument(
        "--extract-with",
        choices=["openai", "anthropic"],
        default=None,
        help="LLM provider for autonomous extraction on Stop (optional)",
    )
    args = parser.parse_args(argv)

    event = parse_hook_event(stdin if stdin is not None else sys.stdin.read())
    if not event:
        return 0
    if event.get("hook_event_name") not in SUPPORTED_EVENTS:
        return 0

    memory: AriadneMemory | None = None
    try:
        memory = _resolve_memory(args.db_path, args.namespace)
        caller = _resolve_caller(args.extract_with)
        output = handle_event(
            memory, event, namespace=args.namespace, caller=caller
        )
        if output:
            json.dump(output, sys.stdout)
            sys.stdout.write("\n")
        return 0
    except Exception as exc:
        logger.warning("ariadne hook failed (fail-open): %s", exc)
        return 0
    finally:
        if memory is not None:
            try:
                memory.close()
            except Exception:  # pragma: no cover - defensive
                pass


# ---------------------------------------------------------------------------
# MCP host configuration generator
# ---------------------------------------------------------------------------

# Each host gets: the JSON fragment shape and the file it belongs in. The
# server command runs the dependency-free stdio MCP server shipped in
# arriadne.integrations.mcp_server.
_HOSTS: dict[str, dict[str, Any]] = {
    "claude-code": {
        "file": ".mcp.json (project root)",
        "wrapper": "mcpServers",
    },
    "claude-desktop": {
        "file": "~/Library/Application Support/Claude/claude_desktop_config.json (macOS) "
        "or ~/.config/claude/claude_desktop_config.json (Linux)",
        "wrapper": "mcpServers",
    },
    "cursor": {
        "file": "~/.cursor/mcp.json",
        "wrapper": "mcpServers",
    },
    "vscode": {
        "file": ".vscode/mcp.json",
        "wrapper": "servers",
    },
    "zed": {
        "file": "~/.config/zed/settings.json",
        "wrapper": "context_servers",
    },
}

_SERVER_ENTRY = {
    "command": sys.executable,
    "args": ["-m", "arriadne.integrations.mcp_server", "--db-path", "{db_path}"],
}


def mcp_server_entry(db_path: str | Path) -> dict[str, Any]:
    """The stdio server entry (command + args) pointing at ``db_path``.

    ``sys.executable`` is resolved at call time so the snippet always uses the
    interpreter that has Ariadne installed.
    """
    entry: dict[str, Any] = json.loads(json.dumps(_SERVER_ENTRY))  # deep copy
    entry["args"] = [a.replace("{db_path}", str(Path(db_path).resolve())) for a in entry["args"]]
    return entry


def mcp_host_config(host: str, db_path: str | Path) -> dict[str, Any]:
    """Return the JSON config registering Ariadne's MCP server in ``host``.

    Supported hosts: claude-code, claude-desktop, cursor, vscode, zed.
    Raises ValueError for unknown hosts (use :data:`MCP_HOSTS` to enumerate).
    """
    if host not in _HOSTS:
        raise ValueError(f"Unknown host {host!r}. Choose one of: {', '.join(_HOSTS)}")
    spec = _HOSTS[host]
    entry = mcp_server_entry(db_path)
    if host == "vscode":
        entry = {"type": "stdio", **entry}
    if host == "zed":
        entry = {**entry, "env": {}}
    wrapper = str(spec["wrapper"])
    return {wrapper: {"ariadne": entry}}


MCP_HOSTS: tuple[str, ...] = tuple(_HOSTS)


def install_snippet(db_path: str | Path, *, extract_with: str | None = None) -> dict[str, Any]:
    """The Claude Code ``hooks`` block that installs Ariadne memory.

    Drop the returned object into ``~/.claude/settings.json`` (user-wide) or
    ``.claude/settings.json`` (per project). The command uses the ``ariadne``
    console script so it survives interpreter changes; ``--db-path`` is
    resolved to an absolute path.
    """
    hook_cmd = f"ariadne hook claude-code --db-path {Path(db_path).resolve()}"
    if extract_with:
        hook_cmd += f" --extract-with {extract_with}"
    command = {"type": "command", "command": hook_cmd}
    return {
        "hooks": {
            "UserPromptSubmit": [{"hooks": [command]}],
            "Stop": [{"hooks": [command]}],
        }
    }


def main(argv: list[str] | None = None) -> int:
    """``python -m arriadne.integrations.claude_code`` — run as a hook."""
    return run_hook(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

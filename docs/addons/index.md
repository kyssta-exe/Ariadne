---
title: "Addons — Ariadne"
description: "Extend Ariadne with addons: domain-specific memory types, custom search pipelines, integrations, and tools. Discover, install, and create addons."
---

# Addons

Ariadne's **addon system** lets you extend the memory engine with domain-specific tools, custom search pipelines, and integrations. Addons plug into Ariadne's core via Python's `entry_points` mechanism — no forking, no monkey-patching.

## What Is an Addon?

An addon is a separate Python package that:

1. **Registers itself** via `pyproject.toml` entry points (e.g. `ariadne.addons`)
2. **Extends Ariadne** with new CLI commands, dashboard panels, API endpoints, or Hermes tools
3. **Shares the same database** — addons read/write the same SQLite `.db` file as core Ariadne

Addons are first-class citizens. They get their own namespace, their own configuration, and full access to `AriadneMemory`.

## Quick Start

```bash
# Install an addon from PyPI
pip install ariadne-finance

# Or from source
git clone https://github.com/kyssta-exe/ariadne-finance.git
cd ariadne-finance
pip install -e .
```

Once installed, the addon's commands and tools appear automatically:

```bash
# CLI commands are available immediately
ariadne finance --help
ariadne finance import-csv transactions.csv

# Python API works too
from ariadne_finance import FinanceAddon
```

## How It Works

Ariadne discovers addons at startup by scanning the `ariadne.addons` entry point group:

```python
# Simplified discovery logic
from importlib.metadata import entry_points

def discover_addons():
    addons = {}
    for ep in entry_points(group="ariadne.addons"):
        addon_cls = ep.load()
        addons[ep.name] = addon_cls()
    return addons
```

Each addon is a class that implements the `AddonProtocol` interface:

```python
from typing import Protocol

class AddonProtocol(Protocol):
    """Base interface all Ariadne addons must implement."""

    name: str                          # Unique identifier (e.g. "finance")
    description: str                   # Human-readable description

    def register_commands(self, parser) -> None:
        """Add CLI subcommands to the main parser."""
        ...

    def get_tools(self) -> list[dict]:
        """Return Hermes-compatible tool definitions."""
        ...
```

## Entry Points Pattern

Addons register via `pyproject.toml`:

```toml
[project.entry-points."ariadne.addons"]
finance = "ariadne_finance.addon:FinanceAddon"
```

This means:

| Field | Example | Description |
|-------|---------|-------------|
| Group | `ariadne.addons` | Must match exactly |
| Key | `finance` | Unique addon identifier |
| Target | `ariadne_finance.addon:FinanceAddon` | `module:ClassName` to import |

::: tip
The entry point key (e.g. `finance`) becomes the addon's namespace in config and CLI commands.
:::

## Creating an Addon

### 1. Scaffold the package

```
ariadne-my-addon/
├── pyproject.toml
├── src/
│   └── ariadne_my_addon/
│       ├── __init__.py
│       └── addon.py
└── README.md
```

### 2. Implement the addon

```python
# src/ariadne_my_addon/addon.py
from arriadne.interface import AriadneMemory

class MyAddon:
    name = "my-addon"
    description = "My custom Ariadne addon"

    def __init__(self):
        self._memory: AriadneMemory | None = None

    def set_memory(self, memory: AriadneMemory) -> None:
        """Called by Ariadne after discovery. Store the shared memory instance."""
        self._memory = memory

    def register_commands(self, parser) -> None:
        """Add CLI subcommands."""
        sub = parser.add_subparsers(dest="addon_command")
        sub.add_parser("my-command", help="Run my custom command")

    def get_tools(self) -> list[dict]:
        """Return Hermes tool definitions."""
        return [
            {
                "name": "my_tool",
                "description": "Does something useful",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            }
        ]
```

### 3. Register the entry point

```toml
# pyproject.toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ariadne-my-addon"
version = "0.1.0"
dependencies = ["ariadne>=0.10.0"]

[project.entry-points."ariadne.addons"]
my-addon = "ariadne_my_addon.addon:MyAddon"
```

### 4. Install and test

```bash
pip install -e .
ariadne my-addon --help  # Your command is now available
```

## Configuration

Addon configuration lives in the same `AriadneConfig` or a separate config file:

```python
# In your addon's __init__.py or addon.py
import os

class MyAddon:
    def __init__(self):
        self.api_key = os.environ.get("ARIADNE_MY_ADDON_API_KEY", "")
        self.max_results = int(os.environ.get("ARIADNE_MY_ADDON_MAX_RESULTS", "100"))
```

::: tip
Use environment variables for secrets. Never commit API keys to the addon package itself.
:::

## Directory Structure

Installed addons live alongside Ariadne's core files:

```
~/.ariadne/
├── memory.db              # Core database (shared with all addons)
├── addons/
│   ├── finance/
│   │   ├── finance.db     # Addon-specific data (optional)
│   │   └── cache/         # Addon cache (optional)
│   └── my-addon/
│       └── ...
└── config.yaml            # Includes addon-specific settings
```

## Existing Addons

| Addon | Description | Install |
|-------|-------------|---------|
| [`finance`](./finance) | Financial records, portfolio tracking, transaction memory | `pip install ariadne-finance` |

## Related

- [Installation](/guide/installation) — Install Ariadne core
- [Configuration](/guide/configuration) — AriadneConfig reference
- [Architecture](/guide/architecture) — How Ariadne stores data
- [CLI Reference](/api/cli) — All CLI commands

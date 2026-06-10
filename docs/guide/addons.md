---
title: "Using Addons — Ariadne Guide"
description: "Discover, install, configure, and manage Ariadne addons. User guide for the addon ecosystem."
---

# Using Addons

Ariadne's addon system lets you extend the memory engine without modifying core code. This guide covers discovering, installing, configuring, and managing addons.

## Discovering Addons

### PyPI

Search for addons on PyPI:

```bash
pip index versions ariadne-finance
pip index versions ariadne-health
```

Addons follow the naming convention `ariadne-<name>`.

### Community List

Browse available addons in the [Addons Directory](https://github.com/kyssta-exe/Ariadne#addons) on GitHub.

### CLI Discovery

```bash
# List all installed addons
ariadne addons list

# Check for updates
ariadne addons check-updates
```

Output:

```
Installed addons:
  finance  v0.2.1  (ariadne-finance)  Financial records and portfolio tracking
  health   v0.1.0  (ariadne-health)   Health and fitness memory tracking

No updates available.
```

## Installing Addons

### From PyPI (Recommended)

```bash
pip install ariadne-finance
```

That's it. The addon registers itself via entry points and is available immediately:

```bash
ariadne finance --help
```

### From Source

```bash
git clone https://github.com/kyssta-exe/ariadne-finance.git
cd ariadne-finance
pip install -e .
```

Development install (includes tests and linting):

```bash
pip install -e ".[dev]"
```

### Installing with Ariadne Extras

Some addons are bundled as extras:

```bash
pip install "ariadne[finance]"       # Install Ariadne + finance addon
pip install "ariadne[finance,health]" # Multiple addons at once
```

::: tip
Using extras ensures version compatibility between Ariadne core and the addon.
:::

### Uninstalling

```bash
pip uninstall ariadne-finance
```

The addon is removed and its commands disappear from the CLI.

## Configuring Addons

Each addon has its own configuration section in `~/.ariadne/config.yaml`:

```yaml
# Core Ariadne settings
db_path: ~/.ariadne/memory.db
embedding_dim: 384

# Addon-specific settings
addons:
  finance:
    default_currency: USD
    default_account: checking
    retention_days: 365

  health:
    units: metric
    tracking_enabled: true
```

### Environment Variables

Addons also read environment variables (useful for secrets):

```bash
export ARIADNE_FINANCE_DEFAULT_CURRENCY=EUR
export ARIADNE_FINANCE_CSV_ENCODING=utf-8
```

::: warning
Never put API keys or secrets in `config.yaml`. Use environment variables or a `.env` file.
:::

### Configuration Priority

1. **Environment variables** (highest priority)
2. **`~/.ariadne/config.yaml`**
3. **Addon defaults** (lowest priority)

## Managing Addons

### List Installed Addons

```bash
ariadne addons list
```

### Check Addon Status

```bash
ariadne addons status
```

Output:

```
Addon Status
═══════════════════════════════════════
  Name       Version   Status    DB Tables
  ─────────  ────────  ────────  ─────────
  finance    0.2.1     ✅ OK     finance_transactions, finance_portfolios, finance_balances
  health     0.1.0     ✅ OK     health_records

All addons healthy.
```

### Enable/Disable

Disable an addon without uninstalling:

```bash
ariadne addons disable finance
ariadne addons enable finance
```

This moves the addon entry point to a disabled list — the package stays installed but Ariadne ignores it.

### Update

```bash
# Check for updates
ariadne addons check-updates

# Update all addons
ariadne addons update

# Update a specific addon
ariadne addons update finance
```

## How Addons Work Under the Hood

```
┌──────────────────────────────────────────────────┐
│                  Ariadne Core                     │
│                                                   │
│  AriadneMemory ──┬── remember()                   │
│                  ├── recall()                     │
│                  └── graph()                      │
│                                                   │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐    │
│  │  finance   │  │  health   │  │  my-addon  │    │
│  │  addon     │  │  addon    │  │  addon     │    │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘    │
│        │              │              │            │
│        └──────────────┼──────────────┘            │
│                       │                           │
│              entry_points discovery               │
│              (ariadne.addons group)               │
└──────────────────────────────────────────────────┘
```

1. **Discovery**: Ariadne scans `entry_points(group="ariadne.addons")` at startup
2. **Registration**: Each addon's `set_memory()` is called with the shared `AriadneMemory` instance
3. **CLI**: Addon commands are registered with the main CLI parser
4. **Dashboard**: Addon routes are registered with the FastAPI server
5. **Hermes**: Addon tools are registered with the Hermes plugin interface

## Writing Your Own Addon

See the [Addon System Overview](/addons/index) for the full guide on creating addons. The short version:

```python
# my_addon/addon.py
class MyAddon:
    name = "my-addon"
    description = "Does something useful"

    def set_memory(self, memory):
        self._memory = memory

    def register_commands(self, parser):
        sub = parser.add_subparsers(dest="my_command")
        sub.add_parser("do-thing", help="Do a thing")

    def get_tools(self):
        return [{"name": "my_tool", "description": "...", "parameters": {...}}]
```

```toml
# pyproject.toml
[project.entry-points."ariadne.addons"]
my-addon = "my_addon.addon:MyAddon"
```

## Troubleshooting

### Addon not showing up

```bash
# Check entry points are registered
python -c "from importlib.metadata import entry_points; print(entry_points(group='ariadne.addons'))"

# If empty, reinstall the addon
pip install -e .
```

### Database table conflicts

Addons should use their own table names prefixed with the addon name (e.g. `finance_transactions`). If you see errors:

```bash
# Check which tables exist
ariadne stats
```

### Version mismatch

```bash
# Check installed version
pip show ariadne-finance

# Check Ariadne version
ariadne --version

# Ensure compatibility
pip install "ariadne>=0.10.0" ariadne-finance
```

## Related

- [Addon System](/addons/index) — Architecture and creating addons
- [Finance Addon](/addons/finance) — Finance addon reference
- [Installation](/guide/installation) — Install Ariadne core
- [Configuration](/guide/configuration) — AriadneConfig reference
- [CLI Reference](/api/cli) — All CLI commands

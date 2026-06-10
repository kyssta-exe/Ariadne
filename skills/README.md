# Ariadne Skills for Hermes

This directory contains Hermes skills for the Ariadne memory system and its addons.

## Available Skills

| Skill | Description | Requires |
|-------|-------------|----------|
| [ariadne](ariadne/SKILL.md) | Core memory system — FAISS + FTS5 + knowledge graph | `ariadne-memory` |
| [ariadne-finance](ariadne-finance/SKILL.md) | Finance research — PDF/Excel extraction, tickers, financial graph | `ariadne-finance` |

## Installation

### Option 1: Copy to Hermes skills directory

```bash
# Install core skill
cp -r ariadne ~/.hermes/skills/ariadne

# Install finance skill
cp -r ariadne-finance ~/.hermes/skills/ariadne-finance
```

### Option 2: Symlink (stays updated with repo)

```bash
ln -s /path/to/Ariadne/skills/ariadne ~/.hermes/skills/ariadne
ln -s /path/to/Ariadne/skills/ariadne-finance ~/.hermes/skills/ariadne-finance
```

### Option 3: From a remote URL (for Hermes agents)

Hermes can install skills from a URL. Point the agent to:

```
https://github.com/kyssta-exe/Ariadne/tree/main/skills
```

The agent can read the manifest.json to discover available skills, then fetch and install individual SKILL.md files.

## Manifest

The `manifest.json` file provides a machine-readable registry of all skills. Hermes agents can use this to:

1. Discover available skills
2. Check requirements
3. Install skills automatically

## Per-Addon Skills

Each addon also includes its skill in its own directory:

```
addons/
  finance/
    skill/
      SKILL.md    # Finance addon skill
```

This allows addons to be self-contained — install the addon package and its skill is available in the repo.

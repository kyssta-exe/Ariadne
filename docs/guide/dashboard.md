---
title: "Dashboard"
description: "Visual dashboard for browsing and managing Ariadne memories."
---

# Dashboard <Badge type="warning" text="BETA" />

The Ariadne Dashboard is a self-contained web interface for browsing,
searching, and managing your memory system. It runs as a single FastAPI
process with zero additional infrastructure.

> **Beta status.** The dashboard is functional but evolving. APIs and UI
> may change between minor versions.

## Starting the dashboard

```bash
# Install dashboard dependencies
pip install "ariadne-memory[dashboard]"

# Launch with default settings
ariadne dashboard

# Custom host and port
ariadne dashboard --host 0.0.0.0 --port 8765

# Point at a specific database
ariadne --db-path /path/to/memory.db dashboard
```

Open `http://localhost:8765` in your browser. The dashboard auto-detects
the local server — no API key or configuration needed.

### Via Python

```python
from arriadne.dashboard.server import create_app
import uvicorn

app = create_app(db_path="arriadne.db")
uvicorn.run(app, host="127.0.0.1", port=8765)
```

## Pages

### Dashboard (home)

The landing page gives you a system overview:

| Metric | Description |
| :----- | :---------- |
| **Memories** | Total active memories |
| **Entities** | Unique named entities extracted from memories |
| **Edges** | Relationships (links) between entities |
| **Vectors** | FAISS vector index size |
| **Avg Importance** | Mean importance score across all memories |
| **DB Size** | On-disk database size |

#### Memory Composition

A horizontal bar chart showing memory distribution by type
(`architecture`, `infrastructure`, `project`, etc.). Bars are sorted
descending – hover for exact counts and percentages.

#### Link Types

A compact ranked list of relationship types between entities, sorted
by frequency. Each row shows a colored indicator, type name, count,
proportional bar, and percentage. Long lists collapse to the top 10
with an expand button.

#### Activity Timeline

A bar chart of memory creation activity over the selected time range
(1D / 7D / 30D / 90D).

#### Quick Actions

- **Create Memory** – add a new memory via a simple form
- **Run Consolidation** – manually trigger memory consolidation
- **Refresh Data** – reload all dashboard data
- **Export Data** – download all memories as JSON

### Memories

Browse all memories in a searchable, filterable grid. Click a memory
card to view its full content, metadata, entities, and lifecycle
status.

### Graph

An interactive force-directed knowledge graph. Nodes represent
memories, edges represent shared entities.

- Drag nodes to rearrange
- Scroll to zoom
- **Click a node** → enters **Focus Mode**, showing its neighborhood
  at a configurable depth (1–5). Click "Exit Focus" to return to the
  full graph.
- Toggle physics simulation on/off with the button in the toolbar.

### Search

Full-text and semantic search against memories. Supports two modes:

| Mode | Description |
| :--- | :---------- |
| **Hybrid** (default) | FAISS vector similarity + FTS5 keyword search fused with Reciprocal Rank Fusion |
| **Keyword** | Pure FTS5 BM25 keyword search |

### Lifecycle

Monitor the cognitive retention model:

- **Hot / Warm / Cold tiers** – memory retention strength distribution
- **Retention curve** – projected decay over time per tier
- **Prune preview** – compare candidate memories for eviction

### Settings

Manage API keys for programmatic access to the Ariadne REST API.

## REST API

The dashboard exposes a REST API at `/api/…` — the same endpoints the
frontend calls. Key endpoints:

| Endpoint | Returns |
| :------- | :------ |
| `GET /api/stats` | Aggregated system statistics |
| `GET /api/composition` | Memory type breakdown |
| `GET /api/link-types` | Entity relationship type breakdown |
| `GET /api/activity?range=7d` | Activity timeline (day buckets) |
| `GET /api/recent?limit=8` | Most recently created memories |
| `GET /api/health-report` | System health and maintenance metrics |
| `GET /api/graph/memory` | Memory graph nodes + edges |
| `GET /api/graph/neighbors?memory_id=N&depth=2` | Neighborhood subgraph |
| `POST /api/remember` | Store a new memory |
| `GET /api/search?q=...` | Hybrid / keyword search |
| `GET /api/export` | Full JSON export of all data |

## Troubleshooting

**Dashboard won't start.** Make sure you have the dashboard extra
installed:

```bash
pip install "ariadne-memory[dashboard]"
```

**Pages are blank / charts don't render.** The dashboard requires a
modern browser (Chrome, Firefox, Safari). Check the browser console
(F12) for JavaScript errors.

**"No such table" errors.** Your database may be from an older version.
Run `ariadne migrate` to apply pending schema migrations.

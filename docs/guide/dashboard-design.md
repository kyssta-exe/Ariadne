---
title: Dashboard Design
description: Technical design for the Ariadne Console — browser-based management dashboard
---


## Screenshots

### Login
![Login](/screenshot-auth.png)

### Dashboard Overview
![Dashboard](/screenshot-dashboard.png)

### Memory Browser
![Memories](/screenshot-memories.png)

### Knowledge Graph
![Graph](/screenshot-graph.png)

### Search
![Search](/screenshot-search.png)

### Settings & API Key Management
![Settings](/screenshot-settings.png)


# Ariadne Console — Browser-Based Management Dashboard

## Executive Summary

A lightweight, self-contained web dashboard embedded in the existing FastAPI server.
No build step, no Node.js, no React. Served as static files from `/dashboard/` route.
Total footprint: ~200KB of HTML/CSS/JS + CDN-loaded libraries.

---

## 1. Architecture Decision

### Option Analysis

| Approach | Pros | Cons | Verdict |
|----------|------|------|---------|
| **Embedded static files in FastAPI** | Single port, single deploy, zero infra | Limited caching | ✅ **Chosen** |
| Standalone SPA (React/Vue) | Modern DX | Build step, Node.js dep, separate server | ❌ |
| Refine/Appsmith admin | Rapid prototyping | Heavy, opinionated, overkill | ❌ |
| Server-rendered (Jinja2) | Simple | No interactivity, poor UX | ❌ |

### Why Embedded Static Files

1. **Zero infrastructure** — matches Ariadne's philosophy (no external DB, no daemon)
2. **Single deployment** — `arriadne serve` gives you API + dashboard on one port
3. **No build step** — edit HTML/JS directly, refresh browser
4. **Existing server** — FastAPI already serves static files via `StaticFiles`
5. **Competitor pattern** — Mem0, Zep, and LiteLLM all serve dashboards from their API server

---

## 2. Tech Stack

### Frontend Libraries (CDN-loaded, zero build)

| Component | Library | Size | Why |
|-----------|---------|------|-----|
| **Reactive UI** | Alpine.js 3.x | ~15KB | React-like reactivity without build step |
| **Server interactions** | HTMX 2.x | ~14KB | Server-driven UI updates, SSE support |
| **Knowledge Graph** | vis-network 9.x | ~200KB | Built-in physics, tooltips, clustering, easy API |
| **Charts/Metrics** | Chart.js 4.x | ~65KB | Lightweight, responsive, many chart types |
| **Icons** | Lucide Icons | ~30KB | Clean, modern icon set |
| **CSS Framework** | Pico CSS 2.x | ~10KB | Classless CSS, minimal, semantic HTML |
| **Utilities** | day.js | ~7KB | Date formatting |

**Total: ~340KB loaded from CDN** (or bundled into single files for offline use)

### Why Not React/Vue/Svelte

- Requires Node.js build toolchain
- Adds ~500KB+ of framework code
- Breaks "zero infrastructure" promise
- Ariadne users are Python developers — don't require JS expertise
- Alpine.js + HTMX covers 90% of needed interactivity

### Why vis-network Over D3.js

| Criteria | vis-network | D3.js | Cytoscape.js |
|----------|:-----------:|:-----:|:------------:|
| Setup complexity | ⭐ Low | ⭐⭐⭐ High | ⭐⭐ Medium |
| Physics simulation | Built-in | Manual | Plugin |
| Tooltips/Labels | Built-in | Manual | Plugin |
| Clustering | Built-in | Manual | Plugin |
| Learning curve | Hours | Days | Hours |
| File size | 200KB | 250KB | 300KB |
| API surface | Simple | Verbose | Moderate |

vis-network is the clear winner for this use case: small-to-medium knowledge graphs
with built-in physics, tooltips, and easy JavaScript API.

### Why Chart.js Over D3.js for Charts

- Declarative API (pass data, get chart)
- Responsive out of the box
- 65KB vs 250KB
- Perfect for the metrics we need (bar, line, doughnut, radar)

---

## 3. File Structure

```
src/arriadne/dashboard/
├── __init__.py          # Module init
├── routes.py            # FastAPI router for dashboard routes
├── static/
│   ├── index.html       # Main SPA shell
│   ├── css/
│   │   └── style.css    # Custom styles (on top of Pico CSS)
│   ├── js/
│   │   ├── app.js       # Alpine.js app initialization + state
│   │   ├── api.js       # API client (fetch wrappers)
│   │   ├── graph.js     # vis-network graph visualization
│   │   ├── charts.js    # Chart.js chart configurations
│   │   └── utils.js     # Date formatting, helpers
│   └── favicon.svg      # Ariadne logo (SVG)
```

**Total files: 10 files, ~500KB combined** (before minification)

---

## 4. Page Layout & Navigation

### Layout Structure

```
┌─────────────────────────────────────────────────────────────┐
│  ◉ Ariadne Console                    [Search...]  ⚙  🔔  │
├──────────┬──────────────────────────────────────────────────┤
│          │                                                  │
│ Dashboard│           Main Content Area                      │
│ Memories │           (routed by sidebar)                    │
│ Graph    │                                                  │
│ Lifecycle│                                                  │
│ Stats    │                                                  │
│ Settings │                                                  │
│          │                                                  │
│          │                                                  │
├──────────┴──────────────────────────────────────────────────┤
│  Status: Healthy │ Memories: 1,234 │ Entities: 89 │ ⏱ 2ms  │
└─────────────────────────────────────────────────────────────┘
```

### Sidebar Navigation

| Route | Page | Description |
|-------|------|-------------|
| `/` | Dashboard | Overview with key metrics + recent activity |
| `/memories` | Memory Browser | Search, filter, view, edit, delete memories |
| `/graph` | Knowledge Graph | Interactive node-link visualization |
| `/lifecycle` | Lifecycle | Hot/Warm/Cold tier management, pruning |
| `/stats` | Statistics | Detailed metrics, charts, performance |
| `/settings` | Settings | Configuration, import/export, maintenance |

---

## 5. Component Design

### 5.1 Dashboard Page (`/`)

The landing page showing system health and key metrics at a glance.

```
┌─────────────────────────────────────────────────────┐
│  System Health        Memories    Entities    Edges  │
│  ● Healthy            1,234       89          342   │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌─ Lifecycle Distribution ─┐  ┌─ Memory Types ──┐ │
│  │  🔴 Hot: 234  (19%)     │  │  semantic: 890  │ │
│  │  🟡 Warm: 567 (46%)     │  │  episodic: 234  │ │
│  │  🔵 Cold: 433 (35%)     │  │  procedural: 110│ │
│  │                          │  └─────────────────┘ │
│  │  [Bar chart: days]       │                      │
│  └──────────────────────────┘                      │
│                                                     │
│  ┌─ Recent Activity ──────────────────────────────┐ │
│  │  2 min ago  Stored: "Deployed v2.1 to prod"   │ │
│  │  5 min ago  Searched: "server config" → 3 hits │ │
│  │  12 min ago Consolidated 4 memories            │ │
│  │  1 hr ago   Community detected: 3 clusters     │ │
│  └─────────────────────────────────────────────────┘ │
│                                                     │
│  ┌─ Search Quality ───────────────────────────────┐ │
│  │  Avg latency: 1.78ms  │  p95: 4.2ms           │ │
│  │  Searches today: 156  │  Results avg: 7.3      │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

**API calls:**
- `GET /health` — system status
- `GET /stats` — memory/entity/edge counts
- `GET /lifecycle` — tier distribution
- `GET /metrics?format=json` — search quality, latency

**Components:**
- `StatCard` — single metric display (icon, value, label, delta)
- `LifecycleChart` — horizontal stacked bar (hot/warm/cold)
- `ActivityFeed` — recent operations list
- `HealthBadge` — green/yellow/red status indicator

---

### 5.2 Memory Browser (`/memories`)

Full-text search with filters, memory list with inline actions.

```
┌─────────────────────────────────────────────────────┐
│  Search: [________________________] [🔍 Search]     │
│                                                     │
│  Filters: [Type ▾] [Tier ▾] [Importance ▾] [Age ▾] │
│           [Entity ▾] [Tenant ▾]                     │
│                                                     │
│  Found 23 memories (1.78ms)        [Export] [Import]│
├─────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐│
│  │ #1234  ⭐ 8.5  🔴 Hot   semantic   2h ago      ││
│  │ "Deployed v2.1 to production server..."         ││
│  │ Entities: VPS, Nginx, Production                ││
│  │ [View] [Edit] [Score] [Graph] [🗑]             ││
│  ├─────────────────────────────────────────────────┤│
│  │ #1230  ⭐ 7.2  🟡 Warm  episodic   1d ago      ││
│  │ "User prefers dark mode in the dashboard..."    ││
│  │ Entities: User, Dashboard                       ││
│  │ [View] [Edit] [Score] [Graph] [🗑]             ││
│  ├─────────────────────────────────────────────────┤│
│  │ #1198  ⭐ 3.1  🔵 Cold  procedural 1w ago      ││
│  │ "npm install --legacy-peer-deps fixes..."       ││
│  │ Entities: npm, Dependencies                     ││
│  │ [View] [Edit] [Score] [Graph] [🗑]             ││
│  └─────────────────────────────────────────────────┘│
│                                                     │
│  [◀ Prev]  Page 1 of 5  [Next ▶]                   │
└─────────────────────────────────────────────────────┘
```

**Memory Detail Modal/Panel:**

```
┌─ Memory #1234 ──────────────────────────────────────┐
│                                                      │
│ Content:                                             │
│ "Deployed v2.1 to production server with zero        │
│  downtime. Key changes: rate limiter, WAL tuning."   │
│                                                      │
│ ┌─ Metadata ──────────────────────────────────────┐  │
│ │ Type:      semantic                             │  │
│ │ Tier:      Hot 🔴                               │  │
│ │ Importance: 8.5/10                              │  │
│ │ Created:   2026-06-04 14:30                     │  │
│ │ Accessed:  3 times (last: 2h ago)               │  │
│ │ Retention: 0.87 (Ebbinghaus)                    │  │
│ │ Tenant:    default                              │  │
│ │ Hash:      a3f2...9c1d                          │  │
│ └──────────────────────────────────────────────────┘ │
│                                                      │
│ ┌─ Importance Score Breakdown ────────────────────┐  │
│ │ Composite:     0.85 ████████████████░░░░        │  │
│ │ Info Density:  0.72 ██████████████░░░░░░        │  │
│ │ Recency:       0.91 ██████████████████░░        │  │
│ │ Access Freq:   0.65 █████████████░░░░░░░        │  │
│ │ Entity Cent.:  0.88 █████████████████░░░        │  │
│ │ LLM Score:     0.80 ████████████████░░░░        │  │
│ └──────────────────────────────────────────────────┘ │
│                                                      │
│ ┌─ Related Entities ─────────────────────────────┐   │
│ │ VPS ──[runs_on]──► Production                  │   │
│ │ Nginx ──[serves]──► VPS                        │   │
│ └──────────────────────────────────────────────────┘ │
│                                                      │
│ [Edit Content] [Update Score] [View in Graph] [🗑]  │
└──────────────────────────────────────────────────────┘
```

**API calls:**
- `POST /search` — hybrid search with filters
- `GET /memories/{id}` — full memory detail
- `GET /memories/{id}/score` — importance score breakdown
- `PATCH /memories/{id}` — update memory
- `DELETE /memories/{id}` — delete memory
- `GET /memories/ranked` — ranked list

**Components:**
- `SearchBar` — query input with search type toggle (hybrid/keyword/vector)
- `FilterBar` — dropdown filters for type, tier, importance, age, entity, tenant
- `MemoryCard` — memory preview with metadata badges
- `MemoryDetail` — full detail panel/modal
- `ScoreBreakdown` — horizontal bar chart of importance factors
- `Pagination` — page navigation

---

### 5.3 Knowledge Graph (`/graph`)

Interactive force-directed graph with physics simulation.

```
┌─────────────────────────────────────────────────────┐
│  Knowledge Graph                    [🔍 Search Node]│
│                                                     │
│  ┌─ Controls ──┐  ┌───────────────────────────────┐│
│  │ Layout:      │  │                               ││
│  │ ○ Force      │  │      (Paris)                  ││
│  │ ○ Hierarchical│  │      /    \                   ││
│  │ ● Circular   │  │  (France)  (Eiffel)           ││
│  │              │  │    |         |                 ││
│  │ Physics:     │  │  (Europe) (Louvre)            ││
│  │ ○ On         │  │                               ││
│  │ ○ Off        │  │     vis-network canvas        ││
│  │              │  │     (interactive, zoomable)   ││
│  │ Depth:       │  │                               ││
│  │ [1] [2] [3]  │  │                               ││
│  │              │  │                               ││
│  │ Color by:    │  │                               ││
│  │ ○ Type       │  │                               ││
│  │ ● Community  │  │                               ││
│  │ ○ Degree     │  │                               ││
│  │              │  │                               ││
│  │ [Detect     ]│  └───────────────────────────────┘│
│  │ [Communities]│                                   │
│  │ [Reset View ]│  ┌─ Node Info ───────────────────┐│
│  │ [Export PNG ]│  │ Paris                         ││
│  └──────────────┘  │ Type: person                  ││
│                     │ Memories: 12                  ││
│                     │ Edges: 5 (centrality: 0.82)  ││
│                     │ Community: "Travel" (#3)      ││
│                     │ [View Memories] [View Timeline]│
│                     └────────────────────────────────┘│
│                                                     │
│  ┌─ Legend ────────────────────────────────────────┐│
│  │ ● person  ● location  ● concept  ● organization││
│  │ ── related  ── located_in  ── works_at         ││
│  │ Node size = memory count │ Edge width = weight  ││
│  └─────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

**Interactive Features:**
- Click node → show entity details + related memories
- Drag node → reposition
- Scroll → zoom in/out
- Double-click node → expand to show connected nodes
- Hover → tooltip with entity name + memory count
- Right-click → context menu (view memories, add edge, delete)

**API calls:**
- `GET /graph/entities` — list all entities (nodes)
- `GET /graph/entity/{name}?hops=2` — entity neighborhood
- `GET /graph/stats` — graph statistics
- `GET /graph/metrics` — community metrics
- `POST /communities/detect` — run community detection
- `POST /graph/connect` — add edge
- `DELETE /graph/entities/{name}` — remove entity

**Components:**
- `GraphCanvas` — vis-network container
- `GraphControls` — layout, physics, depth, color-by controls
- `NodeInfoPanel` — entity detail sidebar
- `CommunityLegend` — color legend for communities
- `GraphStats` — small metrics overlay (nodes, edges, density)

---

### 5.4 Lifecycle Management (`/lifecycle`)

Visualize and manage the three-tier memory lifecycle.

```
┌─────────────────────────────────────────────────────┐
│  Memory Lifecycle                                    │
│                                                     │
│  ┌─ Tier Distribution ─────────────────────────────┐│
│  │                                                  ││
│  │  🔴 Hot (7d)     ████████████░░░░░░░░░  234     ││
│  │  🟡 Warm (30d)   ████████████████████░  567     ││
│  │  🔵 Cold (90d+)  █████████████████░░░░  433     ││
│  │  ⚫ Deleted      █░░░░░░░░░░░░░░░░░░░░   23     ││
│  │                                                  ││
│  │  Total: 1,257 active memories                    ││
│  └──────────────────────────────────────────────────┘│
│                                                     │
│  ┌─ Retention Curve ───────────────────────────────┐│
│  │  1.0 ┤●                                        ││
│  │      │ ●●                                       ││
│  │  0.8 ┤   ●●●                                    ││
│  │      │      ●●●●                                ││
│  │  0.6 ┤          ●●●●●                           ││
│  │      │               ●●●●●●                     ││
│  │  0.4 ┤                    ●●●●●●●●              ││
│  │      │                          ●●●●●●●●        ││
│  │  0.2 ┤                                ●●●●●●●   ││
│  │      │                                       ●●●││
│  │  0.0 ┼───────────────────────────────────────── ││
│  │      0    7d    14d    21d    30d    60d    90d  ││
│  └──────────────────────────────────────────────────┘│
│                                                     │
│  ┌─ Actions ───────────────────────────────────────┐│
│  │                                                  ││
│  │  [▶ Run Lifecycle]  Demote hot→warm, warm→cold  ││
│  │                                                  ││
│  │  [🔍 Preview Prune]  Show memories to prune     ││
│  │  [🗑 Execute Prune]  Remove cold + forgotten    ││
│  │                                                  ││
│  │  [📊 Recompute Scores]  Refresh importance       ││
│  │                                                  ││
│  │  ┌─ Pruning Preview ───────────────────────────┐││
│  │  │ Candidates: 45 memories older than 90 days   │││
│  │  │ with retention < 0.01 and 0 accesses         │││
│  │  │                                              │││
│  │  │ #891  "old debug log entry..."    ret: 0.002 │││
│  │  │ #867  "temp workaround for..."    ret: 0.001 │││
│  │  │ ... (showing 5 of 45)                        │││
│  │  │                                              │││
│  │  │ [Confirm Prune 45]  [Cancel]                 │││
│  │  └──────────────────────────────────────────────┘││
│  └──────────────────────────────────────────────────┘│
│                                                     │
│  ┌─ Promotion Rules ───────────────────────────────┐│
│  │  Cold → Warm:  access_count ≥ 5 AND accessed < 24h│
│  │  Warm → Hot:   access_count ≥ 10 AND accessed < 12h│
│  │                                                  │
│  │  [Edit Rules]                                    ││
│  └──────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

**API calls:**
- `GET /lifecycle` — lifecycle status + tier counts
- `POST /lifecycle/evict` — trigger eviction
- `POST /lifecycle/forget` — forget dead communities
- `POST /importance/recompute` — recompute all scores
- `GET /importance/stats` — importance distribution

**Components:**
- `TierBar` — stacked horizontal bar showing hot/warm/cold
- `RetentionChart` — line chart of Ebbinghaus decay curve
- `ActionPanel` — lifecycle action buttons with confirmations
- `PrunePreview` — list of candidate memories for pruning
- `RuleEditor` — editable promotion/demotion rules

---

### 5.5 Statistics Page (`/stats`)

Comprehensive metrics and performance data.

```
┌─────────────────────────────────────────────────────┐
│  Statistics & Metrics                                │
│                                                     │
│  ┌─ Performance ──────────────────────────────────┐ │
│  │  Search Latency    │  Storage Performance       │ │
│  │  ┌───────────────┐ │  ┌───────────────────────┐│ │
│  │  │ Avg: 1.78ms   │ │  │ Insert rate: 2000/s   ││ │
│  │  │ P50: 1.2ms    │ │  │ Batch rate: 5000/s    ││ │
│  │  │ P95: 4.2ms    │ │  │ FAISS vectors: 1,234  ││ │
│  │  │ P99: 8.1ms    │ │  │ Index type: FlatIP    ││ │
│  │  └───────────────┘ │  └───────────────────────┘│ │
│  └─────────────────────────────────────────────────┘ │
│                                                     │
│  ┌─ Memory Distribution ──────────────────────────┐ │
│  │                                                  │ │
│  │  [Doughnut: by type]    [Bar: by age]           │ │
│  │                                                  │ │
│  │  semantic:  890 (72%)   ▏ 0-1d:   45            │ │
│  │  episodic:  234 (19%)   ▏ 1-7d:  189            │ │
│  │  procedural: 110 (9%)   ▏ 7-30d: 567            │ │
│  │                          ▏ 30d+:  433            │ │
│  └─────────────────────────────────────────────────┘ │
│                                                     │
│  ┌─ Graph Metrics ────────────────────────────────┐  │
│  │                                                  │ │
│  │  Nodes: 89  │  Edges: 342  │  Density: 0.087   │ │
│  │  Components: 3  │  Avg degree: 7.7              │ │
│  │                                                  │ │
│  │  Top Central Nodes:                              │ │
│  │  1. Paris      (centrality: 0.92)               │ │
│  │  2. France     (centrality: 0.85)               │ │
│  │  3. User       (centrality: 0.78)               │ │
│  │                                                  │ │
│  │  Communities: 5 (modularity: 0.67)               │ │
│  └─────────────────────────────────────────────────┘ │
│                                                     │
│  ┌─ Access Log (24h) ─────────────────────────────┐ │
│  │                                                  │ │
│  │  [Line chart: requests/hour over 24h]           │ │
│  │                                                  │ │
│  │  Total searches: 156                             │ │
│  │  Avg results per search: 7.3                    │ │
│  │  Unique queries: 89                              │ │
│  └─────────────────────────────────────────────────┘ │
│                                                     │
│  ┌─ System ───────────────────────────────────────┐  │
│  │  DB size: 12.4 MB  │  FAISS index: 4.2 MB     │  │
│  │  Embedding: ONNX all-MiniLM-L6-v2 (384d)       │  │
│  │  LLM: none (keyword-only mode)                  │  │
│  │  Uptime: 3d 14h 22m                             │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

**API calls:**
- `GET /stats` — comprehensive system stats
- `GET /metrics?format=json` — Prometheus metrics in JSON
- `GET /graph/stats` — graph statistics
- `GET /graph/metrics` — community metrics
- `GET /categories` — category distribution
- `GET /importance/stats` — importance distribution
- `GET /temporal/stats` — temporal graph stats

**Components:**
- `PerformanceCard` — latency metrics with sparklines
- `DistributionChart` — doughnut + bar charts
- `GraphMetricsPanel` — graph statistics display
- `AccessLogChart` — line chart of access patterns
- `SystemInfo` — database and embedding provider info

---

### 5.6 Settings Page (`/settings`)

Configuration, import/export, and maintenance operations.

```
┌─────────────────────────────────────────────────────┐
│  Settings                                            │
│                                                     │
│  ┌─ Configuration ─────────────────────────────────┐│
│  │                                                  ││
│  │  Database:  ariadne.db (12.4 MB)                ││
│  │  FAISS:     FlatIP (1,234 vectors, 384d)       ││
│  │  Embedding: ONNX all-MiniLM-L6-v2              ││
│  │  LLM:       None                                ││
│  │  FTS:       porter unicode61                     ││
│  │                                                  ││
│  │  [View Full Config]                              ││
│  └──────────────────────────────────────────────────┘│
│                                                     │
│  ┌─ Import / Export ───────────────────────────────┐│
│  │                                                  ││
│  │  Export:                                         ││
│  │  [📦 Export JSON]  [📝 Export Markdown]          ││
│  │  [📊 Export Graph DOT]  [📈 Export Graph JSON]   ││
│  │                                                  ││
│  │  Import:                                         ││
│  │  [Upload JSON]  [Upload Markdown]  [Upload Text] ││
│  │                                                  ││
│  │  From other systems:                             ││
│  │  [Import from Mem0]  [Import from ChromaDB]      ││
│  └──────────────────────────────────────────────────┘│
│                                                     │
│  ┌─ Maintenance ──────────────────────────────────┐│
│  │                                                  ││
│  │  [🔄 Rebuild FAISS Index]                       ││
│  │  [🔍 Rebuild Dedup Index]                       ││
│  │  [🗑 Vacuum Database]                           ││
│  │  [📊 Run Benchmark]                             ││
│  │                                                  ││
│  │  Danger Zone:                                    ││
│  │  [⚠️  Delete All Memories] (requires confirm)    ││
│  └──────────────────────────────────────────────────┘│
│                                                     │
│  ┌─ API Reference ────────────────────────────────┐│
│  │                                                  ││
│  │  Interactive API docs available at:              ││
│  │  [📖 Swagger UI]  [📋 ReDoc]                    ││
│  │                                                  ││
│  │  Endpoints: 42  │  Tags: 12                     ││
│  └──────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────┘
```

**API calls:**
- `GET /export?format=json` — export all memories
- `POST /import` — import memories
- `GET /` — server info
- `GET /docs` — Swagger UI link
- `GET /redoc` — ReDoc link

**Components:**
- `ConfigDisplay` — read-only configuration display
- `ImportExportPanel` — file upload/download buttons
- `MaintenanceActions` — maintenance operation buttons with confirmations
- `ApiReference` — links to Swagger/ReDoc

---

## 6. API Routes for Dashboard

The dashboard needs a few additional API endpoints that don't exist yet:

```python
# Additional endpoints needed for dashboard

# Dashboard summary (combines multiple stats into one call)
@app.get("/api/dashboard/summary")
async def dashboard_summary():
    """Single endpoint for dashboard overview - reduces HTTP requests."""
    return {
        "health": {...},
        "stats": {...},
        "lifecycle": {...},
        "recent_activity": [...],
    }

# Activity log (for recent activity feed)
@app.get("/api/dashboard/activity")
async def activity_log(limit: int = 20):
    """Recent operations from access log + request logger."""
    ...

# Graph data optimized for vis-network
@app.get("/api/graph/vis-network")
async def graph_vis_network(max_nodes: int = 200):
    """Graph data formatted for vis-network.js directly."""
    # Returns { nodes: [...], edges: [...] } in vis-network format
    ...

# Batch operations for settings page
@app.post("/api/maintenance/rebuild-index")
async def rebuild_faiss_index():
    """Rebuild FAISS index from scratch."""
    ...

@app.post("/api/maintenance/vacuum")
async def vacuum_database():
    """VACUUM the SQLite database."""
    ...
```

---

## 7. Real-Time Updates Strategy

### Approach: SSE + Polling Hybrid

| Data | Update Method | Frequency | Why |
|------|--------------|-----------|-----|
| Stats counters | Polling | 30s | Low frequency, cached |
| Search results | On-demand | User-triggered | Immediate feedback |
| Graph | On-demand | User-triggered | Expensive to compute |
| Activity feed | SSE | Real-time | Live updates |
| Lifecycle | On-demand | User-triggered | User-initiated action |
| Health status | Polling | 10s | Quick health check |

### Implementation

```javascript
// SSE for activity feed
const events = new EventSource('/api/dashboard/activity/stream');
events.onmessage = (e) => {
    const activity = JSON.parse(e.data);
    Alpine.store('dashboard').addActivity(activity);
};

// Polling for stats
setInterval(async () => {
    const stats = await fetch('/stats').then(r => r.json());
    Alpine.store('dashboard').updateStats(stats);
}, 30000);
```

---

## 8. Styling & Theme

### Color Palette

```css
:root {
    /* Primary */
    --primary: #6366f1;        /* Indigo - Ariadne brand */
    --primary-dark: #4f46e5;

    /* Lifecycle tiers */
    --hot: #ef4444;            /* Red */
    --hot-bg: #fef2f2;
    --warm: #f59e0b;           /* Amber */
    --warm-bg: #fffbeb;
    --cold: #3b82f6;           /* Blue */
    --cold-bg: #eff6ff;

    /* Memory types */
    --semantic: #8b5cf6;       /* Purple */
    --episodic: #10b981;       /* Emerald */
    --procedural: #f97316;     /* Orange */
    --preference: #06b6d4;     /* Cyan */

    /* Community colors */
    --community-1: #6366f1;
    --community-2: #ec4899;
    --community-3: #14b8a6;
    --community-4: #f59e0b;
    --community-5: #8b5cf6;
}
```

### Dark Mode

```css
@media (prefers-color-scheme: dark) {
    :root {
        --bg: #0f172a;
        --surface: #1e293b;
        --text: #e2e8f0;
        --border: #334155;
    }
}
```

### Typography

- **Headings:** Inter (from Google Fonts)
- **Body:** System font stack (fast loading)
- **Code/Data:** JetBrains Mono (from Google Fonts)

---

## 9. Implementation Phases

### Phase 1: Foundation (1-2 days)
- [ ] Create `src/arriadne/dashboard/` module structure
- [ ] Add FastAPI router with static file serving
- [ ] Build `index.html` shell with sidebar navigation
- [ ] Implement `api.js` with fetch wrappers
- [ ] Add basic Alpine.js state management
- [ ] Style with Pico CSS + custom CSS

### Phase 2: Core Pages (2-3 days)
- [ ] Dashboard overview page (stat cards, lifecycle chart)
- [ ] Memory browser page (search, filter, list, detail)
- [ ] Settings page (config display, import/export)
- [ ] Status bar (health, counts, latency)

### Phase 3: Graph Visualization (1-2 days)
- [ ] vis-network integration
- [ ] Graph controls (layout, physics, depth, color)
- [ ] Node click → entity detail panel
- [ ] Community detection trigger + legend
- [ ] Export graph as PNG

### Phase 4: Advanced Features (1-2 days)
- [ ] Lifecycle management page (tier chart, prune preview)
- [ ] Statistics page (charts, metrics, access log)
- [ ] SSE for activity feed
- [ ] Dark mode toggle
- [ ] Responsive design for mobile

### Phase 5: Polish (1 day)
- [ ] Loading states and error handling
- [ ] Keyboard shortcuts
- [ ] Keyboard navigation (accessibility)
- [ ] Performance optimization (lazy loading, caching)
- [ ] Documentation

**Total estimated time: 6-10 days**

---

## 10. Integration with Existing Code

### server.py Changes

```python
# In create_app(), add dashboard static files serving:

from pathlib import Path

# Mount dashboard static files
dashboard_path = Path(__file__).parent / "dashboard" / "static"
if dashboard_path.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_path), html=True), name="dashboard")

# Redirect root to dashboard (optional)
@app.get("/ui")
async def dashboard_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard/")
```

### CLI Changes

```python
# In cmd_serve(), add dashboard flag:

serve_parser.add_argument("--no-dashboard", action="store_true",
                          help="Disable dashboard (API only)")
serve_parser.add_argument("--dashboard-port", type=int, default=None,
                          help="Separate port for dashboard (default: same as API)")
```

### pyproject.toml Changes

```toml
[project.optional-dependencies]
dashboard = []  # No additional dependencies needed!
# All JS libraries loaded from CDN
# For offline use, bundle into static files during build
```

---

## 11. Offline / Self-Contained Mode

For environments without internet access, libraries can be bundled:

```bash
# Download all CDN libraries into static/vendor/
mkdir -p src/arriadne/dashboard/static/vendor
curl -o src/arriadne/dashboard/static/vendor/alpine.min.js \
    https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js
curl -o src/arriadne/dashboard/static/vendor/htmx.min.js \
    https://unpkg.com/htmx.org@2.0.0/dist/htmx.min.js
curl -o src/arriadne/dashboard/static/vendor/vis-network.min.js \
    https://unpkg.com/vis-network/standalone/umd/vis-network.min.js
curl -o src/arriadne/dashboard/static/vendor/chart.min.js \
    https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js
curl -o src/arriadne/dashboard/static/vendor/pico.min.css \
    https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css
```

Then swap CDN URLs for local paths in the HTML.

---

## 12. Security Considerations

1. **API Key Passthrough** — Dashboard sends `Authorization` header with stored key
2. **CORS** — Already configured as `allow_origins=["*"]` in server.py
3. **XSS Prevention** — Alpine.js auto-escapes, Chart.js renders to canvas
4. **CSRF** — Not needed (state-changing operations use POST with API key)
5. **Content Security Policy** — Add CSP headers for CDN resources

---

## 13. Comparison with Competitors

| Feature | Ariadne Console | Mem0 Dashboard | Zep Dashboard | Letta/MemGPT |
|---------|:---:|:---:|:---:|:---:|
| Embedded in API server | ✅ | ✅ | ✅ | ❌ |
| No build step | ✅ | ❌ (React) | ❌ (React) | ❌ (React) |
| Knowledge graph viz | ✅ vis-network | ❌ | ✅ | ❌ |
| Lifecycle management | ✅ 3-tier | ❌ | ✅ | ✅ |
| Community detection UI | ✅ | ❌ | ✅ | ❌ |
| Importance scoring | ✅ 6-factor | ❌ | ❌ | ❌ |
| Temporal timeline | ✅ | ❌ | ✅ | ✅ |
| Import/Export UI | ✅ | ❌ | ❌ | ❌ |
| Dark mode | ✅ | ❌ | ✅ | ❌ |
| Mobile responsive | ✅ | ❌ | ❌ | ❌ |
| API docs link | ✅ Swagger | ✅ | ✅ | ❌ |
| Size (total) | ~500KB | ~2MB | ~3MB | ~2MB |

---

## Appendix A: vis-network Graph Data Format

The `/api/graph/vis-network` endpoint returns data optimized for vis-network:

```json
{
    "nodes": [
        {
            "id": "entity_1",
            "label": "Paris",
            "group": "location",
            "size": 12,
            "color": { "background": "#6366f1", "border": "#4f46e5" },
            "title": "Paris\nType: location\nMemories: 12\nCentrality: 0.92",
            "font": { "size": 14 }
        }
    ],
    "edges": [
        {
            "from": "entity_1",
            "to": "entity_2",
            "label": "located_in",
            "width": 2,
            "arrows": "to",
            "color": { "color": "#94a3b8" },
            "title": "located_in (weight: 0.8)"
        }
    ],
    "communities": {
        "1": { "name": "Travel", "color": "#6366f1" },
        "2": { "name": "Tech", "color": "#ec4899" }
    }
}
```

## Appendix B: Alpine.js State Structure

```javascript
Alpine.store('app', {
    // Navigation
    currentPage: 'dashboard',
    sidebarOpen: true,

    // Auth
    apiKey: localStorage.getItem('ariadne_api_key') || '',

    // Dashboard data
    health: { status: 'unknown', memories: 0, active: 0 },
    stats: { total: 0, by_type: {}, entities: 0, edges: 0 },
    lifecycle: { hot: 0, warm: 0, cold: 0 },
    activity: [],

    // Memory browser
    searchQuery: '',
    searchResults: [],
    searchLoading: false,
    selectedMemory: null,
    filters: { type: '', tier: '', importance: 0, entity: '' },
    page: 1,
    pageSize: 20,

    // Graph
    graphData: { nodes: [], edges: [] },
    selectedNode: null,
    graphLayout: 'force',
    physicsEnabled: true,

    // Theme
    darkMode: window.matchMedia('(prefers-color-scheme: dark)').matches,

    // Methods
    async search() { ... },
    async loadStats() { ... },
    async loadGraph() { ... },
    toggleTheme() { ... },
});
```

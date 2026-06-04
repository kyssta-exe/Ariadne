---
title: "REST API — Ariadne"
description: "Ariadne REST API reference. Store, search, traverse graphs, and manage memories over HTTP. Endpoints for /remember, /recall, /graph, /stats, and more."
---

# REST API

Ariadne includes a built-in HTTP server for accessing the memory system over REST. No external web framework needed -- just start the server and use standard HTTP requests.

## Starting the Server

```python
from arriadne import AriadneMemory
from arriadne.server import start_server

mem = AriadneMemory(db_path="memory.db")
start_server(mem, host="0.0.0.0", port=8420)
```

Or via CLI:

```bash
ariadne serve --db-path memory.db --port 8420
```

## Base URL

```
http://localhost:8420
```

## Authentication

By default, the API has no authentication (designed for local use). For production, set an API key:

```bash
ARIADNE_API_KEY=your-secret-key
```

When set, all requests must include:

```
Authorization: Bearer your-secret-key
```

## Endpoints

### Health Check

```
GET /health
```

Returns server status and basic stats.

**Response:**
```json
{
  "status": "ok",
  "uptime_seconds": 3600,
  "active_memories": 1520,
  "faiss_vectors": 1520
}
```

---

### Store a Memory

```
POST /remember
```

**Request Body:**
```json
{
  "content": "User prefers dark mode in all applications",
  "memory_type": "semantic",
  "importance": 0.8,
  "embedding": [0.1, 0.2, ...],
  "entities": ["user", "preferences"],
  "metadata": {"category": "ui"}
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `content` | string | yes | -- | Text content to store |
| `memory_type` | string | no | `"semantic"` | `semantic`, `episodic`, `procedural` |
| `importance` | float | no | `0.5` | Importance score (0.0-1.0) |
| `embedding` | float[] | no | `null` | 384-dim vector (or configured dimension) |
| `entities` | string[] | no | `null` | Entity names to associate |
| `metadata` | object | no | `null` | JSON-serializable metadata |

**Response:**
```json
{
  "memory_id": 42,
  "status": "created",
  "duplicate_of": null,
  "contradictions": []
}
```

---

### Search Memories

```
POST /recall
```

**Request Body:**
```json
{
  "query": "server configuration",
  "embedding": [0.1, 0.2, ...],
  "k": 10,
  "type_filter": "semantic",
  "importance_min": 0.5
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | yes | -- | Text query for FTS5 |
| `embedding` | float[] | no | `null` | Query embedding for vector search |
| `k` | int | no | `10` | Number of results |
| `type_filter` | string | no | `null` | Filter by memory type |
| `importance_min` | float | no | `null` | Minimum importance threshold |

**Response:**
```json
{
  "results": [
    {
      "id": 42,
      "content": "VPS has 4 cores, 8GB RAM",
      "memory_type": "semantic",
      "importance": 0.8,
      "score": 0.0324,
      "search_type": "hybrid",
      "created_at": 1717500000.0,
      "access_count": 3
    }
  ],
  "count": 1
}
```

---

### Delete a Memory

```
DELETE /memories/{id}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `id` | int | Memory ID |
| `hard` | bool (query) | `true` for permanent delete, `false` for soft-delete |

**Response:**
```json
{
  "forgotten": true
}
```

---

### Update a Memory

```
PATCH /memories/{id}
```

**Request Body:**
```json
{
  "content": "Updated content here",
  "importance": 0.9,
  "metadata": {"updated": true}
}
```

All fields optional. Only provided fields are updated.

**Response:**
```json
{
  "updated": true
}
```

---

### Knowledge Graph Traversal

```
POST /graph
```

**Request Body:**
```json
{
  "entity": "WebApp",
  "hops": 3,
  "edge_type": "depends_on"
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `entity` | string | yes | -- | Starting entity name |
| `hops` | int | no | `1` | Maximum traversal depth |
| `edge_type` | string | no | `null` | Filter by edge type |

**Response:**
```json
{
  "nodes": ["WebApp", "API", "Database", "PostgreSQL"],
  "edges": [
    {"source": "WebApp", "target": "API", "type": "depends_on", "weight": 1.0},
    {"source": "API", "target": "Database", "type": "depends_on", "weight": 1.0}
  ]
}
```

---

### Add Graph Edge

```
POST /edges
```

**Request Body:**
```json
{
  "source": "WebApp",
  "target": "API",
  "edge_type": "depends_on",
  "weight": 1.0
}
```

**Response:**
```json
{
  "created": true
}
```

---

### System Stats

```
GET /stats
```

**Response:**
```json
{
  "total_memories": 1520,
  "active_memories": 1490,
  "deleted_memories": 30,
  "by_type": {"semantic": 800, "episodic": 400, "procedural": 290},
  "total_entities": 350,
  "total_edges": 480,
  "faiss_vectors": 1490,
  "faiss_type": "IndexIVFFlat",
  "db_size_bytes": 5242880
}
```

---

## Python Client Example

```python
import requests

BASE = "http://localhost:8420"

# Store a memory
resp = requests.post(f"{BASE}/remember", json={
    "content": "Deploy to production via make deploy-prod",
    "memory_type": "procedural",
    "importance": 0.9,
    "entities": ["deployment"],
})
print(resp.json())
# {'memory_id': 1, 'status': 'created', ...}

# Search
resp = requests.post(f"{BASE}/recall", json={
    "query": "deploy",
    "k": 5,
})
for r in resp.json()["results"]:
    print(f"  [{r['search_type']}] {r['content'][:60]}")

# Graph traversal
resp = requests.post(f"{BASE}/graph", json={
    "entity": "WebApp",
    "hops": 3,
})
print(resp.json()["nodes"])
# ['WebApp', 'API', 'Database', 'PostgreSQL']
```

## cURL Examples

```bash
# Health check
curl http://localhost:8420/health

# Store a memory
curl -X POST http://localhost:8420/remember \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode", "memory_type": "semantic", "importance": 0.8}'

# Search
curl -X POST http://localhost:8420/recall \
  -H "Content-Type: application/json" \
  -d '{"query": "dark mode", "k": 5}'

# Get stats
curl http://localhost:8420/stats
```

## Error Responses

All endpoints return standard HTTP status codes:

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Invalid request body |
| 401 | Missing or invalid API key |
| 404 | Memory or entity not found |
| 500 | Internal server error |

Error body format:
```json
{
  "error": "Memory not found",
  "status_code": 404
}
```

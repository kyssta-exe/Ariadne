---
title: "Observability — Ariadne"
description: "Monitor Ariadne with built-in metrics, health checks, and logging. Track search latency, memory counts, and system health."
---

# Observability

Ariadne includes built-in metrics, health checks, and structured logging for monitoring your memory system in production.

## Health Check Endpoint

The REST server exposes a health check at `GET /health`:

```bash
curl http://localhost:8420/health
```

```json
{
  "status": "ok",
  "uptime_seconds": 3600,
  "active_memories": 1520,
  "faiss_vectors": 1520,
  "db_size_bytes": 5242880
}
```

Use this for load balancer health checks, container probes, or monitoring dashboards.

## Built-in Metrics

Ariadne tracks performance metrics automatically:

### Search Latency

Every search operation records latency. Access via Python:

```python
from arriadne import AriadneMemory

mem = AriadneMemory(db_path="memory.db")

# Run some searches...
mem.recall("query", k=5)

# Get metrics
metrics = mem.metrics()
print(metrics)
```

```python
{
    "search_count": 142,
    "search_latency_p50_ms": 0.31,
    "search_latency_p95_ms": 0.38,
    "search_latency_p99_ms": 0.52,
    "remember_count": 89,
    "remember_latency_p50_ms": 42.0,
    "dedup_rejected": 12,
    "contradictions_found": 3,
}
```

### Memory System Stats

```python
stats = mem.stats()
```

Returns comprehensive stats about the memory system:

| Field | Description |
|-------|-------------|
| `total_memories` | All memories including soft-deleted |
| `active_memories` | Currently active memories |
| `deleted_memories` | Soft-deleted memories |
| `by_type` | Count by memory type |
| `total_entities` | Knowledge graph entities |
| `total_edges` | Knowledge graph edges |
| `faiss_vectors` | Vectors in the FAISS index |
| `faiss_type` | Current index type (`IndexFlatIP` or `IndexIVFFlat`) |
| `db_size_bytes` | SQLite database file size |

## Structured Logging

Ariadne uses Python's standard `logging` module. Configure it like any other library:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("arriadne")
logger.setLevel(logging.DEBUG)
```

### Log Levels

| Level | What It Logs |
|-------|-------------|
| `DEBUG` | Search queries, index operations, dedup checks |
| `INFO` | Memory creation, graph operations, server start |
| `WARNING` | Fallback to keyword-only search, index upgrade |
| `ERROR` | Database errors, FAISS failures |

### Logging to File

```python
import logging

handler = logging.FileHandler("/var/log/ariadne.log")
handler.setLevel(logging.INFO)
logger = logging.getLogger("arriadne")
logger.addHandler(handler)
```

## Monitoring Search Performance

Track search latency over time to catch regressions:

```python
import time
from arriadne import AriadneMemory

mem = AriadneMemory(db_path="memory.db")

latencies = []
for _ in range(100):
    start = time.perf_counter()
    mem.recall("test query", k=5)
    elapsed = (time.perf_counter() - start) * 1000  # ms
    latencies.append(elapsed)

latencies.sort()
p50 = latencies[len(latencies) // 2]
p95 = latencies[int(len(latencies) * 0.95)]
print(f"p50: {p50:.2f}ms, p95: {p95:.2f}ms")
```

## Production Checklist

- [ ] Health check endpoint configured (`GET /health`)
- [ ] Log level set to `INFO` or `WARNING`
- [ ] API key set via `ARIADNE_API_KEY` (if exposed to network)
- [ ] Monitor `active_memories` count for growth
- [ ] Monitor `db_size_bytes` for disk usage
- [ ] Alert on search latency p95 > 5ms

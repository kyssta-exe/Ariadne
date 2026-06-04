"""
Ariadne Observability — Prometheus Metrics & Structured Logging

Provides:
- Prometheus-compatible /metrics endpoint
- Request metrics (latency histograms, counters, error rates)
- Search quality metrics
- Memory statistics
- Graph metrics
- SQLite WAL metrics
- Structured request logging with timing
"""

from __future__ import annotations

import logging
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("arriadne.observability")


# === Prometheus Text Format ===

class PrometheusMetric:
    """Base class for Prometheus metrics."""

    def __init__(self, name: str, help_text: str, metric_type: str = "gauge"):
        self.name = name
        self.help_text = help_text
        self.metric_type = metric_type
        self._value: float = 0.0
        self._labels: Dict[str, str] = {}

    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        self._value = value
        if labels:
            self._labels = labels

    def inc(self, amount: float = 1.0):
        self._value += amount

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help_text}"]
        lines.append(f"# TYPE {self.name} {self.metric_type}")
        if self._labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in self._labels.items())
            lines.append(f"{self.name}{{{label_str}}} {self._value}")
        else:
            lines.append(f"{self.name} {self._value}")
        return "\n".join(lines)


class PrometheusHistogram:
    """Prometheus histogram with predefined buckets."""

    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0]

    def __init__(self, name: str, help_text: str, buckets: Optional[List[float]] = None):
        self.name = name
        self.help_text = help_text
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts: List[int] = [0] * (len(self.buckets) + 1)  # +Inf bucket
        self._sum: float = 0.0
        self._count: int = 0

    def observe(self, value: float):
        self._sum += value
        self._count += 1
        for i, bucket in enumerate(self.buckets):
            if value <= bucket:
                self._counts[i] += 1
        # +Inf bucket always incremented
        self._counts[-1] += 1

    def render(self, suffix: str = "") -> str:
        full_name = f"{self.name}{suffix}"
        lines = [f"# HELP {full_name} {self.help_text}"]
        lines.append(f"# TYPE {full_name} histogram")
        for i, bucket in enumerate(self.buckets):
            lines.append(f'{full_name}_bucket{{le="{bucket}"}} {self._counts[i]}')
        lines.append(f'{full_name}_bucket{{le="+Inf"}} {self._counts[-1]}')
        lines.append(f"{full_name}_sum {self._sum}")
        lines.append(f"{full_name}_count {self._count}")
        return "\n".join(lines)

    @property
    def avg(self) -> float:
        return self._sum / max(1, self._count)

    @property
    def p50(self) -> float:
        return self._percentile(0.50)

    @property
    def p95(self) -> float:
        return self._percentile(0.95)

    @property
    def p99(self) -> float:
        return self._percentile(0.99)

    def _percentile(self, p: float) -> float:
        if self._count == 0:
            return 0.0
        target = int(self._count * p)
        running = 0
        for i, bucket in enumerate(self.buckets):
            running += self._counts[i]
            if running >= target:
                return bucket
        return self.buckets[-1] if self.buckets else 0.0


class PrometheusCounter:
    """Prometheus counter with label support."""

    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help_text = help_text
        self._values: Dict[str, float] = defaultdict(float)

    def inc(self, amount: float = 1.0, labels: Optional[Dict[str, str]] = None):
        key = self._label_key(labels)
        self._values[key] += amount

    def _label_key(self, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return ""
        return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help_text}"]
        lines.append(f"# TYPE {self.name} counter")
        if not self._values:
            lines.append(f"{self.name} 0")
        for label_key, value in self._values.items():
            if label_key:
                lines.append(f"{self.name}{{{label_key}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


class PrometheusGauge:
    """Prometheus gauge with label support."""

    def __init__(self, name: str, help_text: str):
        self.name = name
        self.help_text = help_text
        self._values: Dict[str, float] = {}

    def set(self, value: float, labels: Optional[Dict[str, str]] = None):
        key = self._label_key(labels)
        self._values[key] = value

    def _label_key(self, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return ""
        return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))

    def render(self) -> str:
        lines = [f"# HELP {self.name} {self.help_text}"]
        lines.append(f"# TYPE {self.name} gauge")
        if not self._values:
            lines.append(f"{self.name} 0")
        for label_key, value in self._values.items():
            if label_key:
                lines.append(f"{self.name}{{{label_key}}} {value}")
            else:
                lines.append(f"{self.name} {value}")
        return "\n".join(lines)


# === Structured Request Logger ===

class RequestLogger:
    """Structured request logger with timing and context."""

    def __init__(self, name: str = "arriadne.request"):
        self.logger = logging.getLogger(name)

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float,
        client_ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        api_key_id: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """Log a completed request with structured fields."""
        level = logging.WARNING if status_code >= 400 else logging.INFO
        if status_code >= 500:
            level = logging.ERROR

        extra = {
            "method": method,
            "path": path,
            "status_code": status_code,
            "latency_ms": round(latency_ms, 2),
        }
        if client_ip:
            extra["client_ip"] = client_ip
        if user_agent:
            extra["user_agent"] = user_agent
        if api_key_id:
            extra["api_key_id"] = api_key_id
        if error:
            extra["error"] = error

        self.logger.log(
            level,
            "%s %s -> %d (%.1fms)",
            method, path, status_code, latency_ms,
            extra=extra,
        )


# === Observability Collector ===

@dataclass
class SQLiteWAMetrics:
    """SQLite WAL mode metrics."""
    wal_pages: int = 0
    wal_checkpoint_count: int = 0
    journal_mode: str = "unknown"
    db_size_bytes: int = 0
    page_count: int = 0
    page_size: int = 0


class ObservabilityCollector:
    """
    Central observability collector with Prometheus metrics.

    Tracks:
    - Request metrics (count, latency, errors by endpoint)
    - Search metrics (latency, result count, quality)
    - Memory statistics
    - Graph metrics
    - SQLite WAL metrics
    - System uptime
    """

    def __init__(self):
        self.start_time = time.time()

        # === Prometheus Counters ===
        self.requests_total = PrometheusCounter(
            "arriadne_requests_total",
            "Total number of HTTP requests",
        )
        self.request_errors_total = PrometheusCounter(
            "arriadne_request_errors_total",
            "Total number of HTTP error responses",
        )
        self.searches_total = PrometheusCounter(
            "arriadne_searches_total",
            "Total number of search requests",
        )
        self.memories_stored_total = PrometheusCounter(
            "arriadne_memories_stored_total",
            "Total number of memories stored",
        )
        self.memories_deleted_total = PrometheusCounter(
            "arriadne_memories_deleted_total",
            "Total number of memories deleted",
        )

        # === Prometheus Histograms ===
        self.request_duration = PrometheusHistogram(
            "arriadne_request_duration_seconds",
            "HTTP request duration in seconds",
        )
        self.search_duration = PrometheusHistogram(
            "arriadne_search_duration_seconds",
            "Search request duration in seconds",
        )
        self.search_results = PrometheusHistogram(
            "arriadne_search_results_count",
            "Number of results returned by search",
            buckets=[0, 1, 2, 5, 10, 20, 50, 100],
        )

        # === Prometheus Gauges ===
        self.memories_total = PrometheusGauge(
            "arriadne_memories_total",
            "Total number of memories in the database",
        )
        self.entities_total = PrometheusGauge(
            "arriadne_entities_total",
            "Total number of entities in the graph",
        )
        self.edges_total = PrometheusGauge(
            "arriadne_edges_total",
            "Total number of edges in the graph",
        )
        self.communities_total = PrometheusGauge(
            "arriadne_communities_total",
            "Total number of detected communities",
        )
        self.uptime = PrometheusGauge(
            "arriadne_uptime_seconds",
            "Server uptime in seconds",
        )

        # === Legacy observability (JSON endpoint compatibility) ===
        self.request_count = 0
        self.error_count = 0
        self.total_latency_ms = 0.0
        self.latencies: List[float] = []
        self.endpoint_stats: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"count": 0, "latency_ms": 0.0, "errors": 0}
        )
        self.search_metrics: Dict[str, Any] = {
            "total_searches": 0,
            "avg_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "total_results_returned": 0,
        }
        self.sqlite_metrics = SQLiteWAMetrics()

    def record_request(self, endpoint: str, latency_ms: float, status: str = "ok"):
        """Record a completed request."""
        self.request_count += 1
        self.total_latency_ms += latency_ms
        self.latencies.append(latency_ms)
        self.endpoint_stats[endpoint]["count"] += 1
        self.endpoint_stats[endpoint]["latency_ms"] += latency_ms

        # Prometheus metrics
        self.requests_total.inc(labels={"endpoint": endpoint})
        self.request_duration.observe(latency_ms / 1000.0)

        if status == "error":
            self.error_count += 1
            self.endpoint_stats[endpoint]["errors"] += 1
            self.request_errors_total.inc(labels={"endpoint": endpoint})

    def record_search(self, latency_ms: float, result_count: int):
        """Record a search operation."""
        self.search_metrics["total_searches"] += 1
        self.search_metrics["total_results_returned"] += result_count

        # Prometheus metrics
        self.searches_total.inc()
        self.search_duration.observe(latency_ms / 1000.0)
        self.search_results.observe(float(result_count))

        # Update percentiles from all latencies
        if self.latencies:
            sorted_lat = sorted(self.latencies)
            n = len(sorted_lat)
            self.search_metrics["p50_latency_ms"] = sorted_lat[n // 2]
            self.search_metrics["p95_latency_ms"] = sorted_lat[int(n * 0.95)]
            self.search_metrics["p99_latency_ms"] = sorted_lat[min(int(n * 0.99), n - 1)]
            self.search_metrics["avg_latency_ms"] = self.total_latency_ms / self.request_count

    def update_memory_stats(self, stats: Dict[str, Any]):
        """Update memory statistics gauges."""
        self.memories_total.set(float(stats.get("total_memories", 0)))
        self.entities_total.set(float(stats.get("total_entities", 0)))
        self.edges_total.set(float(stats.get("total_edges", 0)))

    def update_community_stats(self, count: int):
        """Update community count gauge."""
        self.communities_total.set(float(count))

    def update_sqlite_metrics(self, conn: sqlite3.Connection):
        """Collect SQLite WAL metrics from a connection."""
        try:
            cursor = conn.execute("PRAGMA journal_mode")
            row = cursor.fetchone()
            self.sqlite_metrics.journal_mode = row[0] if row else "unknown"

            cursor = conn.execute("PRAGMA page_count")
            row = cursor.fetchone()
            self.sqlite_metrics.page_count = row[0] if row else 0

            cursor = conn.execute("PRAGMA page_size")
            row = cursor.fetchone()
            self.sqlite_metrics.page_size = row[0] if row else 0

            self.sqlite_metrics.db_size_bytes = (
                self.sqlite_metrics.page_count * self.sqlite_metrics.page_size
            )

            # WAL page count only available in WAL mode
            if self.sqlite_metrics.journal_mode == "wal":
                try:
                    cursor = conn.execute("PRAGMA wal_page_count")
                    row = cursor.fetchone()
                    self.sqlite_metrics.wal_pages = row[0] if row else 0
                except Exception:
                    self.sqlite_metrics.wal_pages = 0
            else:
                self.sqlite_metrics.wal_pages = 0
        except Exception as e:
            logger.debug("Failed to collect SQLite metrics: %s", e)

    def render_prometheus(self) -> str:
        """Render all metrics in Prometheus text exposition format."""
        self.uptime.set(time.time() - self.start_time)

        sections = [
            self.requests_total.render(),
            self.request_errors_total.render(),
            self.searches_total.render(),
            self.memories_stored_total.render(),
            self.memories_deleted_total.render(),
            self.request_duration.render(),
            self.search_duration.render(),
            self.search_results.render(),
            self.memories_total.render(),
            self.entities_total.render(),
            self.edges_total.render(),
            self.communities_total.render(),
            self.uptime.render(),
        ]

        # SQLite metrics
        sqlite_gauge = PrometheusGauge(
            "arriadne_sqlite_db_size_bytes",
            "SQLite database size in bytes",
        )
        sqlite_gauge.set(float(self.sqlite_metrics.db_size_bytes))
        sections.append(sqlite_gauge.render())

        page_count_gauge = PrometheusGauge(
            "arriadne_sqlite_page_count",
            "SQLite page count",
        )
        page_count_gauge.set(float(self.sqlite_metrics.page_count))
        sections.append(page_count_gauge.render())

        wal_pages_gauge = PrometheusGauge(
            "arriadne_sqlite_wal_pages",
            "SQLite WAL page count",
        )
        wal_pages_gauge.set(float(self.sqlite_metrics.wal_pages))
        sections.append(wal_pages_gauge.render())

        return "\n\n".join(sections) + "\n"

    def to_dict(self) -> Dict[str, Any]:
        """Render metrics as JSON dict (backward compatible)."""
        uptime = time.time() - self.start_time
        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": round(self.error_count / max(1, self.request_count), 4),
            "avg_latency_ms": round(self.total_latency_ms / max(1, self.request_count), 2),
            "search_metrics": dict(self.search_metrics),
            "sqlite_metrics": {
                "journal_mode": self.sqlite_metrics.journal_mode,
                "wal_pages": self.sqlite_metrics.wal_pages,
                "page_count": self.sqlite_metrics.page_count,
                "page_size": self.sqlite_metrics.page_size,
                "db_size_bytes": self.sqlite_metrics.db_size_bytes,
            },
            "endpoint_stats": {
                k: {
                    "count": v["count"],
                    "avg_latency_ms": round(v["latency_ms"] / max(1, v["count"]), 2),
                    "errors": v["errors"],
                }
                for k, v in self.endpoint_stats.items()
            },
        }


# Singleton collector
_collector: Optional[ObservabilityCollector] = None


def get_collector() -> ObservabilityCollector:
    """Get the global observability collector."""
    global _collector
    if _collector is None:
        _collector = ObservabilityCollector()
    return _collector

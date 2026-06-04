"""
Ariadne Python Client Library

HTTP client for the Ariadne REST API with:
- Automatic retry with exponential backoff
- Connection pooling
- Async support (aiohttp)
- Type-safe request/response models
- Streaming SSE support
- Local mode (direct AriadneDB access)
- Remote mode (HTTP API)
- Batch operations

Usage:
    from arriadne.client import AriadneClient

    # Remote mode (HTTP API)
    client = AriadneClient("http://localhost:8899")
    result = client.remember("Paris is the capital of France")
    results = client.recall("capital of France", limit=5)

    # Local mode (direct DB access)
    client = AriadneClient(local_db="/path/to/memory.db")
    result = client.remember("Paris is the capital of France")

    # Async usage
    async with AriadneClientAsync("http://localhost:8899") as client:
        results = await client.recall("capital of France")
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger("arriadne.client")


class AriadneClient:
    """
    Synchronous HTTP client for Ariadne REST API.

    Features:
    - Automatic retry with exponential backoff
    - Connection pooling via requests.Session
    - Timeout handling
    - Error normalization
    - Local mode (direct AriadneDB access)
    """

    def __init__(
        self,
        base_url: Optional[str] = "http://localhost:8899",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        local_db: Optional[str] = None,
    ):
        """
        Initialize the client.

        Args:
            base_url: HTTP server URL (for remote mode)
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Base delay between retries
            local_db: Path to local database (for local mode).
                      If set, operates directly on the DB without HTTP.
        """
        self._base_url = (base_url or "http://localhost:8899").rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._session = None
        self._local_mode = local_db is not None
        self._local_db = None
        self._local_mem = None

        if local_db:
            self._init_local(local_db)

    def _init_local(self, db_path: str):
        """Initialize local mode with direct DB access."""
        try:
            from arriadne.interface import AriadneMemory
            self._local_mem = AriadneMemory(db_path=db_path)
            self._local_db = self._local_mem._db
            logger.info("AriadneClient initialized in local mode: %s", db_path)
        except Exception as e:
            logger.error("Failed to initialize local mode: %s", e)
            raise

    @property
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    @property
    def is_local(self) -> bool:
        return self._local_mode

    @property
    def is_remote(self) -> bool:
        return not self._local_mode

    def _get_session(self):
        if self._session is None:
            import requests
            from requests.adapters import HTTPAdapter
            from urllib3.util.retry import Retry

            self._session = requests.Session()
            retry_strategy = Retry(
                total=self._max_retries,
                backoff_factor=self._retry_delay,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "POST", "PUT", "PATCH", "DELETE"],
            )
            adapter = HTTPAdapter(max_retries=retry_strategy, pool_maxsize=10)
            self._session.mount("http://", adapter)
            self._session.mount("https://", adapter)
        return self._session

    def _request(
        self, method: str, path: str, data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request with retry logic."""
        session = self._get_session()
        url = f"{self._base_url}{path}"

        for attempt in range(self._max_retries + 1):
            try:
                response = session.request(
                    method, url,
                    json=data,
                    headers=self._headers,
                    timeout=self._timeout,
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt == self._max_retries:
                    raise
                logger.warning("Request failed (attempt %d/%d): %s", attempt + 1, self._max_retries, e)
                time.sleep(self._retry_delay * (2 ** attempt))

        return {"error": "Max retries exceeded"}

    # === Health & Stats ===

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        if self._local_mode:
            try:
                stats = self._local_mem.stats()
                return {"status": "healthy", "memories": stats.get("total_memories", 0)}
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}
        return self._request("GET", "/health")

    def stats(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        if self._local_mode:
            return self._local_mem.stats()
        return self._request("GET", "/stats")

    # === Memory CRUD ===

    def remember(
        self,
        content: str,
        topic: str = "general",
        importance: int = 5,
        entities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Store a new memory."""
        if self._local_mode:
            return self._local_mem.store(
                content, topic=topic, importance=importance,
                entities=entities or [], metadata=metadata or {},
            )
        data = {
            "content": content,
            "topic": topic,
            "importance": importance,
            "entities": entities or [],
            "metadata": metadata or {},
        }
        return self._request("POST", "/memories", data)

    def get_memory(self, memory_id: str) -> Dict[str, Any]:
        """Get a memory by ID."""
        if self._local_mode:
            result = self._local_mem.get(memory_id)
            if not result:
                raise ValueError(f"Memory {memory_id} not found")
            return result
        return self._request("GET", f"/memories/{memory_id}")

    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        topic: Optional[str] = None,
        importance: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update an existing memory."""
        if self._local_mode:
            updates = {}
            if content is not None:
                updates["content"] = content
            if topic is not None:
                updates["topic"] = topic
            if importance is not None:
                updates["importance"] = importance
            if metadata is not None:
                updates["metadata"] = metadata
            return self._local_mem.update(memory_id, **updates)
        data = {}
        if content is not None:
            data["content"] = content
        if topic is not None:
            data["topic"] = topic
        if importance is not None:
            data["importance"] = importance
        if metadata is not None:
            data["metadata"] = metadata
        return self._request("PATCH", f"/memories/{memory_id}", data)

    def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        """Delete a memory."""
        if self._local_mode:
            success = self._local_mem.delete(memory_id)
            return {"deleted": success, "id": memory_id}
        return self._request("DELETE", f"/memories/{memory_id}")

    # === Search ===

    def search(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.5,
        use_hybrid: bool = True,
        include_graph: bool = False,
    ) -> Dict[str, Any]:
        """Search memories with hybrid retrieval."""
        if self._local_mode:
            t0 = time.monotonic()
            results = self._local_mem.search(
                query, limit=limit, threshold=threshold,
                use_hybrid=use_hybrid, include_graph=include_graph,
            )
            latency_ms = (time.monotonic() - t0) * 1000
            return {"results": results, "count": len(results), "latency_ms": round(latency_ms, 2)}
        data = {
            "query": query,
            "limit": limit,
            "threshold": threshold,
            "use_hybrid": use_hybrid,
            "include_graph": include_graph,
        }
        return self._request("POST", "/search", data)

    def recall(
        self,
        query: str,
        limit: int = 10,
        threshold: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Search and return just the results list (convenience method)."""
        result = self.search(query, limit=limit, threshold=threshold)
        return result.get("results", [])

    def search_stream(
        self,
        query: str,
        limit: int = 10,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream search results via SSE.

        Yields result dicts as they arrive.
        """
        if self._local_mode:
            results = self._local_mem.search(query, limit=limit)
            for i, result in enumerate(results):
                yield {"result": result, "rank": i + 1, "done": i == len(results) - 1}
            return

        import requests
        session = self._get_session()
        url = f"{self._base_url}/search/stream"
        params = {"query": query, "limit": limit}

        response = session.get(
            url, params=params, headers=self._headers,
            timeout=self._timeout, stream=True,
        )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith("data: "):
                data = json.loads(line[6:])
                if data.get("done"):
                    break
                yield data

    def batch_search(
        self,
        queries: List[str],
        limit: int = 10,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Execute multiple search queries in a single request."""
        if self._local_mode:
            t0 = time.monotonic()
            all_results = []
            for q in queries:
                results = self._local_mem.search(q, limit=limit, threshold=threshold)
                all_results.append({"query": q, "results": results, "count": len(results)})
            latency_ms = (time.monotonic() - t0) * 1000
            return {
                "queries": all_results,
                "total_results": sum(r["count"] for r in all_results),
                "latency_ms": round(latency_ms, 2),
            }
        data = {"queries": queries, "limit": limit, "threshold": threshold}
        return self._request("POST", "/batch/search", data)

    # === Conversation ===

    def extract(
        self,
        messages: List[Dict[str, str]],
        auto_store: bool = True,
    ) -> Dict[str, Any]:
        """Extract memories from a conversation."""
        if self._local_mode:
            extracted = self._local_mem.extract_from_conversation(messages)
            stored = []
            if auto_store:
                for m in extracted:
                    result = self._local_mem.store(
                        m.text, topic=m.topic, importance=m.importance,
                        entities=m.entities, metadata={"attributed_to": m.attributed_to},
                    )
                    stored.append(result)
            return {
                "extracted": len(extracted),
                "stored": len(stored),
                "memories": [
                    {"text": m.text, "topic": m.topic, "importance": m.importance, "entities": m.entities}
                    for m in extracted
                ],
            }
        data = {
            "messages": messages,
            "auto_store": auto_store,
        }
        return self._request("POST", "/extract", data)

    # === Graph ===

    def get_entities(
        self, entity_type: Optional[str] = None, limit: int = 100,
    ) -> Dict[str, Any]:
        """Get entities from the knowledge graph."""
        if self._local_mode:
            entities = self._local_mem.get_entities(entity_type=entity_type, limit=limit)
            return {"entities": entities, "count": len(entities)}
        params = {"limit": limit}
        if entity_type:
            params["entity_type"] = entity_type
        session = self._get_session()
        response = session.get(
            f"{self._base_url}/graph/entities",
            params=params, headers=self._headers, timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def get_entity_graph(self, entity_name: str, hops: int = 2) -> Dict[str, Any]:
        """Get graph connections for an entity."""
        if self._local_mode:
            return self._local_mem.graph_search(entity_name, hops=hops)
        return self._request("GET", f"/graph/entity/{entity_name}?hops={hops}")

    def connect_entities(
        self, source: str, target: str, relation: str = "related",
        weight: float = 1.0,
    ) -> Dict[str, Any]:
        """Connect two entities in the graph."""
        if self._local_mode:
            self._local_mem.graph_add_edge(source, target, relation, weight=weight)
            return {"connected": True, "source": source, "target": target, "relation": relation}
        data = {
            "source": source,
            "target": target,
            "relation": relation,
            "weight": weight,
        }
        return self._request("POST", "/graph/connect", data)

    # === Lifecycle ===

    def run_lifecycle(self) -> Dict[str, Any]:
        """Run memory lifecycle management."""
        if self._local_mode:
            if hasattr(self._local_mem, "run_lifecycle"):
                return self._local_mem.run_lifecycle()
            return {"error": "Lifecycle not configured"}
        return self._request("GET", "/lifecycle")

    def consolidate(self, method: str = "similarity", dry_run: bool = False) -> Dict[str, Any]:
        """Run memory consolidation."""
        if self._local_mode:
            if hasattr(self._local_mem, "_consolidator"):
                return self._local_mem.consolidate_with_llm(method=method, dry_run=dry_run)
            return {"error": "Consolidation not configured"}
        params = {"method": method, "dry_run": str(dry_run).lower()}
        session = self._get_session()
        response = session.post(
            f"{self._base_url}/consolidate",
            params=params, headers=self._headers, timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    # === Batch Operations ===

    def import_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Import multiple memories at once."""
        if self._local_mode:
            results = []
            for m in memories:
                result = self._local_mem.store(
                    m.get("content", ""),
                    topic=m.get("topic", "general"),
                    importance=m.get("importance", 5),
                    entities=m.get("entities", []),
                    metadata=m.get("metadata", {}),
                )
                results.append(result)
            return {"imported": len(results), "memories": results}
        return self._request("POST", "/import", memories)

    def export_memories(self, format: str = "json") -> Dict[str, Any]:
        """Export all memories."""
        if self._local_mode:
            all_memories = self._local_mem.search("*", limit=10000, threshold=0.0)
            return {"memories": all_memories, "count": len(all_memories), "format": format}
        session = self._get_session()
        response = session.get(
            f"{self._base_url}/export",
            params={"format": format},
            headers=self._headers, timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    # === Community (new) ===

    def get_communities(self) -> Dict[str, Any]:
        """Get all detected communities."""
        return self._request("GET", "/communities")

    def detect_communities(self) -> Dict[str, Any]:
        """Run community detection."""
        return self._request("POST", "/communities/detect")

    def get_community(self, community_id: int) -> Dict[str, Any]:
        """Get a specific community."""
        return self._request("GET", f"/communities/{community_id}")

    # === Scoring (new) ===

    def score_memory(self, memory_id: str) -> Dict[str, Any]:
        """Get importance score breakdown for a memory."""
        return self._request("GET", f"/memories/{memory_id}/score")

    def rank_memories(self, limit: int = 10) -> Dict[str, Any]:
        """Get top memories ranked by importance."""
        return self._request("GET", f"/memories/ranked?limit={limit}")

    # === Metrics ===

    def metrics(self, format: str = "json") -> Dict[str, Any]:
        """Get server metrics."""
        if self._local_mode:
            return {"error": "Metrics not available in local mode"}
        if format == "json":
            return self._request("GET", "/metrics?format=json")
        # For Prometheus format, return raw text
        session = self._get_session()
        response = session.get(
            f"{self._base_url}/metrics?format=prometheus",
            headers=self._headers, timeout=self._timeout,
        )
        response.raise_for_status()
        return {"text": response.text}

    # === Temporal Graph ===

    def get_temporal_facts(
        self, subject: Optional[str] = None, current_only: bool = True,
    ) -> Dict[str, Any]:
        """Query temporal facts."""
        params = {"current_only": str(current_only).lower()}
        if subject:
            params["subject"] = subject
        session = self._get_session()
        response = session.get(
            f"{self._base_url}/temporal/facts",
            params=params, headers=self._headers, timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()

    def add_temporal_fact(
        self, text: str, subject: str, predicate: str, obj: str,
    ) -> Dict[str, Any]:
        """Add a temporal fact."""
        data = {
            "text": text, "subject": subject,
            "predicate": predicate, "obj": obj,
        }
        return self._request("POST", "/temporal/facts", data)

    def get_timeline(self, subject: str) -> Dict[str, Any]:
        """Get timeline for a subject."""
        return self._request("GET", f"/temporal/timeline/{subject}")

    # === Close / Context Manager ===

    def close(self) -> None:
        """Close the HTTP session and local resources."""
        if self._session:
            self._session.close()
            self._session = None
        if self._local_mem:
            self._local_mem.close()
            self._local_mem = None

    def __enter__(self) -> AriadneClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()


class AriadneClientAsync:
    """
    Async HTTP client for Ariadne REST API.

    Usage:
        async with AriadneClientAsync("http://localhost:8899") as client:
            result = await client.remember("Paris is the capital of France")
            results = await client.recall("capital of France")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8899",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._session = None

    @property
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def _get_session(self):
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout),
                headers=self._headers,
            )
        return self._session

    async def _request(
        self, method: str, path: str, data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make an async HTTP request with retry logic."""
        session = await self._get_session()
        url = f"{self._base_url}{path}"

        for attempt in range(self._max_retries + 1):
            try:
                async with session.request(method, url, json=data) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except Exception as e:
                if attempt == self._max_retries:
                    raise
                import asyncio
                await asyncio.sleep(0.5 * (2 ** attempt))

        return {"error": "Max retries exceeded"}

    async def health(self) -> Dict[str, Any]:
        return await self._request("GET", "/health")

    async def stats(self) -> Dict[str, Any]:
        return await self._request("GET", "/stats")

    async def remember(
        self, content: str, topic: str = "general", importance: int = 5,
        entities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        data = {
            "content": content, "topic": topic, "importance": importance,
            "entities": entities or [], "metadata": metadata or {},
        }
        return await self._request("POST", "/memories", data)

    async def recall(
        self, query: str, limit: int = 10, threshold: float = 0.5,
    ) -> Dict[str, Any]:
        data = {
            "query": query, "limit": limit, "threshold": threshold,
            "use_hybrid": True, "include_graph": False,
        }
        return await self._request("POST", "/search", data)

    async def get_memory(self, memory_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/memories/{memory_id}")

    async def update_memory(
        self, memory_id: str, content: Optional[str] = None,
        topic: Optional[str] = None, importance: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        data = {}
        if content is not None:
            data["content"] = content
        if topic is not None:
            data["topic"] = topic
        if importance is not None:
            data["importance"] = importance
        if metadata is not None:
            data["metadata"] = metadata
        return await self._request("PATCH", f"/memories/{memory_id}", data)

    async def delete_memory(self, memory_id: str) -> Dict[str, Any]:
        return await self._request("DELETE", f"/memories/{memory_id}")

    async def search(
        self, query: str, limit: int = 10, threshold: float = 0.5,
    ) -> Dict[str, Any]:
        data = {
            "query": query, "limit": limit, "threshold": threshold,
            "use_hybrid": True, "include_graph": False,
        }
        return await self._request("POST", "/search", data)

    async def extract(
        self, messages: List[Dict[str, str]], auto_store: bool = True,
    ) -> Dict[str, Any]:
        data = {"messages": messages, "auto_store": auto_store}
        return await self._request("POST", "/extract", data)

    async def metrics(self) -> Dict[str, Any]:
        return await self._request("GET", "/metrics?format=json")

    async def close(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> AriadneClientAsync:
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

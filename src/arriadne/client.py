"""
Ariadne Python Client Library

Production-ready HTTP client for the Ariadne REST API with:
- Automatic retry with exponential backoff
- Connection pooling (requests.Session / httpx.AsyncClient)
- Async support (httpx)
- Type-safe request/response models
- Streaming SSE support
- Local mode (direct AriadneDB access)
- Remote mode (HTTP API)
- Auto-detect mode (local if same process, remote if URL provided)
- Multi-tenancy (X-Tenant-ID header or API key-based tenant)
- API key authentication (Bearer token)
- Key management (create, list, revoke, rotate)
- Specific exception classes for error handling
- Batch operations
- Context manager support

Usage:
    from arriadne.client import AriadneClient

    # Remote mode (HTTP API)
    client = AriadneClient("http://localhost:8899")
    result = client.remember("Paris is the capital of France")
    results = client.recall("capital of France", limit=5)

    # API key authentication
    client = AriadneClient("http://localhost:8899", api_key="your-api-key")
    result = client.remember("Authenticated memory")

    # Key management
    new_key = client.create_key(name="my-app", scopes=["read", "write"])
    keys = client.list_keys()
    client.revoke_key(key_id="key_123")
    rotated = client.rotate_key(key_id="key_123")

    # Local mode (direct DB access)
    client = AriadneClient(local_db="/path/to/memory.db")
    result = client.remember("Paris is the capital of France")

    # Auto-detect mode
    client = AriadneClient.auto_detect("http://localhost:8899", local_db="memory.db")

    # Async usage
    async with AriadneClientAsync("http://localhost:8899") as client:
        results = await client.recall("capital of France")

    # Multi-tenancy (via header - for token auth without API keys)
    client = AriadneClient("http://localhost:8899", tenant_id="team_alpha")
    client.remember("Team secret memory")  # automatically tagged with tenant
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger("arriadne.client")


# === Exception Classes ===

class AriadneError(Exception):
    """Base exception for Ariadne client errors."""
    pass


class AriadneConnectionError(AriadneError):
    """Failed to connect to the Ariadne server."""
    pass


class AriadneTimeoutError(AriadneError):
    """Request timed out."""
    pass


class AriadneAuthError(AriadneError):
    """Authentication failed (invalid or missing API key)."""
    pass


class AriadneNotFoundError(AriadneError):
    """Requested resource not found."""
    pass


class AriadneRateLimitError(AriadneError):
    """Rate limit exceeded."""
    pass


class AriadneServerError(AriadneError):
    """Server returned an error response."""
    pass


# === Sync Client ===

class AriadneClient:
    """
    Synchronous HTTP client for Ariadne REST API.

    Features:
    - Automatic retry with exponential backoff
    - Connection pooling via requests.Session
    - Timeout handling
    - Error normalization with specific exceptions
    - Local mode (direct AriadneDB access)
    - Auto-detect mode
    - Multi-tenancy support
    """

    def __init__(
        self,
        base_url: Optional[str] = "http://localhost:8899",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        local_db: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ):
        """
        Initialize the client.

        Args:
            base_url: HTTP server URL (for remote mode)
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
            retry_delay: Base delay between retries
            local_db: Path to local database (for local mode)
            tenant_id: Multi-tenant isolation key
        """
        self._base_url = (base_url or "http://localhost:8899").rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._tenant_id = tenant_id
        self._session = None
        self._local_mode = local_db is not None
        self._local_db = None
        self._local_mem = None

        if local_db:
            self._init_local(local_db)

    @classmethod
    def auto_detect(
        cls,
        base_url: Optional[str] = "http://localhost:8899",
        local_db: Optional[str] = None,
        **kwargs,
    ) -> "AriadneClient":
        """Create a client with auto-detection.

        If local_db is provided, operates in local mode (no HTTP overhead).
        Otherwise, operates in remote mode via HTTP API.
        """
        return cls(base_url=base_url, local_db=local_db, **kwargs)

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
            # When using API key auth, tenant is derived from the key
            # Don't send X-Tenant-ID header
        elif self._tenant_id:
            # Only send tenant header when not using API key auth
            headers["X-Tenant-ID"] = self._tenant_id
        return headers

    @property
    def is_local(self) -> bool:
        return self._local_mode

    @property
    def is_remote(self) -> bool:
        return not self._local_mode

    @property
    def tenant_id(self) -> Optional[str]:
        return self._tenant_id

    @tenant_id.setter
    def tenant_id(self, value: Optional[str]):
        self._tenant_id = value

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
        """Make an HTTP request with retry logic and error handling."""
        session = self._get_session()
        url = f"{self._base_url}{path}"

        try:
            response = session.request(
                method, url,
                json=data,
                headers=self._headers,
                timeout=self._timeout,
            )

            # Handle specific HTTP errors
            if response.status_code == 401:
                raise AriadneAuthError(f"Authentication failed: {response.text}")
            elif response.status_code == 404:
                raise AriadneNotFoundError(f"Not found: {path}")
            elif response.status_code == 429:
                raise AriadneRateLimitError("Rate limit exceeded")
            elif response.status_code >= 500:
                raise AriadneServerError(f"Server error ({response.status_code}): {response.text}")

            response.raise_for_status()
            return response.json()

        except AriadneError:
            raise
        except ImportError:
            raise AriadneConnectionError("requests library not installed. Install with: pip install requests")
        except Exception as e:
            if "ConnectionError" in type(e).__name__ or "ConnectionRefused" in str(e):
                raise AriadneConnectionError(f"Cannot connect to {self._base_url}: {e}")
            elif "Timeout" in type(e).__name__:
                raise AriadneTimeoutError(f"Request timed out after {self._timeout}s")
            raise

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
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Store a new memory."""
        tid = tenant_id or self._tenant_id
        if self._local_mode:
            kwargs = {}
            if tid:
                kwargs["tenant_id"] = tid
            return self._local_mem.store(
                content, topic=topic, importance=importance,
                entities=entities or [], metadata=metadata or {},
                **kwargs,
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
                raise AriadneNotFoundError(f"Memory {memory_id} not found")
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

    # === API Key Management ===

    def create_key(
        self,
        name: str,
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new API key.

        Args:
            name: Human-readable name for the key
            scopes: List of allowed scopes (e.g., ["read", "write", "admin"])
            expires_in_days: Number of days until key expires (None = no expiry)
            metadata: Optional metadata to attach to the key

        Returns:
            Dict with key details including the secret (shown once)

        Example:
            result = client.create_key(name="my-app", scopes=["read", "write"])
            secret = result["secret"]  # Save this - it won't be shown again
        """
        if self._local_mode:
            raise AriadneError("Key management is not available in local mode")
        data: Dict[str, Any] = {"name": name}
        if scopes:
            data["scopes"] = scopes
        if expires_in_days is not None:
            data["expires_in_days"] = expires_in_days
        if metadata:
            data["metadata"] = metadata
        return self._request("POST", "/keys", data)

    def list_keys(self, include_revoked: bool = False) -> Dict[str, Any]:
        """
        List all API keys.

        Args:
            include_revoked: If True, include revoked keys in results

        Returns:
            Dict with list of key metadata (secrets are never returned)
        """
        if self._local_mode:
            raise AriadneError("Key management is not available in local mode")
        params = f"?include_revoked={str(include_revoked).lower()}"
        return self._request("GET", f"/keys{params}")

    def revoke_key(self, key_id: str) -> Dict[str, Any]:
        """
        Revoke an API key.

        Args:
            key_id: The ID of the key to revoke

        Returns:
            Dict with revocation status
        """
        if self._local_mode:
            raise AriadneError("Key management is not available in local mode")
        return self._request("DELETE", f"/keys/{key_id}")

    def rotate_key(self, key_id: str) -> Dict[str, Any]:
        """
        Rotate an API key (generates a new secret, invalidates the old one).

        Args:
            key_id: The ID of the key to rotate

        Returns:
            Dict with new key details including the new secret (shown once)
        """
        if self._local_mode:
            raise AriadneError("Key management is not available in local mode")
        return self._request("POST", f"/keys/{key_id}/rotate")

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

    def __repr__(self) -> str:
        mode = "local" if self._local_mode else "remote"
        auth = "api_key" if self._api_key else "none"
        return f"AriadneClient(mode={mode}, url={self._base_url}, auth={auth})"


# === Async Client ===

class AriadneClientAsync:
    """
    Async HTTP client for Ariadne REST API.

    Uses httpx for async HTTP requests with:
    - Automatic retry with exponential backoff
    - Connection pooling
    - Timeout handling
    - Multi-tenancy support

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
        tenant_id: Optional[str] = None,
    ):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._max_retries = max_retries
        self._tenant_id = tenant_id
        self._client = None

    @property
    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
            # When using API key auth, tenant is derived from the key
            # Don't send X-Tenant-ID header
        elif self._tenant_id:
            # Only send tenant header when not using API key auth
            headers["X-Tenant-ID"] = self._tenant_id
        return headers

    @property
    def tenant_id(self) -> Optional[str]:
        return self._tenant_id

    @tenant_id.setter
    def tenant_id(self, value: Optional[str]):
        self._tenant_id = value

    async def _get_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                headers=self._headers,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._client

    async def _request(
        self, method: str, path: str, data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make an async HTTP request with retry logic."""
        client = await self._get_client()
        url = f"{self._base_url}{path}"

        for attempt in range(self._max_retries + 1):
            try:
                if method == "GET":
                    response = await client.get(url)
                elif method == "POST":
                    response = await client.post(url, json=data)
                elif method == "PATCH":
                    response = await client.patch(url, json=data)
                elif method == "DELETE":
                    response = await client.delete(url)
                else:
                    response = await client.request(method, url, json=data)

                if response.status_code == 401:
                    raise AriadneAuthError("Authentication failed")
                elif response.status_code == 404:
                    raise AriadneNotFoundError(f"Not found: {path}")
                elif response.status_code == 429:
                    raise AriadneRateLimitError("Rate limit exceeded")
                elif response.status_code >= 500:
                    raise AriadneServerError(f"Server error ({response.status_code})")

                response.raise_for_status()
                return response.json()

            except AriadneError:
                raise
            except Exception as e:
                pass  # store error for next attempt
                if attempt < self._max_retries:
                    import asyncio
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
                if "ConnectError" in type(e).__name__ or "connect" in str(e).lower():
                    raise AriadneConnectionError(f"Cannot connect to {self._base_url}")
                elif "TimeoutException" in type(e).__name__:
                    raise AriadneTimeoutError(f"Request timed out after {self._timeout}s")
                raise

        return {"error": "Max retries exceeded"}

    # === Core Methods ===

    async def health(self) -> Dict[str, Any]:
        return await self._request("GET", "/health")

    async def stats(self) -> Dict[str, Any]:
        return await self._request("GET", "/stats")

    async def remember(
        self,
        content: str,
        topic: str = "general",
        importance: int = 5,
        entities: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
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

    async def get_entities(
        self, entity_type: Optional[str] = None, limit: int = 100,
    ) -> Dict[str, Any]:
        params = f"?limit={limit}"
        if entity_type:
            params += f"&entity_type={entity_type}"
        return await self._request("GET", f"/graph/entities{params}")

    async def get_entity_graph(self, entity_name: str, hops: int = 2) -> Dict[str, Any]:
        return await self._request("GET", f"/graph/entity/{entity_name}?hops={hops}")

    async def connect_entities(
        self, source: str, target: str, relation: str = "related",
        weight: float = 1.0,
    ) -> Dict[str, Any]:
        data = {"source": source, "target": target, "relation": relation, "weight": weight}
        return await self._request("POST", "/graph/connect", data)

    async def run_lifecycle(self) -> Dict[str, Any]:
        return await self._request("GET", "/lifecycle")

    async def batch_search(
        self, queries: List[str], limit: int = 10, threshold: float = 0.5,
    ) -> Dict[str, Any]:
        data = {"queries": queries, "limit": limit, "threshold": threshold}
        return await self._request("POST", "/batch/search", data)

    async def import_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        client = await self._get_client()
        url = f"{self._base_url}/import"
        response = await client.post(url, json=memories)
        response.raise_for_status()
        return response.json()

    async def metrics(self) -> Dict[str, Any]:
        return await self._request("GET", "/metrics?format=json")

    async def get_communities(self) -> Dict[str, Any]:
        return await self._request("GET", "/communities")

    async def detect_communities(self) -> Dict[str, Any]:
        return await self._request("POST", "/communities/detect")

    async def score_memory(self, memory_id: str) -> Dict[str, Any]:
        return await self._request("GET", f"/memories/{memory_id}/score")

    async def rank_memories(self, limit: int = 10) -> Dict[str, Any]:
        return await self._request("GET", f"/memories/ranked?limit={limit}")

    # === API Key Management ===

    async def create_key(
        self,
        name: str,
        scopes: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new API key.

        Args:
            name: Human-readable name for the key
            scopes: List of allowed scopes (e.g., ["read", "write", "admin"])
            expires_in_days: Number of days until key expires (None = no expiry)
            metadata: Optional metadata to attach to the key

        Returns:
            Dict with key details including the secret (shown once)
        """
        data: Dict[str, Any] = {"name": name}
        if scopes:
            data["scopes"] = scopes
        if expires_in_days is not None:
            data["expires_in_days"] = expires_in_days
        if metadata:
            data["metadata"] = metadata
        return await self._request("POST", "/keys", data)

    async def list_keys(self, include_revoked: bool = False) -> Dict[str, Any]:
        """
        List all API keys.

        Args:
            include_revoked: If True, include revoked keys in results

        Returns:
            Dict with list of key metadata (secrets are never returned)
        """
        params = f"?include_revoked={str(include_revoked).lower()}"
        return await self._request("GET", f"/keys{params}")

    async def revoke_key(self, key_id: str) -> Dict[str, Any]:
        """
        Revoke an API key.

        Args:
            key_id: The ID of the key to revoke

        Returns:
            Dict with revocation status
        """
        return await self._request("DELETE", f"/keys/{key_id}")

    async def rotate_key(self, key_id: str) -> Dict[str, Any]:
        """
        Rotate an API key (generates a new secret, invalidates the old one).

        Args:
            key_id: The ID of the key to rotate

        Returns:
            Dict with new key details including the new secret (shown once)
        """
        return await self._request("POST", f"/keys/{key_id}/rotate")

    # === Close / Context Manager ===

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> AriadneClientAsync:
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    def __repr__(self) -> str:
        auth = "api_key" if self._api_key else "none"
        return f"AriadneClientAsync(url={self._base_url}, auth={auth})"

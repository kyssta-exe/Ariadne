---
title: "Integration Guide — Ariadne"
description: "Connect any AI agent to a self-hosted Ariadne memory server. Python clients, REST API, prompt-based integration, LangChain, LlamaIndex, CrewAI, multi-tenant setups, and security best practices."
---

# Integration Guide

Connect **any** AI agent to a self-hosted Ariadne memory server in under 5 minutes. This guide covers Python clients, raw REST API, prompt-based integration, framework adapters (LangChain, LlamaIndex, CrewAI), multi-agent setups, and production security.

---

## Quick Start

Three steps from zero to a connected agent:

### 1. Start the Server

```bash
pip install arriadne
ariadne serve --port 8899
```

The server starts in **open mode** (no auth required) by default. You'll see:

```
INFO:     Started server process [12345]
INFO:     Uvicorn running on http://0.0.0.0:8899
```

### 2. Create an API Key

In a separate terminal:

```bash
curl -X POST http://localhost:8899/auth/keys \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "my-agent",
    "tenant_id": "default",
    "scopes": ["read", "write"]
  }'
```

Response — **save the `key` value, it won't be shown again**:

```json
{
  "id": "a1b2c3d4e5f6",
  "key": "ak_a1b2c3d4e5f67890abcdef1234567890",
  "key_prefix": "ak_a1b2c3...",
  "agent_name": "my-agent",
  "tenant_id": "default",
  "scopes": ["read", "write"],
  "rate_limit_rpm": 120,
  "created_at": 1717526400.0,
  "expires_at": null
}
```

### 3. Connect Your Agent

```bash
# Store a memory
curl -X POST http://localhost:8899/memories \
  -H "Authorization: Bearer ak_a1b2c3d4e5f67890abcdef1234567890" \
  -H "Content-Type: application/json" \
  -d '{"content": "User prefers dark mode", "topic": "preferences", "importance": 7}'

# Search memories
curl -X POST http://localhost:8899/search \
  -H "Authorization: Bearer ak_a1b2c3d4e5f67890abcdef1234567890" \
  -H "Content-Type: application/json" \
  -d '{"query": "dark mode", "limit": 5}'
```

That's it. Your agent has persistent memory.

---

## Python Agents

Ariadne ships with a production-ready Python client. It handles retries, connection pooling, timeouts, and error normalization automatically.

### Install

```bash
pip install ariadne
```

### Sync Client

```python
from arriadne.client import AriadneClient

# Connect with API key
client = AriadneClient(
    base_url="http://localhost:8899",
    api_key="ak_a1b2c3d4e5f67890abcdef1234567890",
)

# Store a memory
result = client.remember(
    content="The API gateway listens on port 8443",
    topic="infrastructure",
    importance=8,
    entities=["API", "gateway", "port"],
    metadata={"source": "devops-agent"},
)
print(result)
# {"id": "42", "content": "The API gateway listens on port 8443", ...}

# Search memories
results = client.search(
    query="port 8443",
    limit=5,
    threshold=0.3,
)
for r in results["results"]:
    print(f"  score={r['score']:.3f} | {r['content'][:60]}")

# Quick recall (returns just the list)
memories = client.recall("gateway configuration", limit=3)

# Get a specific memory
memory = client.get_memory("42")

# Update a memory
client.update_memory("42", content="The API gateway listens on port 8443 (HTTPS)")

# Delete a memory
client.delete_memory("42")

# Extract memories from a conversation automatically
result = client.extract(messages=[
    {"role": "user", "content": "My favorite language is Python"},
    {"role": "assistant", "content": "Noted! I'll remember that."},
], auto_store=True)
print(result)
# {"extracted": 1, "stored": 1, "memories": [...]}

# Batch search (multiple queries, single request)
result = client.batch_search(
    queries=["server config", "database settings", "deployment process"],
    limit=5,
)
for q in result["queries"]:
    print(f"Query: {q['query']} → {q['count']} results")

# Check health
print(client.health())  # {"status": "healthy", "memories": 42}

# Check stats
print(client.stats())  # {"total_memories": 42, "active_memories": 40, ...}
```

### Async Client

For async frameworks (FastAPI, asyncio, etc.):

```python
import asyncio
from arriadne.client import AriadneClientAsync

async def main():
    async with AriadneClientAsync(
        base_url="http://localhost:8899",
        api_key="ak_a1b2c3d4e5f67890abcdef1234567890",
    ) as client:
        # Store
        await client.remember("Async memory from background task")

        # Search
        result = await client.search("background task", limit=5)
        for r in result["results"]:
            print(f"  {r['score']:.3f} | {r['content'][:60]}")

asyncio.run(main())
```

### Context Manager (Sync)

```python
with AriadneClient(
    base_url="http://localhost:8899",
    api_key="ak_a1b2c3d4e5f67890abcdef1234567890",
) as client:
    client.remember("Session-scoped memory")
    # Session is automatically closed when exiting the block
```

### Local Mode (No Server)

If your agent runs in the same process as Ariadne, skip the HTTP server entirely:

```python
from arriadne.client import AriadneClient

client = AriadneClient(local_db="agent_memory.db")

# Same API, but hits SQLite directly — no HTTP overhead
client.remember("Direct local memory")
results = client.search("direct local")
```

### Auto-Detect Mode

Try local first, fall back to remote:

```python
client = AriadneClient.auto_detect(
    base_url="http://localhost:8899",
    local_db="agent_memory.db",
)
# Uses local_db if the file exists; otherwise connects via HTTP
```

### Error Handling

The client raises specific exceptions you can catch:

```python
from arriadne.client import (
    AriadneClient,
    AriadneError,
    AriadneConnectionError,
    AriadneAuthError,
    AriadneNotFoundError,
    AriadneRateLimitError,
    AriadneServerError,
    AriadneTimeoutError,
)

client = AriadneClient("http://localhost:8899", api_key="ak_...")

try:
    client.remember("Important memory")
except AriadneAuthError:
    print("Invalid API key — check your credentials")
except AriadneRateLimitError:
    print("Too many requests — slow down")
except AriadneConnectionError:
    print("Cannot reach Ariadne server — is it running?")
except AriadneServerError as e:
    print(f"Server error: {e}")
except AriadneError as e:
    print(f"General Ariadne error: {e}")
```

---

## Non-Python Agents (REST API)

Any language that can make HTTP requests can connect to Ariadne. The entire API is REST with JSON payloads.

### Authentication

All authenticated requests require a Bearer token in the `Authorization` header:

```
Authorization: Bearer ak_<your-key>
```

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/memories` | Store a memory |
| `GET` | `/memories/{id}` | Get a memory by ID |
| `PATCH` | `/memories/{id}` | Update a memory |
| `DELETE` | `/memories/{id}` | Delete a memory |
| `POST` | `/search` | Search memories (hybrid) |
| `GET` | `/search/stream` | Stream search results (SSE) |
| `POST` | `/batch/search` | Multiple queries in one request |
| `POST` | `/extract` | Extract memories from a conversation |
| `GET` | `/graph/entities` | List entities |
| `GET` | `/graph/entity/{name}` | Get entity connections |
| `POST` | `/graph/connect` | Connect two entities |
| `POST` | `/communities/detect` | Run community detection |
| `GET` | `/communities` | List communities |
| `GET` | `/health` | Health check (no auth) |
| `GET` | `/stats` | Memory statistics |
| `GET` | `/metrics` | Prometheus/JSON metrics |

All authenticated endpoints also support versioned paths: `/api/v1/memories`, `/api/v1/search`, etc.

### Store a Memory

```bash
curl -X POST http://localhost:8899/memories \
  -H "Authorization: Bearer ak_a1b2c3d4e5f67890abcdef1234567890" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "The database runs on PostgreSQL 16",
    "topic": "infrastructure",
    "importance": 8,
    "entities": ["PostgreSQL", "database", "version"],
    "metadata": {"source": "deployment-agent", "env": "production"}
  }'
```

**Response:**

```json
{
  "id": "42",
  "content": "The database runs on PostgreSQL 16",
  "topic": "infrastructure",
  "importance": 8,
  "entities": ["PostgreSQL", "database", "version"],
  "metadata": {"source": "deployment-agent", "env": "production"},
  "created_at": 1717526400.0,
  "tenant_id": "default"
}
```

### Search Memories

```bash
curl -X POST http://localhost:8899/search \
  -H "Authorization: Bearer ak_a1b2c3d4e5f67890abcdef1234567890" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "what database do we use",
    "limit": 5,
    "threshold": 0.3,
    "use_hybrid": true,
    "include_graph": false
  }'
```

**Response:**

```json
{
  "results": [
    {
      "id": "42",
      "content": "The database runs on PostgreSQL 16",
      "topic": "infrastructure",
      "score": 0.87,
      "search_type": "hybrid",
      "entities": ["PostgreSQL", "database", "version"]
    }
  ],
  "count": 1,
  "latency_ms": 3.2
}
```

### Extract from Conversation

Send a conversation and Ariadne automatically extracts and stores memories:

```bash
curl -X POST http://localhost:8899/extract \
  -H "Authorization: Bearer ak_a1b2c3d4e5f67890abcdef1234567890" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I deployed v2.3 to staging last Friday"},
      {"role": "assistant", "content": "Got it. v2.3 is on staging."},
      {"role": "user", "content": "The deploy went smoothly but the API was slow"},
      {"role": "assistant", "content": "I'll remember the performance issue."}
    ],
    "auto_store": true
  }'
```

**Response:**

```json
{
  "extracted": 3,
  "stored": 3,
  "memories": [
    {"text": "User deployed v2.3 to staging last Friday", "topic": "deployment", "importance": 7, "entities": ["v2.3", "staging"]},
    {"text": "API was slow after v2.3 deployment", "topic": "performance", "importance": 8, "entities": ["API", "v2.3", "performance"]}
  ]
}
```

### Stream Search Results (SSE)

```bash
curl -N "http://localhost:8899/search/stream?query=deployment&limit=5" \
  -H "Authorization: Bearer ak_a1b2c3d4e5f67890abcdef1234567890"
```

**Response (streamed):**

```
data: {"result": {"id": "42", "content": "Deployed v2.3 to staging...", "score": 0.92}, "rank": 1, "done": false}

data: {"result": {"id": "38", "content": "Last deploy was Friday...", "score": 0.81}, "rank": 2, "done": true}
```

### Batch Search

```bash
curl -X POST http://localhost:8899/batch/search \
  -H "Authorization: Bearer ak_a1b2c3d4e5f67890abcdef1234567890" \
  -H "Content-Type: application/json" \
  -d '{
    "queries": ["server config", "database settings", "deployment process"],
    "limit": 5,
    "threshold": 0.3
  }'
```

### Health Check (No Auth Required)

```bash
curl http://localhost:8899/health
```

```json
{"status": "healthy", "memories": 42, "active_memories": 40, "uptime": 1717526400.0}
```

### Complete Example: Node.js Agent

```javascript
const ARIADNE_URL = "http://localhost:8899";
const API_KEY = "ak_a1b2c3d4e5f67890abcdef1234567890";

const headers = {
  "Authorization": `Bearer ${API_KEY}`,
  "Content-Type": "application/json",
};

async function remember(content, topic = "general") {
  const res = await fetch(`${ARIADNE_URL}/memories`, {
    method: "POST",
    headers,
    body: JSON.stringify({ content, topic, importance: 5 }),
  });
  return res.json();
}

async function recall(query, limit = 5) {
  const res = await fetch(`${ARIADNE_URL}/search`, {
    method: "POST",
    headers,
    body: JSON.stringify({ query, limit, threshold: 0.3 }),
  });
  return res.json();
}

// Usage
const result = await remember("Node.js agent is active", "agent-status");
const memories = await recall("agent status", 3);
console.log("Found:", memories.count, "memories");
```

---

## Prompt-Based Integration

You can give **any** LLM agent instructions to use Ariadne's REST API directly. This works with agents that have HTTP/curl tools (AutoGPT, OpenDevin, Claude Code, Hermes, etc.).

### System Prompt Template

Copy this into your agent's system prompt or tool configuration:

```
You have access to a persistent memory system via the Ariadne API.

SERVER: http://localhost:8899
AUTH: Bearer ak_a1b2c3d4e5f67890abcdef1234567890

## Store a Memory
POST /memories
{
  "content": "<what you want to remember>",
  "topic": "<category: general|infrastructure|preferences|bugs|...>",
  "importance": <1-10, where 10 is critical>,
  "entities": ["<named entities>"],
  "metadata": {"source": "your-agent-name"}
}

## Search Memories
POST /search
{
  "query": "<what you're looking for>",
  "limit": 5,
  "threshold": 0.3
}

## Extract from Conversation
POST /extract
{
  "messages": [{"role": "user|assistant", "content": "..."}],
  "auto_store": true
}

## Rules:
- Always search BEFORE answering user questions (you might already know the answer)
- Store important facts, preferences, and decisions immediately
- Use importance=8-10 for critical info, 5-7 for normal, 1-4 for minor
- Include entities for better graph connectivity
- Set metadata.source to your agent name for traceability
```

### Example: Hermes Agent with Ariadne

For Hermes Agent, add this to your agent's configuration:

```yaml
# In your agent config
tools:
  - name: ariadne_remember
    description: "Store a memory in Ariadne"
    method: POST
    url: "http://localhost:8899/memories"
    headers:
      Authorization: "Bearer ak_a1b2c3d4e5f67890abcdef1234567890"
    body:
      content: "{{input}}"
      topic: "general"
      importance: 5

  - name: ariadne_recall
    description: "Search Ariadne memories"
    method: POST
    url: "http://localhost:8899/search"
    headers:
      Authorization: "Bearer ak_a1b2c3d4e5f67890abcdef1234567890"
    body:
      query: "{{input}}"
      limit: 5
      threshold: 0.3
```

---

## LangChain Integration

Ariadne integrates with LangChain as a custom retriever and memory backend.

### As a Retriever

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from arriadne.client import AriadneClient
from typing import List

class AriadneRetriever(BaseRetriever):
    """LangChain retriever backed by Ariadne memory."""

    base_url: str = "http://localhost:8899"
    api_key: str = ""
    search_limit: int = 5
    threshold: float = 0.3

    def _get_relevant_documents(self, query: str) -> List[Document]:
        client = AriadneClient(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        result = client.search(
            query=query,
            limit=self.search_limit,
            threshold=self.threshold,
        )
        return [
            Document(
                page_content=r["content"],
                metadata={
                    "id": r["id"],
                    "score": r["score"],
                    "topic": r.get("topic", "general"),
                    "entities": r.get("entities", []),
                },
            )
            for r in result.get("results", [])
        ]

# Usage
retriever = AriadneRetriever(
    base_url="http://localhost:8899",
    api_key="ak_a1b2c3d4e5f67890abcdef1234567890",
    search_limit=10,
)

docs = retriever.get_relevant_documents("database configuration")
for doc in docs:
    print(f"  [{doc.metadata['score']:.3f}] {doc.page_content[:80]}")
```

### As a Chat Message History (Memory)

```python
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from arriadne.client import AriadneClient
from typing import List
import json

class AriadneChatHistory(BaseChatMessageHistory):
    """LangChain chat memory backed by Ariadne."""

    def __init__(
        self,
        session_id: str,
        base_url: str = "http://localhost:8899",
        api_key: str = "",
    ):
        self.session_id = session_id
        self.client = AriadneClient(base_url=base_url, api_key=api_key)

    @property
    def messages(self) -> List[BaseMessage]:
        result = self.client.search(
            query=f"session {self.session_id}",
            limit=100,
            threshold=0.0,
        )
        messages = []
        for r in result.get("results", []):
            if r.get("metadata", {}).get("session_id") == self.session_id:
                role = r["metadata"].get("role", "user")
                if role == "user":
                    messages.append(HumanMessage(content=r["content"]))
                else:
                    messages.append(AIMessage(content=r["content"]))
        return messages

    def add_user_message(self, content: str) -> None:
        self.client.remember(
            content=content,
            topic="conversation",
            importance=5,
            metadata={"session_id": self.session_id, "role": "user"},
        )

    def add_ai_message(self, content: str) -> None:
        self.client.remember(
            content=content,
            topic="conversation",
            importance=5,
            metadata={"session_id": self.session_id, "role": "assistant"},
        )

    def clear(self) -> None:
        pass  # Memories are managed by lifecycle/eviction
```

### As a LangChain Tool

```python
from langchain_core.tools import tool

@tool
def search_memories(query: str) -> str:
    """Search the agent's persistent memory for relevant information."""
    client = AriadneClient(
        base_url="http://localhost:8899",
        api_key="ak_a1b2c3d4e5f67890abcdef1234567890",
    )
    result = client.search(query, limit=5, threshold=0.3)
    memories = result.get("results", [])
    if not memories:
        return "No relevant memories found."
    lines = []
    for i, m in enumerate(memories, 1):
        lines.append(f"{i}. [{m['score']:.2f}] {m['content'][:200]}")
    return "\n".join(lines)

@tool
def store_memory(content: str, importance: int = 5) -> str:
    """Store important information in persistent memory."""
    client = AriadneClient(
        base_url="http://localhost:8899",
        api_key="ak_a1b2c3d4e5f67890abcdef1234567890",
    )
    result = client.remember(content=content, importance=importance)
    return f"Memory stored with id={result['id']}"
```

---

## LlamaIndex Integration

### Custom LlamaIndex Retriever

```python
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import QueryBundle, NodeWithScore, TextNode
from arriadne.client import AriadneClient
from typing import List

class AriadneRetriever(BaseRetriever):
    """LlamaIndex retriever backed by Ariadne."""

    base_url: str = "http://localhost:8899"
    api_key: str = ""
    search_limit: int = 10
    threshold: float = 0.3

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        client = AriadneClient(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        result = client.search(
            query=query_bundle.query_str,
            limit=self.search_limit,
            threshold=self.threshold,
        )
        return [
            NodeWithScore(
                node=TextNode(
                    text=r["content"],
                    metadata={
                        "id": r["id"],
                        "topic": r.get("topic", "general"),
                        "entities": r.get("entities", []),
                    },
                ),
                score=r["score"],
            )
            for r in result.get("results", [])
        ]

# Usage
retriever = AriadneRetriever(
    base_url="http://localhost:8899",
    api_key="ak_a1b2c3d4e5f67890abcdef1234567890",
)

from llama_index.core import QueryBundle

results = retriever.retrieve(
    QueryBundle(query_str="what is the deployment process")
)
for node in results:
    print(f"  [{node.score:.3f}] {node.text[:80]}")
```

### As a LlamaIndex Knowledge Base

```python
from arriadne.client import AriadneClient

client = AriadneClient(
    base_url="http://localhost:8899",
    api_key="ak_a1b2c3d4e5f67890abcdef1234567890",
)

# Populate the knowledge base
client.remember(
    "CI pipeline: lint → test → build → deploy to staging → manual approval → production",
    topic="deployment", importance=9,
    entities=["CI", "pipeline", "deployment"],
)
client.remember(
    "Production database: PostgreSQL 16 on AWS RDS, multi-AZ, encrypted at rest",
    topic="infrastructure", importance=9,
    entities=["PostgreSQL", "AWS RDS", "production"],
)

# Query from LlamaIndex
results = client.search("how to deploy to production", limit=3)
```

---

## CrewAI and Custom Frameworks

### CrewAI Agent with Ariadne Memory

```python
from crewai import Agent, Task, Crew
from arriadne.client import AriadneClient

# Initialize the shared memory client
memory = AriadneClient(
    base_url="http://localhost:8899",
    api_key="ak_a1b2c3d4e5f67890abcdef1234567890",
)

# Before a task: search for context
def get_context(query: str) -> str:
    results = memory.search(query, limit=5, threshold=0.3)
    if not results["results"]:
        return "No prior context found."
    context = []
    for r in results["results"]:
        context.append(f"- {r['content']}")
    return "\n".join(context)

# After a task: store findings
def store_findings(content: str, topic: str = "general"):
    memory.remember(content=content, topic=topic, importance=6)

# Create an agent with memory context
researcher = Agent(
    role="Research Analyst",
    goal="Find and analyze information",
    backstory="""You are a research analyst.
When starting work, check the team's memory for prior findings.
After completing analysis, store your findings in memory.""",
)

# Create a task with memory context injection
task = Task(
    description=f"""
Research the following topic and provide a detailed analysis.

Prior context:
{get_context('recent analysis of system performance')}

After completing your analysis, summarize your key findings.
""",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()

# Store the results
store_findings(
    content=f"Analysis completed: {result}",
    topic="analysis",
)
```

### Generic Framework Integration Pattern

Any framework can integrate with this pattern:

```python
from arriadne.client import AriadneClient

class AgentMemory:
    """Generic memory wrapper for any agent framework."""

    def __init__(self, base_url="http://localhost:8899", api_key="", agent_name="agent"):
        self.client = AriadneClient(base_url=base_url, api_key=api_key)
        self.agent_name = agent_name

    def before_think(self, query: str, limit: int = 5) -> list:
        """Call before your agent thinks. Returns relevant memories."""
        result = self.client.search(query, limit=limit, threshold=0.3)
        return result.get("results", [])

    def after_act(self, action: str, result: str, importance: int = 5):
        """Call after your agent acts. Stores what happened."""
        self.client.remember(
            content=f"{action}: {result}",
            topic="agent-actions",
            importance=importance,
            metadata={"agent": self.agent_name},
        )

    def remember(self, content: str, topic: str = "general", importance: int = 5):
        """Store an explicit memory."""
        self.client.remember(
            content=content, topic=topic, importance=importance,
            metadata={"agent": self.agent_name},
        )

# Usage in any framework
mem = AgentMemory(agent_name="my-custom-agent")

# Before doing work
context = mem.before_think("user's database preferences")
# Inject context into your agent's prompt...

# After completing work
mem.after_act("configured database", "Set up PostgreSQL connection for production")
```

---

## Multi-Agent Setups

Ariadne supports **multi-tenant isolation** — different agents can have isolated memory spaces within the same server.

### How It Works

Each API key is bound to a `tenant_id`. When an agent authenticates with its key, all memories are automatically scoped to that tenant. Search results are filtered to the tenant's scope.

```
┌─────────────────────────────────────────────┐
│              Ariadne Server                  │
│                                             │
│  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Tenant: alpha │  │ Tenant: beta         │ │
│  │ (Research)   │  │ (Deployment)         │ │
│  │              │  │                      │ │
│  │ memories     │  │ memories             │ │
│  │ entities     │  │ entities             │ │
│  │ graph        │  │ graph                │ │
│  └──────────────┘  └──────────────────────┘ │
│                                             │
│  API Key: ak_alpha_... → tenant: alpha      │
│  API Key: ak_beta_...  → tenant: beta       │
│  API Key: ak_admin_... → tenant: default    │
└─────────────────────────────────────────────┘
```

### Creating Tenant-Scoped Keys

```bash
# Research agent → tenant "research"
curl -X POST http://localhost:8899/auth/keys \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "research-agent",
    "tenant_id": "research",
    "scopes": ["read", "write"]
  }'

# Deployment agent → tenant "devops"
curl -X POST http://localhost:8899/auth/keys \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "devops-agent",
    "tenant_id": "devops",
    "scopes": ["read", "write"]
  }'

# Admin key → can manage everything
curl -X POST http://localhost:8899/auth/keys \
  -H "Content-Type: application/json" \
  -d '{
    "agent_name": "admin",
    "tenant_id": "default",
    "scopes": ["read", "write", "admin"]
  }'
```

### Python: Multiple Agents, One Server

```python
from arriadne.client import AriadneClient

# Research agent
research = AriadneClient(
    base_url="http://localhost:8899",
    api_key="ak_alpha_...",  # Bound to tenant "research"
)
research.remember("Found that latency increases with index size", topic="research")
# Only visible to "research" tenant

# DevOps agent
devops = AriadneClient(
    base_url="http://localhost:8899",
    api_key="ak_beta_...",   # Bound to tenant "devops"
)
devops.remember("Deploy pipeline runs on GitHub Actions", topic="devops")
# Only visible to "devops" tenant

# Each agent searches its own isolated memory
research_results = research.search("latency")
devops_results = devops.search("deploy pipeline")
# They don't see each other's memories
```

### Non-Python: Multi-Tenant via REST

```bash
# Research agent stores a memory
curl -X POST http://localhost:8899/memories \
  -H "Authorization: Bearer ak_alpha_..." \
  -H "Content-Type: application/json" \
  -d '{"content": "Research finding: model accuracy improves with more data", "topic": "research"}'

# DevOps agent stores a memory
curl -X POST http://localhost:8899/memories \
  -H "Authorization: Bearer ak_beta_..." \
  -H "Content-Type: application/json" \
  -d '{"content": "Deploy command: make deploy-prod ENV=production", "topic": "devops"}'

# Research agent searches — only sees research memories
curl -X POST http://localhost:8899/search \
  -H "Authorization: Bearer ak_alpha_..." \
  -H "Content-Type: application/json" \
  -d '{"query": "model accuracy", "limit": 5}'
```

### Cross-Tenant Access (Admin)

Admin keys can access all tenants:

```python
admin = AriadneClient(
    base_url="http://localhost:8899",
    api_key="ak_admin_...",  # tenant="default", scopes=["read","write","admin"]
)
# Admin can search across all tenants and manage keys
all_keys = admin.list_keys()  # Requires admin scope
```

---

## Security Best Practices

### Key Management

**Rotate keys regularly.** The `ak_` prefix + 32-char hex gives 128 bits of entropy, but regular rotation limits exposure.

```python
# Create a key
new_key = client.create_key(
    name="production-agent",
    scopes=["read", "write"],
    expires_in_days=90,  # Auto-expires in 90 days
)
print(f"Save this key: {new_key['key']}")

# Rotate an existing key (generates new secret, invalidates old)
rotated = client.rotate_key(key_id="a1b2c3d4e5f6")
print(f"New key: {rotated['key']}")

# List keys (never exposes secrets, only prefixes)
keys = client.list_keys()
for k in keys["keys"]:
    print(f"  {k['key_prefix']} | {k['agent_name']} | scopes={k['scopes']}")

# Revoke a compromised key
client.revoke_key(key_id="a1b2c3d4e5f6")
```

**REST API for key management:**

```bash
# List keys (admin only)
curl http://localhost:8899/auth/keys \
  -H "Authorization: Bearer ak_admin_..."

# Rotate a key
curl -X POST http://localhost:8899/auth/keys/a1b2c3d4e5f6/rotate \
  -H "Authorization: Bearer ak_admin_..."

# Revoke a key
curl -X DELETE http://localhost:8899/auth/keys/a1b2c3d4e5f6 \
  -H "Authorization: Bearer ak_admin_..."
```

### Scope Control

Use the minimum scopes needed for each agent:

| Scope | Allows |
|-------|--------|
| `read` | `GET` requests (search, stats, health, list) |
| `write` | `POST`/`PATCH`/`DELETE` (store, update, delete, extract) |
| `admin` | Auth endpoints (create/revoke/rotate keys) |

**Read-only agent** (can search but not modify):

```bash
curl -X POST http://localhost:8899/auth/keys \
  -d '{"agent_name": "reader", "tenant_id": "default", "scopes": ["read"]}'
```

**Write-only agent** (can store but not read back — unusual but possible):

```bash
curl -X POST http://localhost:8899/auth/keys \
  -d '{"agent_name": "logger", "tenant_id": "default", "scopes": ["write"]}'
```

### Rate Limiting

The server enforces per-key rate limits (default: 120 requests/minute). Configure per-key:

```bash
curl -X POST http://localhost:8899/auth/keys \
  -d '{
    "agent_name": "high-throughput-agent",
    "tenant_id": "default",
    "scopes": ["read", "write"],
    "rate_limit_rpm": 500
  }'
```

When rate limited, the server returns:

```json
{
  "error": "Rate limit exceeded",
  "retry_after": 60,
  "rate_key": "a1b2c3d4"
}
```

Handle this in your agent with exponential backoff:

```python
from arriadne.client import AriadneClient, AriadneRateLimitError
import time

client = AriadneClient("http://localhost:8899", api_key="ak_...")

for attempt in range(3):
    try:
        result = client.remember("Important memory")
        break
    except AriadneRateLimitError:
        wait = 2 ** attempt * 10
        print(f"Rate limited, waiting {wait}s...")
        time.sleep(wait)
```

### Environment Variables

Never hardcode API keys in source code:

```bash
export ARIADNE_URL="http://localhost:8899"
export ARIADNE_API_KEY="ak_a1b2c3d4e5f67890abcdef1234567890"
```

```python
import os

client = AriadneClient(
    base_url=os.environ["ARIADNE_URL"],
    api_key=os.environ["ARIADNE_API_KEY"],
)
```

### Network Security

For production, bind to localhost and use a reverse proxy:

```bash
# Bind to localhost only
ariadne serve --host 127.0.0.1 --port 8899

# Or use TLS via a reverse proxy (nginx, Caddy, etc.)
```

### Key Storage in Secrets Managers

For production deployments, fetch keys from your secrets manager:

```python
# AWS Secrets Manager
import boto3

secrets = boto3.client("secretsmanager")
api_key = secrets.get_secret_value(SecretId="ariadne/api-key")["SecretString"]

# HashiCorp Vault
# Or use your framework's built-in secret management
```

---

## Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install Ariadne and dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir ariadne uvicorn

# Copy application
COPY src/ src/

# Create data directory
RUN mkdir -p /data

# Expose port
EXPOSE 8899

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8899/health')" || exit 1

# Start server with persistent database
CMD ["python", "-m", "ariadne.server", \
     "--host", "0.0.0.0", \
     "--port", "8899", \
     "--db-path", "/data/ariadne.db"]
```

### Docker Compose

```yaml
version: "3.8"

services:
  ariadne:
    build: .
    ports:
      - "8899:8899"
    volumes:
      - ariadne-data:/data
    environment:
      - ARIADNE_DB_PATH=/data/ariadne.db
      - ARIADNE_RATE_LIMIT=300
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8899/health')"]
      interval: 30s
      timeout: 5s
      retries: 3

  # Your agent (example)
  my-agent:
    image: my-agent:latest
    environment:
      - ARIADNE_URL=http://ariadne:8899
      - ARIADNE_API_KEY=${ARIADNE_API_KEY}
    depends_on:
      ariadne:
        condition: service_healthy

volumes:
  ariadne-data:
```

### Run It

```bash
# Start the server
docker compose up -d

# Create an API key
docker compose exec ariadne python -c "
from arriadne.auth.keys import APIKeyManager
km = APIKeyManager('/data/ariadne.db')
result = km.create_key('docker-agent', tenant_id='default', scopes=['read','write'])
print(f'API Key: {result[\"key\"]}')
"

# Store a memory
curl -X POST http://localhost:8899/memories \
  -H "Authorization: Bearer ak_..." \
  -H "Content-Type: application/json" \
  -d '{"content": "Docker deployment is running", "topic": "devops", "importance": 8}'
```

### Production Docker Compose with Reverse Proxy

```yaml
version: "3.8"

services:
  ariadne:
    build: .
    expose:
      - "8899"  # Internal only
    volumes:
      - ariadne-data:/data
    environment:
      - ARIADNE_RATE_LIMIT=600
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - ariadne
    restart: unless-stopped

volumes:
  ariadne-data:
```

### nginx.conf

```nginx
events { worker_connections 1024; }

http {
    upstream ariadne {
        server ariadne:8899;
    }

    server {
        listen 443 ssl;
        ssl_certificate     /etc/nginx/certs/fullchain.pem;
        ssl_certificate_key /etc/nginx/certs/privkey.pem;

        location / {
            proxy_pass http://ariadne;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # SSE support for /search/stream
            proxy_buffering off;
            proxy_cache off;
        }
    }
}
```

---

## API Reference Cheat Sheet

### Store

```http
POST /memories
Content-Type: application/json
Authorization: Bearer ak_...

{
  "content": "string (required)",
  "topic": "string (default: general)",
  "importance": "int 1-10 (default: 5)",
  "entities": ["string"],
  "metadata": {"key": "value"}
}
```

### Search

```http
POST /search
Content-Type: application/json
Authorization: Bearer ak_...

{
  "query": "string (required)",
  "limit": "int 1-100 (default: 10)",
  "threshold": "float 0-1 (default: 0.5)",
  "use_hybrid": "bool (default: true)",
  "include_graph": "bool (default: false)"
}
```

### Auth

```http
POST /auth/keys
Content-Type: application/json
Authorization: Bearer ak_... (admin scope)

{
  "agent_name": "string (required)",
  "tenant_id": "string (default: default)",
  "scopes": ["read", "write", "admin"],
  "rate_limit_rpm": "int (default: 120)",
  "expires_in_seconds": "int (optional)"
}
```

### Error Codes

| Code | Meaning |
|------|---------|
| 401 | Invalid or missing API key |
| 403 | Insufficient scope for this endpoint |
| 404 | Resource not found |
| 429 | Rate limit exceeded (check `Retry-After` header) |
| 500 | Server error |

---

## Troubleshooting

**"Cannot connect to server"**
→ Check the server is running: `curl http://localhost:8899/health`

**"Invalid or missing API key" (401)**
→ Verify the key hasn't been revoked: `curl -H "Authorization: Bearer ak_..." http://localhost:8899/auth/keys`

**"Admin scope required" (403)**
→ The endpoint requires an admin-scoped key. Create one with `"scopes": ["read", "write", "admin"]`

**"Rate limit exceeded" (429)**
→ Wait for the `Retry-After` period, or create a key with a higher `rate_limit_rpm`

**Tenant isolation not working**
→ When using API key auth, the tenant is derived from the key, not the `X-Tenant-ID` header. Don't send `X-Tenant-ID` when using Bearer tokens.

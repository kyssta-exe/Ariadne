# Memory Types

Ariadne supports three memory types that model different categories of human-like memory. Each type is stored with a `memory_type` tag and can be filtered independently during search.

| Type | Use Case | Example |
|------|----------|---------|
| `semantic` | Facts, preferences, knowledge | "User prefers dark mode" |
| `episodic` | Events, experiences, temporal | "Met with client on 2026-01-15" |
| `procedural` | Skills, workflows, how-tos | "Deploy with `make deploy-prod`" |

## Semantic Memory

Semantic memories store factual knowledge, preferences, and general information — things that are true independent of time. Think of it as your agent's "knowledge base."

### Storing Facts

```python
from arriadne import AriadneMemory

mem = AriadneMemory(db_path="memory.db", embedding_dim=384)

# Store a fact
result = mem.remember(
    content="Paris is the capital of France",
    memory_type="semantic",
    importance=0.9,
    entities=["Paris", "France"],
)
print(result)
# {'memory_id': 1, 'status': 'created'}
```

### Storing Preferences

```python
# User preferences
mem.remember(
    content="User prefers vim over nano for editing",
    memory_type="semantic",
    importance=0.8,
    entities=["user"],
    metadata={"category": "editor_preference"},
)

mem.remember(
    content="User's home directory is /home/alex",
    memory_type="semantic",
    importance=0.9,
    entities=["user", "alex"],
)
```

### Retrieving Semantic Facts

```python
# Find facts about a topic
results = mem.recall("capital of France", k=5, type_filter="semantic")
for r in results:
    print(f"  [{r['score']:.4f}] {r['content']}")
```

## Episodic Memory

Episodic memories capture events with temporal context — things that happened at a specific time. These are naturally less persistent than semantic facts and are subject to stronger forgetting curves.

### Storing Events

```python
import time

now = time.time()

# Store a meeting event
mem.remember(
    content="Met with the DevOps team to discuss migration timeline",
    memory_type="episodic",
    importance=0.6,
    entities=["DevOps", "migration"],
    metadata={
        "event_type": "meeting",
        "timestamp": now,
        "duration_minutes": 45,
    },
)

# Store a deployment event
mem.remember(
    content="Deployed v2.3.1 to production at 14:00 UTC",
    memory_type="episodic",
    importance=0.7,
    entities=["production", "v2.3.1"],
    metadata={
        "event_type": "deployment",
        "version": "2.3.1",
    },
)
```

### Filtering by Time Range

```python
# Recall only recent events
one_hour_ago = time.time() - 3600
now = time.time()

results = mem.recall(
    "deployment",
    type_filter="episodic",
    time_range=(one_hour_ago, now),
)
```

## Procedural Memory

Procedural memories encode skills, workflows, and step-by-step procedures — things your agent knows how to do. These benefit from high importance scores since they guide future actions.

### Storing Workflows

```python
# Store a deployment workflow
mem.remember(
    content="""Deployment procedure:
    1. Run tests with pytest
    2. Build Docker image
    3. Push to registry
    4. Update k8s deployment
    5. Verify health checks""",
    memory_type="procedural",
    importance=0.9,
    entities=["deployment", "kubernetes", "docker"],
    metadata={
        "steps": 5,
        "automation_level": "semi-automatic",
    },
)
```

### Storing Skills

```python
# Store a debugging skill
mem.remember(
    content="To debug memory leaks: use tracemalloc to take snapshots, compare diffs over time, identify growing allocations",
    memory_type="procedural",
    importance=0.7,
    entities=["python", "debugging"],
)

# Store a git workflow
mem.remember(
    content="For hotfixes: create branch from main, cherry-pick fix, bump patch version, deploy, merge back to main",
    memory_type="procedural",
    importance=0.8,
    entities=["git", "hotfix"],
)
```

## Working with Memory Types

### Filtering by Type in Search

```python
# Only search semantic facts
facts = mem.recall("Python version", type_filter="semantic")

# Only search procedures
procedures = mem.recall("how to deploy", type_filter="procedural")

# Search across all types (default)
all_results = mem.recall("deployment")
```

### Updating Memory Type

```python
# If you stored an event but later realize it's a fact
mem.update(
    memory_id=42,
    metadata={"memory_type": "semantic"},  # Metadata can override type
)
```

### Type Distribution Stats

```python
stats = mem.stats()
print(stats["by_type"])
# {'semantic': 150, 'episodic': 45, 'procedural': 30}
```


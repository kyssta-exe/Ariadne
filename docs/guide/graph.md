# Knowledge Graph

Ariadne's knowledge graph connects memories through **entities** and **typed relationships**, enabling multi-hop traversal that finds connections vector search alone might miss.

## Overview

The knowledge graph is stored in SQLite alongside the memory tables:

```
entities (id, name, entity_type)
    │
    ├── edges (source_id, target_id, edge_type, weight)
    │
    └── memory_entities (memory_id, entity_id)
```

## Entity Types

Entities represent the concepts, people, and objects in your memory system:

| Type | Description | Example |
|------|-------------|---------|
| `person` | People, users, contacts | "Alice", "user_alex" |
| `project` | Projects, initiatives | "Ariadne", "migration" |
| `component` | Software components | "FAISS", "SQLite", "FTS5" |
| `infrastructure` | Servers, services | "production", "VPS" |
| `concept` | Abstract ideas | "deployment", "security" |
| `skill` | Abilities, techniques | "python", "debugging" |

When you add an entity without specifying a type, it defaults to `"general"`.

## Relationship Types

Edges connect entities with typed, weighted relationships:

| Type | Meaning | Example |
|------|---------|---------|
| `uses` | A depends on or employs B | project → component |
| `replaces` | A supersedes B | new → old |
| `forked_from` | A is derived from B | fork → original |
| `depends_on` | A requires B | component → service |
| `related` | Generic association | any → any |
| `authored_by` | Created by | project → person |
| `deployed_to` | Deployed on | service → infrastructure |

## Adding Entities and Edges

### Via the High-Level API

```python
from arriadne import AriadneMemory

mem = AriadneMemory(db_path="memory.db", embedding_dim=384)

# Add edges between entities
mem.add_edge("Alice", "user", "person", edge_type="authored_by", weight=1.0)
mem.add_edge("Hermes", "project", "Ariadne", "component", edge_type="uses")
mem.add_edge("Ariadne", "component", "FAISS", "component", edge_type="uses")
mem.add_edge("Ariadne", "component", "SQLite", "component", edge_type="uses")
mem.add_edge("SQLite", "component", "FTS5", "component", edge_type="uses")
```

### Via the Storage API

```python
from arriadne import AriadneDB, AriadneConfig

db = AriadneDB(AriadneConfig(db_path="memory.db"))
db.open()

# Create edges directly
db.add_edge("production", "VPS", "infrastructure", edge_type="deployed_to")
db.add_edge("Hermes", "AI", "project", edge_type="depends_on", weight=0.8)

db.close()
```

### Automatic Entity Association

When you use the `entities` parameter in `remember()`, entities are automatically created and linked:

```python
mem.remember(
    content="Deployed v2.3.1 to production using Kubernetes",
    memory_type="episodic",
    importance=0.7,
    entities=["production", "Kubernetes", "v2.3.1"],
)
# Automatically creates entities and links them to the memory
```

## Graph Traversal (BFS)

Ariadne uses **Breadth-First Search** implemented via SQLite **recursive CTEs** (Common Table Expressions) for efficient multi-hop traversal.

### Basic Traversal

```python
# Traverse 1 hop from "Ariadne"
result = mem.graph("Ariadne", hops=1)
print(result["nodes"])  # ['Ariadne', 'FAISS', 'SQLite']
print(result["edges"])  # [{'source': 'Ariadne', 'target': 'FAISS', ...}, ...]
```

### Multi-Hop Traversal

```python
# Traverse 3 hops
result = mem.graph("Alice", hops=3)
print(f"Found {len(result['nodes'])} nodes, {len(result['edges'])} edges")

for edge in result["edges"]:
    print(f"  {edge['source']} --[{edge['type']}]--> {edge['target']}")
```

### Filter by Edge Type

```python
# Only follow "uses" relationships
result = mem.graph("Ariadne", hops=3, edge_type="uses")

# Only follow "depends_on" relationships
result = mem.graph("production", hops=2, edge_type="depends_on")
```

### Traversal Depth Limit

The maximum traversal depth is controlled by `max_graph_depth` in the config (default: 10 hops):

```python
from arriadne import AriadneConfig

config = AriadneConfig(
    db_path="memory.db",
    max_graph_depth=5,  # Limit to 5 hops
)
```

## How Recursive CTEs Work

Ariadne's graph traversal uses this SQL pattern:

```sql
WITH RECURSIVE graph_traverse(node_id, depth) AS (
    -- Base case: start from the source entity
    SELECT ?, 0
    
    UNION
    
    -- Recursive case: follow edges in both directions
    SELECT CASE
        WHEN gt.depth < ? THEN e.target_id
        ELSE e.source_id
    END, gt.depth + 1
    FROM graph_traverse gt
    JOIN edges e ON (
        e.source_id = gt.node_id
        OR e.target_id = gt.node_id
    )
    WHERE gt.depth < ?
    -- Optional: AND e.edge_type = ?
)
SELECT DISTINCT gt.node_id, en.name
FROM graph_traverse gt
JOIN entities en ON en.id = gt.node_id
```

This bidirectional traversal follows edges in both directions, allowing you to discover relationships regardless of their orientation.

## Complete Example

```python
from arriadne import AriadneMemory
import numpy as np

mem = AriadneMemory(db_path="memory.db", embedding_dim=384)

# Build a project dependency graph
mem.add_edge("WebApp", "API", "project", edge_type="depends_on")
mem.add_edge("API", "Database", "project", edge_type="depends_on")
mem.add_edge("API", "Cache", "project", edge_type="depends_on")
mem.add_edge("Cache", "Redis", "component", edge_type="uses")
mem.add_edge("Database", "PostgreSQL", "component", edge_type="uses")
mem.add_edge("WebApp", "Nginx", "component", edge_type="uses")

# Store related memories with entity tags
mem.remember(
    content="Migrated Database from MySQL to PostgreSQL on 2026-01-15",
    memory_type="episodic",
    importance=0.8,
    entities=["Database", "PostgreSQL", "MySQL"],
)

# Traverse the full dependency tree
result = mem.graph("WebApp", hops=3)
print("Nodes:", result["nodes"])
# ['WebApp', 'API', 'Nginx', 'Database', 'Cache', 'PostgreSQL', 'Redis']

# Find only direct infrastructure dependencies
result = mem.graph("WebApp", hops=2, edge_type="uses")
print("Uses relationships:", result["nodes"])
```

---

<div style="text-align: center; padding: 20px 0;">
  <a href="https://mantes.net" class="mantes-badge" target="_blank">
    Powered by <strong>Mantes</strong>
  </a>
</div>

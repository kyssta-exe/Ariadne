# Dedup & Contradiction Detection

MinHash LSH deduplication and negation-based contradiction detection APIs.

## Deduplicator

MinHash LSH-based near-duplicate detection for text content.

### Constructor

```python
from arriadne import Deduplicator

dedup = Deduplicator(threshold=0.8, num_perm=128)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.8` | Jaccard similarity threshold (0.0–1.0) |
| `num_perm` | `int` | `128` | Number of MinHash permutations |

::: tip
Higher `num_perm` improves accuracy but uses more memory. 128 is a good default; use 64 for speed or 256 for precision.
:::

### `add()`

Add content to the deduplication index.

```python
doc_id = dedup.add(content, doc_id="memory_42")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | *required* | Text to index |
| `doc_id` | `str \| None` | `None` | Document ID (auto-generated if None) |

**Returns:** `str` — The document ID.

### `remove()`

Remove a document from the index.

```python
removed = dedup.remove("memory_42")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `doc_id` | `str` | *required* | Document ID to remove |

**Returns:** `bool` — `True` if removed, `False` if not found.

### `is_duplicate()`

Check if content is a near-duplicate of any indexed content.

```python
is_dup = dedup.is_duplicate("Deploy to production using kubectl")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | *required* | Content to check |

**Returns:** `bool` — `True` if a duplicate exists above threshold.

### `find_duplicates()`

Find all near-duplicates with similarity scores.

```python
duplicates = dedup.find_duplicates("Deploy to production via kubectl")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | *required* | Content to find duplicates for |

**Returns:** `list[dict]`

```python
[
    {
        "id": str,              # Document ID
        "content": str,         # Original content
        "similarity": float,    # Jaccard similarity (0.0–1.0)
    }
]
```

Results are sorted by similarity (descending).

### `find_related()`

Find loosely related content using a lower effective threshold.

```python
related = dedup.find_related("deployment configuration", limit=10)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | *required* | Content to find related items for |
| `limit` | `int` | `10` | Maximum results |

**Returns:** `list[dict]` — Same format as `find_duplicates()`.

### `size`

Property returning the number of indexed documents.

```python
print(f"Indexed: {dedup.size}")
```

---

## ContradictionDetector

Detects contradictions between text statements using negation pattern matching and fact extraction.

### Constructor

```python
from arriadne import ContradictionDetector

detector = ContradictionDetector()
```

### `detect_contradictions()`

Detect contradictions between two text statements.

```python
contradictions = detector.detect_contradictions(
    "Python is a compiled language",
    "Python is not a compiled language",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text_a` | `str` | *required* | First text statement |
| `text_b` | `str` | *required* | Second text statement |

**Returns:** `list[dict]`

```python
[
    {
        "subject": str,           # e.g., "python"
        "predicate": str,         # e.g., "a compiled language"
        "statement_a": str,       # Original clause from text_a
        "statement_b": str,       # Original clause from text_b
        "negated_in_a": bool,     # Whether fact_a is negated
        "negated_in_b": bool,     # Whether fact_b is negated
    }
]
```

### `is_contradictory()`

Quick boolean check for contradictions.

```python
is_contra = detector.is_contradictory(
    "Redis is a relational database",
    "Redis is not a relational database",
)
print(is_contra)  # True
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text_a` | `str` | *required* | First text |
| `text_b` | `str` | *required* | Second text |

**Returns:** `bool` — `True` if contradictions found.

### `extract_facts()`

Extract factual claims from text.

```python
facts = detector.extract_facts(
    "Python is dynamic. Java is compiled. Go has garbage collection."
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | `str` | *required* | Input text |

**Returns:** `list[dict]`

```python
[
    {
        "subject": str,       # e.g., "python"
        "predicate": str,     # e.g., "dynamic"
        "negated": bool,      # Whether the fact is negated
        "original": str,      # Original clause text
    }
]
```

### Negation Patterns

The detector recognizes these negation patterns:

| Pattern | Example |
|---------|---------|
| `not`, `no`, `never`, `neither`, `nor` | "Python is not slow" |
| `without`, `cannot`, `can't`, `won't` | "Can't use Java" |
| `wouldn't`, `shouldn't`, `don't` | "Don't support threads" |
| `doesn't`, `didn't`, `isn't`, `aren't` | "Isn't compiled" |
| `wasn't`, `weren't`, `hasn't`, `haven't` | "Hasn't been deprecated" |
| `hadn't` | "Hadn't been tested" |

### Fact Patterns

Facts are extracted using these patterns:

| Pattern | Example |
|---------|---------|
| `X is/are/was/were Y` | "Python is dynamic" |
| `X has/have/had Y` | "Go has garbage collection" |
| `X can/could/may/might Y` | "SQLite can run in-memory" |
| `X does/did Y` | "Node.js does event-driven I/O" |

---

## Integration with AriadneMemory

Both `Deduplicator` and `ContradictionDetector` are used automatically by `AriadneMemory`:

```python
from arriadne import AriadneMemory

mem = AriadneMemory(db_path="memory.db")

# Automatic dedup + contradiction detection
result = mem.remember("Python is a compiled language", importance=0.8)
# Status: created

result = mem.remember("Python is a compiled language", importance=0.8)
# Status: duplicate

result = mem.remember("Python is not a compiled language", importance=0.5)
# Status: created, with contradictions detected
```

---

## Advanced: Standalone Usage

```python
from arriadne import Deduplicator, ContradictionDetector

# Build a standalone dedup index
dedup = Deduplicator(threshold=0.8, num_perm=128)

# Index a corpus
documents = [
    "Deploy to production using kubectl apply",
    "Deploy to prod via kubectl apply -f",
    "User prefers dark mode",
    "User likes dark theme",
]

for i, doc in enumerate(documents):
    dedup.add(doc, doc_id=f"doc_{i}")

# Find duplicates
dups = dedup.find_duplicates("Deploy to production using kubectl")
for d in dups:
    print(f"  {d['id']}: {d['similarity']:.4f}")

# Detect contradictions between any two texts
contra = ContradictionDetector()
result = contra.detect_contradictions(
    "PostgreSQL is a NoSQL database",
    "PostgreSQL is not a NoSQL database",
)
print(f"Contradictions: {len(result)}")
```

---
title: "Deduplication — Ariadne"
description: "MinHash LSH near-duplicate detection at 0.12ms. Automatic deduplication on memory insert."
---


Ariadne uses **MinHash LSH** (Locality-Sensitive Hashing) for near-duplicate detection and a **negation-based pattern matcher** for contradiction detection. Both run automatically during `remember()`.

## MinHash LSH for Near-Duplicate Detection

### How It Works

1. **Tokenize** the content into lowercase words
2. **Create shingles** — 2-word sliding windows over the token list
3. **Compute MinHash** — a compact signature that approximates Jaccard similarity
4. **LSH lookup** — find candidates with similar signatures in sub-linear time
5. **Exact Jaccard** — compute true similarity only for LSH candidates

### Configuration

```python
from arriadne import AriadneConfig, AriadneMemory

config = AriadneConfig(
    dedup_threshold=0.8,    # Jaccard similarity threshold for "duplicate"
    dedup_num_perm=128,     # Number of MinHash permutations (higher = more accurate)
)

mem = AriadneMemory(config=config)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dedup_threshold` | 0.8 | Minimum Jaccard similarity to consider a duplicate |
| `dedup_num_perm` | 128 | Number of MinHash permutations (64–256 typical) |

### Automatic Dedup in remember()

Every call to `remember()` automatically:

1. Checks the dedup index for near-duplicates
2. If found, returns `status: "duplicate"` without creating a new memory
3. If not found, adds the memory and indexes it

```python
# First call — creates the memory
result = mem.remember(
    content="Python is a high-level programming language",
    memory_type="semantic",
    importance=0.8,
)
print(result)
# {'memory_id': 1, 'status': 'created'}

# Duplicate call — detected and rejected
result = mem.remember(
    content="Python is a high-level programming language",
    memory_type="semantic",
    importance=0.8,
)
print(result)
# {'memory_id': None, 'status': 'duplicate', 'duplicate_of': 1}

# Near-duplicate — also caught
result = mem.remember(
    content="Python is a high level programming language",
    memory_type="semantic",
)
print(result)
# {'memory_id': None, 'status': 'duplicate', 'duplicate_of': 1}
```

### Using the Deduplicator Directly

```python
from arriadne import Deduplicator

# Initialize with custom settings
dedup = Deduplicator(threshold=0.8, num_perm=128)

# Add content to the index
dedup.add("Deploy to production using kubectl", doc_id="deploy-1")
dedup.add("Deploy to production using kubectl apply", doc_id="deploy-2")
dedup.add("User prefers dark mode", doc_id="pref-1")

# Check for duplicates
print(dedup.is_duplicate("Deploy to production via kubectl"))
# True — very similar to deploy-1

# Find all near-duplicates with similarity scores
duplicates = dedup.find_duplicates("Deploy to production using kubectl")
for d in duplicates:
    print(f"  {d['id']}: similarity={d['similarity']:.4f}")
    print(f"  {d['content']}")

# Find loosely related content
related = dedup.find_related("deployment configuration", limit=5)
for r in related:
    print(f"  {r['id']}: {r['similarity']:.4f} | {r['content'][:50]}")
```

### Managing the Dedup Index

```python
# Add with auto-generated ID
doc_id = dedup.add("Some content")

# Remove from index
dedup.remove("deploy-1")

# Check index size
print(f"Indexed documents: {dedup.size}")
```

## Contradiction Detection

The `ContradictionDetector` finds conflicting statements by extracting factual claims and checking for negation patterns.

### How It Works

1. **Split** text into individual clauses (split on `and`, `but`, `,`, `.`, `;`)
2. **Extract facts** from each clause using patterns like `X is Y`, `X has Y`, `X can Y`
3. **Normalize** predicates by removing negation words
4. **Compare** facts across two texts
5. **Flag contradictions** where same subject + same predicate + different negation

### Negation Patterns Detected

Ariadne recognizes 20+ negation patterns:

| Pattern | Example |
|---------|---------|
| `not`, `no`, `never`, `neither`, `nor` | "Python is not slow" |
| `without`, `cannot`, `can't`, `won't` | "Can't use Java" |
| `doesn't`, `didn't`, `isn't`, `aren't` | "Isn't compiled" |
| `wasn't`, `weren't`, `hasn't`, `haven't` | "Hasn't been deprecated" |

### Detecting Contradictions

```python
from arriadne import ContradictionDetector

detector = ContradictionDetector()

# Simple contradiction
contradictions = detector.detect_contradictions(
    "Python is a compiled language",
    "Python is not a compiled language",
)
print(contradictions)
# [{'subject': 'python', 'predicate': 'a compiled language',
#   'statement_a': 'Python is a compiled language',
#   'statement_b': 'Python is not a compiled language',
#   'negated_in_a': False, 'negated_in_b': True}]

# No contradiction
contradictions = detector.detect_contradictions(
    "Python is a compiled language",
    "Java is a compiled language",
)
print(contradictions)
# [] — different subjects
```

### Quick Contradiction Check

```python
# Fast boolean check
is_contra = detector.is_contradictory(
    "Redis is a relational database",
    "Redis is not a relational database",
)
print(is_contra)  # True
```

### Extracting Facts

```python
# Extract all factual claims from text
facts = detector.extract_facts(
    "Python is dynamic. Java is compiled. Go has garbage collection."
)
for fact in facts:
    print(f"  {fact['subject']} | {fact['predicate']} | negated={fact['negated']}")
# python | dynamic | negated=False
# java | compiled | negated=False
# go | garbage collection | negated=False
```

### Automatic Contradiction Detection in remember()

When you call `remember()`, Ariadne automatically:

1. Searches FTS5 for semantically related existing memories
2. Runs `ContradictionDetector.detect_contradictions()` against each result
3. Returns any contradictions in the result

```python
# Store an initial fact
mem.remember("Redis is an in-memory database", importance=0.9)

# Store a contradicting fact
result = mem.remember(
    "Redis is not an in-memory database",
    importance=0.5,
)
print(result)
# {'memory_id': None, 'status': 'created', 'contradictions': [
#     {'subject': 'redis', 'existing_memory_id': 1,
#      'statement_a': 'Redis is not an in-memory database',
#      'statement_b': 'Redis is an in-memory database', ...}
# ]}
```

## Combining Dedup and Contradiction Detection

```python
from arriadne import AriadneMemory

mem = AriadneMemory(db_path="memory.db", embedding_dim=384)

# Add some facts
mem.remember("PostgreSQL supports JSONB", importance=0.8)
mem.remember("MySQL supports JSONB", importance=0.7)
mem.remember("SQLite is embedded", importance=0.9)

# Check what we have
stats = mem.stats()
print(f"Active memories: {stats['active_memories']}")
print(f"Dedup index size: {stats['dedup_index_size']}")
```


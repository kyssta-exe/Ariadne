"""Deterministic retrieval-accuracy evaluation for Ariadne.

Measures end-to-end answer retrieval over a synthetic memory corpus — the
companion to the latency benchmarks: "fast" is meaningless without "correct".

Why synthetic: it is fully deterministic (same score on every machine, no
dataset download, no embedding-model dependency when run without the
``embeddings`` extra), while still exercising the properties that matter for
agent memory:

- exact keyword recall
- paraphrase recall (query shares meaning, not words)
- temporal supersession (a fact changed; only the current value may answer)
- distractor density (the answer must outrank N similar-but-wrong facts)
- namespace isolation (a fact in namespace B must never answer a query in A)

With sentence-transformers installed, pass ``--embeddings`` to additionally
evaluate the semantic (vector) path; otherwise the deterministic hashing
embedder below provides a fixed, dependency-free vector signal so the hybrid
path is still measured.

Run:  python benchmarks/accuracy_eval.py [--n 200] [--k 5] [--embeddings]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from arriadne import AriadneConfig, AriadneMemory

DIM = 256
rng = np.random.default_rng(2026)

FIRST = ["Kai", "Mira", "Theo", "Zara", "Owen", "Lena", "Ravi", "Nora", "Felix", "Iris"]
LAST = ["Hale", "Novak", "Osei", "Marin", "Kaur", "Lind", "Farkas", "Duarte"]


def person_name(i: int) -> str:
    """Unique subject per index: unambiguous questions need unique names."""
    combo_first = FIRST[i % len(FIRST)]
    combo_last = LAST[(i // len(FIRST)) % len(LAST)]
    generation = i // (len(FIRST) * len(LAST))
    suffix = "" if generation == 0 else f" {generation + 1}"
    return f"{combo_first} {combo_last}{suffix}"
# 30 cities: each name needs up to ~12 unique cities (facts + temporal moves),
# so the uniqueness loop can always terminate.
CITIES = [
    "Oslo", "Kyoto", "Lima", "Cairo", "Austin", "Chennai", "Porto", "Sofia", "Hanoi", "Bogota",
    "Turin", "Dakar", "Riga", "Muscat", "Suva", "Karachi", "Bilbao", "Goa", "Perth", "Accra",
    "Yerevan", "Trieste", "Luanda", "Kigali", "Namur", "Osaka", "Petra", "Quito", "Siena", "Tromso",
]
SKILLS = [
    "rust", "kubernetes", "design", "databases", "trading", "cartography", "baking", "biology"
]
COLORS = ["teal", "amber", "crimson", "indigo", "jade", "slate"]
FOODS = ["ramen", "tapas", "felafel", "pierogi", "curry", "tacos"]
PARA = {
    "lives in": "makes their home in",
    "works on": "spends their days on",
    "favorite food is": "always orders",
    "favorite color is": "paints everything",
}
DISTRACTOR_VERBS = [
    "enjoys hiking near", "reads about", "dreams of", "photographs", "writes letters about"
]


class HashingEmbedder:
    """Deterministic char-trigram hashing embedder (dependency-free).

    Not a semantic model — but paraphrases share most trigrams, so cosine
    similarity is a stable, reproducible proxy that makes the hybrid path
    meaningful in CI without downloading models.
    """

    dim = DIM

    def __init__(self) -> None:
        self._cache: dict[str, list[float]] = {}

    def _embed(self, text: str) -> list[float]:
        vec = np.zeros(DIM, dtype=np.float32)
        normalized = re.sub(r"\W+", " ", text.lower()).strip()
        grams = [normalized[i : i + 3] for i in range(max(0, len(normalized) - 2))]
        for gram in grams:
            bucket = int.from_bytes(hashlib.md5(gram.encode()).digest()[:4], "big") % DIM
            sign = 1.0 if hashlib.md5(gram.encode()).digest()[4] % 2 else -1.0
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 1e-10:
            vec /= norm
        return vec.tolist()

    def __call__(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = self._embed(text)
        return self._cache[text]


def build_corpus(n: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build (memories, questions). Questions carry the expected memory id."""
    memories: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    rs = np.random.default_rng(99)

    # Deterministic fact template pool. Contents must be globally unique:
    # duplicate rows would collide with the supersession targets and silently
    # turn into hash-duplicates, corrupting the temporal questions.
    used_contents: set[str] = set()

    def unique_city(name: str) -> str:
        while True:
            city = CITIES[rs.integers(len(CITIES))]
            if f"{name} lives in {city}." not in used_contents:
                return city

    for i in range(n):
        name = person_name(i)
        city = unique_city(name)
        skill = SKILLS[rs.integers(len(SKILLS))]
        food = FOODS[rs.integers(len(FOODS))]
        color = COLORS[rs.integers(len(COLORS))]

        facts = [
            (f"{name} lives in {city}.", f"Where does {name} live?", city, "exact"),
            (
                f"{name} works on {skill}.",
                f"What does {name} spend their days on?",
                skill,
                "paraphrase",
            ),
            (
                f"{name}'s favorite food is {food}.",
                f"What does {name} always order?",
                food,
                "paraphrase",
            ),
            (
                f"{name}'s favorite color is {color}.",
                f"What is {name}'s favorite color?",
                color,
                "exact",
            ),
        ]
        for content, query, answer, kind in facts:
            if content in used_contents:
                continue
            used_contents.add(content)
            memories.append({"content": content, "importance": 0.7})
            questions.append(
                {
                    "query": query,
                    "answer": answer,
                    "kind": kind,
                    "content": content,
                    "subject": name,
                }
            )

        # Distractors: similar shape, wrong subject.
        other = person_name((i + 3) % n) if n > 3 else person_name(i)
        memories.append(
            {
                "content": (
                    f"{other} {DISTRACTOR_VERBS[rs.integers(len(DISTRACTOR_VERBS))]} {city}."
                ),
                "importance": 0.5,
            }
        )

    # Temporal supersession: the first 20 people "move".
    moved = 0
    for q in questions:
        if moved >= 20:
            break
        if q["kind"] == "exact" and "live" in q["query"]:
            name = q["subject"]  # full unique name, not just the first word
            old_content = q["content"]
            while True:
                new_city = CITIES[rng.integers(len(CITIES))]
                new_content = f"{name} lives in {new_city}."
                if new_content not in used_contents:
                    break
            used_contents.add(new_content)
            memories.append(
                {"content": new_content, "importance": 0.7, "supersedes_content": old_content}
            )
            q["answer"] = new_city
            q["kind"] = "temporal"
            q["superseded_content"] = old_content
            moved += 1

    # Namespace isolation facts.
    for i in range(10):
        memories.append(
            {"content": f"{person_name(i)} keeps a diary about {FOODS[i % len(FOODS)]}.",
             "namespace": "other-user"}
        )

    return memories, questions


def evaluate(mode: str, n: int, k: int) -> dict[str, Any]:
    """Run the eval in one of three modes: 'fts', 'hashing', or 'real'.

    'real' uses SentenceTransformerEmbedder (requires the embeddings extra);
    'hashing' uses the deterministic trigram embedder; 'fts' uses no embedder.
    """
    tmp = tempfile.mkdtemp(prefix="ariadne_eval_")
    cfg_kwargs: dict[str, Any] = {"embedding_dim": DIM}
    if mode == "real":
        from arriadne.embeddings import SentenceTransformerEmbedder

        embedder = SentenceTransformerEmbedder("all-MiniLM-L6-v2")
        cfg_kwargs["embedding_dim"] = embedder.dim
    elif mode == "hashing":
        embedder: Any = HashingEmbedder()
    else:
        embedder = None
        cfg_kwargs["semantic_dedup"] = False
    mem = AriadneMemory(
        config=AriadneConfig(db_path=Path(tmp) / "eval.db", **cfg_kwargs),
        embedder=embedder,
    )

    memories, questions = build_corpus(n)
    content_to_id: dict[str, int] = {}
    # Store base facts only: temporal "new value" memories must be created via
    # supersede() (with supersedes_id), not bulk-stored — a bulk-stored copy
    # makes the later supersede() a hash-duplicate no-op and the old value
    # never gets retired.
    base = [m for m in memories if "supersedes_content" not in m]
    stored = mem.remember_many(base)
    for m, res in zip(base, stored, strict=True):
        content_to_id[m["content"]] = res["memory_id"]

    for m in memories:
        old_content = m.get("supersedes_content")
        if old_content and old_content in content_to_id:
            res = mem.supersede(
                old_memory_id=content_to_id[old_content],
                new_content=m["content"],
            )
            if res.get("status") == "created":
                content_to_id[m["content"]] = res["memory_id"]

    correct = 0
    by_kind: dict[str, list[int]] = {}
    for q in questions:
        # Scoped recall, as a real multi-tenant deployment queries: namespace
        # None means "all namespaces" by design, which would make the
        # isolation check meaningless (other-user facts may legitimately rank).
        results = mem.recall(q["query"], k=k, namespace="default")
        name = q["subject"]  # unique subject of the question
        # Hit = the expected answer appears in a returned, live memory that is
        # about the same subject (another person's identical answer is a miss).
        hit = any(
            q["answer"] in (r.get("content") or "")
            and (r.get("content") or "").startswith(name)
            and not r.get("is_deleted")
            for r in results
        )
        # Temporal queries must additionally NOT be answered by the value
        # that was superseded (the whole point of temporal recall).
        if hit and q["kind"] == "temporal":
            superseded_id = content_to_id.get(q.get("superseded_content", ""), -1)
            hit = not any(r["id"] == superseded_id for r in results)
        by_kind.setdefault(q["kind"], []).append(1 if hit else 0)
        correct += 1 if hit else 0

    # Namespace isolation must hold: scoped recall never returns other-user
    # facts, even though they describe the same people.
    leak = any(
        r.get("namespace") == "other-user"
        for q in questions[:20]
        for r in mem.recall(q["query"], k=k, namespace="default")
    )

    report = {
        "questions": len(questions),
        "recall_at_k": correct / len(questions),
        "by_kind": {kind: float(np.mean(v)) for kind, v in by_kind.items()},
        "namespace_leak": leak,
        "mode": mode,
        "k": k,
        "n": n,
    }
    mem.close()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="corpus size multiplier")
    parser.add_argument("--k", type=int, default=5, help="recall depth")
    parser.add_argument(
        "--embeddings",
        action="store_true",
        help="use sentence-transformers (real semantic embeddings)",
    )
    parser.add_argument(
        "--hashing",
        action="store_true",
        help="use the deterministic trigram hashing embedder (hybrid path without models)",
    )
    args = parser.parse_args()

    mode = "fts"
    if args.embeddings:
        mode = "real"
    elif args.hashing:
        mode = "hashing"
    if mode == "real":
        import importlib.util

        if importlib.util.find_spec("sentence_transformers") is None:
            print("sentence-transformers not installed; falling back to hashing embedder")
            mode = "hashing"

    report = evaluate(mode, args.n, args.k)
    print("\n=== Ariadne accuracy eval ===")
    print(f"mode:            {report['mode']}")
    print(f"questions:       {report['questions']}  (k={report['k']}, n={report['n']})")
    print(f"answer recall@{report['k']}:  {report['recall_at_k']:.3f}")
    for kind, score in sorted(report["by_kind"].items()):
        print(f"  {kind:<12s} {score:.3f}")
    print(f"namespace leak:  {report['namespace_leak']}  (must be False)")


if __name__ == "__main__":
    main()

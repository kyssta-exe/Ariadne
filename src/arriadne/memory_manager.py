"""Autonomous memory management for Ariadne.

This is the intelligence layer that turns Ariadne from a *storage engine* into a
*memory product*. It lets an agent (or a host framework) feed in raw
conversation turns and have Ariadne decide what is worth remembering, extract
structured facts and relationships, resolve conflicts, and keep the store tidy —
without the caller hand-crafting every ``remember()`` call.

Everything here works with the plain Python API and is backend-agnostic: it needs
only a callable that turns text into a structured extract (``LLMCaller``), an
:class:`~arriadne.interface.AriadneMemory`, and optionally an embedder.

The design keeps the *core dependency-light*: ``memory_manager`` imports nothing
heavy. The default ``LLMCaller`` uses only ``json`` + the ``re`` standard library
for its deterministic fallback, and provider adapters (OpenAI, Anthropic, ...) are
import-guarded / optional.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extraction model
# ---------------------------------------------------------------------------


@dataclass
class ExtractedMemory:
    """A single candidate memory with optional structured form.

    Attributes:
        content: Human-readable sentence capturing the memory.
        kind: ``semantic`` | ``episodic`` | ``procedural`` | ``preference``.
        importance: 0.0-1.0 salience estimate.
        subject: Optional subject for a KV fact (e.g. ``"user"``, ``"project"``).
        attribute: Optional attribute for a KV fact (e.g. ``"name"``, ``"language"``).
        value: Optional value for a KV fact (e.g. ``"Kyssta"``).
        entities: Optional list of entity names to attach (feeds the graph).
        metadata: Optional extra metadata.
    """

    content: str
    kind: str = "semantic"
    importance: float = 0.5
    subject: str | None = None
    attribute: str | None = None
    value: str | None = None
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_fact(self) -> bool:
        return bool(self.subject and self.attribute and self.value is not None)


@dataclass
class ExtractedRelation:
    """A relationship between two entities for the knowledge graph."""

    source: str
    target: str
    edge_type: str = "related"
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    """Structured output of a single extraction pass."""

    memories: list[ExtractedMemory] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.memories or self.relations)

    def as_dict(self) -> dict[str, Any]:
        return {
            "memories": [
                {
                    "content": m.content,
                    "kind": m.kind,
                    "importance": m.importance,
                    "subject": m.subject,
                    "attribute": m.attribute,
                    "value": m.value,
                    "entities": list(m.entities),
                }
                for m in self.memories
            ],
            "relations": [
                {
                    "source": r.source,
                    "target": r.target,
                    "edge_type": r.edge_type,
                    "weight": r.weight,
                }
                for r in self.relations
            ],
        }


# ---------------------------------------------------------------------------
# LLM caller protocol
# ---------------------------------------------------------------------------


def _build_extraction_prompt(user: str, assistant: str) -> str:
    """Build the memory-extraction prompt with literal braces properly escaped.

    The prompt describes a JSON shape containing literal ``{{ ... }}`` braces
    and embeds the user/assistant turn via an f-string substitution. Using a
    helper avoids any string-literal foot-guns.
    """
    kinds_hint = (
        '"semantic": a timeless fact or insight,\n'
        '"episodic": an event / experience with a time context,\n'
        '"procedural": a how-to / workflow / skill,\n'
        '"preference": something the user prefers (still stored as semantic).'
    )
    return f"""You are a memory extraction engine. Read the conversation turn below and
extract what is worth remembering long-term, such that the agent can answer
future questions without re-seeing this exact turn.

Rules:
- Only extract durable, useful information. Skip chatter, greetings, and transient content.
- For each memory, choose kind: {kinds_hint}
- Assign importance 0.0-1.0 (higher = more important / longer-lived).
- Set entities to short entity names (lowercase) where relevant.
- For factual claims about a subject's attribute (e.g. "the user's name is Kyssta"),
  also populate subject / attribute / value so the store can upsert them.
- Optionally emit relations between entities as objects of the form
  {{"source": <entity>, "target": <entity>, "edge_type": <label>, "weight": <0..1>}}.

Return a STRICT JSON object (no prose, no markdown fences) with this shape:
{{"memories": [{{"content": str, "kind": str, "importance": float,
  "subject": str|null, "attribute": str|null, "value": str|null, "entities": [str]}}],
  "relations": [{{"source": str, "target": str, "edge_type": str, "weight": float}}]}}

Conversation turn:
{{
"user": {user!r},
"assistant": {assistant!r}
}}
"""


def _extract_json(text: str) -> dict[str, Any]:
    """Tolerantly parse a JSON object out of an LLM reply.

    Handles optional markdown fences and stray prose around the JSON.
    """
    s = text.strip()
    # Strip markdown code fences.
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    # Find the first {...} block if there's surrounding prose.
    if not s.startswith("{"):
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            s = s[start : end + 1]
    try:
        parsed = json.loads(s)
    except json.JSONDecodeError:
        logger.warning("LLM returned non-JSON; falling back to empty extraction")
        return {}
    return parsed if isinstance(parsed, dict) else {}


class LLMCaller(Protocol):
    """Anything callable that maps a prompt string to a text response."""

    def __call__(self, prompt: str) -> str: ...


class _FallbackCaller:
    """Deterministic, dependency-free fallback used when no LLM is configured.

    It never fabricates content: it extracts nothing (usable as a test stand-in)
    so a misconfigured system stays correct rather than injecting noise.
    """

    def __call__(self, prompt: str) -> str:
        return json.dumps({"memories": [], "relations": []})


def _normalise_importance(value: Any, default: float = 0.5) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = default
    return max(0.0, min(1.0, v))


def _parse_extraction(text: str) -> ExtractionResult:
    """Turn an LLM JSON reply into an :class:`ExtractionResult`.

    Tolerant: per-item errors are skipped, never fatal.
    """
    data = _extract_json(text)
    result = ExtractionResult()

    for m in data.get("memories", []):
        if not isinstance(m, dict) or not isinstance(m.get("content"), str):
            continue
        content = m["content"].strip()
        if not content:
            continue
        kind = str(m.get("kind", "semantic")).strip() or "semantic"
        if kind not in {"semantic", "episodic", "procedural", "preference"}:
            kind = "semantic"
        entities = [
            str(e).strip().lower()
            for e in m.get("entities", [])
            if isinstance(e, str) and e.strip()
        ]
        result.memories.append(
            ExtractedMemory(
                content=content,
                kind=kind,
                importance=_normalise_importance(m.get("importance")),
                subject=(str(m["subject"]).strip() or None) if m.get("subject") else None,
                attribute=(str(m["attribute"]).strip() or None) if m.get("attribute") else None,
                value=(str(m["value"]) if m.get("value") is not None else None),
                entities=entities,
            )
        )

    for r in data.get("relations", []):
        if not isinstance(r, dict):
            continue
        source = str(r.get("source", "")).strip().lower()
        target = str(r.get("target", "")).strip().lower()
        if not source or not target or source == target:
            continue
        weight = _normalise_importance(r.get("weight"), default=1.0)
        result.relations.append(
            ExtractedRelation(
                source=source,
                target=target,
                edge_type=str(r.get("edge_type") or "related").strip() or "related",
                weight=weight,
            )
        )

    return result


# ---------------------------------------------------------------------------
# The manager
# ---------------------------------------------------------------------------


class LLMMemoryManager:
    """Encapsulates the autonomous memory lifecycle over an :class:`AriadneMemory`.

    Args:
        memory: The backing :class:`~arriadne.interface.AriadneMemory`.
        caller: An :class:`LLMCaller` (``prompt -> str``). Defaults to a
            dependency-free fallback that extracts nothing.
        min_importance: Memories below this importance are not written by
            ``process_turn`` (a cheap noise gate before writing).
        dedupe_before_write: When True, skip writes that the store already
            flags as duplicates (exact-hash) to avoid fat temp tables.

    Example:
        >>> mem = AriadneMemory(db_path="memory.db", embedder=your_embedder)
        >>> mgr = LLMMemoryManager(mem, caller=my_llm)
        >>> summary = mgr.process_turn("My name is Kyssta", "Nice to meet you!")
    """

    def __init__(
        self,
        memory: Any,
        caller: LLMCaller | None = None,
        min_importance: float = 0.3,
        default_namespace: str = "default",
    ) -> None:
        self.memory = memory
        self.caller = caller if caller is not None else _FallbackCaller()
        self.min_importance = min_importance
        self.default_namespace = default_namespace

    # -- Raw extraction ----------------------------------------------------

    def extract(
        self,
        user: str = "",
        assistant: str = "",
        *,
        prompt: str | None = None,
    ) -> ExtractionResult:
        """Run one extraction pass over a conversation turn.

        Pass ``prompt`` to bypass the default '{}'-format prompt entirely.
        """
        if prompt is None:
            prompt = _build_extraction_prompt(user, assistant)
        response = self.caller(prompt)
        return _parse_extraction(response)

    # -- Best-effort KV fact upsert -----------------------------------------

    def set_fact(
        self,
        subject: str,
        attribute: str,
        value: str,
        *,
        importance: float = 0.6,
        namespace: str | None = None,
        memory_type: str = "semantic",
    ) -> dict[str, Any]:
        """Store a subject.attribute = value fact, superseding any older value.

        Uses Ariadne's temporal machinery (``remember`` with ``valid_from`` now and
        ``supersedes_id`` pointing at the prior value) so history is preserved and
        current recall returns only the latest value.

        Returns the ``remember`` result dict.
        """
        import time

        memory = self.memory
        namespace = namespace or self.default_namespace

        # Find an existing fact with the same subject+attribute that is still active.
        recall = memory.recall(f"{subject} {attribute}", k=10, namespace=namespace)
        prior_id: int | None = None
        for r in recall:
            meta = r.get("metadata") or {}
            if (
                meta.get("fact_subject") == subject
                and meta.get("fact_attribute") == attribute
                and not r.get("is_deleted")
            ):
                prior_id = r["id"]
                break

        content = f"{subject} {attribute} is {value}."
        now = time.time()
        kwargs: dict[str, Any] = {
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            "namespace": namespace,
            "event_at": now,
            "valid_from": now,
            "entities": [subject, attribute],
            "metadata": {
                "fact_subject": subject,
                "fact_attribute": attribute,
                "fact_value": value,
            },
        }

        if prior_id is not None:
            kwargs["supersedes_id"] = prior_id

        result = memory.remember(**kwargs)
        return result

    # -- Turn processing -----------------------------------------------------

    def process_turn(
        self,
        user: str,
        assistant: str,
        *,
        namespace: str | None = None,
        record_episode: bool = True,
        add_relations: bool = True,
        extract_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Process one conversation turn: record it, extract, and persist.

        Steps:
        1. Optionally record the raw turn as an immutable episode (provenance).
        2. Run the extraction pass.
        3. Write each memory (dedup-aware) with its entities attached.
        4. Link extracted relations into the knowledge graph.
        5. Return a summary of what was stored.

        Returns a dict with ``episode_id``, ``stored`` (list of ids), ``duplicates``,
        ``skipped``, and ``relations_added``.
        """
        ns = namespace or self.default_namespace
        memory = self.memory
        summary: dict[str, Any] = {
            "episode_id": None,
            "stored": [],
            "duplicates": [],
            "skipped": [],
            "relations_added": 0,
        }

        if record_episode and (user.strip() or assistant.strip()):
            ep = memory.record_episode(
                content=f"user: {user}\nassistant: {assistant}".strip(),
                role="turn",
                namespace=ns,
            )
            summary["episode_id"] = ep.get("episode_id")

        result = self.extract(user, assistant, prompt=extract_prompt)

        for m in result.memories:
            if m.importance < self.min_importance:
                summary["skipped"].append(m.content)
                continue
            call = memory.remember(
                content=m.content,
                memory_type=m.kind if m.kind != "preference" else "semantic",
                importance=m.importance,
                entities=m.entities or None,
                metadata={
                    **(m.metadata or {}),
                    "extracted": True,
                    "kind": m.kind,
                    "fact_subject": m.subject,
                    "fact_attribute": m.attribute,
                    "fact_value": m.value,
                },
                namespace=ns,
            )
            status = call.get("status")
            if status == "created" and call.get("memory_id") is not None:
                summary["stored"].append(call["memory_id"])
            else:
                summary["duplicates"].append(
                    {"content": m.content, "duplicate_of": call.get("duplicate_of")}
                )

        if add_relations:
            for rel in result.relations:
                try:
                    memory.add_edge(rel.source, rel.target, rel.edge_type, rel.weight)
                    summary["relations_added"] += 1
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Failed to add relation %s->%s: %s", rel.source, rel.target, exc)

        return summary

"""Memory curator — retention, conflict resolution, and consolidation for Ariadne.

Complements :mod:`arriadne.memory_manager` (which handles *ingest*: turning
turns into clean memories). This module owns the *retention and hygiene* side
of the lifecycle:

- **Decay / eviction**: expire low-priority, stale memories so the store stays
  bounded.
- **Conflict resolution**: use :class:`~arriadne.dedup.ContradictionDetector`
  to find contradictory pairs and supersede the weaker one, preserving history.
- **Consolidation orchestration**: wraps the store's ``consolidate()`` as a
  single entry point for housekeeping.

It is dependency-light: no LLM is required for the deterministic paths.
``MemoryCurator`` works directly with :class:`~arriadne.interface.AriadneMemory`;
a thin :class:`CuratorAddon` subclass also satisfies the :class:`~arriadne.addons.BaseAddon`
contract for auto-discovery via entry points.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from arriadne.addons import BaseAddon
from arriadne.dedup import ContradictionDetector

logger = logging.getLogger(__name__)


@dataclass
class CurateReport:
    """Human-readable summary of a curation pass."""

    decayed: int = 0
    consolidated: int = 0
    contradictions_resolved: int = 0
    actions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "decayed": self.decayed,
            "consolidated": self.consolidated,
            "contradictions_resolved": self.contradictions_resolved,
            "actions": list(self.actions),
        }


class MemoryCurator:
    """Runs retention + conflict-resolution passes over an :class:`AriadneMemory`.

    Args:
        memory: The backing :class:`~arriadne.interface.AriadneMemory`.
        default_namespace: Namespace to scope decay/conflict scans to when the
            caller doesn't pass one.
        decay_ttl_seconds: Hard age (seconds) after which un-accessed memories
            are considered decay candidates. ``None`` disables time-based decay.
        decay_importance_threshold: Only memories with importance below this are
            eligible for decay (protects important facts).
        resolve_contradictions: When ``True``, scan for contradictory pairs and
            supersede the older/lower-confidence statement.
    """

    def __init__(
        self,
        memory: Any,
        *,
        default_namespace: str = "default",
        decay_ttl_seconds: float | None = 86400.0 * 30,  # 30 days
        decay_importance_threshold: float = 0.4,
        resolve_contradictions: bool = True,
        allow_assistant_overwrite_user: bool = False,
    ) -> None:
        self.memory = memory
        self.default_namespace = default_namespace
        self.decay_ttl_seconds = decay_ttl_seconds
        self.decay_importance_threshold = decay_importance_threshold
        # Kept private so it does not shadow the public method below.
        self._resolve_contradictions = resolve_contradictions
        # When False (default), a contradiction pair where the older statement
        # is user-sourced and the newer one is not (e.g. an assistant guess)
        # is left alone instead of letting machine-authored text erase what
        # the user explicitly said.
        self.allow_assistant_overwrite_user = allow_assistant_overwrite_user
        self._contradiction = ContradictionDetector()

    # -- Decay -------------------------------------------------------------

    def decay(
        self,
        *,
        namespace: str | None = None,
        ttl_seconds: float | None = None,
        importance_threshold: float | None = None,
    ) -> int:
        """Soft-delete stale, low-importance memories to keep the store bounded.

        A memory is decayed when BOTH:
          - it was last accessed before ``time.time() - ttl_seconds``, and
          - its importance is below ``importance_threshold``.

        Returns the number of memories soft-deleted.
        """
        ttl = self.decay_ttl_seconds if ttl_seconds is None else ttl_seconds
        imp_cut = (
            self.decay_importance_threshold
            if importance_threshold is None
            else importance_threshold
        )
        if ttl is None:
            return 0

        ns = namespace or self.default_namespace
        db = self.memory._db
        assert db.conn is not None
        cutoff = time.time() - ttl
        params: list[Any] = [cutoff, imp_cut]
        ns_sql = ""
        if ns:
            ns_sql = " AND namespace = ?"
            params.append(ns)

        rows = db.conn.execute(
            f"""SELECT id FROM memories
                WHERE is_deleted = 0
                  AND accessed_at < ?
                  AND importance < ?
                  {ns_sql}""",
            params,
        ).fetchall()
        ids = [int(r[0]) for r in rows]
        for mid in ids:
            self.memory.forget(mid, hard=False)
        if ids:
            logger.info("Decayed %d stale memories (namespace=%r)", len(ids), ns)
        return len(ids)

    # -- Conflict resolution -----------------------------------------------

    def resolve_contradictions(
        self,
        *,
        namespace: str | None = None,
        k_scan: int = 200,
    ) -> int:
        """Find contradictory statement pairs and supersede the weaker one.

        For each candidate memory we compare against a short suffix of the
        sorted list (newer-first), avoiding a full O(n^2) scan while still
        catching recent contradictions. When a contradiction is detected, the
        older / lower-importance statement is soft-deleted and linked via
        ``supersede_id`` so provenance survives.

        Returns the number of contradictory pairs resolved.
        """
        ns = namespace or self.default_namespace
        db = self.memory._db
        assert db.conn is not None
        params: list[Any] = []
        ns_sql = ""
        if ns:
            ns_sql = " AND namespace = ?"
            params.append(ns)

        rows = db.conn.execute(
            f"""SELECT id, content, created_at, importance
                FROM memories
                WHERE is_deleted = 0
                  {ns_sql}
                ORDER BY created_at DESC
                LIMIT ?""",
            (*params, k_scan),
        ).fetchall()

        mems: list[dict[str, Any]] = [
            {
                "id": int(r[0]),
                "content": str(r[1]),
                "created_at": float(r[2]) if r[2] is not None else 0.0,
                "importance": float(r[3] or 0.5),
            }
            for r in rows
        ]

        resolved = 0
        for i, cand in enumerate(mems):
            if db.get_memory(cand["id"]) is None:
                continue
            # Compare against nearby newer entries only (cheap window).
            for other in mems[i + 1 : i + 20]:
                if db.get_memory(other["id"]) is None:
                    continue
                conflicts = self._contradiction.detect_contradictions(
                    cand["content"], other["content"]
                )
                if not conflicts:
                    continue
                # The *newer* one wins; the older gets linked as superseded and
                # soft-deleted. The newer memory is linked in place rather than
                # re-written — re-remembering identical content would bounce off
                # the dedup layer and leave the supersession chain dangling.
                newer, older = (
                    (cand, other) if cand["created_at"] >= other["created_at"] else (other, cand)
                )
                if not self._authority_allows(newer["id"], older["id"]):
                    logger.debug(
                        "Skipping contradiction %d vs %d: user-sourced fact "
                        "protected from non-user overwrite",
                        older["id"],
                        newer["id"],
                    )
                    continue
                linked = db.link_supersession(new_id=newer["id"], old_id=older["id"])
                self.memory.forget(older["id"], hard=False)
                resolved += 1
                logger.info(
                    "Resolved contradiction: superseded %d with %d (linked=%s)",
                    older["id"],
                    newer["id"],
                    linked,
                )
                break  # one resolution per candidate keeps the scan linear-ish

        return resolved

    def _authority_allows(self, newer_id: int, older_id: int) -> bool:
        """Guard against non-user content overwriting user-stated facts.

        Returns False when the older memory carries a ``user``-authored source
        and the newer one does not, unless the curator was constructed with
        ``allow_assistant_overwrite_user=True``.
        """
        if self.allow_assistant_overwrite_user:
            return True
        sources = self.memory._db.get_sources_for_memories([newer_id, older_id])
        older_is_user = any(s.get("source") == "user" for s in sources.get(older_id, []))
        newer_is_user = any(s.get("source") == "user" for s in sources.get(newer_id, []))
        return not (older_is_user and not newer_is_user)

    # -- Consolidated pass ---------------------------------------------------

    def curate(
        self,
        *,
        namespace: str | None = None,
        run_consolidate: bool = True,
    ) -> CurateReport:
        """Run the full curation cycle in a deterministic order.

        Order matters: decay first (bound the store), then contradiction
        resolution, then consolidation.
        """
        ns = namespace or self.default_namespace
        report = CurateReport()
        report.decayed = self.decay(namespace=ns)
        if self._resolve_contradictions:
            report.contradictions_resolved = self.resolve_contradictions(namespace=ns)
        if run_consolidate:
            try:
                report.consolidated = self.memory.consolidate()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Consolidation failed: %s", exc)
        if report.decayed or report.contradictions_resolved or report.consolidated:
            logger.info(
                "Curated: decayed=%d contradictions=%d consolidated=%d",
                report.decayed,
                report.contradictions_resolved,
                report.consolidated,
            )
        return report


class CuratorAddon(BaseAddon):
    """Registers :class:`MemoryCurator` as a discoverable Ariadne addon.

    Register this via the ``ariadne.addons`` entry-point group to expose a
    ``ariadne curate`` CLI command automatically.
    """

    @property
    def name(self) -> str:
        return "ariadne-curator"

    @property
    def version(self) -> str:
        return "0.1.0"

    @property
    def description(self) -> str:
        return "Retention, conflict-resolution, and consolidation pass over Ariadne memory."

    def get_cli_commands(self) -> list[Any]:
        from arriadne.addons import CLICommand

        def _curate(args: Any) -> None:
            from arriadne.config import AriadneConfig
            from arriadne.interface import AriadneMemory

            mem = AriadneMemory(config=AriadneConfig(db_path=args.db_path))
            curator = MemoryCurator(mem)
            report = curator.curate()
            print(report.as_dict())
            mem.close()

        return [
            CLICommand(
                name="curate",
                help_text="Run the memory curation cycle (decay + conflict + consolidate)",
                handler=_curate,
            )
        ]

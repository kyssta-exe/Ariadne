"""
Temporal Knowledge Graph

Every fact in Ariadne now has temporal awareness:
- valid_at: When this fact became true
- invalid_at: When this fact stopped being true (superseded/contradicted)
- expired_at: When this fact was soft-deleted
- reference_time: The time context of the conversation where it was mentioned

Inspired by Zep/Graphiti's temporal edge model.

Features:
- Temporal queries: "What was true about X at time T?"
- Automatic invalidation: New facts that supersede old ones
- Soft deletion: Never hard-delete, always preserve history
- Fact versioning: Track how facts evolve over time
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("arriadne.temporal")


@dataclass
class TemporalFact:
    """A fact with full temporal provenance."""

    fact_id: str
    text: str
    subject: str  # Main entity
    predicate: str  # Relationship/type
    object: str  # Target entity or value
    valid_at: float  # When this became true (epoch)
    invalid_at: Optional[float] = None  # When this became false
    expired_at: Optional[float] = None  # When this was soft-deleted
    reference_time: Optional[float] = None  # When it was observed
    confidence: float = 1.0
    source_memory_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_current(self) -> bool:
        """Is this fact currently valid?"""
        now = time.time()
        if self.expired_at and self.expired_at < now:
            return False
        if self.invalid_at and self.invalid_at < now:
            return False
        return True

    @property
    def age_seconds(self) -> float:
        """Age of this fact in seconds."""
        return time.time() - self.valid_at

    def invalidate(self, at_time: Optional[float] = None) -> None:
        """Mark this fact as no longer valid."""
        self.invalid_at = at_time or time.time()
        logger.info(f"Invalidated fact {self.fact_id}: {self.text[:50]}...")

    def expire(self, at_time: Optional[float] = None) -> None:
        """Soft-delete this fact."""
        self.expired_at = at_time or time.time()


class TemporalGraph:
    """
    Temporal knowledge graph with time-aware facts.

    Supports:
    - Adding facts with temporal metadata
    - Querying facts valid at a specific time
    - Automatic invalidation when superseded
    - Fact versioning (track evolution)
    """

    def __init__(self, db_conn: Any):
        """
        Initialize temporal graph with a SQLite connection.

        Args:
            db_conn: sqlite3.Connection for the memory database
        """
        self._conn = db_conn
        self._init_temporal_tables()

    def _init_temporal_tables(self) -> None:
        """Create temporal tables if they don't exist."""
        cursor = self._conn.cursor()

        # Temporal facts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS temporal_facts (
                fact_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                valid_at REAL NOT NULL,
                invalid_at REAL,
                expired_at REAL,
                reference_time REAL,
                confidence REAL DEFAULT 1.0,
                source_memory_ids TEXT DEFAULT '[]',
                metadata TEXT DEFAULT '{}',
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)

        # Indexes for temporal queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tf_subject ON temporal_facts(subject)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tf_valid_at ON temporal_facts(valid_at)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_tf_current ON temporal_facts(invalid_at, expired_at)
        """)

        # Fact relationships (contradiction, update, etc.)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fact_relationships (
                from_fact_id TEXT NOT NULL,
                to_fact_id TEXT NOT NULL,
                relationship TEXT NOT NULL,
                reasoning TEXT DEFAULT '',
                created_at REAL DEFAULT (strftime('%s', 'now')),
                PRIMARY KEY (from_fact_id, to_fact_id, relationship)
            )
        """)

        self._conn.commit()

    def add_fact(
        self,
        text: str,
        subject: str,
        predicate: str,
        obj: str,
        valid_at: Optional[float] = None,
        reference_time: Optional[float] = None,
        confidence: float = 1.0,
        source_memory_id: Optional[str] = None,
    ) -> TemporalFact:
        """
        Add a temporal fact to the graph.

        Automatically checks for contradictions and invalidates old facts.
        """
        import hashlib

        fact_id = hashlib.md5(
            f"{subject}:{predicate}:{obj}:{text}".encode()
        ).hexdigest()[:16]

        valid_at = valid_at or time.time()
        reference_time = reference_time or valid_at

        fact = TemporalFact(
            fact_id=fact_id,
            text=text,
            subject=subject,
            predicate=predicate,
            object=obj,
            valid_at=valid_at,
            reference_time=reference_time,
            confidence=confidence,
            source_memory_ids=[source_memory_id] if source_memory_id else [],
        )

        # Check for existing facts about the same subject+predicate
        existing = self._find_facts(subject=subject, predicate=predicate, current_only=True)

        for old_fact in existing:
            if old_fact.fact_id != fact_id:
                # Invalidate the old fact (superseded)
                old_fact.invalidate(at_time=valid_at)
                self._save_fact(old_fact)

                # Record the relationship
                self._add_relationship(
                    from_fact_id=fact_id,
                    to_fact_id=old_fact.fact_id,
                    relationship="updates",
                    reasoning=f"Newer fact supersedes: {text[:50]}",
                )

        self._save_fact(fact)
        return fact

    def find_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        at_time: Optional[float] = None,
        current_only: bool = True,
        limit: int = 100,
    ) -> List[TemporalFact]:
        """Find facts matching the given criteria."""
        return self._find_facts(
            subject=subject,
            predicate=predicate,
            obj=obj,
            at_time=at_time,
            current_only=current_only,
            limit=limit,
        )

    def get_fact_history(self, fact_id: str) -> List[TemporalFact]:
        """Get the full history of a fact, including invalidated versions."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM temporal_facts WHERE subject = ("
            "SELECT subject FROM temporal_facts WHERE fact_id = ?"
            ") ORDER BY valid_at ASC",
            (fact_id,),
        )
        rows = cursor.fetchall()
        return [self._row_to_fact(row) for row in rows]

    def get_timeline(
        self,
        subject: str,
        limit: int = 50,
    ) -> List[TemporalFact]:
        """Get all facts about a subject, ordered by time."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM temporal_facts WHERE subject = ? ORDER BY valid_at DESC LIMIT ?",
            (subject, limit),
        )
        rows = cursor.fetchall()
        return [self._row_to_fact(row) for row in rows]

    def get_superseded_facts(
        self,
        subject: Optional[str] = None,
        limit: int = 100,
    ) -> List[Tuple[TemporalFact, TemporalFact]]:
        """Get pairs of (old_fact, new_fact) where old was superseded."""
        cursor = self._conn.cursor()
        query = """
            SELECT
                f1.fact_id, f1.text, f1.subject, f1.predicate, f1.object,
                f1.valid_at, f1.invalid_at, f1.expired_at, f1.reference_time,
                f1.confidence, f1.source_memory_ids, f1.metadata,
                f2.fact_id, f2.text, f2.subject, f2.predicate, f2.object,
                f2.valid_at, f2.invalid_at, f2.expired_at, f2.reference_time,
                f2.confidence, f2.source_memory_ids, f2.metadata
            FROM fact_relationships r
            JOIN temporal_facts f1 ON r.from_fact_id = f1.fact_id
            JOIN temporal_facts f2 ON r.to_fact_id = f2.fact_id
            WHERE r.relationship = 'updates'
        """
        params: list = []
        if subject:
            query += " AND f1.subject = ?"
            params.append(subject)
        query += " ORDER BY f2.valid_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        results = []
        for row in rows:
            new_fact = self._row_to_fact(row[:12])
            old_fact = self._row_to_fact(row[12:])
            results.append((old_fact, new_fact))
        return results

    def invalidate_fact(self, fact_id: str) -> bool:
        """Soft-invalidate a fact."""
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE temporal_facts SET invalid_at = ? WHERE fact_id = ?",
            (time.time(), fact_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def expire_fact(self, fact_id: str) -> bool:
        """Soft-delete a fact."""
        cursor = self._conn.cursor()
        cursor.execute(
            "UPDATE temporal_facts SET expired_at = ? WHERE fact_id = ?",
            (time.time(), fact_id),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def stats(self) -> Dict[str, Any]:
        """Get statistics about the temporal graph."""
        cursor = self._conn.cursor()
        now = time.time()

        cursor.execute("SELECT COUNT(*) FROM temporal_facts")
        total = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM temporal_facts WHERE invalid_at IS NULL AND expired_at IS NULL"
        )
        current = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(*) FROM temporal_facts WHERE invalid_at IS NOT NULL AND (expired_at IS NULL OR expired_at > ?)",
            (now,),
        )
        superseded = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM fact_relationships")
        relationships = cursor.fetchone()[0]

        cursor.execute(
            "SELECT COUNT(DISTINCT subject) FROM temporal_facts WHERE invalid_at IS NULL AND expired_at IS NULL"
        )
        subjects = cursor.fetchone()[0]

        return {
            "total_facts": total,
            "current_facts": current,
            "superseded_facts": superseded,
            "relationships": relationships,
            "unique_subjects": subjects,
        }

    def _find_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        at_time: Optional[float] = None,
        current_only: bool = True,
        limit: int = 100,
    ) -> List[TemporalFact]:
        """Internal fact search."""
        conditions = []
        params: list = []

        if subject:
            conditions.append("subject = ?")
            params.append(subject)
        if predicate:
            conditions.append("predicate = ?")
            params.append(predicate)
        if obj:
            conditions.append("object = ?")
            params.append(obj)
        if current_only:
            conditions.append("invalid_at IS NULL")
            conditions.append("expired_at IS NULL")
        if at_time:
            conditions.append("valid_at <= ?")
            params.append(at_time)
            conditions.append("(invalid_at IS NULL OR invalid_at > ?)")
            params.append(at_time)

        where = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM temporal_facts WHERE {where} ORDER BY valid_at DESC LIMIT ?"
        params.append(limit)

        cursor = self._conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [self._row_to_fact(row) for row in rows]

    def _save_fact(self, fact: TemporalFact) -> None:
        """Save a fact to the database."""
        cursor = self._conn.cursor()
        cursor.execute(
            """INSERT OR REPLACE INTO temporal_facts
            (fact_id, text, subject, predicate, object, valid_at, invalid_at,
             expired_at, reference_time, confidence, source_memory_ids, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact.fact_id,
                fact.text,
                fact.subject,
                fact.predicate,
                fact.object,
                fact.valid_at,
                fact.invalid_at,
                fact.expired_at,
                fact.reference_time,
                fact.confidence,
                json.dumps(fact.source_memory_ids),
                json.dumps(fact.metadata),
            ),
        )
        self._conn.commit()

    def _add_relationship(
        self,
        from_fact_id: str,
        to_fact_id: str,
        relationship: str,
        reasoning: str = "",
    ) -> None:
        """Record a relationship between two facts."""
        cursor = self._conn.cursor()
        cursor.execute(
            """INSERT OR IGNORE INTO fact_relationships
            (from_fact_id, to_fact_id, relationship, reasoning)
            VALUES (?, ?, ?, ?)""",
            (from_fact_id, to_fact_id, relationship, reasoning),
        )
        self._conn.commit()

    def _row_to_fact(self, row: tuple) -> TemporalFact:
        """Convert a database row to a TemporalFact."""
        return TemporalFact(
            fact_id=row[0],
            text=row[1],
            subject=row[2],
            predicate=row[3],
            object=row[4],
            valid_at=row[5],
            invalid_at=row[6],
            expired_at=row[7],
            reference_time=row[8],
            confidence=row[9] or 1.0,
            source_memory_ids=json.loads(row[10]) if row[10] else [],
            metadata=json.loads(row[11]) if row[11] else {},
        )

"""Multi-agent API key authentication for Ariadne.

Provides APIKeyManager for creating, validating, revoking, listing, and
rotating API keys stored in SQLite. Keys are prefixed with ``ak_`` and
hashed with SHA-256 before storage.

AuthContext is the per-request context attached to ``request.state.auth``
by the server middleware.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import sqlite3
import time
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("arriadne.auth")


@dataclass
class AuthContext:
    """Per-request authentication context attached to request.state.auth."""

    key_id: str
    agent_name: str
    tenant_id: str
    scopes: List[str] = field(default_factory=lambda: ["read", "write"])
    rate_limit_rpm: int = 120


class APIKeyManager:
    """Manages API keys in a SQLite database.

    The ``api_keys`` table is created automatically on first use.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_table()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _ensure_table(self) -> None:
        conn = self._get_conn()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id              TEXT PRIMARY KEY,
                key_hash        TEXT UNIQUE NOT NULL,
                key_prefix      TEXT NOT NULL,
                agent_name      TEXT NOT NULL,
                tenant_id       TEXT NOT NULL DEFAULT 'default',
                scopes          TEXT NOT NULL DEFAULT '["read","write"]',
                rate_limit_rpm  INTEGER NOT NULL DEFAULT 120,
                created_at      REAL NOT NULL,
                expires_at      REAL,
                revoked_at      REAL,
                last_used_at    REAL,
                use_count       INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.commit()

    @staticmethod
    def _hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def has_keys(self) -> bool:
        """Return True if at least one non-revoked key exists."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) FROM api_keys WHERE revoked_at IS NULL"
        ).fetchone()
        return row[0] > 0

    def create_key(
        self,
        agent_name: str,
        tenant_id: str = "default",
        scopes: Optional[List[str]] = None,
        rate_limit_rpm: int = 120,
        expires_in_seconds: Optional[int] = None,
    ) -> dict:
        """Create a new API key and return its details (raw key shown once)."""
        if scopes is None:
            scopes = ["read", "write"]

        key_id = secrets.token_hex(8)
        raw_key = f"ak_{secrets.token_hex(32)}"
        key_hash = self._hash_key(raw_key)
        key_prefix = raw_key[:12] + "..."

        now = time.time()
        expires_at = now + expires_in_seconds if expires_in_seconds else None

        conn = self._get_conn()
        conn.execute(
            """INSERT INTO api_keys
               (id, key_hash, key_prefix, agent_name, tenant_id, scopes,
                rate_limit_rpm, created_at, expires_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                key_id,
                key_hash,
                key_prefix,
                agent_name,
                tenant_id,
                json.dumps(scopes),
                rate_limit_rpm,
                now,
                expires_at,
            ),
        )
        conn.commit()

        logger.info("Created API key %s for agent=%s tenant=%s", key_id, agent_name, tenant_id)
        return {
            "id": key_id,
            "key": raw_key,
            "key_prefix": key_prefix,
            "agent_name": agent_name,
            "tenant_id": tenant_id,
            "scopes": scopes,
            "rate_limit_rpm": rate_limit_rpm,
            "created_at": now,
            "expires_at": expires_at,
        }

    def validate(self, raw_key: str) -> Optional[AuthContext]:
        """Validate a raw API key. Returns AuthContext or None."""
        if not raw_key.startswith("ak_"):
            return None

        key_hash = self._hash_key(raw_key)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ? AND revoked_at IS NULL",
            (key_hash,),
        ).fetchone()

        if row is None:
            return None

        now = time.time()
        if row["expires_at"] is not None and now > row["expires_at"]:
            return None

        conn.execute(
            "UPDATE api_keys SET last_used_at = ?, use_count = use_count + 1 WHERE id = ?",
            (now, row["id"]),
        )
        conn.commit()

        return AuthContext(
            key_id=row["id"],
            agent_name=row["agent_name"],
            tenant_id=row["tenant_id"],
            scopes=json.loads(row["scopes"]),
            rate_limit_rpm=row["rate_limit_rpm"],
        )

    def revoke(self, key_id: str) -> bool:
        """Revoke a key by id. Returns True if a key was revoked."""
        conn = self._get_conn()
        cursor = conn.execute(
            "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL",
            (time.time(), key_id),
        )
        conn.commit()
        if cursor.rowcount > 0:
            logger.info("Revoked API key %s", key_id)
        return cursor.rowcount > 0

    def list_keys(self) -> List[dict]:
        """Return metadata for all keys (raw key values are never returned)."""
        conn = self._get_conn()
        rows = conn.execute(
            """SELECT id, key_prefix, agent_name, tenant_id, scopes,
                      rate_limit_rpm, created_at, expires_at,
                      revoked_at, last_used_at, use_count
               FROM api_keys ORDER BY created_at DESC"""
        ).fetchall()
        return [
            {
                "id": r["id"],
                "key_prefix": r["key_prefix"],
                "agent_name": r["agent_name"],
                "tenant_id": r["tenant_id"],
                "scopes": json.loads(r["scopes"]),
                "rate_limit_rpm": r["rate_limit_rpm"],
                "created_at": r["created_at"],
                "expires_at": r["expires_at"],
                "revoked_at": r["revoked_at"],
                "last_used_at": r["last_used_at"],
                "use_count": r["use_count"],
            }
            for r in rows
        ]

    def rotate(self, key_id: str) -> Optional[dict]:
        """Rotate a key: revoke the old one, create a new one with same settings.

        Returns the new key details, or None if the old key was not found.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM api_keys WHERE id = ? AND revoked_at IS NULL",
            (key_id,),
        ).fetchone()

        if row is None:
            return None

        conn.execute(
            "UPDATE api_keys SET revoked_at = ? WHERE id = ?",
            (time.time(), key_id),
        )
        conn.commit()

        new_key = self.create_key(
            agent_name=row["agent_name"],
            tenant_id=row["tenant_id"],
            scopes=json.loads(row["scopes"]),
            rate_limit_rpm=row["rate_limit_rpm"],
        )
        logger.info("Rotated API key %s → %s", key_id, new_key["id"])
        return new_key

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

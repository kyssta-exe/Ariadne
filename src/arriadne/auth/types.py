"""Authentication types for Ariadne."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True, slots=True)
class AuthContext:
    """Authenticated context after successful API key validation.

    Attributes:
        key_id: Unique identifier for the API key (e.g. 'ak_myagent_abc123...').
        agent_name: Name of the agent that owns this key.
        tenant_id: Tenant identifier for multi-tenant isolation.
        scopes: List of permission scopes granted to this key.
        rate_limit_rpm: Maximum requests per minute for this key.
    """

    key_id: str
    agent_name: str
    tenant_id: str
    scopes: List[str] = field(default_factory=lambda: ["read", "write"])
    rate_limit_rpm: int = 60

    def has_scope(self, scope: str) -> bool:
        """Check if this context has a specific scope."""
        return scope in self.scopes or "admin" in self.scopes

    def has_any_scope(self, *scopes: str) -> bool:
        """Check if this context has any of the given scopes."""
        if "admin" in self.scopes:
            return True
        return any(s in self.scopes for s in scopes)

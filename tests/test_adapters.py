"""Tests for the Claude Code hook adapter, MCP host configs, and the
dashboard's bearer-token auth + Prometheus metrics endpoint."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from arriadne.interface import AriadneMemory


@pytest.fixture
def mem() -> AriadneMemory:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    memory = AriadneMemory(db_path=db_path, embedding_dim=8)
    memory.remember("deploy target is staging", importance=0.9)
    yield memory
    memory.close()
    for suffix in ["", "-wal", "-shm"]:
        Path(db_path + suffix).unlink(missing_ok=True)


# ── Claude Code hook adapter ────────────────────────────────────────────


class TestHookParsing:
    def test_parse_valid_event(self) -> None:
        from arriadne.integrations.claude_code import parse_hook_event

        event = parse_hook_event('{"hook_event_name": "Stop", "session_id": "s"}')
        assert event["hook_event_name"] == "Stop"

    def test_parse_tolerates_garbage(self) -> None:
        from arriadne.integrations.claude_code import parse_hook_event

        assert parse_hook_event("not json at all") == {}
        assert parse_hook_event("") == {}
        assert parse_hook_event(None) == {}
        assert parse_hook_event(b"\xff\xfe") == {}
        assert parse_hook_event("[1,2,3]") == {}  # non-object top level


class TestHookHandlers:
    def test_user_prompt_submit_injects_context_and_records(self, mem: AriadneMemory) -> None:
        from arriadne.integrations.claude_code import handle_user_prompt_submit

        event = {
            "hook_event_name": "UserPromptSubmit",
            "prompt": "where do we deploy?",
            "session_id": "sess-1",
        }
        out = handle_user_prompt_submit(mem, event, k=3)
        assert "additionalContext" in out["hookSpecificOutput"]
        assert "deploy target is staging" in out["hookSpecificOutput"]["additionalContext"]
        # The prompt itself is recorded as a provenance episode.
        latest = mem._db.get_latest_episode(role="claude_code_user_prompt")
        assert latest is not None and latest["session_id"] == "sess-1"

    def test_user_prompt_submit_empty_prompt_noop(self, mem: AriadneMemory) -> None:
        from arriadne.integrations.claude_code import handle_user_prompt_submit

        assert handle_user_prompt_submit(mem, {"prompt": ""}) == {}
        assert handle_user_prompt_submit(mem, {}) == {}

    def test_stop_records_reply_and_pairs_turn(self, mem: AriadneMemory) -> None:
        from arriadne.integrations.claude_code import (
            handle_stop,
            handle_user_prompt_submit,
        )

        handle_user_prompt_submit(
            mem, {"hook_event_name": "UserPromptSubmit", "prompt": "hi", "session_id": "s9"}
        )
        result = handle_stop(
            mem,
            {
                "hook_event_name": "Stop",
                "session_id": "s9",
                "last_message": "Hello! How can I help?",
            },
        )
        assert result["recorded_reply"] is True
        latest = mem._db.get_latest_episode(role="claude_code_assistant")
        assert latest is not None and "How can I help" in latest["content"]

    def test_handle_event_dispatches_and_noops(self, mem: AriadneMemory) -> None:
        from arriadne.integrations.claude_code import handle_event

        assert handle_event(mem, {"hook_event_name": "PreToolUse"}) == {}
        assert handle_event(mem, {}) == {}
        out = handle_event(
            mem, {"hook_event_name": "UserPromptSubmit", "prompt": "where do we deploy?"}
        )
        assert "hookSpecificOutput" in out


class TestHookRunner:
    def test_run_hook_end_to_end(self, mem: AriadneMemory, capsys: pytest.CaptureFixture) -> None:
        from arriadne.integrations.claude_code import run_hook

        payload = json.dumps(
            {"hook_event_name": "UserPromptSubmit", "prompt": "where do we deploy?"}
        )
        code = run_hook(
            ["--db-path", str(mem._config.db_path)], stdin=payload
        )
        assert code == 0
        out = capsys.readouterr().out.strip()
        assert out  # context was injected
        assert "additionalContext" in json.loads(out)["hookSpecificOutput"]

    def test_run_hook_fail_open_on_garbage(
        self, mem: AriadneMemory, capsys: pytest.CaptureFixture
    ) -> None:
        from arriadne.integrations.claude_code import run_hook

        assert run_hook(["--db-path", str(mem._config.db_path)], stdin="{{{nope") == 0
        assert capsys.readouterr().out == ""

    def test_run_hook_unknown_event_noop(
        self, mem: AriadneMemory, capsys: pytest.CaptureFixture
    ) -> None:
        from arriadne.integrations.claude_code import run_hook

        payload = json.dumps({"hook_event_name": "SomeFutureEvent"})
        assert run_hook(["--db-path", str(mem._config.db_path)], stdin=payload) == 0
        assert capsys.readouterr().out == ""


# ── MCP host configs ────────────────────────────────────────────────────


class TestMcpHostConfigs:
    def test_claude_code_shape(self) -> None:
        from arriadne.integrations.claude_code import mcp_host_config

        cfg = mcp_host_config("claude-code", "mem.db")
        entry = cfg["mcpServers"]["ariadne"]
        assert entry["command"].endswith("python")
        assert "-m" in entry["args"] and "arriadne.integrations.mcp_server" in entry["args"]
        assert any(str(Path("mem.db").resolve()) in a for a in entry["args"])

    def test_all_hosts_produce_valid_json(self) -> None:
        from arriadne.integrations.claude_code import MCP_HOSTS, mcp_host_config

        for host in MCP_HOSTS:
            blob = json.dumps(mcp_host_config(host, "mem.db"))
            assert "mcp_server" in blob

    def test_unknown_host_raises(self) -> None:
        from arriadne.integrations.claude_code import mcp_host_config

        with pytest.raises(ValueError):
            mcp_host_config("not-a-host", "mem.db")

    def test_install_snippet_registers_both_hooks(self) -> None:
        from arriadne.integrations.claude_code import install_snippet

        snippet = install_snippet("mem.db")
        assert set(snippet["hooks"]) == {"UserPromptSubmit", "Stop"}
        cmd = snippet["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
        assert cmd.startswith("ariadne hook claude-code")

    def test_cli_mcp_command(self, capsys: pytest.CaptureFixture) -> None:
        from arriadne.cli import main

        code = main(["mcp", "--host", "cursor"])
        assert code == 0
        out = capsys.readouterr().out
        assert "mcpServers" in out and "ariadne" in out

    def test_cli_mcp_unknown_host_fails(self, capsys: pytest.CaptureFixture) -> None:
        from arriadne.cli import main

        assert main(["mcp", "--host", "nope"]) == 1


# ── Dashboard auth + metrics ────────────────────────────────────────────


@pytest.fixture
def dash(tmp_path: Path):
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from arriadne.dashboard.server import create_app

    db = tmp_path / "dash.db"
    app = create_app(db_path=str(db), auth_token="s3cret-token")
    client = fastapi_testclient.TestClient(app)
    return client, app


class TestDashboardAuth:
    def test_api_requires_token(self, dash) -> None:
        client, _ = dash
        resp = client.get("/api/stats")
        assert resp.status_code == 401
        assert resp.headers["www-authenticate"] == "Bearer"

    def test_api_accepts_valid_token(self, dash) -> None:
        client, _ = dash
        resp = client.get(
            "/api/stats", headers={"Authorization": "Bearer s3cret-token"}
        )
        assert resp.status_code == 200
        assert "active_memories" in resp.json()

    def test_wrong_token_rejected(self, dash) -> None:
        client, _ = dash
        resp = client.get("/api/stats", headers={"Authorization": "Bearer wrong"})
        assert resp.status_code == 401

    def test_health_open_without_token(self, dash) -> None:
        client, _ = dash
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_no_token_means_open_api(self, tmp_path: Path) -> None:
        fastapi_testclient = pytest.importorskip("fastapi.testclient")
        from arriadne.dashboard.server import create_app

        app = create_app(db_path=str(tmp_path / "open.db"), auth_token=None)
        client = fastapi_testclient.TestClient(app)
        assert client.get("/api/stats").status_code == 200


class TestDashboardMetrics:
    def test_metrics_prometheus_text(self, dash) -> None:
        client, _ = dash
        # Generate traffic: an authorized hit, an unauthorized probe, /metrics.
        client.get("/health")
        client.get("/api/stats")  # no token -> 401, counted
        client.get("/api/stats", headers={"Authorization": "Bearer s3cret-token"})
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        body = resp.text
        assert "# TYPE ariadne_memories_active gauge" in body
        assert "ariadne_memories_active 0" in body  # fresh store
        assert "ariadne_dashboard_http_requests_total" in body
        # Both the authenticated request and the rejected probe were counted.
        assert 'status="401"' in body and 'status="200"' in body

    def test_metrics_can_be_disabled(self, tmp_path: Path) -> None:
        fastapi_testclient = pytest.importorskip("fastapi.testclient")
        from arriadne.dashboard.server import create_app

        app = create_app(db_path=str(tmp_path / "nom.db"), enable_metrics=False)
        client = fastapi_testclient.TestClient(app)
        assert client.get("/metrics").status_code == 404

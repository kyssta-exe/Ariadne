"""CLI `doctor` diagnostics command."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from arriadne import AriadneConfig, AriadneMemory
from arriadne.cli import main


def test_doctor_passes_healthy_store(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "m.db"
    with AriadneMemory(config=AriadneConfig(db_path=db_path, embedding_dim=4)) as mem:
        mem.remember("healthy", embedding=np.array([1, 0, 0, 0], dtype=np.float32))

    rc = main(["--db-path", str(db_path), "doctor", "--dim", "4"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "8/8 checks passed" in out
    assert "[PASS] sqlite integrity: ok" in out


def test_doctor_fails_missing_database(tmp_path: Path, capsys) -> None:
    rc = main(["--db-path", str(tmp_path / "nope.db"), "doctor"])
    out = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL] database file exists" in out

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from etl_sar.cli import app


ROOT = Path(__file__).resolve().parents[1]


def test_formal_dry_run_writes_complete_matrix_without_myosuite(tmp_path) -> None:
    output = tmp_path / "matrix.json"
    result = CliRunner().invoke(
        app,
        [
            "formal-dry-run",
            "--hand-config",
            str(ROOT / "configs" / "formal_hand.yaml"),
            "--leg-config",
            str(ROOT / "configs" / "formal_leg.yaml"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"] == {
        "source_stage_count": 10,
        "target_job_count": 30,
        "total_declared_interactions": 525_000_000,
    }

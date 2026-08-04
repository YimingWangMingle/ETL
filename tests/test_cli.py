from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from etl_sar.cli import app


def write_config(path: Path) -> Path:
    config = path / "hand.yaml"
    config.write_text(
        """
name: hand_quick
limb: hand
seed: 7
source:
  env_id: myoHandReorient8-v0
  role: source
  limb: hand
target:
  env_id: myoHandReorient100-v0
  role: target
  limb: hand
""".strip(),
        encoding="utf-8",
    )
    return config


def test_inspect_prints_roles_and_exact_tasks(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["inspect", "--config", str(write_config(tmp_path))])

    assert result.exit_code == 0
    assert "ETL core" in result.stdout
    assert "SAR transfer" in result.stdout
    assert "SB3 engineering" in result.stdout
    assert "myoHandReorient8-v0" in result.stdout
    assert "myoHandReorient100-v0" in result.stdout


def test_help_lists_complete_pipeline_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in (
        "explore", "fit-representation", "transfer", "evaluate", "compare"
    ):
        assert command in result.stdout

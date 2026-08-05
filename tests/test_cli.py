from __future__ import annotations

from pathlib import Path

import pytest

from typer.testing import CliRunner

from etl_sar.cli import _resolve_sar_scale, app


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


def test_explore_rejects_minimum_over_maximum_budget(tmp_path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "explore",
            "--config",
            str(write_config(tmp_path)),
            "--run-dir",
            str(tmp_path / "run"),
            "--timesteps",
            "32",
            "--min-timesteps",
            "64",
        ],
    )

    assert result.exit_code != 0
    assert "cannot exceed" in result.output


def test_runtime_sar_scale_overrides_bundle_value() -> None:
    assert _resolve_sar_scale(1.0, 0.0) == 0.0
    assert _resolve_sar_scale(1.0, None) == 1.0


def test_runtime_sar_scale_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _resolve_sar_scale(1.0, 1.1)

    with pytest.raises(ValueError, match="between 0 and 1"):
        _resolve_sar_scale(1.0, -0.1)

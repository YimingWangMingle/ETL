from __future__ import annotations

from typer.testing import CliRunner

from etl_sar.cli import app


def test_evaluate_cli_accepts_only_paired_checkpoint_manifest() -> None:
    result = CliRunner().invoke(app, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "--pair-manifest" in result.output
    assert "--model-path" not in result.output


def test_transfer_reports_manifest_as_checkpoint_contract(monkeypatch, tmp_path) -> None:
    result = CliRunner().invoke(app, ["transfer", "--help"])
    assert result.exit_code == 0
    assert "paired" in result.output.lower()

from typer.testing import CliRunner

from etl_sar.cli import app


def test_formal_etl_evaluate_requires_paired_manifest_and_seed_bank_options() -> None:
    result = CliRunner().invoke(app, ["formal-etl-evaluate", "--help"])
    assert result.exit_code == 0
    assert "--pair-manifest" in result.output
    assert "--episodes" in result.output
    assert "--environment-steps" in result.output

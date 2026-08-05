from typer.testing import CliRunner

from etl_sar.cli import app


def test_transfer_exposes_formal_evaluation_and_resume_options() -> None:
    result = CliRunner().invoke(app, ["transfer", "--help"])
    assert result.exit_code == 0
    assert "--evaluation-episodes" in result.output
    assert "--evaluation-seed" in result.output
    assert "--resume-manifest" in result.output

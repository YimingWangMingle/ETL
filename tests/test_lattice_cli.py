from typer.testing import CliRunner

from etl_sar.cli import app


def test_formal_lattice_train_exposes_fixed_protocol_options() -> None:
    result = CliRunner().invoke(app, ["formal-lattice-train", "--help"])
    assert result.exit_code == 0
    for option in (
        "--domain",
        "--env-id",
        "--timesteps",
        "--checkpoint-interval",
        "--evaluation-episodes",
        "--final-episodes",
        "--num-envs",
        "--run-dir",
    ):
        assert option in result.output

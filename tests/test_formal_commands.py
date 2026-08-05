from pathlib import Path

from etl_sar.formal.config import FormalDomainConfig
from etl_sar.formal.matrix import ExperimentMatrix, Method
from etl_sar.formal.runner import commands_for_job


ROOT = Path(__file__).resolve().parents[1]


def jobs():
    matrix = ExperimentMatrix.from_configs(
        [
            FormalDomainConfig.from_yaml(ROOT / "configs" / "formal_hand.yaml"),
            FormalDomainConfig.from_yaml(ROOT / "configs" / "formal_leg.yaml"),
        ]
    )
    return matrix.target_jobs


def test_etl_commands_share_source_and_differ_only_by_sar_switch(tmp_path) -> None:
    no_sar = next(job for job in jobs() if job.method is Method.ETL_NO_SAR)
    sar = next(job for job in jobs() if job.method is Method.ETL_SAR)
    no_sar_commands = commands_for_job(no_sar, output_root=tmp_path)
    sar_commands = commands_for_job(sar, output_root=tmp_path)
    assert no_sar_commands[0:2] == sar_commands[0:2]
    assert "0.0" in no_sar_commands[2]
    assert "1.0" in sar_commands[2]
    assert "--evaluation-episodes" in sar_commands[2]
    assert "20" in sar_commands[2]
    assert "--eval-freq" in sar_commands[2]
    assert "250000" in sar_commands[2]


def test_lattice_command_uses_target_only_budget_and_official_domain(tmp_path) -> None:
    leg = next(
        job
        for job in jobs()
        if job.method is Method.LATTICE and job.domain == "leg"
    )
    commands = commands_for_job(leg, output_root=tmp_path)
    assert len(commands) == 1
    command = commands[0]
    assert command[1] == "formal-lattice-train"
    assert "15000000" in command
    assert "myoLegRoughTerrainWalk-v0" in command

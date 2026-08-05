from __future__ import annotations

from pathlib import Path

from etl_sar.formal.config import FormalDomainConfig
from etl_sar.formal.matrix import ExperimentMatrix, Method
from etl_sar.formal.runner import JobState, required_resume_artifacts


ROOT = Path(__file__).resolve().parents[1]


def matrix() -> ExperimentMatrix:
    return ExperimentMatrix.from_configs(
        [
            FormalDomainConfig.from_yaml(ROOT / "configs" / "formal_hand.yaml"),
            FormalDomainConfig.from_yaml(ROOT / "configs" / "formal_leg.yaml"),
        ]
    )


def test_resume_requirements_are_method_specific() -> None:
    etl = next(job for job in matrix().target_jobs if job.method is Method.ETL_SAR)
    lattice_leg = next(
        job
        for job in matrix().target_jobs
        if job.method is Method.LATTICE and job.domain == "leg"
    )
    assert required_resume_artifacts(etl) == {
        "latest_pair.json",
        "latest_policy.zip",
        "latest_action_model.pt",
        "transition_count.json",
    }
    assert required_resume_artifacts(lattice_leg) == {
        "latest_policy.zip",
        "latest_replay_buffer.pkl",
        "latest_vecnormalize.pkl",
        "transition_count.json",
    }


def test_job_state_skips_only_verified_complete_manifest(tmp_path) -> None:
    job = matrix().target_jobs[0]
    state = JobState(job=job, run_dir=tmp_path)
    assert state.should_skip() is False
    assert state.resume_from == 0
    (tmp_path / "transition_count.json").write_text(
        '{"transitions": 250000}', encoding="utf-8"
    )
    for name in required_resume_artifacts(job) - {"transition_count.json"}:
        (tmp_path / name).write_bytes(b"checkpoint")

    assert state.resume_from == 250_000

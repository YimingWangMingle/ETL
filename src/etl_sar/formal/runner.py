from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from etl_sar.formal.manifest import verify_manifest
from etl_sar.formal.matrix import Method, TargetJob


def commands_for_job(
    job: TargetJob,
    *,
    output_root: str | Path,
    executable: str = "etl-sar",
) -> list[list[str]]:
    root = Path(output_root)
    target_dir = root / "jobs" / job.job_id
    if job.method is Method.LATTICE:
        return [
            [
                executable,
                "formal-lattice-train",
                "--domain",
                job.domain,
                "--env-id",
                job.environment_id,
                "--seed",
                str(job.seed),
                "--timesteps",
                str(job.target_transitions),
                "--checkpoint-interval",
                str(job.checkpoint_interval),
                "--evaluation-episodes",
                str(job.intermediate_episodes),
                "--final-episodes",
                str(job.final_episodes),
                "--num-envs",
                str(job.num_envs),
                "--run-dir",
                str(target_dir),
            ]
        ]
    if job.source_stage_id is None:
        raise ValueError("ETL target job is missing its source stage")
    source_dir = root / "sources" / job.source_stage_id
    experiment = source_dir / "experiment.yaml"
    explore_dir = source_dir / "explore"
    bundle_dir = source_dir / "bundle"
    sar_scale = "1.0" if job.method is Method.ETL_SAR else "0.0"
    evaluation_seed = 10_000_000 + (0 if job.domain == "hand" else 1_000_000)
    evaluation_seed += job.seed * 10_000
    return [
        [
            executable,
            "explore",
            "--config",
            str(experiment),
            "--run-dir",
            str(explore_dir),
            "--timesteps",
            str(job.source_transitions),
            "--min-timesteps",
            str(job.source_transitions),
            "--min-success-actions",
            "20",
        ],
        [
            executable,
            "fit-representation",
            "--config",
            str(experiment),
            "--data-dir",
            str(explore_dir / "data"),
            "--explore-checkpoint",
            str(explore_dir / "representation.pt"),
            "--output-dir",
            str(bundle_dir),
        ],
        [
            executable,
            "transfer",
            "--config",
            str(experiment),
            "--bundle",
            str(bundle_dir / "representation_bundle.pt"),
            "--run-dir",
            str(target_dir),
            "--timesteps",
            str(job.target_transitions),
            "--sar-scale",
            sar_scale,
            "--eval-freq",
            str(job.checkpoint_interval),
            "--evaluation-episodes",
            str(job.intermediate_episodes),
            "--evaluation-seed",
            str(evaluation_seed),
        ],
    ]


def required_resume_artifacts(job: TargetJob) -> set[str]:
    if job.method in {Method.ETL_NO_SAR, Method.ETL_SAR}:
        return {
            "latest_pair.json",
            "latest_policy.zip",
            "latest_action_model.pt",
            "transition_count.json",
        }
    required = {
        "latest_policy.zip",
        "latest_vecnormalize.pkl",
        "transition_count.json",
    }
    if job.domain == "leg":
        required.add("latest_replay_buffer.pkl")
    return required


@dataclass(frozen=True)
class JobState:
    job: TargetJob
    run_dir: Path

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def should_skip(self) -> bool:
        if not self.manifest_path.is_file():
            return False
        try:
            verify_manifest(self.manifest_path)
        except (ValueError, OSError, json.JSONDecodeError):
            return False
        return True

    @property
    def resume_from(self) -> int:
        path = self.run_dir / "transition_count.json"
        if not path.is_file():
            return 0
        payload = json.loads(path.read_text(encoding="utf-8"))
        transitions = int(payload["transitions"])
        if transitions < 0 or transitions > self.job.target_transitions:
            raise ValueError("resume transition count is outside the target budget")
        if transitions and not all(
            (self.run_dir / name).is_file()
            for name in required_resume_artifacts(self.job) - {"transition_count.json"}
        ):
            raise ValueError("resume checkpoint is incomplete for method")
        return transitions

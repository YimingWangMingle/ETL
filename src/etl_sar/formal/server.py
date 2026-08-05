from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import yaml

from etl_sar.formal.aggregate import write_aggregate
from etl_sar.formal.config import FormalDomainConfig
from etl_sar.formal.manifest import RunManifest, finalize_manifest, verify_manifest
from etl_sar.formal.matrix import ExperimentMatrix, Method, SourceStage, TargetJob
from etl_sar.formal.runner import commands_for_job


LATTICE_COMMIT = "846d02fa993b9b80ce5ecb806463e0a05711bad3"
TASKS = {
    "hand": ("myoHandReorient8-v0", "myoHandReorient100-v0"),
    "leg": ("myoLegWalk-v0", "myoLegRoughTerrainWalk-v0"),
}


def select_source(matrix: ExperimentMatrix, index: int) -> SourceStage:
    try:
        return matrix.source_stages[index]
    except IndexError as error:
        raise ValueError(f"source array index must be 0..{len(matrix.source_stages) - 1}") from error


def select_target(matrix: ExperimentMatrix, index: int) -> TargetJob:
    try:
        return matrix.target_jobs[index]
    except IndexError as error:
        raise ValueError(f"target array index must be 0..{len(matrix.target_jobs) - 1}") from error


def write_experiment_config(source: SourceStage, path: str | Path) -> Path:
    source_env, target_env = TASKS[source.domain]
    payload = {
        "name": f"formal_{source.domain}_seed{source.seed}",
        "limb": source.domain,
        "seed": source.seed,
        "source": {
            "env_id": source_env,
            "role": "source",
            "limb": source.domain,
        },
        "target": {
            "env_id": target_env,
            "role": "target",
            "limb": source.domain,
        },
        "representation": {"latent_dim": 20, "mixture_components": 20},
        "synergy": {"components": 20, "rho": 0.2, "enabled_scale": 1.0},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def load_matrix(hand_config: Path, leg_config: Path) -> ExperimentMatrix:
    return ExperimentMatrix.from_configs(
        [
            FormalDomainConfig.from_yaml(hand_config),
            FormalDomainConfig.from_yaml(leg_config),
        ]
    )


def _run(command: Sequence[str]) -> None:
    subprocess.run(list(command), check=True)


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _packages() -> dict[str, str]:
    names = ("myosuite", "mujoco", "gymnasium", "stable-baselines3", "sb3-contrib", "torch")
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def environment_fingerprint(env_id: str) -> str:
    import gymnasium as gym
    import myosuite  # noqa: F401

    env = gym.make(env_id)
    try:
        model = getattr(env.unwrapped, "model", None)
        payload = {
            "env_id": env_id,
            "spec": str(env.spec),
            "observation_space": str(env.observation_space),
            "action_space": str(env.action_space),
            "model": {
                name: int(getattr(model, name))
                for name in ("nq", "nv", "nu", "nbody", "njnt")
                if model is not None and hasattr(model, name)
            },
            "packages": _packages(),
        }
    finally:
        env.close()
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _manifest(
    *,
    repo_root: Path,
    job_id: str,
    method: str,
    domain: str,
    seed: int,
    source_transitions: int,
    target_transitions: int,
    environment_id: str,
    command: Sequence[str],
) -> RunManifest:
    return RunManifest(
        schema_version=1,
        job_id=job_id,
        method=method,
        domain=domain,
        seed=seed,
        source_transitions=source_transitions,
        target_transitions=target_transitions,
        git_commit=_git_commit(repo_root),
        lattice_commit=LATTICE_COMMIT,
        environment_id=environment_id,
        environment_fingerprint=environment_fingerprint(environment_id),
        command=list(command),
        python_version=platform.python_version(),
        packages=_packages(),
        hardware={
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
    )


def run_source(
    source: SourceStage,
    *,
    matrix: ExperimentMatrix,
    output_root: Path,
    repo_root: Path,
    executable: str,
) -> None:
    source_dir = output_root / "sources" / source.stage_id
    manifest_path = source_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            verify_manifest(manifest_path)
            return
        except ValueError:
            pass
    write_experiment_config(source, source_dir / "experiment.yaml")
    representative = next(
        job
        for job in matrix.target_jobs
        if job.source_stage_id == source.stage_id and job.method is Method.ETL_NO_SAR
    )
    commands = commands_for_job(
        representative, output_root=output_root, executable=executable
    )[:2]
    started = time.perf_counter()
    for command in commands:
        _run(command)
    elapsed = time.perf_counter() - started
    artifacts = [
        source_dir / "experiment.yaml",
        source_dir / "explore" / "representation.pt",
        source_dir / "bundle" / "representation_bundle.pt",
        source_dir / "bundle" / "synergy.joblib",
    ]
    manifest = _manifest(
        repo_root=repo_root,
        job_id=source.stage_id,
        method="etl_source",
        domain=source.domain,
        seed=source.seed,
        source_transitions=source.transitions,
        target_transitions=0,
        environment_id=source.environment_id,
        command=[part for command in commands for part in command],
    )
    finalize_manifest(
        manifest,
        manifest_path,
        artifacts=artifacts,
        resource_usage={
            "wall_seconds": elapsed,
            "transitions_per_second": source.transitions / max(elapsed, 1e-9),
        },
    )


def run_target(
    job: TargetJob,
    *,
    output_root: Path,
    repo_root: Path,
    executable: str,
) -> None:
    run_dir = output_root / "jobs" / job.job_id
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        try:
            verify_manifest(manifest_path)
            return
        except ValueError:
            pass
    commands = commands_for_job(job, output_root=output_root, executable=executable)
    if job.method is not Method.LATTICE:
        source_manifest = output_root / "sources" / str(job.source_stage_id) / "manifest.json"
        verify_manifest(source_manifest)
        commands = [commands[2]]
        latest_pair = run_dir / "latest_pair.json"
        if latest_pair.is_file():
            commands[0] += ["--resume-manifest", str(latest_pair)]
    started = time.perf_counter()
    for command in commands:
        _run(command)
    if job.method is not Method.LATTICE:
        source_dir = output_root / "sources" / str(job.source_stage_id)
        evaluation = [
            executable,
            "formal-etl-evaluate",
            "--config",
            str(source_dir / "experiment.yaml"),
            "--bundle",
            str(source_dir / "bundle" / "representation_bundle.pt"),
            "--pair-manifest",
            str(run_dir / "latest_pair.json"),
            "--output-dir",
            str(run_dir / "evaluation" / "final"),
            "--episodes",
            str(job.final_episodes),
            "--environment-steps",
            str(job.target_transitions),
        ]
        _run(evaluation)
        commands.append(evaluation)
    elapsed = time.perf_counter() - started
    artifacts = [
        run_dir / "latest_policy.zip",
        run_dir / "evaluation" / "final" / "episodes.csv",
        run_dir / "evaluation" / "final" / "summary.json",
    ]
    if job.method is Method.LATTICE:
        artifacts.append(run_dir / "latest_vecnormalize.pkl")
        if job.domain == "leg":
            artifacts.append(run_dir / "latest_replay_buffer.pkl")
    else:
        artifacts.extend(
            [run_dir / "latest_action_model.pt", run_dir / "latest_pair.json"]
        )
    manifest = _manifest(
        repo_root=repo_root,
        job_id=job.job_id,
        method=job.method.value,
        domain=job.domain,
        seed=job.seed,
        source_transitions=job.source_transitions,
        target_transitions=job.target_transitions,
        environment_id=job.environment_id,
        command=[part for command in commands for part in command],
    )
    finalize_manifest(
        manifest,
        manifest_path,
        artifacts=artifacts,
        resource_usage={
            "wall_seconds": elapsed,
            "transitions_per_second": job.target_transitions / max(elapsed, 1e-9),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Formal ETL/SAR/Lattice server runner")
    parser.add_argument("mode", choices=("dry-run", "source", "target"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("runs/formal"))
    parser.add_argument("--hand-config", type=Path, default=Path("configs/formal_hand.yaml"))
    parser.add_argument("--leg-config", type=Path, default=Path("configs/formal_leg.yaml"))
    parser.add_argument("--executable", default="etl-sar")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    matrix = load_matrix(args.hand_config, args.leg_config)
    if args.mode == "dry-run":
        print(json.dumps(matrix.to_mapping(), indent=2))
        return 0
    if args.mode == "source":
        run_source(
            select_source(matrix, args.index),
            matrix=matrix,
            output_root=args.output_root,
            repo_root=repo_root,
            executable=args.executable,
        )
    else:
        run_target(
            select_target(matrix, args.index),
            output_root=args.output_root,
            repo_root=repo_root,
            executable=args.executable,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from etl_sar.dmc.config import DMCTransferConfig
from etl_sar.dmc.matrix import DMCExperimentMatrix, DMCMethod
from etl_sar.dmc.runtime import train_source_stage, train_target_job
from etl_sar.formal.metrics import normalized_auc


def load_matrix(humanoid_config: Path, dog_config: Path) -> DMCExperimentMatrix:
    return DMCExperimentMatrix.from_configs(
        (
            DMCTransferConfig.from_yaml(humanoid_config),
            DMCTransferConfig.from_yaml(dog_config),
        )
    )


def run_source(
    matrix: DMCExperimentMatrix,
    *,
    index: int,
    output_root: Path,
    device: str,
) -> Path:
    try:
        stage = matrix.sources[index]
    except IndexError as error:
        raise ValueError(f"source index must be 0..{len(matrix.sources) - 1}") from error
    config = matrix.config_for(stage.domain)
    run_dir = output_root / "sources" / stage.stage_id
    (run_dir / "config.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config.to_mapping(), indent=2), encoding="utf-8"
    )
    return train_source_stage(config, run_dir=run_dir, device=device)


def run_target(
    matrix: DMCExperimentMatrix,
    *,
    index: int,
    output_root: Path,
    device: str,
) -> Path:
    try:
        job = matrix.jobs[index]
    except IndexError as error:
        raise ValueError(f"target index must be 0..{len(matrix.jobs) - 1}") from error
    config = matrix.config_for(job.domain)
    bundle = None
    if job.source_stage_id is not None:
        bundle = (
            output_root
            / "sources"
            / job.source_stage_id
            / "representation_bundle.pt"
        )
        if not bundle.is_file():
            raise ValueError(f"source bundle is missing for {job.job_id}")
    run_dir = output_root / "jobs" / job.job_id
    (run_dir / "job.json").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "job.json").write_text(
        json.dumps(job.to_mapping(), indent=2), encoding="utf-8"
    )
    return train_target_job(
        config,
        method=job.method,
        run_dir=run_dir,
        bundle_path=bundle,
        device=device,
    )


def aggregate(matrix: DMCExperimentMatrix, output_root: Path) -> Path:
    rows: list[dict[str, object]] = []
    for job in matrix.jobs:
        evaluation = output_root / "jobs" / job.job_id / "evaluation"
        curve: list[tuple[int, float]] = []
        for directory in evaluation.glob("checkpoint_*"):
            try:
                transitions = int(directory.name.removeprefix("checkpoint_"))
            except ValueError:
                continue
            summary_path = directory / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                curve.append((transitions, float(summary["mean_return"])))
        curve.sort()
        if not curve or curve[-1][0] != job.total_budget:
            raise ValueError(f"incomplete learning curve for {job.job_id}")
        transitions = [0]
        returns = [0.0]
        if job.source_budget:
            transitions.append(job.source_budget)
            returns.append(0.0)
        transitions.extend(point[0] for point in curve)
        returns.extend(point[1] for point in curve)
        final = json.loads(
            (evaluation / "final" / "summary.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "domain": job.domain,
                "method": job.method.value,
                "seed": job.seed,
                "return_auc": normalized_auc(
                    transitions=transitions,
                    values=returns,
                    total_budget=job.total_budget,
                ),
                "final_mean_return": float(final["mean_return"]),
            }
        )
    domains: dict[str, object] = {}
    for config in matrix.configs:
        by_method = {
            str(row["method"]): row for row in rows if row["domain"] == config.domain
        }
        treatment = by_method[DMCMethod.ETL_SAR.value]
        domains[config.domain] = {
            "methods": by_method,
            "etl_sar_deltas": {
                baseline.value: {
                    "return_auc": float(treatment["return_auc"])
                    - float(by_method[baseline.value]["return_auc"]),
                    "final_mean_return": float(treatment["final_mean_return"])
                    - float(by_method[baseline.value]["final_mean_return"]),
                }
                for baseline in (DMCMethod.ETL_NO_SAR, DMCMethod.LATTICE)
            },
        }
    result_dir = output_root / "aggregate"
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output = result_dir / "summary.json"
    output.write_text(
        json.dumps(
            {
                "analysis_mode": "descriptive_single_seed",
                "primary_metric": "mean_episode_return",
                "learning_metric": "return_auc_over_charged_interactions",
                "statistical_significance": None,
                "domains": domains,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DMC ETL/SAR/Lattice pilot")
    parser.add_argument("mode", choices=("dry-run", "source", "target", "run", "aggregate"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("runs/dmc_transfer_pilot"))
    parser.add_argument(
        "--humanoid-config",
        type=Path,
        default=Path("configs/dmc_humanoid_pilot.yaml"),
    )
    parser.add_argument(
        "--dog-config", type=Path, default=Path("configs/dmc_dog_pilot.yaml")
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    matrix = load_matrix(args.humanoid_config, args.dog_config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "dry-run":
        print(json.dumps(matrix.to_mapping(), indent=2))
    elif args.mode == "source":
        print(run_source(matrix, index=args.index, output_root=args.output_root, device=args.device))
    elif args.mode == "target":
        print(run_target(matrix, index=args.index, output_root=args.output_root, device=args.device))
    elif args.mode == "aggregate":
        print(aggregate(matrix, args.output_root))
    else:
        for index, source in enumerate(matrix.sources):
            print(f"[source {index + 1}/{len(matrix.sources)}] {source.stage_id}", flush=True)
            run_source(matrix, index=index, output_root=args.output_root, device=args.device)
        for index, job in enumerate(matrix.jobs):
            print(f"[target {index + 1}/{len(matrix.jobs)}] {job.job_id}", flush=True)
            run_target(matrix, index=index, output_root=args.output_root, device=args.device)
        print(aggregate(matrix, args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

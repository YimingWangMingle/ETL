from __future__ import annotations

import csv
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from etl_sar.formal.matrix import ExperimentMatrix
from etl_sar.formal.metrics import normalized_auc
from etl_sar.formal.statistics import aggregate_values, holm_adjust, paired_effect


@dataclass(frozen=True)
class SeedResult:
    domain: str
    method: str
    seed: int
    normalized_auc: float
    final_primary: float


def _paired_sign_flip_p(treatment: list[float], baseline: list[float]) -> float:
    differences = np.asarray(treatment) - np.asarray(baseline)
    observed = abs(float(np.mean(differences)))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        values.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(values) >= observed - 1e-15))


def _expected_seeds(
    rows: tuple[SeedResult, ...],
    expected_seeds_by_domain: Mapping[str, Sequence[int]] | None,
) -> dict[str, tuple[int, ...]]:
    domains = ("hand", "leg")
    if expected_seeds_by_domain is None:
        expected = {
            domain: tuple(sorted({row.seed for row in rows if row.domain == domain}))
            for domain in domains
        }
    else:
        if set(expected_seeds_by_domain) != set(domains):
            raise ValueError("expected seeds must be provided for hand and leg")
        expected = {
            domain: tuple(int(seed) for seed in expected_seeds_by_domain[domain])
            for domain in domains
        }
    for domain, seeds in expected.items():
        if not seeds or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
            raise ValueError(f"expected seeds for {domain} are invalid")
    return expected


def _ordered_methods(
    rows: tuple[SeedResult, ...], domain: str, expected_seeds: tuple[int, ...]
) -> dict[str, list[SeedResult]]:
    domain_rows = [row for row in rows if row.domain == domain]
    by_method: dict[str, list[SeedResult]] = {}
    for method in ("etl_no_sar", "etl_sar", "lattice"):
        ordered = sorted(
            (row for row in domain_rows if row.method == method),
            key=lambda row: row.seed,
        )
        if [row.seed for row in ordered] != list(expected_seeds):
            raise ValueError(
                f"{domain}/{method} must contain seeds {list(expected_seeds)}"
            )
        by_method[method] = ordered
    return by_method


def _summarize_single_seed(
    rows: tuple[SeedResult, ...], expected: Mapping[str, tuple[int, ...]]
) -> dict[str, Any]:
    domains: dict[str, Any] = {}
    for domain in ("hand", "leg"):
        by_method = _ordered_methods(rows, domain, expected[domain])
        methods = {
            method: {
                "seed": results[0].seed,
                "normalized_auc": results[0].normalized_auc,
                "final_primary": results[0].final_primary,
            }
            for method, results in by_method.items()
        }
        treatment = by_method["etl_sar"][0]
        comparisons = {}
        for baseline_name in ("etl_no_sar", "lattice"):
            baseline = by_method[baseline_name][0]
            comparisons[baseline_name] = {
                "auc_delta": treatment.normalized_auc - baseline.normalized_auc,
                "final_delta": treatment.final_primary - baseline.final_primary,
            }
        domains[domain] = {"methods": methods, "comparisons": comparisons}
    return {
        "analysis_mode": "descriptive_single_seed",
        "domains": domains,
        "protocol_success": None,
    }


def summarize_seed_results(
    records: Iterable[SeedResult],
    *,
    bootstrap_seed: int,
    expected_seeds_by_domain: Mapping[str, Sequence[int]] | None = None,
) -> dict[str, Any]:
    rows = tuple(records)
    expected = _expected_seeds(rows, expected_seeds_by_domain)
    if all(len(seeds) == 1 for seeds in expected.values()):
        return _summarize_single_seed(rows, expected)
    if any(len(seeds) == 1 for seeds in expected.values()):
        raise ValueError("hand and leg must use the same aggregation mode")
    domains: dict[str, Any] = {}
    protocol_success = True
    for domain in ("hand", "leg"):
        methods: dict[str, Any] = {}
        by_method = _ordered_methods(rows, domain, expected[domain])
        for method in ("etl_no_sar", "etl_sar", "lattice"):
            ordered = by_method[method]
            methods[method] = {
                "auc": asdict(
                    aggregate_values(
                        [row.normalized_auc for row in ordered],
                        bootstrap_seed=bootstrap_seed,
                    )
                ),
                "final_primary": asdict(
                    aggregate_values(
                        [row.final_primary for row in ordered],
                        bootstrap_seed=bootstrap_seed + 1,
                    )
                ),
            }
        treatment = by_method["etl_sar"]
        raw_p: dict[str, float] = {}
        comparisons: dict[str, Any] = {}
        for offset, baseline_name in enumerate(("etl_no_sar", "lattice")):
            baseline = by_method[baseline_name]
            treatment_auc = [row.normalized_auc for row in treatment]
            baseline_auc = [row.normalized_auc for row in baseline]
            treatment_final = [row.final_primary for row in treatment]
            baseline_final = [row.final_primary for row in baseline]
            auc_effect = paired_effect(
                treatment_auc,
                baseline_auc,
                bootstrap_seed=bootstrap_seed + 10 + offset,
            )
            final_effect = paired_effect(
                treatment_final,
                baseline_final,
                bootstrap_seed=bootstrap_seed + 20 + offset,
            )
            raw_p[baseline_name] = _paired_sign_flip_p(treatment_auc, baseline_auc)
            comparisons[baseline_name] = {
                "auc_effect": asdict(auc_effect),
                "final_effect": asdict(final_effect),
                "raw_p": raw_p[baseline_name],
            }
            protocol_success = protocol_success and (
                auc_effect.mean > 0
                and auc_effect.ci_low > 0
                and final_effect.mean >= 0
            )
        adjusted = holm_adjust(raw_p)
        for baseline_name, value in adjusted.items():
            comparisons[baseline_name]["holm_p"] = value
        domains[domain] = {"methods": methods, "comparisons": comparisons}
    return {"domains": domains, "protocol_success": bool(protocol_success)}


def collect_seed_results(
    matrix: ExperimentMatrix, output_root: str | Path
) -> list[SeedResult]:
    root = Path(output_root)
    records: list[SeedResult] = []
    for job in matrix.target_jobs:
        evaluation_dir = root / "jobs" / job.job_id / "evaluation"
        curve = []
        for directory in evaluation_dir.glob("checkpoint_*"):
            try:
                transitions = int(directory.name.removeprefix("checkpoint_"))
            except ValueError:
                continue
            summary_path = directory / "summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                curve.append((transitions, float(summary["mean_primary"])))
        curve.sort()
        if not curve:
            raise ValueError(f"missing learning curve for {job.job_id}")
        transitions = [0, *[point[0] for point in curve]]
        values = [0.0, *[point[1] for point in curve]]
        if transitions[-1] != job.target_transitions:
            raise ValueError(f"learning curve does not end at budget for {job.job_id}")
        final = json.loads(
            (evaluation_dir / "final" / "summary.json").read_text(encoding="utf-8")
        )
        records.append(
            SeedResult(
                domain=job.domain,
                method=job.method.value,
                seed=job.seed,
                normalized_auc=normalized_auc(
                    transitions=transitions,
                    values=values,
                    total_budget=job.target_transitions,
                ),
                final_primary=float(final["mean_primary"]),
            )
        )
    return records


def write_aggregate(
    matrix: ExperimentMatrix,
    output_root: str | Path,
    *,
    bootstrap_seed: int = 20260806,
) -> Path:
    root = Path(output_root)
    records = collect_seed_results(matrix, root)
    aggregate = summarize_seed_results(
        records,
        bootstrap_seed=bootstrap_seed,
        expected_seeds_by_domain={
            config.domain: config.seeds for config in matrix.configs
        },
    )
    result_dir = root / "aggregate"
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "per_seed.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)
    output = result_dir / "summary.json"
    output.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    return output

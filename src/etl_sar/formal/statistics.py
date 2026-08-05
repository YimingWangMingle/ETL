from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AggregateStats:
    count: int
    mean: float
    se: float
    median: float
    ci_low: float
    ci_high: float


def aggregate_values(
    values: Sequence[float],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int = 10_000,
) -> AggregateStats:
    data = np.asarray(values, dtype=np.float64)
    if data.ndim != 1 or len(data) == 0 or not np.isfinite(data).all():
        raise ValueError("statistics require a nonempty finite one-dimensional sample")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(data), size=(bootstrap_samples, len(data)))
    boot_means = data[indices].mean(axis=1)
    sample_std = float(np.std(data, ddof=1)) if len(data) > 1 else 0.0
    return AggregateStats(
        count=len(data),
        mean=float(np.mean(data)),
        se=sample_std / float(np.sqrt(len(data))),
        median=float(np.median(data)),
        ci_low=float(np.quantile(boot_means, 0.025)),
        ci_high=float(np.quantile(boot_means, 0.975)),
    )


def paired_effect(
    treatment: Sequence[float],
    baseline: Sequence[float],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int = 10_000,
) -> AggregateStats:
    if len(treatment) != len(baseline):
        raise ValueError("paired comparison requires equal-length seed results")
    differences = np.asarray(treatment, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    return aggregate_values(
        differences,
        bootstrap_seed=bootstrap_seed,
        bootstrap_samples=bootstrap_samples,
    )


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        if not 0.0 <= value <= 1.0:
            raise ValueError("p-values must lie in [0, 1]")
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted

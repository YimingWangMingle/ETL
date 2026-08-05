from __future__ import annotations

import pytest

from etl_sar.formal.statistics import (
    aggregate_values,
    holm_adjust,
    paired_effect,
)


def test_aggregate_reports_mean_se_median_and_reproducible_bootstrap_ci() -> None:
    first = aggregate_values([1.0, 2.0, 3.0, 4.0, 5.0], bootstrap_seed=17)
    second = aggregate_values([1.0, 2.0, 3.0, 4.0, 5.0], bootstrap_seed=17)
    assert first == second
    assert first.mean == pytest.approx(3.0)
    assert first.median == pytest.approx(3.0)
    assert first.se == pytest.approx((2.5 / 5.0) ** 0.5)
    assert first.ci_low <= first.mean <= first.ci_high


def test_paired_effect_bootstraps_seed_aligned_differences() -> None:
    result = paired_effect(
        treatment=[3.0, 5.0, 7.0, 9.0, 11.0],
        baseline=[1.0, 2.0, 3.0, 4.0, 5.0],
        bootstrap_seed=5,
    )
    assert result.mean == pytest.approx(4.0)
    assert result.median == pytest.approx(4.0)
    assert result.ci_low > 0


def test_holm_adjusts_two_primary_comparisons() -> None:
    adjusted = holm_adjust({"etl_no_sar": 0.01, "lattice": 0.04})
    assert adjusted["etl_no_sar"] == pytest.approx(0.02)
    assert adjusted["lattice"] == pytest.approx(0.04)


def test_pairing_requires_same_number_of_seeds() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        paired_effect([1.0, 2.0], [1.0], bootstrap_seed=0)

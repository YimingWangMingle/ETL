import json

import pytest

from etl_sar.formal.aggregate import SeedResult, summarize_seed_results


def test_aggregate_compares_sar_with_both_baselines_and_declares_protocol_success() -> None:
    records = []
    for domain in ("hand", "leg"):
        for seed in range(5):
            records.extend(
                [
                    SeedResult(domain, "etl_no_sar", seed, 0.40 + seed * 0.01, 0.50),
                    SeedResult(domain, "lattice", seed, 0.45 + seed * 0.01, 0.55),
                    SeedResult(domain, "etl_sar", seed, 0.70 + seed * 0.01, 0.70),
                ]
            )

    result = summarize_seed_results(records, bootstrap_seed=123)

    assert result["protocol_success"] is True
    for domain in ("hand", "leg"):
        comparisons = result["domains"][domain]["comparisons"]
        assert set(comparisons) == {"etl_no_sar", "lattice"}
        assert comparisons["etl_no_sar"]["auc_effect"]["ci_low"] > 0
        assert comparisons["lattice"]["holm_p"] <= 1.0


def test_single_seed_aggregate_reports_deltas_without_inference() -> None:
    records = []
    for domain in ("hand", "leg"):
        records.extend(
            [
                SeedResult(domain, "etl_no_sar", 0, 0.40, 0.50),
                SeedResult(domain, "lattice", 0, 0.45, 0.55),
                SeedResult(domain, "etl_sar", 0, 0.70, 0.70),
            ]
        )

    result = summarize_seed_results(
        records,
        bootstrap_seed=123,
        expected_seeds_by_domain={"hand": (0,), "leg": (0,)},
    )

    assert result["analysis_mode"] == "descriptive_single_seed"
    assert result["protocol_success"] is None
    for domain in ("hand", "leg"):
        summary = result["domains"][domain]
        assert summary["methods"]["etl_sar"] == {
            "seed": 0,
            "normalized_auc": 0.70,
            "final_primary": 0.70,
        }
        assert summary["comparisons"]["etl_no_sar"][
            "auc_delta"
        ] == pytest.approx(0.30)
        assert summary["comparisons"]["lattice"][
            "final_delta"
        ] == pytest.approx(0.15)

    encoded = json.dumps(result)
    for forbidden in ("raw_p", "holm_p", "ci_low", "ci_high", '"se"'):
        assert forbidden not in encoded

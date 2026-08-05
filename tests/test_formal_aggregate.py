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

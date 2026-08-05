from __future__ import annotations

from pathlib import Path

import pytest

from etl_sar.formal.config import FormalDomainConfig
from etl_sar.formal.matrix import ExperimentMatrix, Method


ROOT = Path(__file__).resolve().parents[1]


def load_both() -> list[FormalDomainConfig]:
    return [
        FormalDomainConfig.from_yaml(ROOT / "configs" / "formal_hand.yaml"),
        FormalDomainConfig.from_yaml(ROOT / "configs" / "formal_leg.yaml"),
    ]


def test_formal_configs_lock_tasks_budgets_and_evaluation_schedule() -> None:
    hand, leg = load_both()
    assert (hand.source_env, hand.target_env) == (
        "myoHandReorient8-v0",
        "myoHandReorient100-v0",
    )
    assert (hand.source_budget, hand.target_budget, hand.lattice_budget) == (
        1_000_000,
        19_000_000,
        20_000_000,
    )
    assert hand.final_episodes == 500
    assert (leg.source_env, leg.target_env) == (
        "myoLegWalk-v0",
        "myoLegRoughTerrainWalk-v0",
    )
    assert (leg.source_budget, leg.target_budget, leg.lattice_budget) == (
        1_500_000,
        13_500_000,
        15_000_000,
    )
    assert leg.final_episodes == 100
    assert hand.seeds == leg.seeds == (0, 1, 2, 3, 4)
    assert hand.checkpoint_interval == leg.checkpoint_interval == 250_000
    assert hand.intermediate_episodes == leg.intermediate_episodes == 20


def test_equal_total_interaction_budget_is_enforced() -> None:
    hand, leg = load_both()
    for config in (hand, leg):
        assert config.source_budget + config.target_budget == config.lattice_budget

    invalid = hand.to_mapping()
    invalid["target_budget"] -= 1
    with pytest.raises(ValueError, match="equal total interactions"):
        FormalDomainConfig.from_mapping(invalid)


def test_matrix_expands_to_ten_sources_and_thirty_targets() -> None:
    matrix = ExperimentMatrix.from_configs(load_both())
    assert len(matrix.source_stages) == 10
    assert len(matrix.target_jobs) == 30
    assert matrix.total_declared_interactions == 525_000_000
    assert {job.method for job in matrix.target_jobs} == set(Method)


def test_etl_pair_shares_source_artifact_for_each_domain_and_seed() -> None:
    matrix = ExperimentMatrix.from_configs(load_both())
    for domain in ("hand", "leg"):
        for seed in range(5):
            matched = [
                job
                for job in matrix.target_jobs
                if job.domain == domain and job.seed == seed and job.method != Method.LATTICE
            ]
            assert len(matched) == 2
            assert matched[0].source_stage_id == matched[1].source_stage_id


def test_vector_transition_accounting_rejects_silent_overshoot() -> None:
    hand = load_both()[0]
    assert hand.vector_iterations(hand.target_budget, num_envs=16) == 1_187_500
    with pytest.raises(ValueError, match="divisible"):
        hand.vector_iterations(hand.target_budget - 1, num_envs=16)


def test_seed_banks_are_deterministic_and_disjoint() -> None:
    matrix = ExperimentMatrix.from_configs(load_both())
    first = matrix.seed_bank(domain="hand", seed=2, episodes=20)
    second = matrix.seed_bank(domain="hand", seed=2, episodes=20)
    training = matrix.training_seed_bank(domain="hand", seed=2, count=20)
    assert first == second
    assert len(first) == len(set(first)) == 20
    assert set(first).isdisjoint(training)

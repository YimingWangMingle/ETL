from __future__ import annotations

from pathlib import Path

import pytest

from etl_sar.formal.config import FormalDomainConfig
from etl_sar.formal.matrix import ExperimentMatrix, Method


ROOT = Path(__file__).resolve().parents[1]


def load_short_configs() -> tuple[FormalDomainConfig, FormalDomainConfig]:
    return (
        FormalDomainConfig.from_yaml(
            ROOT / "configs" / "single_seed_10h_hand.yaml"
        ),
        FormalDomainConfig.from_yaml(
            ROOT / "configs" / "single_seed_10h_leg.yaml"
        ),
    )


def test_single_seed_matrix_locks_budget_and_methods() -> None:
    hand, leg = load_short_configs()
    assert hand.seeds == leg.seeds == (0,)
    assert (hand.source_budget, hand.target_budget, hand.lattice_budget) == (
        80_000,
        1_520_000,
        1_600_000,
    )
    assert (leg.source_budget, leg.target_budget, leg.lattice_budget) == (
        120_000,
        1_080_000,
        1_200_000,
    )
    matrix = ExperimentMatrix.from_configs((hand, leg))
    assert len(matrix.source_stages) == 2
    assert len(matrix.target_jobs) == 6
    assert matrix.total_declared_interactions == 8_400_000
    assert {job.method for job in matrix.target_jobs} == set(Method)


def test_single_seed_checkpoints_end_at_each_budget() -> None:
    for config in load_short_configs():
        assert config.target_budget % config.checkpoint_interval == 0
        assert config.lattice_budget % config.checkpoint_interval == 0
        assert config.intermediate_episodes == 10
        assert config.final_episodes == 50
        assert config.num_envs == 16


def test_single_seed_domains_keep_the_shared_myo_tasks() -> None:
    hand, leg = load_short_configs()
    assert (hand.source_env, hand.target_env) == (
        "myoHandReorient8-v0",
        "myoHandReorient100-v0",
    )
    assert (leg.source_env, leg.target_env) == (
        "myoLegWalk-v0",
        "myoLegRoughTerrainWalk-v0",
    )


@pytest.mark.parametrize("seeds", ([], [0, 0], [-1]))
def test_formal_config_rejects_invalid_seed_sets(seeds: list[int]) -> None:
    config = FormalDomainConfig.from_yaml(ROOT / "configs" / "formal_hand.yaml")
    invalid = config.to_mapping()
    invalid["seeds"] = seeds

    with pytest.raises(
        ValueError,
        match="formal seeds must be nonempty, unique, and nonnegative",
    ):
        FormalDomainConfig.from_mapping(invalid)

from __future__ import annotations

from pathlib import Path

import pytest

from etl_sar.dmc.config import DMCTransferConfig
from etl_sar.dmc.matrix import DMCExperimentMatrix, DMCMethod


ROOT = Path(__file__).resolve().parents[1]


def load_configs() -> tuple[DMCTransferConfig, DMCTransferConfig]:
    return (
        DMCTransferConfig.from_yaml(ROOT / "configs" / "dmc_humanoid_pilot.yaml"),
        DMCTransferConfig.from_yaml(ROOT / "configs" / "dmc_dog_pilot.yaml"),
    )


def test_dmc_pilot_locks_tasks_budgets_and_representation() -> None:
    humanoid, dog = load_configs()
    assert (humanoid.source_task, humanoid.target_task) == ("walk", "run")
    assert (dog.source_task, dog.target_task) == ("walk", "run")
    assert {humanoid.domain, dog.domain} == {"humanoid", "dog"}
    for config in (humanoid, dog):
        assert config.seed == 0
        assert (config.source_budget, config.target_budget, config.lattice_budget) == (
            200_000,
            800_000,
            1_000_000,
        )
        assert config.checkpoint_interval == 100_000
        assert config.intermediate_episodes == 10
        assert config.final_episodes == 50
        assert config.etl_latent_dim == 4
        assert config.etl_components == 4
        assert config.sar_components == 4


def test_dmc_matrix_contains_exactly_three_methods_per_domain() -> None:
    matrix = DMCExperimentMatrix.from_configs(load_configs())
    assert len(matrix.sources) == 2
    assert len(matrix.jobs) == 6
    assert {job.method for job in matrix.jobs} == set(DMCMethod)
    assert all(job.total_budget == 1_000_000 for job in matrix.jobs)
    assert matrix.total_declared_interactions == 6_000_000


def test_dmc_config_rejects_unfair_total_budget() -> None:
    config = load_configs()[0]
    payload = config.to_mapping()
    payload["target_budget"] -= 1
    with pytest.raises(ValueError, match="equal total interactions"):
        DMCTransferConfig.from_mapping(payload)

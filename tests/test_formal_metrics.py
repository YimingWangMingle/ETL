from __future__ import annotations

import numpy as np
import pytest

from etl_sar.formal.metrics import (
    EpisodeMetrics,
    normalized_auc,
    resolve_root_x_qpos_address,
)


class Joint:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeModel:
    njnt = 3
    jnt_type = np.asarray([3, 0, 3])
    jnt_qposadr = np.asarray([0, 7, 14])

    def joint(self, index: int) -> Joint:
        return Joint(("knee", "root", "ankle")[index])


def test_leg_root_address_uses_joint_metadata_not_qpos_zero() -> None:
    assert resolve_root_x_qpos_address(FakeModel()) == 7


def test_root_address_rejects_model_without_free_joint() -> None:
    model = FakeModel()
    model.jnt_type = np.asarray([3, 3, 3])
    with pytest.raises(ValueError, match="free joint"):
        resolve_root_x_qpos_address(model)


def test_normalized_auc_uses_declared_budget() -> None:
    assert normalized_auc(
        transitions=[0, 250_000, 500_000],
        values=[0.0, 0.5, 1.0],
        total_budget=500_000,
    ) == pytest.approx(0.5)


def test_episode_metrics_include_primary_and_secondary_fields() -> None:
    metrics = EpisodeMetrics(
        episode=3,
        seed=10003,
        episode_return=12.5,
        length=150,
        success=True,
        terminated=False,
        fall=False,
        object_drop=False,
        forward_distance=2.25,
        velocity_tracking_error=0.15,
        muscle_effort=0.04,
    )
    row = metrics.to_mapping()
    assert row["episode"] == 3
    assert row["seed"] == 10003
    assert row["return"] == pytest.approx(12.5)
    assert row["forward_distance"] == pytest.approx(2.25)
    assert row["muscle_effort"] == pytest.approx(0.04)

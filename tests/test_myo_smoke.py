from __future__ import annotations

import gymnasium as gym
import pytest

from etl_sar.config import ExperimentConfig
from etl_sar.envs import validate_environment


pytest.importorskip("myosuite", reason="MyoSuite is not installed in the test environment")


@pytest.mark.myo
@pytest.mark.parametrize(
    "env_id",
    (
        "myoHandReorient8-v0",
        "myoHandReorient100-v0",
        "myoLegWalk-v0",
        "myoLegRoughTerrainWalk-v0",
        "myoLegHillyTerrainWalk-v0",
        "myoLegStairTerrainWalk-v0",
    ),
)
def test_registered_myo_environment_contract(env_id: str) -> None:
    try:
        env = gym.make(env_id)
    except gym.error.Error as error:
        pytest.fail(f"required MyoSuite environment {env_id!r} is not registered: {error}")
    try:
        validate_environment(env)
    finally:
        env.close()


@pytest.mark.myo
def test_leg_quick_target_is_registered() -> None:
    config = ExperimentConfig.from_yaml("configs/leg_quick.yaml")

    assert config.target.env_id in gym.registry, (
        f"configured SAR uneven target {config.target.env_id!r} is not registered"
    )

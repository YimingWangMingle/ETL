from __future__ import annotations

import gymnasium as gym
import pytest

from etl_sar.envs import validate_environment


pytest.importorskip("myosuite", reason="MyoSuite is not installed in the test environment")


@pytest.mark.myo
@pytest.mark.parametrize(
    "env_id",
    ("myoHandReorient8-v0", "myoHandReorient100-v0", "myoLegWalk-v0"),
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

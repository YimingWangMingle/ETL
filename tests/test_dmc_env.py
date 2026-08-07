from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from etl_sar.dmc.env import DMCGymEnv, flatten_observation


class ArraySpec:
    def __init__(self, shape, minimum=-1.0, maximum=1.0):
        self.shape = shape
        self.minimum = np.broadcast_to(minimum, shape)
        self.maximum = np.broadcast_to(maximum, shape)
        self.dtype = np.dtype(np.float64)


class FakeDMCEnv:
    def __init__(self) -> None:
        self.actions = []
        self.count = 0

    def observation_spec(self):
        return OrderedDict((('position', ArraySpec((2,))), ('speed', ArraySpec((1,)))))

    def action_spec(self):
        return ArraySpec((2,), minimum=[-2.0, -1.0], maximum=[2.0, 1.0])

    def reset(self):
        self.count = 0
        return SimpleNamespace(
            observation=OrderedDict(position=np.array([1.0, 2.0]), speed=np.array([3.0])),
            reward=None,
            discount=1.0,
            last=lambda: False,
        )

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.count += 1
        last = self.count >= 2
        return SimpleNamespace(
            observation=OrderedDict(
                position=np.array([self.count, self.count + 1.0]),
                speed=np.array([4.0]),
            ),
            reward=1.5,
            discount=0.0 if last else 1.0,
            last=lambda: last,
        )


def test_flatten_observation_preserves_mapping_order() -> None:
    value = OrderedDict((('b', np.array([2.0, 3.0])), ('a', np.array(1.0))))
    assert flatten_observation(value).tolist() == [2.0, 3.0, 1.0]


def test_dmc_adapter_repeats_actions_and_accumulates_rewards() -> None:
    backend = FakeDMCEnv()
    env = DMCGymEnv(backend, action_repeat=2)
    observation, _ = env.reset(seed=7)
    assert observation.shape == (3,)
    assert env.action_space.low.tolist() == pytest.approx([-2.0, -1.0])
    next_observation, reward, terminated, truncated, info = env.step(
        np.array([0.25, -0.5], dtype=np.float32)
    )
    assert next_observation.shape == (3,)
    assert reward == pytest.approx(3.0)
    assert terminated is True
    assert truncated is False
    assert info["dmc/internal_steps"] == 2
    assert len(backend.actions) == 2

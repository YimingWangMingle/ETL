from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


def flatten_observation(observation: Any) -> np.ndarray:
    if isinstance(observation, Mapping):
        values = [np.asarray(value).reshape(-1) for value in observation.values()]
        if not values:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(values).astype(np.float32, copy=False)
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def _observation_size(specification: Any) -> int:
    if isinstance(specification, Mapping):
        return sum(int(np.prod(spec.shape, dtype=int)) for spec in specification.values())
    return int(np.prod(specification.shape, dtype=int))


class DMCGymEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, backend: Any, *, action_repeat: int = 1) -> None:
        if action_repeat <= 0:
            raise ValueError("action_repeat must be positive")
        self.backend = backend
        self.action_repeat = int(action_repeat)
        action_spec = backend.action_spec()
        if len(action_spec.shape) != 1:
            raise ValueError("DMC action spec must be one-dimensional")
        self.action_space = Box(
            low=np.broadcast_to(action_spec.minimum, action_spec.shape).astype(
                np.float32
            ),
            high=np.broadcast_to(action_spec.maximum, action_spec.shape).astype(
                np.float32
            ),
            dtype=np.float32,
        )
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(_observation_size(backend.observation_spec()),),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            task = getattr(self.backend, "task", None)
            if task is not None and hasattr(task, "_random"):
                task._random = np.random.RandomState(seed)
        timestep = self.backend.reset()
        return flatten_observation(timestep.observation), {}

    def step(self, action: np.ndarray):
        bounded = np.clip(
            np.asarray(action, dtype=np.float32),
            self.action_space.low,
            self.action_space.high,
        )
        reward = 0.0
        internal_steps = 0
        timestep = None
        for _ in range(self.action_repeat):
            timestep = self.backend.step(bounded)
            internal_steps += 1
            reward += float(timestep.reward or 0.0)
            if timestep.last():
                break
        if timestep is None:  # pragma: no cover - constructor rejects this state
            raise RuntimeError("DMC adapter did not execute an action")
        last = bool(timestep.last())
        discount = 1.0 if timestep.discount is None else float(timestep.discount)
        terminated = bool(last and discount == 0.0)
        truncated = bool(last and not terminated)
        return (
            flatten_observation(timestep.observation),
            reward,
            terminated,
            truncated,
            {"dmc/internal_steps": internal_steps, "dmc/discount": discount},
        )

    def render(self):
        return self.backend.physics.render()

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()


def make_dmc_env(
    domain: str,
    task: str,
    *,
    seed: int,
    action_repeat: int,
) -> DMCGymEnv:
    try:
        from dm_control import suite
    except ImportError as error:  # pragma: no cover - dependency message
        raise ImportError(
            "DeepMind Control Suite is required; install the project 'dmc' extra"
        ) from error
    backend = suite.load(
        domain_name=domain,
        task_name=task,
        task_kwargs={"random": seed},
    )
    return DMCGymEnv(backend, action_repeat=action_repeat)

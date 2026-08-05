from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecNormalize,
    sync_envs_normalization,
)

from etl_sar.formal.evaluate import FormalEvaluationSummary, evaluate_formal
from etl_sar.formal.metrics import resolve_root_x_qpos_address


class FormalMetricInfoWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, *, domain: str) -> None:
        super().__init__(env)
        self.root_address = (
            resolve_root_x_qpos_address(self.unwrapped.model)
            if domain == "leg"
            else None
        )

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        enriched = dict(info)
        if self.root_address is not None:
            enriched["formal/root_x"] = float(
                self.unwrapped.data.qpos[self.root_address]
            )
        return observation, reward, terminated, truncated, enriched


class SingleVecEnvGymAdapter(gym.Env):
    def __init__(self, vecnormalize: VecNormalize, raw_env: gym.Env) -> None:
        super().__init__()
        if vecnormalize.num_envs != 1:
            raise ValueError("evaluation adapter requires exactly one vector environment")
        self.vecnormalize = vecnormalize
        self.raw_env = raw_env
        self.observation_space = vecnormalize.observation_space
        self.action_space = vecnormalize.action_space

    @property
    def model(self):
        return self.raw_env.unwrapped.model

    @property
    def data(self):
        return self.raw_env.unwrapped.data

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.vecnormalize.seed(seed)
        observation = self.vecnormalize.reset()[0]
        return observation, {}

    def step(self, action):
        observations, rewards, dones, infos = self.vecnormalize.step(
            np.asarray(action)[None, ...]
        )
        info = infos[0]
        truncated = bool(info.get("TimeLimit.truncated", False))
        terminated = bool(dones[0] and not truncated)
        return observations[0], float(rewards[0]), terminated, truncated, info

    def close(self) -> None:
        self.vecnormalize.close()


def evaluate_lattice_model(
    *,
    model: Any,
    training_vecnormalize: VecNormalize,
    env_factory: Callable[[], gym.Env],
    domain: str,
    seeds: Iterable[int],
    output_dir: str | Path,
    environment_steps: int,
    recurrent: bool,
) -> FormalEvaluationSummary:
    wrapped = FormalMetricInfoWrapper(env_factory(), domain=domain)
    dummy = DummyVecEnv([lambda: wrapped])
    evaluation_vec = VecNormalize(dummy, training=False, norm_reward=False)
    sync_envs_normalization(training_vecnormalize, evaluation_vec)
    evaluation_vec.training = False
    evaluation_vec.norm_reward = False
    adapter = SingleVecEnvGymAdapter(evaluation_vec, wrapped)
    return evaluate_formal(
        model,
        adapter,
        domain=domain,
        seeds=seeds,
        output_dir=output_dir,
        environment_steps=environment_steps,
        recurrent=recurrent,
    )

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box

from etl_sar.action_model import ActionModelOutput, ETLSARActionModel
from etl_sar.types import TaskMetadata


@dataclass(frozen=True)
class EnvironmentContract:
    observation_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray


def validate_environment(
    env: gym.Env,
    *,
    expected_action_dim: int | None = None,
) -> EnvironmentContract:
    if not isinstance(env.action_space, Box) or len(env.action_space.shape) != 1:
        raise ValueError("environment action space must be a one-dimensional Box")
    if not isinstance(env.observation_space, Box) or len(env.observation_space.shape) != 1:
        raise ValueError("environment observation space must be a one-dimensional Box")
    action_dim = int(env.action_space.shape[0])
    if expected_action_dim is not None and action_dim != expected_action_dim:
        raise ValueError(
            f"environment action dimension {action_dim} does not match expected "
            f"action dimension {expected_action_dim}"
        )
    return EnvironmentContract(
        observation_dim=int(env.observation_space.shape[0]),
        action_dim=action_dim,
        action_low=np.asarray(env.action_space.low, dtype=np.float32),
        action_high=np.asarray(env.action_space.high, dtype=np.float32),
    )


class TaskRegistry:
    def __init__(self, tasks: Mapping[str, TaskMetadata]) -> None:
        self._tasks = dict(tasks)

    def resolve_exact(self, name: str) -> TaskMetadata:
        try:
            return self._tasks[name]
        except KeyError as error:
            raise KeyError(f"task {name!r} is not registered") from error


class LatentActionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, action_model: ETLSARActionModel) -> None:
        contract = validate_environment(
            env,
            expected_action_dim=int(action_model.action_low.numel()),
        )
        if contract.action_dim != action_model.synergy_basis.shape[0]:
            raise ValueError("action dimension does not match synergy basis")
        super().__init__(env)
        self.action_model = action_model.eval()
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(action_model.latent_dim,),
            dtype=np.float32,
        )
        self.last_output: ActionModelOutput | None = None

    def step(self, action: np.ndarray):
        latent = np.asarray(action, dtype=np.float32)
        if latent.shape != self.action_space.shape:
            raise ValueError(
                f"latent action shape {latent.shape} does not match {self.action_space.shape}"
            )
        with torch.no_grad():
            tensor = torch.as_tensor(latent).unsqueeze(0)
            output = self.action_model(tensor)
        self.last_output = output
        executed_action = output.full_action[0].cpu().numpy().astype(np.float32)
        observation, reward, terminated, truncated, info = self.env.step(executed_action)
        enriched = dict(info)
        enriched.update(
            {
                "etl_sar/latent_action": latent.copy(),
                "etl_sar/etl_action": output.etl_action[0].cpu().numpy().copy(),
                "etl_sar/sar_action": output.sar_action[0].cpu().numpy().copy(),
                "etl_sar/executed_action": executed_action.copy(),
                "etl_sar/contribution_ratio": float(output.contribution_ratio[0].cpu()),
            }
        )
        return observation, reward, terminated, truncated, enriched

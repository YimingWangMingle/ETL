from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn

from etl_sar.action_model import ETLSARActionModel
from etl_sar.envs import LatentActionWrapper, TaskRegistry, validate_environment
from etl_sar.types import Limb, TaskMetadata, TaskRole


class DummyMuscleEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self, action_dim: int = 6) -> None:
        self.observation_space = spaces.Box(-10, 10, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(action_dim,), dtype=np.float32)
        self.last_action = np.zeros(action_dim, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {"success": False}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32)
        return np.ones(4, dtype=np.float32), 1.0, False, False, {"success": True}


def make_action_model(action_dim: int = 6) -> ETLSARActionModel:
    decoder = nn.Sequential(nn.Linear(20, action_dim), nn.Tanh())
    return ETLSARActionModel(
        decoder=decoder,
        synergy_basis=torch.randn(action_dim, 20),
        latent_dim=20,
        action_low=-torch.ones(action_dim),
        action_high=torch.ones(action_dim),
    )


def test_latent_wrapper_exposes_twenty_dimensional_action_space() -> None:
    base = DummyMuscleEnv(action_dim=6)
    wrapped = LatentActionWrapper(base, make_action_model(action_dim=6))

    assert wrapped.action_space.shape == (20,)
    wrapped.reset(seed=7)
    _, _, _, _, info = wrapped.step(np.zeros(20, dtype=np.float32))

    assert base.last_action.shape == (6,)
    assert info["etl_sar/latent_action"].shape == (20,)
    assert info["etl_sar/executed_action"].shape == (6,)
    assert info["etl_sar/contribution_ratio"] <= 0.200001


def test_validate_environment_rejects_action_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="action dimension"):
        validate_environment(DummyMuscleEnv(action_dim=6), expected_action_dim=39)


def test_registry_resolves_only_exact_names() -> None:
    metadata = TaskMetadata(
        env_id="myoHandReorient8-v0",
        role=TaskRole.SOURCE,
        limb=Limb.HAND,
    )
    registry = TaskRegistry({"hand_reorient8": metadata})

    assert registry.resolve_exact("hand_reorient8") is metadata
    with pytest.raises(KeyError, match="not registered"):
        registry.resolve_exact("hand_reorient")


def test_wrapper_rejects_decoder_action_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="action dimension"):
        LatentActionWrapper(DummyMuscleEnv(action_dim=8), make_action_model(action_dim=6))

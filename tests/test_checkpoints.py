from __future__ import annotations

import shutil

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import PPO

from etl_sar.action_model import ETLSARActionModel
from etl_sar.checkpoints import load_checkpoint_pair, save_checkpoint_pair
from etl_sar.gmvae import GMVAE


class LatentEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(20,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(3, dtype=np.float32), 0.0, True, False, {}


def make_action_model() -> ETLSARActionModel:
    torch.manual_seed(11)
    gmvae = GMVAE(action_dim=6, latent_dim=20, components=20, hidden_dims=(16,))
    return ETLSARActionModel(
        decoder=gmvae.decoder,
        synergy_basis=torch.randn(6, 20),
        latent_dim=20,
        action_low=-torch.ones(6),
        action_high=torch.ones(6),
    )


def test_checkpoint_pair_round_trip_preserves_policy_and_muscle_actions(tmp_path) -> None:
    policy = PPO("MlpPolicy", LatentEnv(), n_steps=8, batch_size=4, seed=3)
    action_model = make_action_model()
    with torch.no_grad():
        action_model.decoder[-2].bias.add_(0.125)
    observation = np.asarray([0.2, -0.1, 0.4], dtype=np.float32)
    latent_before, _ = policy.predict(observation, deterministic=True)
    with torch.no_grad():
        muscle_before = action_model(torch.as_tensor(latent_before).unsqueeze(0)).full_action

    manifest_path = save_checkpoint_pair(
        policy=policy,
        action_model=action_model,
        directory=tmp_path,
        name="latest",
        transitions=1234,
    )

    restored_action_model = make_action_model()
    restored_policy, manifest = load_checkpoint_pair(
        manifest_path,
        policy_loader=PPO.load,
        action_model=restored_action_model,
    )
    latent_after, _ = restored_policy.predict(observation, deterministic=True)
    with torch.no_grad():
        muscle_after = restored_action_model(
            torch.as_tensor(latent_after).unsqueeze(0)
        ).full_action

    assert manifest.transitions == 1234
    np.testing.assert_array_equal(latent_after, latent_before)
    torch.testing.assert_close(muscle_after, muscle_before, rtol=0, atol=0)


def test_checkpoint_pair_rejects_modified_policy(tmp_path) -> None:
    manifest = save_checkpoint_pair(
        policy=PPO("MlpPolicy", LatentEnv(), n_steps=8, batch_size=4),
        action_model=make_action_model(),
        directory=tmp_path,
        name="latest",
        transitions=8,
    )
    policy_path = tmp_path / "latest_policy.zip"
    policy_path.write_bytes(policy_path.read_bytes() + b"modified")

    with pytest.raises(ValueError, match="policy hash"):
        load_checkpoint_pair(
            manifest,
            policy_loader=PPO.load,
            action_model=make_action_model(),
        )


def test_checkpoint_pair_rejects_cross_paired_action_model(tmp_path) -> None:
    policy = PPO("MlpPolicy", LatentEnv(), n_steps=8, batch_size=4)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_manifest = save_checkpoint_pair(
        policy=policy,
        action_model=make_action_model(),
        directory=first,
        name="best",
        transitions=8,
    )
    save_checkpoint_pair(
        policy=policy,
        action_model=make_action_model(),
        directory=second,
        name="best",
        transitions=16,
    )
    shutil.copy2(second / "best_action_model.pt", first / "best_action_model.pt")

    with pytest.raises(ValueError, match="action-model hash"):
        load_checkpoint_pair(
            first_manifest,
            policy_loader=PPO.load,
            action_model=make_action_model(),
        )

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from etl_sar.action_model import ETLSARActionModel
from etl_sar.bdr import StateEncoder
from etl_sar.data import TrajectoryStore
from etl_sar.gmvae import GMVAE
from etl_sar.representation import RepresentationTrainer
from etl_sar.trainers import DirectionalExplorationWrapper, ExploreTrainer
from etl_sar.types import Limb


class TinySourceEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self) -> None:
        self.observation_space = spaces.Box(-2, 2, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(6,), dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(4, dtype=np.float32), {"success": False}

    def step(self, action):
        self.steps += 1
        action = np.asarray(action, dtype=np.float32)
        observation = np.full(4, self.steps / 6, dtype=np.float32)
        reward = float(0.5 - np.square(action).mean())
        terminated = self.steps >= 6
        return observation, reward, terminated, False, {"success": terminated}


def make_representation() -> RepresentationTrainer:
    gmvae = GMVAE(action_dim=6, latent_dim=20, components=20, hidden_dims=(16,))
    action_model = ETLSARActionModel(
        decoder=gmvae.decoder,
        synergy_basis=torch.randn(6, 20),
        latent_dim=20,
        action_low=-torch.ones(6),
        action_high=torch.ones(6),
    )
    return RepresentationTrainer(gmvae=gmvae, action_model=action_model, learning_rate=1e-3)


def test_directional_wrapper_augments_observation_and_reward() -> None:
    encoder = StateEncoder(observation_dim=4, latent_dim=20, hidden_dims=(16,))
    wrapped = DirectionalExplorationWrapper(
        TinySourceEnv(),
        state_encoder=encoder,
        direction_count=20,
        bonus_scale=0.5,
        seed=7,
    )

    observation, _ = wrapped.reset(seed=7)
    _, reward, _, _, info = wrapped.step(np.zeros(6, dtype=np.float32))

    assert observation.shape == (24,)
    assert info["etl/direction"].shape == (20,)
    assert info["etl/behavior"].shape == (20,)
    assert reward == info["etl/extrinsic_reward"] + 0.5 * info["etl/directional_bonus"]


def test_explore_trainer_collects_actions_and_interleaves_gmvae(tmp_path) -> None:
    store = TrajectoryStore(
        tmp_path / "data",
        limb=Limb.HAND,
        source_task="reorient8",
        action_dim=6,
    )
    trainer = ExploreTrainer(
        env_factory=TinySourceEnv,
        state_encoder=StateEncoder(4, 20, hidden_dims=(16,)),
        representation=make_representation(),
        trajectory_store=store,
        limb=Limb.HAND,
        source_task="reorient8",
        run_dir=tmp_path / "run",
        total_timesteps=48,
        n_steps=16,
        batch_size=8,
        representation_update_interval=12,
        seed=3,
    )

    artifacts = trainer.run()

    assert artifacts.latest_checkpoint.exists()
    assert artifacts.representation_checkpoint.exists()
    assert store.action_pool().shape[0] > 0
    assert store.success_pool().shape[0] > 0
    assert trainer.gmvae_update_steps
    assert trainer.bdr_update_steps

from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

from etl_sar.action_model import ETLSARActionModel
from etl_sar.gmvae import GMVAE
from etl_sar.representation import RepresentationTrainer
from etl_sar.trainers import (
    CheckpointMetadata,
    FiniteTrainingCallback,
    TransferTrainer,
    validate_checkpoint_metadata,
)
from etl_sar.types import Limb


class TinyMuscleEnv(gym.Env[np.ndarray, np.ndarray]):
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
        reward = float(1.0 - np.square(action).mean())
        terminated = self.steps >= 8
        observation = np.full(4, self.steps / 8, dtype=np.float32)
        return observation, reward, terminated, False, {"success": reward > 0.5}


def make_components():
    torch.manual_seed(3)
    gmvae = GMVAE(action_dim=6, latent_dim=20, components=20, hidden_dims=(16,))
    action_model = ETLSARActionModel(
        decoder=gmvae.decoder,
        synergy_basis=torch.randn(6, 20),
        latent_dim=20,
        action_low=-torch.ones(6),
        action_high=torch.ones(6),
    )
    representation = RepresentationTrainer(
        gmvae=gmvae,
        action_model=action_model,
        learning_rate=1e-3,
    )
    return action_model, representation


def test_transfer_trainer_runs_latent_ppo_and_writes_checkpoints(tmp_path) -> None:
    action_model, representation = make_components()
    trainer = TransferTrainer(
        env_factory=TinyMuscleEnv,
        action_model=action_model,
        representation=representation,
        run_dir=tmp_path,
        total_timesteps=64,
        decoder_freeze_steps=32,
        n_steps=16,
        batch_size=8,
        seed=4,
    )

    result = trainer.run()

    assert result.latest_checkpoint.exists()
    assert result.best_checkpoint.exists()
    assert trainer.model is not None
    assert trainer.model.action_space.shape == (20,)


def test_decoder_updates_start_after_freeze_steps(tmp_path) -> None:
    action_model, representation = make_components()
    trainer = TransferTrainer(
        env_factory=TinyMuscleEnv,
        action_model=action_model,
        representation=representation,
        run_dir=tmp_path,
        total_timesteps=64,
        decoder_freeze_steps=32,
        n_steps=16,
        batch_size=8,
        seed=5,
    )

    trainer.run()

    assert trainer.decoder_update_steps
    assert min(trainer.decoder_update_steps) >= 32


def test_finite_callback_rejects_nan_and_inf() -> None:
    callback = FiniteTrainingCallback()

    with pytest.raises(FloatingPointError, match="NaN|Inf"):
        callback.check({"loss": float("nan")})
    with pytest.raises(FloatingPointError, match="NaN|Inf"):
        callback.check({"action": np.asarray([float("inf")])})


def test_checkpoint_metadata_rejects_wrong_limb(tmp_path) -> None:
    path = tmp_path / "metadata.json"
    CheckpointMetadata(
        schema_version=1,
        limb=Limb.LEG,
        source_task="flat_walk",
        target_task="uneven",
        action_dim=80,
        latent_dim=20,
        data_fingerprint="abc",
    ).save(path)

    with pytest.raises(ValueError, match="checkpoint limb"):
        validate_checkpoint_metadata(
            path,
            expected=CheckpointMetadata(
                schema_version=1,
                limb=Limb.HAND,
                source_task="reorient8",
                target_task="reorient100",
                action_dim=39,
                latent_dim=20,
                data_fingerprint="xyz",
            ),
        )


def test_transfer_trainer_accepts_custom_evaluation_frequency(tmp_path) -> None:
    action_model, representation = make_components()

    trainer = TransferTrainer(
        env_factory=TinyMuscleEnv,
        action_model=action_model,
        representation=representation,
        run_dir=tmp_path,
        total_timesteps=64,
        decoder_freeze_steps=32,
        n_steps=16,
        batch_size=8,
        eval_freq=37,
        seed=6,
    )

    assert trainer.eval_freq == 37


def test_transfer_trainer_rejects_nonpositive_evaluation_frequency(
    tmp_path,
) -> None:
    action_model, representation = make_components()

    with pytest.raises(ValueError, match="eval_freq must be positive"):
        TransferTrainer(
            env_factory=TinyMuscleEnv,
            action_model=action_model,
            representation=representation,
            run_dir=tmp_path,
            total_timesteps=64,
            decoder_freeze_steps=32,
            n_steps=16,
            batch_size=8,
            eval_freq=0,
            seed=6,
        )

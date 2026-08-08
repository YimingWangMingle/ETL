from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import SAC

from etl_sar.lattice.policies import (
    LatticeRecurrentActorCriticPolicy,
    LatticeSACPolicy,
)
from etl_sar.lattice.trainers import hand_model_kwargs, leg_model_kwargs

sb3_contrib = pytest.importorskip("sb3_contrib")
RecurrentPPO = sb3_contrib.RecurrentPPO


class ContinuousEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1, 1, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(5, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return (
            np.zeros(5, dtype=np.float32),
            0.0,
            self.steps >= 8,
            False,
            {},
        )


def test_hand_kwargs_match_official_reorient_configuration() -> None:
    kwargs = hand_model_kwargs(seed=4, device="cpu")
    assert kwargs["policy"] is LatticeRecurrentActorCriticPolicy
    assert kwargs["n_steps"] == 128
    assert kwargs["batch_size"] == 32
    assert kwargs["learning_rate"] == pytest.approx(2.55673e-5)
    assert kwargs["ent_coef"] == pytest.approx(3.62109e-6)
    assert kwargs["clip_range"] == pytest.approx(0.3)
    assert kwargs["gamma"] == pytest.approx(0.99)
    assert kwargs["gae_lambda"] == pytest.approx(0.9)
    assert kwargs["max_grad_norm"] == pytest.approx(0.7)
    assert kwargs["vf_coef"] == pytest.approx(0.835671)
    assert kwargs["n_epochs"] == 10
    assert kwargs["use_sde"] is False
    assert kwargs["sde_sample_freq"] == 1
    assert kwargs["policy_kwargs"]["use_lattice"] is True
    assert kwargs["policy_kwargs"]["log_std_init"] == 0.0
    assert kwargs["policy_kwargs"]["std_reg"] == 0.0


def test_leg_kwargs_match_official_walker_configuration() -> None:
    kwargs = leg_model_kwargs(seed=4, device="cpu")
    assert kwargs["policy"] is LatticeSACPolicy
    assert kwargs["learning_rate"] == pytest.approx(3e-4)
    assert kwargs["buffer_size"] == 300_000
    assert kwargs["learning_starts"] == 10_000
    assert kwargs["batch_size"] == 256
    assert kwargs["tau"] == pytest.approx(0.02)
    assert kwargs["gamma"] == pytest.approx(0.98)
    assert kwargs["train_freq"] == (8, "step")
    assert kwargs["gradient_steps"] == 8
    assert kwargs["ent_coef"] == "auto"
    assert kwargs["target_entropy"] == "auto"
    assert kwargs["use_sde"] is False
    assert kwargs["sde_sample_freq"] == 1
    assert kwargs["policy_kwargs"]["net_arch"] == {"pi": [400, 300], "qf": [400, 300]}
    assert kwargs["policy_kwargs"]["activation_fn"] is torch.nn.GELU
    assert kwargs["policy_kwargs"]["use_lattice"] is True


def test_recurrent_ppo_constructs_with_lattice_distribution() -> None:
    kwargs = hand_model_kwargs(seed=1, device="cpu")
    kwargs.update(n_steps=8, batch_size=4, n_epochs=1)
    model = RecurrentPPO(env=ContinuousEnv(), **kwargs)
    assert model.policy.action_dist.__class__.__name__ == "LatticeNoiseDistribution"
    action, _ = model.predict(np.zeros(5, dtype=np.float32), deterministic=True)
    assert action.shape == (3,)


def test_sac_constructs_with_lattice_distribution() -> None:
    kwargs = leg_model_kwargs(seed=1, device="cpu")
    kwargs.update(buffer_size=32, learning_starts=1, batch_size=2)
    model = SAC(env=ContinuousEnv(), **kwargs)
    assert model.policy.actor.action_dist.__class__.__name__ == "SquashedLatticeNoiseDistribution"
    action, _ = model.predict(np.zeros(5, dtype=np.float32), deterministic=True)
    assert action.shape == (3,)


def test_sac_lattice_sde_std_is_compatible_with_sb3_logging() -> None:
    model = SAC(
        LatticeSACPolicy,
        env=ContinuousEnv(),
        buffer_size=32,
        learning_starts=1,
        batch_size=2,
        train_freq=(1, "step"),
        gradient_steps=1,
        use_sde=True,
        sde_sample_freq=1,
        policy_kwargs={"use_lattice": True, "use_expln": True},
        verbose=0,
        seed=1,
        device="cpu",
    )

    model.learn(total_timesteps=8, log_interval=1)

    std = model.actor.get_std()
    assert isinstance(std, torch.Tensor)
    assert torch.isfinite(std).all()

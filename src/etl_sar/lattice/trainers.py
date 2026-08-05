from __future__ import annotations

from typing import Any

from torch import nn

from etl_sar.lattice.policies import (
    LatticeRecurrentActorCriticPolicy,
    LatticeSACPolicy,
)


def hand_model_kwargs(*, seed: int, device: str = "auto") -> dict[str, Any]:
    return {
        "policy": LatticeRecurrentActorCriticPolicy,
        "device": device,
        "batch_size": 32,
        "n_steps": 128,
        "learning_rate": 2.55673e-5,
        "ent_coef": 3.62109e-6,
        "clip_range": 0.3,
        "gamma": 0.99,
        "gae_lambda": 0.9,
        "max_grad_norm": 0.7,
        "vf_coef": 0.835671,
        "n_epochs": 10,
        "use_sde": False,
        "sde_sample_freq": 1,
        "seed": seed,
        "policy_kwargs": {
            "use_lattice": True,
            "use_expln": True,
            "ortho_init": False,
            "log_std_init": 0.0,
            "activation_fn": nn.ReLU,
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "std_clip": (1e-3, 10),
            "expln_eps": 1e-6,
            "full_std": False,
            "std_reg": 0.0,
            "alpha": 1,
        },
    }


def leg_model_kwargs(*, seed: int, device: str = "auto") -> dict[str, Any]:
    return {
        "policy": LatticeSACPolicy,
        "device": device,
        "learning_rate": 3e-4,
        "buffer_size": 300_000,
        "learning_starts": 10_000,
        "batch_size": 256,
        "tau": 0.02,
        "gamma": 0.98,
        "train_freq": (8, "step"),
        "gradient_steps": 8,
        "action_noise": None,
        "replay_buffer_class": None,
        "ent_coef": "auto",
        "target_update_interval": 1,
        "target_entropy": "auto",
        "seed": seed,
        "use_sde": False,
        "sde_sample_freq": 1,
        "policy_kwargs": {
            "use_lattice": True,
            "use_expln": True,
            "log_std_init": 0.0,
            "activation_fn": nn.GELU,
            "net_arch": {"pi": [400, 300], "qf": [400, 300]},
            "std_clip": (1e-3, 10),
            "expln_eps": 1e-6,
            "clip_mean": 2.0,
            "std_reg": 0.0,
            "alpha": 1,
        },
    }

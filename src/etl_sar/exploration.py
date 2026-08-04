from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from etl_sar.bdr import StateEncoder, behavior_metric_loss, directional_bonus, sample_unit_directions
from etl_sar.data import TrajectoryEpisode, TrajectoryStore
from etl_sar.representation import RepresentationTrainer
from etl_sar.types import Limb


class DirectionalExplorationWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        state_encoder: StateEncoder,
        direction_count: int,
        bonus_scale: float,
        seed: int,
    ) -> None:
        if not isinstance(env.observation_space, Box) or len(env.observation_space.shape) != 1:
            raise ValueError("directional exploration requires a one-dimensional Box observation")
        super().__init__(env)
        self.state_encoder = state_encoder
        self.bonus_scale = float(bonus_scale)
        self._rng = np.random.default_rng(seed)
        with torch.no_grad():
            probe = torch.zeros(1, int(env.observation_space.shape[0]))
            latent_dim = int(state_encoder(probe).shape[-1])
        self._directions = sample_unit_directions(direction_count, latent_dim, seed).numpy()
        self._direction = self._directions[0]
        self._observation: np.ndarray | None = None
        self.observation_space = Box(
            low=np.concatenate([env.observation_space.low, -np.ones(latent_dim)]).astype(
                np.float32
            ),
            high=np.concatenate([env.observation_space.high, np.ones(latent_dim)]).astype(
                np.float32
            ),
            dtype=np.float32,
        )

    def _augment(self, observation: np.ndarray) -> np.ndarray:
        return np.concatenate([observation, self._direction]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self._direction = self._directions[
            int(self._rng.integers(len(self._directions)))
        ].astype(np.float32)
        self._observation = np.asarray(observation, dtype=np.float32)
        enriched = dict(info)
        enriched["etl/direction"] = self._direction.copy()
        return self._augment(self._observation), enriched

    def step(self, action):
        if self._observation is None:
            raise RuntimeError("reset must be called before step")
        previous = self._observation.copy()
        observation, extrinsic, terminated, truncated, info = self.env.step(action)
        next_observation = np.asarray(observation, dtype=np.float32)
        with torch.no_grad():
            phi_previous = self.state_encoder(torch.as_tensor(previous).unsqueeze(0))
            phi_next = self.state_encoder(torch.as_tensor(next_observation).unsqueeze(0))
            bonus = float(
                directional_bonus(
                    phi_previous,
                    phi_next,
                    torch.as_tensor(self._direction).unsqueeze(0),
                )[0]
            )
        self._observation = next_observation
        enriched = dict(info)
        enriched.update(
            {
                "etl/direction": self._direction.copy(),
                "etl/behavior": phi_next[0].cpu().numpy().copy(),
                "etl/current_observation": previous,
                "etl/next_observation": next_observation.copy(),
                "etl/extrinsic_reward": float(extrinsic),
                "etl/directional_bonus": bonus,
                "etl/executed_action": np.asarray(action, dtype=np.float32).copy(),
            }
        )
        reward = float(extrinsic) + self.bonus_scale * bonus
        return self._augment(next_observation), reward, terminated, truncated, enriched


class _ExploreCallback(BaseCallback):
    def __init__(
        self,
        *,
        state_encoder: StateEncoder,
        representation: RepresentationTrainer,
        trajectory_store: TrajectoryStore,
        limb: Limb,
        source_task: str,
        update_interval: int,
        gmvae_update_steps: list[int],
        bdr_update_steps: list[int],
    ) -> None:
        super().__init__(verbose=0)
        self.state_encoder = state_encoder
        self.representation = representation
        self.trajectory_store = trajectory_store
        self.limb = limb
        self.source_task = source_task
        self.update_interval = update_interval
        self.gmvae_update_steps = gmvae_update_steps
        self.bdr_update_steps = bdr_update_steps
        self.encoder_optimizer = torch.optim.Adam(state_encoder.parameters(), lr=1e-3)
        self.episode: list[dict[str, Any]] = []
        self.bdr_buffer: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", np.zeros(len(infos), dtype=bool))
        for info, done in zip(infos, dones):
            if "etl/current_observation" not in info:
                continue
            item = {
                "observation": info["etl/current_observation"],
                "action": info["etl/executed_action"],
                "reward": info["etl/extrinsic_reward"],
                "next_observation": info["etl/next_observation"],
                "behavior": info["etl/behavior"],
                "success": bool(info.get("success", False)),
                "done": bool(done),
                "truncated": bool(info.get("TimeLimit.truncated", False)),
            }
            self.episode.append(item)
            self.bdr_buffer.append(item)
            if done:
                self._flush_episode()
        self._update_bdr()
        if (
            self.num_timesteps % self.update_interval == 0
            and self.trajectory_store.action_pool().shape[0] > 0
        ):
            self.representation.update_gmvae(
                self.trajectory_store,
                steps=1,
                batch_size=32,
            )
            self.gmvae_update_steps.append(self.num_timesteps)
        return True

    def _flush_episode(self) -> None:
        episode = self.episode
        if not episode:
            return
        self.trajectory_store.append_episode(
            TrajectoryEpisode(
                limb=self.limb,
                source_task=self.source_task,
                observations=np.stack([item["observation"] for item in episode]),
                sampled_actions=np.stack([item["action"] for item in episode]),
                executed_actions=np.stack([item["action"] for item in episode]),
                rewards=np.asarray([item["reward"] for item in episode], dtype=np.float32),
                next_observations=np.stack(
                    [item["next_observation"] for item in episode]
                ),
                terminated=np.asarray(
                    [item["done"] and not item["truncated"] for item in episode],
                    dtype=np.bool_,
                ),
                truncated=np.asarray(
                    [item["truncated"] for item in episode], dtype=np.bool_
                ),
                behaviors=np.stack([item["behavior"] for item in episode]),
                success=any(item["success"] for item in episode),
            )
        )
        self.episode = []

    def _update_bdr(self) -> None:
        if len(self.bdr_buffer) < 4:
            return
        batch = self.bdr_buffer[-4:]
        observations = torch.as_tensor(
            np.stack([item["observation"] for item in batch]), dtype=torch.float32
        )
        next_observations = torch.as_tensor(
            np.stack([item["next_observation"] for item in batch]), dtype=torch.float32
        )
        rewards = torch.as_tensor([item["reward"] for item in batch], dtype=torch.float32)
        loss = behavior_metric_loss(
            self.state_encoder(observations),
            self.state_encoder(next_observations),
            rewards,
            gamma=0.99,
            bdr_weight=0.1,
            epsilon=0.05,
            margin=0.5,
        ).total
        self.encoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.encoder_optimizer.step()
        self.bdr_update_steps.append(self.num_timesteps)
        self.bdr_buffer.clear()


@dataclass(frozen=True)
class ExplorationArtifacts:
    latest_checkpoint: Path
    representation_checkpoint: Path


class ExploreTrainer:
    def __init__(
        self,
        *,
        env_factory: Callable[[], gym.Env],
        state_encoder: StateEncoder,
        representation: RepresentationTrainer,
        trajectory_store: TrajectoryStore,
        limb: Limb,
        source_task: str,
        run_dir: str | Path,
        total_timesteps: int,
        n_steps: int,
        batch_size: int,
        representation_update_interval: int,
        seed: int = 0,
        bonus_scale: float = 1.0,
    ) -> None:
        self.env_factory = env_factory
        self.state_encoder = state_encoder
        self.representation = representation
        self.trajectory_store = trajectory_store
        self.limb = limb
        self.source_task = source_task
        self.run_dir = Path(run_dir)
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.representation_update_interval = representation_update_interval
        self.seed = seed
        self.bonus_scale = bonus_scale
        self.gmvae_update_steps: list[int] = []
        self.bdr_update_steps: list[int] = []
        self.model: PPO | None = None

    def run(self) -> ExplorationArtifacts:
        self.run_dir.mkdir(parents=True, exist_ok=True)

        def make_env() -> gym.Env:
            return Monitor(
                DirectionalExplorationWrapper(
                    self.env_factory(),
                    state_encoder=self.state_encoder,
                    direction_count=20,
                    bonus_scale=self.bonus_scale,
                    seed=self.seed,
                )
            )

        env = DummyVecEnv([make_env])
        callback = _ExploreCallback(
            state_encoder=self.state_encoder,
            representation=self.representation,
            trajectory_store=self.trajectory_store,
            limb=self.limb,
            source_task=self.source_task,
            update_interval=self.representation_update_interval,
            gmvae_update_steps=self.gmvae_update_steps,
            bdr_update_steps=self.bdr_update_steps,
        )
        self.model = PPO(
            "MlpPolicy",
            env,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            seed=self.seed,
            verbose=0,
            tensorboard_log=str(self.run_dir / "tensorboard"),
        )
        self.model.learn(total_timesteps=self.total_timesteps, callback=callback)
        latest_base = self.run_dir / "latest_explorer"
        self.model.save(latest_base)
        env.close()
        representation_checkpoint = self.run_dir / "representation.pt"
        torch.save(
            {
                "schema_version": 1,
                "limb": self.limb.value,
                "source_task": self.source_task,
                "latent_dim": 20,
                "state_encoder": self.state_encoder.state_dict(),
                "gmvae": self.representation.gmvae.state_dict(),
                "action_model": self.representation.action_model.state_dict(),
            },
            representation_checkpoint,
        )
        return ExplorationArtifacts(
            latest_checkpoint=latest_base.with_suffix(".zip"),
            representation_checkpoint=representation_checkpoint,
        )

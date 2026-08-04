from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from etl_sar.action_model import ETLSARActionModel
from etl_sar.envs import LatentActionWrapper
from etl_sar.representation import RepresentationTrainer
from etl_sar.types import Limb


@dataclass(frozen=True)
class CheckpointMetadata:
    schema_version: int
    limb: Limb
    source_task: str
    target_task: str
    action_dim: int
    latent_dim: int
    data_fingerprint: str

    def save(self, path: str | Path) -> None:
        payload = asdict(self)
        payload["limb"] = self.limb.value
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CheckpointMetadata":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["limb"] = Limb(payload["limb"])
        return cls(**payload)


def validate_checkpoint_metadata(
    path: str | Path,
    *,
    expected: CheckpointMetadata,
) -> CheckpointMetadata:
    actual = CheckpointMetadata.load(path)
    fields = (
        "schema_version",
        "limb",
        "source_task",
        "target_task",
        "action_dim",
        "latent_dim",
        "data_fingerprint",
    )
    for field in fields:
        if getattr(actual, field) != getattr(expected, field):
            label = field.replace("_", " ")
            raise ValueError(
                f"checkpoint {label} {getattr(actual, field)!r} does not match "
                f"{getattr(expected, field)!r}"
            )
    return actual


class FiniteTrainingCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__(verbose=0)

    def check(self, values: Mapping[str, Any]) -> None:
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                finite = bool(torch.isfinite(value).all())
            else:
                try:
                    finite = bool(np.isfinite(np.asarray(value)).all())
                except (TypeError, ValueError):
                    continue
            if not finite:
                raise FloatingPointError(f"NaN|Inf detected in training value {name!r}")

    def _on_step(self) -> bool:
        values: dict[str, Any] = {}
        for name in ("rewards", "actions", "values", "log_probs"):
            if name in self.locals:
                values[name] = self.locals[name]
        self.check(values)
        return True


class DecoderFineTuneCallback(BaseCallback):
    def __init__(
        self,
        *,
        representation: RepresentationTrainer,
        freeze_steps: int,
        update_steps: list[int],
        update_interval: int = 8,
    ) -> None:
        super().__init__(verbose=0)
        self.representation = representation
        self.freeze_steps = freeze_steps
        self.update_steps = update_steps
        self.update_interval = update_interval
        self._latent: list[np.ndarray] = []
        self._executed: list[np.ndarray] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "etl_sar/latent_action" in info and "etl_sar/executed_action" in info:
                self._latent.append(np.asarray(info["etl_sar/latent_action"], dtype=np.float32))
                self._executed.append(
                    np.asarray(info["etl_sar/executed_action"], dtype=np.float32)
                )
        if self.num_timesteps < self.freeze_steps:
            return True
        if len(self._latent) < self.update_interval:
            return True
        latent = torch.as_tensor(np.stack(self._latent), dtype=torch.float32)
        executed = torch.as_tensor(np.stack(self._executed), dtype=torch.float32)
        self.representation.fine_tune_decoder(
            latent=latent,
            executed_actions=executed,
        )
        self.update_steps.append(self.num_timesteps)
        self._latent.clear()
        self._executed.clear()
        return True


@dataclass(frozen=True)
class TrainingArtifacts:
    latest_checkpoint: Path
    best_checkpoint: Path
    evaluation_log: Path


class TransferTrainer:
    def __init__(
        self,
        *,
        env_factory: Callable[[], gym.Env],
        action_model: ETLSARActionModel,
        representation: RepresentationTrainer,
        run_dir: str | Path,
        total_timesteps: int,
        decoder_freeze_steps: int,
        n_steps: int = 256,
        batch_size: int = 64,
        learning_rate: float = 3e-4,
        seed: int = 0,
    ) -> None:
        self.env_factory = env_factory
        self.action_model = action_model
        self.representation = representation
        self.run_dir = Path(run_dir)
        self.total_timesteps = total_timesteps
        self.decoder_freeze_steps = decoder_freeze_steps
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.decoder_update_steps: list[int] = []
        self.model: PPO | None = None

    def _make_wrapped_env(self) -> gym.Env:
        return Monitor(LatentActionWrapper(self.env_factory(), self.action_model))

    def run(self) -> TrainingArtifacts:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        train_env = DummyVecEnv([self._make_wrapped_env])
        eval_env = DummyVecEnv([self._make_wrapped_env])
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(self.run_dir),
            log_path=str(self.run_dir),
            eval_freq=max(self.n_steps, 1),
            n_eval_episodes=1,
            deterministic=True,
            verbose=0,
        )
        decoder_callback = DecoderFineTuneCallback(
            representation=self.representation,
            freeze_steps=self.decoder_freeze_steps,
            update_steps=self.decoder_update_steps,
            update_interval=max(min(self.n_steps // 2, 8), 1),
        )
        callbacks = CallbackList(
            [FiniteTrainingCallback(), decoder_callback, eval_callback]
        )
        self.model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            seed=self.seed,
            verbose=0,
            tensorboard_log=str(self.run_dir / "tensorboard"),
        )
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=callbacks,
            reset_num_timesteps=True,
        )
        latest_base = self.run_dir / "latest_model"
        self.model.save(latest_base)
        latest = latest_base.with_suffix(".zip")
        best = self.run_dir / "best_model.zip"
        if not best.exists():
            self.model.save(self.run_dir / "best_model")
        train_env.close()
        eval_env.close()
        return TrainingArtifacts(
            latest_checkpoint=latest,
            best_checkpoint=best,
            evaluation_log=self.run_dir / "evaluations.npz",
        )

# Imported after trainer definitions to keep exploration lifecycle isolated.
from etl_sar.exploration import DirectionalExplorationWrapper, ExploreTrainer

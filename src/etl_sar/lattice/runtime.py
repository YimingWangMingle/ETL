from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Callable, Type

import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv, VecNormalize

from etl_sar.lattice.trainers import hand_model_kwargs, leg_model_kwargs


class StopAtTransitionCallback(BaseCallback):
    def __init__(self, *, target: int) -> None:
        super().__init__(verbose=0)
        if target <= 0:
            raise ValueError("target transition count must be positive")
        self.target = int(target)

    def _on_training_start(self) -> None:
        if self.training_env is None:
            raise RuntimeError("transition callback has no training environment")
        if self.target % self.training_env.num_envs:
            raise ValueError("target transitions must be divisible by num_envs")

    def _on_step(self) -> bool:
        return self.num_timesteps < self.target


def checkpoint_boundaries(
    completed: int, total: int, interval: int
) -> tuple[int, ...]:
    if completed < 0 or total < completed:
        raise ValueError("checkpoint range is invalid")
    if interval <= 0:
        raise ValueError("checkpoint interval must be positive")
    next_boundary = ((completed // interval) + 1) * interval
    boundaries = list(range(next_boundary, total + 1, interval))
    if not boundaries or boundaries[-1] != total:
        boundaries.append(total)
    return tuple(boundaries)


def save_lattice_checkpoint(
    *,
    model: BaseAlgorithm,
    vecnormalize: VecNormalize,
    run_dir: str | Path,
    name: str = "latest",
    include_replay: bool = True,
    write_transition_count: bool = True,
) -> set[Path]:
    if not name or Path(name).name != name:
        raise ValueError("Lattice checkpoint name must be a path component")
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    temporary_policy_base = output / f".policy_{token}"
    temporary_policy = temporary_policy_base.with_suffix(".zip")
    temporary_vec = output / f".vecnormalize_{token}.pkl"
    temporary_replay = output / f".replay_{token}.pkl"
    temporary_count = output / f".count_{token}.json"
    policy_path = output / f"{name}_policy.zip"
    vec_path = output / f"{name}_vecnormalize.pkl"
    replay_path = output / f"{name}_replay_buffer.pkl"
    count_path = output / "transition_count.json"
    artifacts = {policy_path, vec_path}
    if write_transition_count:
        artifacts.add(count_path)
    try:
        model.save(temporary_policy_base)
        vecnormalize.save(temporary_vec)
        if include_replay and hasattr(model, "save_replay_buffer"):
            model.save_replay_buffer(temporary_replay)
        if write_transition_count:
            temporary_count.write_text(
                json.dumps({"transitions": int(model.num_timesteps)}, indent=2),
                encoding="utf-8",
            )
        os.replace(temporary_policy, policy_path)
        os.replace(temporary_vec, vec_path)
        if temporary_replay.is_file():
            os.replace(temporary_replay, replay_path)
            artifacts.add(replay_path)
        if write_transition_count:
            os.replace(temporary_count, count_path)
    finally:
        for path in (
            temporary_policy,
            temporary_vec,
            temporary_replay,
            temporary_count,
        ):
            path.unlink(missing_ok=True)
    return artifacts


def load_lattice_checkpoint(
    *,
    algorithm: Type[BaseAlgorithm],
    run_dir: str | Path,
    env: VecEnv,
    device: str = "auto",
    require_replay_buffer: bool = False,
) -> tuple[BaseAlgorithm, VecNormalize]:
    source = Path(run_dir)
    policy_path = source / "latest_policy.zip"
    vec_path = source / "latest_vecnormalize.pkl"
    count_path = source / "transition_count.json"
    required = (policy_path, vec_path, count_path)
    missing = [path.name for path in required if not path.is_file()]
    replay_path = source / "latest_replay_buffer.pkl"
    if require_replay_buffer and not replay_path.is_file():
        missing.append(replay_path.name)
    if missing:
        raise ValueError(f"incomplete Lattice checkpoint: {', '.join(missing)}")
    vecnormalize = VecNormalize.load(vec_path, env)
    vecnormalize.training = True
    model = algorithm.load(policy_path, env=vecnormalize, device=device)
    if replay_path.is_file() and hasattr(model, "load_replay_buffer"):
        model.load_replay_buffer(replay_path)
    expected = int(json.loads(count_path.read_text(encoding="utf-8"))["transitions"])
    if int(model.num_timesteps) != expected:
        raise ValueError("Lattice policy transition count does not match checkpoint")
    return model, vecnormalize


class LatticeCheckpointCallback(BaseCallback):
    def __init__(
        self,
        *,
        run_dir: Path,
        vecnormalize: VecNormalize,
        interval: int,
        evaluate: Callable[[BaseAlgorithm, VecNormalize, int], None] | None = None,
    ) -> None:
        super().__init__(verbose=0)
        self.run_dir = run_dir
        self.vecnormalize = vecnormalize
        self.interval = interval
        self.evaluate = evaluate
        self.next_checkpoint = interval

    def _on_training_start(self) -> None:
        if self.interval <= 0:
            raise ValueError("checkpoint interval must be positive")
        self.next_checkpoint = (
            (self.model.num_timesteps // self.interval) + 1
        ) * self.interval

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_checkpoint:
            return True
        save_lattice_checkpoint(
            model=self.model,
            vecnormalize=self.vecnormalize,
            run_dir=self.run_dir,
            name=f"checkpoint_{self.num_timesteps}",
            include_replay=False,
            write_transition_count=False,
        )
        save_lattice_checkpoint(
            model=self.model,
            vecnormalize=self.vecnormalize,
            run_dir=self.run_dir,
        )
        if self.evaluate is not None:
            self.evaluate(self.model, self.vecnormalize, self.num_timesteps)
        self.next_checkpoint += self.interval
        return True


def make_myo_vector_env(
    *, env_id: str, seed: int, num_envs: int, monitor_dir: str | Path
) -> SubprocVecEnv:
    monitor = Path(monitor_dir)
    monitor.mkdir(parents=True, exist_ok=True)

    def factory(rank: int):
        def build():
            import myosuite  # noqa: F401

            env = gym.make(env_id)
            env.reset(seed=seed + rank)
            return Monitor(env, filename=str(monitor / f"env_{rank}"))

        return build

    return SubprocVecEnv([factory(rank) for rank in range(num_envs)])


def train_lattice_job(
    *,
    domain: str,
    env_id: str,
    seed: int,
    total_transitions: int,
    checkpoint_interval: int,
    run_dir: str | Path,
    num_envs: int = 16,
    device: str = "auto",
    evaluate: Callable[[BaseAlgorithm, VecNormalize, int], None] | None = None,
) -> BaseAlgorithm:
    if domain not in {"hand", "leg"}:
        raise ValueError("Lattice domain must be hand or leg")
    if total_transitions % num_envs or checkpoint_interval % num_envs:
        raise ValueError("Lattice budgets must be divisible by num_envs")
    output = Path(run_dir)
    output.mkdir(parents=True, exist_ok=True)
    base_env = make_myo_vector_env(
        env_id=env_id,
        seed=seed,
        num_envs=num_envs,
        monitor_dir=output / "monitor",
    )
    algorithm: Type[BaseAlgorithm] = RecurrentPPO if domain == "hand" else SAC
    count_path = output / "transition_count.json"
    if count_path.is_file():
        model, vecnormalize = load_lattice_checkpoint(
            algorithm=algorithm,
            run_dir=output,
            env=base_env,
            device=device,
            require_replay_buffer=domain == "leg",
        )
        completed = int(model.num_timesteps)
    else:
        vecnormalize = VecNormalize(base_env)
        kwargs = (
            hand_model_kwargs(seed=seed, device=device)
            if domain == "hand"
            else leg_model_kwargs(seed=seed, device=device)
        )
        model = algorithm(
            env=vecnormalize,
            tensorboard_log=str(output / "tensorboard"),
            verbose=1,
            **kwargs,
        )
        completed = 0
    if completed > total_transitions:
        raise ValueError("Lattice resume checkpoint exceeds the declared budget")
    if completed < total_transitions:
        callbacks = CallbackList(
            [
                LatticeCheckpointCallback(
                    run_dir=output,
                    vecnormalize=vecnormalize,
                    interval=checkpoint_interval,
                    evaluate=evaluate,
                ),
                StopAtTransitionCallback(target=total_transitions),
            ]
        )
        model.learn(
            total_timesteps=total_transitions - completed,
            callback=callbacks,
            reset_num_timesteps=completed == 0,
        )
    save_lattice_checkpoint(
        model=model,
        vecnormalize=vecnormalize,
        run_dir=output,
    )
    return model

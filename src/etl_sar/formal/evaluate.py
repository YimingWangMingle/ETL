from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import gymnasium as gym
import numpy as np

from etl_sar.formal.metrics import EpisodeMetrics, resolve_root_x_qpos_address
from etl_sar.protocols import task_succeeded


@dataclass(frozen=True)
class FormalEvaluationSummary:
    domain: str
    episodes: int
    environment_steps: int
    mean_primary: float
    mean_return: float
    median_return: float
    success_rate: float
    mean_episode_length: float
    termination_rate: float
    fall_rate: float
    object_drop_rate: float
    mean_velocity_tracking_error: float | None
    mean_muscle_effort: float


def _predict(
    model: Any,
    observation: np.ndarray,
    *,
    recurrent: bool,
    state: Any,
    episode_start: bool,
) -> tuple[np.ndarray, Any]:
    if recurrent:
        return model.predict(
            observation,
            state=state,
            episode_start=np.asarray([episode_start], dtype=bool),
            deterministic=True,
        )
    action, next_state = model.predict(observation, deterministic=True)
    return action, next_state


def _resolve_mujoco_model_data(env: gym.Env) -> tuple[Any, Any]:
    unwrapped = env.unwrapped
    candidates = (
        unwrapped,
        getattr(unwrapped, "sim", None),
        getattr(unwrapped, "mj_sim", None),
    )
    model = next(
        (
            candidate.model
            for candidate in candidates
            if candidate is not None and hasattr(candidate, "model")
        ),
        None,
    )
    data = next(
        (
            candidate.data
            for candidate in candidates
            if candidate is not None and hasattr(candidate, "data")
        ),
        None,
    )
    if model is None or data is None:
        raise AttributeError(
            "environment does not expose MuJoCo model/data directly or through sim"
        )
    return model, data


def evaluate_formal(
    model: Any,
    env: gym.Env,
    *,
    domain: str,
    seeds: Iterable[int],
    output_dir: str | Path,
    environment_steps: int,
    recurrent: bool = False,
) -> FormalEvaluationSummary:
    if domain not in {"hand", "leg"}:
        raise ValueError("formal evaluation domain must be hand or leg")
    seed_bank = tuple(int(seed) for seed in seeds)
    if not seed_bank or len(seed_bank) != len(set(seed_bank)):
        raise ValueError("formal evaluation requires a nonempty unique seed bank")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mujoco_model = mujoco_data = None
    if domain == "leg":
        mujoco_model, mujoco_data = _resolve_mujoco_model_data(env)
    root_address = resolve_root_x_qpos_address(mujoco_model) if mujoco_model is not None else None
    rows: list[EpisodeMetrics] = []
    try:
        for episode, seed in enumerate(seed_bank):
            observation, _ = env.reset(seed=seed)
            start_x = (
                float(mujoco_data.qpos[root_address])
                if root_address is not None
                else None
            )
            state = None
            episode_start = True
            terminated = truncated = False
            episode_return = 0.0
            length = 0
            success = False
            object_drop = False
            velocity_errors: list[float] = []
            efforts: list[float] = []
            last_root_x = None
            while not (terminated or truncated):
                action, state = _predict(
                    model,
                    observation,
                    recurrent=recurrent,
                    state=state,
                    episode_start=episode_start,
                )
                episode_start = False
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                length += 1
                success = success or task_succeeded(info)
                object_drop = object_drop or bool(
                    info.get("object_drop", info.get("dropped", False))
                )
                if "velocity" in info and "target_velocity" in info:
                    velocity_errors.append(
                        abs(float(info["velocity"]) - float(info["target_velocity"]))
                    )
                if "formal/root_x" in info:
                    last_root_x = float(info["formal/root_x"])
                efforts.append(float(np.mean(np.square(np.asarray(action, dtype=float)))))
            end_x = (
                (
                    last_root_x
                    if last_root_x is not None
                    else float(mujoco_data.qpos[root_address])
                )
                if root_address is not None
                else None
            )
            rows.append(
                EpisodeMetrics(
                    episode=episode,
                    seed=seed,
                    episode_return=episode_return,
                    length=length,
                    success=success,
                    terminated=terminated,
                    fall=bool(domain == "leg" and terminated and not success),
                    object_drop=object_drop,
                    forward_distance=(end_x - start_x) if end_x is not None else None,
                    velocity_tracking_error=(
                        float(np.mean(velocity_errors)) if velocity_errors else None
                    ),
                    muscle_effort=float(np.mean(efforts)),
                )
            )
    finally:
        env.close()

    mappings = [row.to_mapping() for row in rows]
    fieldnames = list(mappings[0])
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mappings)

    returns = np.asarray([row.episode_return for row in rows], dtype=float)
    primary = np.asarray(
        [
            float(row.success) if domain == "hand" else float(row.forward_distance)
            for row in rows
        ]
    )
    velocity = [
        row.velocity_tracking_error
        for row in rows
        if row.velocity_tracking_error is not None
    ]
    summary = FormalEvaluationSummary(
        domain=domain,
        episodes=len(rows),
        environment_steps=environment_steps,
        mean_primary=float(np.mean(primary)),
        mean_return=float(np.mean(returns)),
        median_return=float(np.median(returns)),
        success_rate=float(np.mean([row.success for row in rows])),
        mean_episode_length=float(np.mean([row.length for row in rows])),
        termination_rate=float(np.mean([row.terminated for row in rows])),
        fall_rate=float(np.mean([row.fall for row in rows])),
        object_drop_rate=float(np.mean([row.object_drop for row in rows])),
        mean_velocity_tracking_error=(float(np.mean(velocity)) if velocity else None),
        mean_muscle_effort=float(np.mean([row.muscle_effort for row in rows])),
    )
    (output / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2), encoding="utf-8"
    )
    return summary

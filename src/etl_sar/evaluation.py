from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from etl_sar.protocols import task_succeeded


@dataclass(frozen=True)
class EvaluationSummary:
    episodes: int
    environment_steps: int
    mean_return: float
    return_std: float
    success_rate: float
    environment_id: str | None = None
    evaluation_seed: int | None = None
    sar_scale: float | None = None

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationSummary":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def evaluate_checkpoint(
    model: Any,
    env: gym.Env,
    *,
    episodes: int,
    output_dir: str | Path,
    environment_steps: int,
    seed: int = 10_000,
    environment_id: str | None = None,
    sar_scale: float | None = None,
) -> EvaluationSummary:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    policy = getattr(model, "policy", None)
    before = (
        {name: value.detach().clone() for name, value in policy.state_dict().items()}
        if policy is not None and hasattr(policy, "state_dict")
        else None
    )
    returns: list[float] = []
    successes: list[bool] = []
    rows: list[dict[str, Any]] = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            terminated = truncated = False
            episode_return = 0.0
            episode_steps = 0
            success = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += float(reward)
                episode_steps += 1
                success = success or task_succeeded(info)
            returns.append(episode_return)
            successes.append(success)
            rows.append(
                {
                    "episode": episode,
                    "return": episode_return,
                    "steps": episode_steps,
                    "success": int(success),
                }
            )
    finally:
        env.close()
    if before is not None:
        after = policy.state_dict()
        if any(not torch.equal(value, after[name]) for name, value in before.items()):
            raise RuntimeError("evaluation changed model parameters")
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("episode", "return", "steps", "success"))
        writer.writeheader()
        writer.writerows(rows)
    summary = EvaluationSummary(
        episodes=episodes,
        environment_steps=environment_steps,
        mean_return=float(np.mean(returns)),
        return_std=float(np.std(returns)),
        success_rate=float(np.mean(successes)),
        environment_id=environment_id,
        evaluation_seed=seed,
        sar_scale=sar_scale,
    )
    summary.save(output / "summary.json")
    return summary


def compare_runs(
    baseline: EvaluationSummary,
    extension: EvaluationSummary,
) -> dict[str, float | int | str | None]:
    if baseline.episodes != extension.episodes:
        raise ValueError("comparison requires equal episode counts")
    if baseline.environment_steps != extension.environment_steps:
        raise ValueError("comparison requires equal environment-step budgets")
    if (
        baseline.environment_id is not None
        and extension.environment_id is not None
        and baseline.environment_id != extension.environment_id
    ):
        raise ValueError("comparison requires equal environment IDs")
    if (
        baseline.evaluation_seed is not None
        and extension.evaluation_seed is not None
        and baseline.evaluation_seed != extension.evaluation_seed
    ):
        raise ValueError("comparison requires equal evaluation seeds")
    return {
        "episodes": baseline.episodes,
        "environment_steps": baseline.environment_steps,
        "environment_id": baseline.environment_id or extension.environment_id,
        "evaluation_seed": baseline.evaluation_seed or extension.evaluation_seed,
        "baseline_sar_scale": baseline.sar_scale,
        "extension_sar_scale": extension.sar_scale,
        "baseline_success_rate": baseline.success_rate,
        "extension_success_rate": extension.success_rate,
        "success_rate_delta": extension.success_rate - baseline.success_rate,
        "baseline_mean_return": baseline.mean_return,
        "extension_mean_return": extension.mean_return,
        "mean_return_delta": extension.mean_return - baseline.mean_return,
    }

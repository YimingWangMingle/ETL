from __future__ import annotations

import json

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn

from etl_sar.evaluation import EvaluationSummary, compare_runs, evaluate_checkpoint


class EvalEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        done = self.steps == 3
        return np.zeros(3, dtype=np.float32), 1.0, done, False, {"success": done}


class SolvedEvalEnv(EvalEnv):
    def step(self, action):
        observation, reward, terminated, truncated, _ = super().step(action)
        return observation, reward, terminated, truncated, {"solved": terminated}


class Predictor:
    def __init__(self) -> None:
        self.policy = nn.Linear(3, 2)

    def predict(self, observation, deterministic=True):
        with torch.no_grad():
            action = self.policy(torch.as_tensor(observation).float()).numpy()
        return action, None


def test_evaluation_never_changes_parameters_and_writes_outputs(tmp_path) -> None:
    model = Predictor()
    before = {name: value.clone() for name, value in model.policy.state_dict().items()}

    summary = evaluate_checkpoint(
        model,
        EvalEnv(),
        episodes=2,
        output_dir=tmp_path,
        environment_steps=64,
    )

    assert summary.success_rate == pytest.approx(1.0)
    assert summary.mean_return == pytest.approx(3.0)
    assert (tmp_path / "episodes.csv").exists()
    assert (tmp_path / "summary.json").exists()
    assert all(
        torch.equal(before[name], value)
        for name, value in model.policy.state_dict().items()
    )


def test_evaluation_accepts_myosuite_solved_flag(tmp_path) -> None:
    summary = evaluate_checkpoint(
        Predictor(),
        SolvedEvalEnv(),
        episodes=2,
        output_dir=tmp_path,
        environment_steps=64,
    )

    assert summary.success_rate == pytest.approx(1.0)


def test_compare_runs_reports_equal_budget_delta() -> None:
    baseline = EvaluationSummary(
        episodes=10,
        environment_steps=1000,
        mean_return=2.0,
        return_std=0.2,
        success_rate=0.4,
    )
    extension = EvaluationSummary(
        episodes=10,
        environment_steps=1000,
        mean_return=3.0,
        return_std=0.3,
        success_rate=0.6,
    )

    result = compare_runs(baseline, extension)

    assert result["success_rate_delta"] == pytest.approx(0.2)
    assert result["mean_return_delta"] == pytest.approx(1.0)


def test_compare_runs_rejects_unequal_budgets() -> None:
    baseline = EvaluationSummary(1, 100, 1.0, 0.0, 0.5)
    extension = EvaluationSummary(1, 200, 2.0, 0.0, 1.0)

    with pytest.raises(ValueError, match="equal environment-step budgets"):
        compare_runs(baseline, extension)

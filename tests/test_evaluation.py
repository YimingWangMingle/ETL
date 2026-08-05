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


def matched_summary(**overrides) -> EvaluationSummary:
    values = {
        "episodes": 10,
        "environment_steps": 20_000,
        "mean_return": 2.0,
        "return_std": 0.2,
        "success_rate": 0.4,
        "environment_id": "myoHandReorient100-v0",
        "evaluation_seed": 10_007,
        "sar_scale": 0.0,
    }
    values.update(overrides)
    return EvaluationSummary(**values)


def test_evaluation_records_matched_protocol_metadata(tmp_path) -> None:
    summary = evaluate_checkpoint(
        Predictor(),
        EvalEnv(),
        episodes=2,
        output_dir=tmp_path,
        environment_steps=64,
        seed=123,
        environment_id="myoHandReorient100-v0",
        sar_scale=0.0,
    )

    assert summary.environment_id == "myoHandReorient100-v0"
    assert summary.evaluation_seed == 123
    assert summary.sar_scale == 0.0
    assert EvaluationSummary.load(tmp_path / "summary.json") == summary


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"episodes": 11}, "episode counts"),
        ({"environment_steps": 19_000}, "environment-step budgets"),
        ({"environment_id": "other-task"}, "environment IDs"),
        ({"evaluation_seed": 10_008}, "evaluation seeds"),
    ),
)
def test_compare_runs_rejects_mismatched_protocol(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        compare_runs(
            matched_summary(),
            matched_summary(sar_scale=1.0, **overrides),
        )


def test_compare_runs_includes_matched_protocol_metadata() -> None:
    result = compare_runs(
        matched_summary(),
        matched_summary(
            mean_return=3.0,
            success_rate=0.6,
            sar_scale=1.0,
        ),
    )

    assert result["episodes"] == 10
    assert result["environment_id"] == "myoHandReorient100-v0"
    assert result["evaluation_seed"] == 10_007
    assert result["baseline_sar_scale"] == 0.0
    assert result["extension_sar_scale"] == 1.0


def test_old_evaluation_summary_json_remains_loadable(tmp_path) -> None:
    path = tmp_path / "old-summary.json"
    path.write_text(
        json.dumps(
            {
                "episodes": 1,
                "environment_steps": 64,
                "mean_return": 1.0,
                "return_std": 0.0,
                "success_rate": 0.0,
            }
        ),
        encoding="utf-8",
    )

    summary = EvaluationSummary.load(path)

    assert summary.environment_id is None
    assert summary.evaluation_seed is None
    assert summary.sar_scale is None

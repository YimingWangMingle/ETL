from __future__ import annotations

import csv

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from etl_sar.formal.evaluate import evaluate_formal


class Joint:
    name = "root"


class Model:
    njnt = 1
    jnt_type = np.asarray([0])
    jnt_qposadr = np.asarray([4])

    def joint(self, index):
        return Joint()


class Data:
    qpos = np.zeros(12, dtype=np.float64)


class LegEvalEnv(gym.Env[np.ndarray, np.ndarray]):
    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self.model = Model()
        self.data = Data()
        self.steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.data.qpos[4] = float((0 if seed is None else seed) % 3)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        self.data.qpos[4] += 0.5
        terminated = self.steps == 2
        return (
            np.zeros(3, dtype=np.float32),
            1.0,
            terminated,
            False,
            {
                "solved": terminated,
                "velocity": 0.8,
                "target_velocity": 1.0,
            },
        )


class RecurrentPredictor:
    def __init__(self) -> None:
        self.episode_starts = []

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        self.episode_starts.append(bool(episode_start[0]))
        next_state = np.asarray([0 if state is None else int(state[0]) + 1])
        return np.asarray([0.5, -0.5], dtype=np.float32), next_state


def test_formal_evaluation_writes_raw_leg_metrics_and_resets_recurrent_state(tmp_path) -> None:
    predictor = RecurrentPredictor()
    result = evaluate_formal(
        predictor,
        LegEvalEnv(),
        domain="leg",
        seeds=[101, 202],
        output_dir=tmp_path,
        environment_steps=250_000,
        recurrent=True,
    )

    assert result.episodes == 2
    assert result.mean_primary == pytest.approx(1.0)
    assert result.mean_return == pytest.approx(2.0)
    assert predictor.episode_starts == [True, False, True, False]
    with (tmp_path / "episodes.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
    assert float(rows[0]["forward_distance"]) == pytest.approx(1.0)
    assert float(rows[0]["velocity_tracking_error"]) == pytest.approx(0.2)
    assert float(rows[0]["muscle_effort"]) == pytest.approx(0.25)
    assert (tmp_path / "summary.json").is_file()

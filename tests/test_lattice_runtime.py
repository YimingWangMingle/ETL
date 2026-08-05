from __future__ import annotations

import json

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from etl_sar.lattice.runtime import load_lattice_checkpoint, save_lattice_checkpoint
from etl_sar.lattice.trainers import leg_model_kwargs
from tests.test_lattice_policies import ContinuousEnv


def make_env() -> VecNormalize:
    return VecNormalize(DummyVecEnv([ContinuousEnv]), norm_reward=False)


def test_sac_checkpoint_round_trip_restores_replay_and_transition_counter(tmp_path) -> None:
    env = make_env()
    kwargs = leg_model_kwargs(seed=2, device="cpu")
    kwargs.update(buffer_size=64, learning_starts=1, batch_size=2, gradient_steps=1)
    model = SAC(env=env, **kwargs)
    model.learn(total_timesteps=16)

    save_lattice_checkpoint(model=model, vecnormalize=env, run_dir=tmp_path)

    restored_env = make_env()
    restored, restored_vec = load_lattice_checkpoint(
        algorithm=SAC,
        run_dir=tmp_path,
        env=restored_env.venv,
        device="cpu",
        require_replay_buffer=True,
    )
    assert restored.num_timesteps == 16
    assert restored.replay_buffer.size() == model.replay_buffer.size()
    assert restored_vec.training is True
    assert json.loads((tmp_path / "transition_count.json").read_text())["transitions"] == 16

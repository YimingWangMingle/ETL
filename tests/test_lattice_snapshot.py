from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from etl_sar.lattice.runtime import save_lattice_checkpoint
from etl_sar.lattice.trainers import leg_model_kwargs
from tests.test_lattice_policies import ContinuousEnv


def test_intermediate_snapshot_keeps_policy_and_normalization_without_replay(tmp_path) -> None:
    env = VecNormalize(DummyVecEnv([ContinuousEnv]), norm_reward=False)
    kwargs = leg_model_kwargs(seed=1, device="cpu")
    kwargs.update(buffer_size=32, learning_starts=100, batch_size=2)
    model = SAC(env=env, **kwargs)
    model.learn(total_timesteps=8)

    artifacts = save_lattice_checkpoint(
        model=model,
        vecnormalize=env,
        run_dir=tmp_path,
        name="checkpoint_8",
        include_replay=False,
        write_transition_count=False,
    )

    assert {path.name for path in artifacts} == {
        "checkpoint_8_policy.zip",
        "checkpoint_8_vecnormalize.pkl",
    }
    assert not (tmp_path / "checkpoint_8_replay_buffer.pkl").exists()

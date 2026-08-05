from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from etl_sar.lattice.runtime import StopAtTransitionCallback
from etl_sar.lattice.trainers import leg_model_kwargs
from tests.test_lattice_policies import ContinuousEnv


def test_callback_stops_vector_training_at_exact_transition_budget() -> None:
    env = DummyVecEnv([ContinuousEnv, ContinuousEnv])
    kwargs = leg_model_kwargs(seed=0, device="cpu")
    kwargs.update(buffer_size=64, learning_starts=100, batch_size=2)
    model = SAC(env=env, **kwargs)
    model.learn(total_timesteps=100, callback=StopAtTransitionCallback(target=10))
    assert model.num_timesteps == 10

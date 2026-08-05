from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from etl_sar.lattice.evaluation import evaluate_lattice_model
from tests.test_formal_evaluate import LegEvalEnv, RecurrentPredictor


def test_lattice_vecnormalize_evaluation_preserves_terminal_leg_distance(tmp_path) -> None:
    training_stats = VecNormalize(
        DummyVecEnv([LegEvalEnv]), training=True, norm_reward=True
    )
    result = evaluate_lattice_model(
        model=RecurrentPredictor(),
        training_vecnormalize=training_stats,
        env_factory=LegEvalEnv,
        domain="leg",
        seeds=[101, 202],
        output_dir=tmp_path,
        environment_steps=250_000,
        recurrent=True,
    )
    assert result.mean_primary == 1.0

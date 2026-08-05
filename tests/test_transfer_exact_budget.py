from etl_sar.trainers import TransferTrainer
from tests.test_trainers import TinyMuscleEnv, make_components


def test_transfer_stops_at_exact_interaction_budget_inside_rollout(tmp_path) -> None:
    action_model, representation = make_components()
    trainer = TransferTrainer(
        env_factory=TinyMuscleEnv,
        action_model=action_model,
        representation=representation,
        run_dir=tmp_path,
        total_timesteps=20,
        decoder_freeze_steps=10,
        n_steps=16,
        batch_size=8,
        eval_freq=16,
    )
    trainer.run()
    assert trainer.model is not None
    assert trainer.model.num_timesteps == 20

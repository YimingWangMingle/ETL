from etl_sar.trainers import TransferTrainer
from tests.test_trainers import TinyMuscleEnv, make_components


def test_transfer_writes_unified_metrics_at_each_formal_checkpoint(tmp_path) -> None:
    action_model, representation = make_components()
    trainer = TransferTrainer(
        env_factory=TinyMuscleEnv,
        action_model=action_model,
        representation=representation,
        run_dir=tmp_path,
        total_timesteps=16,
        decoder_freeze_steps=8,
        n_steps=16,
        batch_size=8,
        eval_freq=16,
        evaluation_episodes=2,
        evaluation_seed=50_000,
        formal_domain="hand",
    )
    trainer.run()
    assert (tmp_path / "evaluation" / "checkpoint_16" / "episodes.csv").is_file()
    assert (tmp_path / "evaluation" / "checkpoint_16" / "summary.json").is_file()

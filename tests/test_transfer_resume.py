from __future__ import annotations

from tests.test_trainers import TinyMuscleEnv, make_components

from etl_sar.trainers import TransferTrainer


def test_transfer_resume_continues_counter_from_paired_checkpoint(tmp_path) -> None:
    first_model, first_representation = make_components()
    first = TransferTrainer(
        env_factory=TinyMuscleEnv,
        action_model=first_model,
        representation=first_representation,
        run_dir=tmp_path / "first",
        total_timesteps=32,
        decoder_freeze_steps=16,
        n_steps=16,
        batch_size=8,
        eval_freq=16,
        evaluation_episodes=2,
        evaluation_seed=50_000,
        seed=3,
    )
    artifacts = first.run()

    resumed_model, resumed_representation = make_components()
    resumed = TransferTrainer(
        env_factory=TinyMuscleEnv,
        action_model=resumed_model,
        representation=resumed_representation,
        run_dir=tmp_path / "resumed",
        total_timesteps=64,
        decoder_freeze_steps=16,
        n_steps=16,
        batch_size=8,
        eval_freq=16,
        evaluation_episodes=2,
        evaluation_seed=50_000,
        resume_manifest=artifacts.latest_manifest,
        seed=3,
    )
    result = resumed.run()

    assert resumed.model is not None
    assert resumed.model.num_timesteps == 64
    assert result.latest_manifest.exists()
    assert (tmp_path / "resumed" / "checkpoint_48_pair.json").exists()
    assert (tmp_path / "resumed" / "checkpoint_64_pair.json").exists()

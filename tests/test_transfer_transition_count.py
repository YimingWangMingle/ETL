import json

from etl_sar.trainers import TransferTrainer
from tests.test_trainers import TinyMuscleEnv, make_components


def test_transfer_writes_atomic_resume_transition_count(tmp_path) -> None:
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
    )
    trainer.run()
    assert json.loads((tmp_path / "transition_count.json").read_text())["transitions"] == 16

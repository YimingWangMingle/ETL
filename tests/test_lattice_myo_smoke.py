from __future__ import annotations

import pytest

pytestmark = pytest.mark.myo


@pytest.mark.parametrize(
    ("domain", "env_id", "expects_replay"),
    [
        ("hand", "myoHandReorient100-v0", False),
        ("leg", "myoLegRoughTerrainWalk-v0", True),
    ],
)
def test_official_lattice_policy_completes_short_myo_training_segment(
    tmp_path, domain, env_id, expects_replay
) -> None:
    pytest.importorskip("myosuite")
    from etl_sar.lattice.runtime import train_lattice_job

    model = train_lattice_job(
        domain=domain,
        env_id=env_id,
        seed=0,
        total_transitions=16,
        checkpoint_interval=16,
        run_dir=tmp_path,
        num_envs=1,
        device="cpu",
    )

    assert model.num_timesteps == 16
    assert (tmp_path / "latest_policy.zip").is_file()
    assert (tmp_path / "latest_vecnormalize.pkl").is_file()
    assert (tmp_path / "latest_replay_buffer.pkl").is_file() is expects_replay

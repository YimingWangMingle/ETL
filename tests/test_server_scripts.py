from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_portable_runner_supports_dry_run_source_and_target_modes() -> None:
    script = (ROOT / "scripts" / "run_formal_server.sh").read_text(encoding="utf-8")
    assert "etl_sar.formal.server" in script
    assert "SLURM_ARRAY_TASK_ID" in script
    assert "dry-run" in script


def test_slurm_submission_orders_sources_before_targets() -> None:
    script = (ROOT / "scripts" / "submit_formal_slurm.sh").read_text(encoding="utf-8")
    assert "--array=0-9" in script
    assert "--array=0-29" in script
    assert "afterok" in script

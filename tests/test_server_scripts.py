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


def test_single_seed_launcher_is_isolated_sequential_and_cuda_checked() -> None:
    script = (ROOT / "scripts" / "run_single_seed_10h.sh").read_text(
        encoding="utf-8"
    )
    assert "configs/single_seed_10h_hand.yaml" in script
    assert "configs/single_seed_10h_leg.yaml" in script
    assert "runs/single_seed_10h" in script
    assert "SHORT_OUTPUT_ROOT" in script
    assert "torch.cuda.is_available()" in script
    assert "for index in 0 1" in script
    assert "for index in 0 1 2 3 4 5" in script
    assert "etl_sar.formal.server" in script
    assert "etl_sar.formal.aggregate_cli" in script
    assert '--hand-config "${HAND_CONFIG}"' in script
    assert '--leg-config "${LEG_CONFIG}"' in script
    assert "sbatch" not in script
    assert "&" not in script


def test_readme_documents_the_single_seed_server_profile() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "run_single_seed_10h.sh" in readme
    assert "SHORT_OUTPUT_ROOT" in readme
    assert "8.4M" in readme
    assert "seed=0" in readme
    assert "RTX 4090" in readme
    assert "descriptive" in readme
    assert "conda create -n etl-lattice-sar python=3.11" in readme
    assert "python -m pytest -m myo -q" in readme


def test_shell_scripts_use_unix_line_endings() -> None:
    scripts = sorted((ROOT / "scripts").glob("*.sh"))
    assert scripts
    for script in scripts:
        content = script.read_bytes()
        assert b"\r\n" not in content, f"{script.name} contains CRLF line endings"

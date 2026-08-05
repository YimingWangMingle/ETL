from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_minimal_pilot.ps1"


def test_minimal_pilot_script_contains_both_matched_domains() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "configs/hand_quick.yaml" in text
    assert "configs/leg_quick.yaml" in text
    assert "--sar-scale 0.0" in text
    assert "--sar-scale 1.0" in text
    assert text.count("--timesteps 20000") >= 4
    assert text.count("--episodes 10") >= 4
    assert text.count("latest_pair.json") >= 8
    assert text.count("--pair-manifest") >= 8
    assert "latest_model.zip" not in text
    assert "--model-path" not in text


def test_minimal_pilot_script_uses_validated_completion_markers() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "Get-FileHash" in text
    assert "stage.complete.json" in text
    assert '$ErrorActionPreference = "Stop"' in text
    assert "Test-Path -LiteralPath" in text


def test_minimal_pilot_whatif_prints_all_stages_without_creating_runs(
    tmp_path,
) -> None:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-File",
            str(SCRIPT),
            "-EtlSar",
            "etl-sar",
            "-RunRoot",
            str(tmp_path / "pilot"),
            "-WhatIf",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "pilot").exists()
    for stage in (
        "hand/source",
        "hand/representation",
        "hand/baseline",
        "hand/extension",
        "hand/baseline-evaluation",
        "hand/extension-evaluation",
        "hand/comparison",
        "leg/source",
        "leg/representation",
        "leg/baseline",
        "leg/extension",
        "leg/baseline-evaluation",
        "leg/extension-evaluation",
        "leg/comparison",
        "pilot-summary",
    ):
        assert f"[WhatIf] {stage}" in result.stdout

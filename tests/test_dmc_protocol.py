from __future__ import annotations

from pathlib import Path

from etl_sar.dmc.runtime import common_sac_kwargs, lattice_sac_kwargs


ROOT = Path(__file__).resolve().parents[1]


def test_lattice_changes_only_policy_exploration_settings() -> None:
    common = common_sac_kwargs(seed=0, device="cpu")
    lattice = lattice_sac_kwargs(seed=0, device="cpu")
    assert {
        key: value for key, value in lattice.items() if key != "policy_kwargs"
    } == {
        key: value for key, value in common.items() if key != "policy_kwargs"
    }
    assert lattice["policy_kwargs"]["use_lattice"] is True
    assert common["use_sde"] is True


def test_server_launcher_is_sequential_and_uses_isolated_output() -> None:
    script = ROOT / "scripts" / "run_dmc_transfer_pilot.sh"
    raw = script.read_bytes()
    text = raw.decode("utf-8")
    assert b"\r\n" not in raw
    assert "runs/dmc_transfer_pilot" in text
    assert "python -m etl_sar.dmc.server" in text
    assert "&" not in "\n".join(
        line for line in text.splitlines() if not line.startswith("#!")
    )

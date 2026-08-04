from __future__ import annotations

from pathlib import Path

import pytest

from etl_sar.config import ExperimentConfig
from etl_sar.types import Limb


def test_default_representation_is_sar_protocol() -> None:
    config = ExperimentConfig.minimal(
        limb=Limb.HAND,
        source_env="source-v0",
        target_env="target-v0",
    )

    assert config.representation.latent_dim == 20
    assert config.representation.mixture_components == 20
    assert config.synergy.components == 20
    assert config.synergy.rho == pytest.approx(0.20)
    config.validate()


def test_rejects_cross_limb_task_metadata() -> None:
    config = ExperimentConfig.minimal(
        limb=Limb.HAND,
        source_env="source-v0",
        target_env="target-v0",
    )
    config.target.limb = Limb.LEG

    with pytest.raises(ValueError, match="limb"):
        config.validate()


def test_yaml_round_trip_preserves_roles(tmp_path: Path) -> None:
    path = tmp_path / "experiment.yaml"
    path.write_text(
        """
name: hand_quick
limb: hand
seed: 7
source:
  env_id: myoHandReorient8-v0
  role: source
  limb: hand
target:
  env_id: myoHandReorient100-v0
  role: target
  limb: hand
""".strip(),
        encoding="utf-8",
    )

    config = ExperimentConfig.from_yaml(path)

    assert config.name == "hand_quick"
    assert config.source.env_id == "myoHandReorient8-v0"
    assert config.target.env_id == "myoHandReorient100-v0"
    assert config.seed == 7


def test_rejects_non_twenty_default_alignment() -> None:
    config = ExperimentConfig.minimal(
        limb=Limb.LEG,
        source_env="source-v0",
        target_env="target-v0",
    )
    config.representation.latent_dim = 8

    with pytest.raises(ValueError, match="20 aligned"):
        config.validate()

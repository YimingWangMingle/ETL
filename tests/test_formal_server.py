from pathlib import Path

from etl_sar.config import ExperimentConfig
from etl_sar.formal.config import FormalDomainConfig
from etl_sar.formal.matrix import ExperimentMatrix
from etl_sar.formal.server import select_source, select_target, write_experiment_config


ROOT = Path(__file__).resolve().parents[1]


def matrix() -> ExperimentMatrix:
    return ExperimentMatrix.from_configs(
        [
            FormalDomainConfig.from_yaml(ROOT / "configs" / "formal_hand.yaml"),
            FormalDomainConfig.from_yaml(ROOT / "configs" / "formal_leg.yaml"),
        ]
    )


def test_server_array_indices_are_stable_and_bounded() -> None:
    experiment = matrix()
    assert select_source(experiment, 0).stage_id == "hand-source-seed0"
    assert select_source(experiment, 9).stage_id == "leg-source-seed4"
    assert select_target(experiment, 0).job_id == "hand-etl_no_sar-seed0"
    assert select_target(experiment, 29).job_id == "leg-lattice-seed4"


def test_server_writes_seed_specific_existing_etl_config(tmp_path) -> None:
    source = select_source(matrix(), 7)
    path = write_experiment_config(source, tmp_path / "experiment.yaml")
    config = ExperimentConfig.from_yaml(path)
    assert config.seed == 2
    assert config.source.env_id == "myoLegWalk-v0"
    assert config.target.env_id == "myoLegRoughTerrainWalk-v0"

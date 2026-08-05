from __future__ import annotations

import argparse
from pathlib import Path

from etl_sar.formal.aggregate import write_aggregate
from etl_sar.formal.config import FormalDomainConfig
from etl_sar.formal.matrix import ExperimentMatrix


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate formal ETL/Lattice results")
    parser.add_argument("--output-root", type=Path, default=Path("runs/formal"))
    parser.add_argument("--hand-config", type=Path, default=Path("configs/formal_hand.yaml"))
    parser.add_argument("--leg-config", type=Path, default=Path("configs/formal_leg.yaml"))
    args = parser.parse_args()
    matrix = ExperimentMatrix.from_configs(
        [
            FormalDomainConfig.from_yaml(args.hand_config),
            FormalDomainConfig.from_yaml(args.leg_config),
        ]
    )
    print(write_aggregate(matrix, args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

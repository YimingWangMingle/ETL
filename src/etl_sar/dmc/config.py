from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DMCTransferConfig:
    domain: str
    source_task: str
    target_task: str
    seed: int
    source_budget: int
    target_budget: int
    lattice_budget: int
    action_repeat: int
    checkpoint_interval: int
    intermediate_episodes: int
    final_episodes: int
    etl_latent_dim: int
    etl_components: int
    sar_components: int
    sar_rho: float
    source_top_fraction: float
    representation_steps: int
    sar_steps: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DMCTransferConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("DMC YAML must contain a mapping")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "DMCTransferConfig":
        required = set(cls.__dataclass_fields__)
        missing = required - raw.keys()
        extra = raw.keys() - required
        if missing or extra:
            raise ValueError(
                f"DMC config keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        config = cls(
            domain=str(raw["domain"]),
            source_task=str(raw["source_task"]),
            target_task=str(raw["target_task"]),
            seed=int(raw["seed"]),
            source_budget=int(raw["source_budget"]),
            target_budget=int(raw["target_budget"]),
            lattice_budget=int(raw["lattice_budget"]),
            action_repeat=int(raw["action_repeat"]),
            checkpoint_interval=int(raw["checkpoint_interval"]),
            intermediate_episodes=int(raw["intermediate_episodes"]),
            final_episodes=int(raw["final_episodes"]),
            etl_latent_dim=int(raw["etl_latent_dim"]),
            etl_components=int(raw["etl_components"]),
            sar_components=int(raw["sar_components"]),
            sar_rho=float(raw["sar_rho"]),
            source_top_fraction=float(raw["source_top_fraction"]),
            representation_steps=int(raw["representation_steps"]),
            sar_steps=int(raw["sar_steps"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.domain not in {"humanoid", "dog"}:
            raise ValueError("DMC domain must be humanoid or dog")
        if (self.source_task, self.target_task) != ("walk", "run"):
            raise ValueError("DMC pilot requires walk-to-run transfer")
        if self.seed < 0:
            raise ValueError("DMC seed must be nonnegative")
        if self.source_budget + self.target_budget != self.lattice_budget:
            raise ValueError("DMC comparison requires equal total interactions")
        positive = (
            self.source_budget,
            self.target_budget,
            self.lattice_budget,
            self.action_repeat,
            self.checkpoint_interval,
            self.intermediate_episodes,
            self.final_episodes,
            self.etl_latent_dim,
            self.etl_components,
            self.sar_components,
            self.representation_steps,
            self.sar_steps,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("DMC numeric settings must be positive")
        if self.target_budget % self.checkpoint_interval:
            raise ValueError("ETL target budget must end on a checkpoint")
        if self.lattice_budget % self.checkpoint_interval:
            raise ValueError("Lattice budget must end on a checkpoint")
        if not 0.0 <= self.sar_rho <= 1.0:
            raise ValueError("SAR rho must be in [0, 1]")
        if not 0.0 < self.source_top_fraction <= 1.0:
            raise ValueError("source top fraction must be in (0, 1]")

    @property
    def total_budget(self) -> int:
        return self.lattice_budget

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

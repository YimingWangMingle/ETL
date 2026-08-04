from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from etl_sar.types import Limb, TaskRole


@dataclass
class TaskConfig:
    env_id: str
    role: TaskRole
    limb: Limb


@dataclass
class RepresentationConfig:
    latent_dim: int = 20
    mixture_components: int = 20


@dataclass
class SynergyConfig:
    components: int = 20
    rho: float = 0.20
    enabled_scale: float = 1.0


@dataclass
class ExperimentConfig:
    name: str
    limb: Limb
    source: TaskConfig
    target: TaskConfig
    seed: int = 0
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    synergy: SynergyConfig = field(default_factory=SynergyConfig)

    @classmethod
    def minimal(
        cls,
        *,
        limb: Limb,
        source_env: str,
        target_env: str,
    ) -> "ExperimentConfig":
        return cls(
            name=f"{limb.value}_quick",
            limb=limb,
            source=TaskConfig(source_env, TaskRole.SOURCE, limb),
            target=TaskConfig(target_env, TaskRole.TARGET, limb),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("experiment YAML must contain a mapping")
        config = cls._from_mapping(raw)
        config.validate()
        return config

    @classmethod
    def _from_mapping(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        limb = Limb(raw["limb"])

        def task(name: str) -> TaskConfig:
            value = raw[name]
            return TaskConfig(
                env_id=str(value["env_id"]),
                role=TaskRole(value["role"]),
                limb=Limb(value["limb"]),
            )

        representation = RepresentationConfig(**raw.get("representation", {}))
        synergy = SynergyConfig(**raw.get("synergy", {}))
        return cls(
            name=str(raw["name"]),
            limb=limb,
            source=task("source"),
            target=task("target"),
            seed=int(raw.get("seed", 0)),
            representation=representation,
            synergy=synergy,
        )

    def validate(self) -> None:
        if self.source.limb != self.limb or self.target.limb != self.limb:
            raise ValueError("source and target limb must match experiment limb")
        aligned = (
            self.representation.latent_dim,
            self.representation.mixture_components,
            self.synergy.components,
        )
        if aligned != (20, 20, 20):
            raise ValueError("default ETL/SAR protocol requires 20 aligned components")

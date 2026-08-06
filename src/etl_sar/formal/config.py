from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FormalDomainConfig:
    domain: str
    source_env: str
    target_env: str
    action_dim: int
    source_budget: int
    target_budget: int
    lattice_budget: int
    seeds: tuple[int, ...]
    num_envs: int
    checkpoint_interval: int
    intermediate_episodes: int
    final_episodes: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FormalDomainConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
        if not isinstance(raw, dict):
            raise ValueError("formal YAML must contain a mapping")
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "FormalDomainConfig":
        required = {
            "domain",
            "source_env",
            "target_env",
            "action_dim",
            "source_budget",
            "target_budget",
            "lattice_budget",
            "seeds",
            "num_envs",
            "checkpoint_interval",
            "intermediate_episodes",
            "final_episodes",
        }
        missing = required - raw.keys()
        extra = raw.keys() - required
        if missing or extra:
            raise ValueError(
                f"formal config keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        config = cls(
            domain=str(raw["domain"]),
            source_env=str(raw["source_env"]),
            target_env=str(raw["target_env"]),
            action_dim=int(raw["action_dim"]),
            source_budget=int(raw["source_budget"]),
            target_budget=int(raw["target_budget"]),
            lattice_budget=int(raw["lattice_budget"]),
            seeds=tuple(int(seed) for seed in raw["seeds"]),
            num_envs=int(raw["num_envs"]),
            checkpoint_interval=int(raw["checkpoint_interval"]),
            intermediate_episodes=int(raw["intermediate_episodes"]),
            final_episodes=int(raw["final_episodes"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.domain not in {"hand", "leg"}:
            raise ValueError("formal domain must be hand or leg")
        if (
            not self.seeds
            or len(self.seeds) != len(set(self.seeds))
            or any(seed < 0 for seed in self.seeds)
        ):
            raise ValueError("formal seeds must be nonempty, unique, and nonnegative")
        if self.source_budget + self.target_budget != self.lattice_budget:
            raise ValueError("formal comparison requires equal total interactions")
        positive = (
            self.action_dim,
            self.source_budget,
            self.target_budget,
            self.lattice_budget,
            self.num_envs,
            self.checkpoint_interval,
            self.intermediate_episodes,
            self.final_episodes,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("formal numeric settings must be positive")
        self.vector_iterations(self.target_budget, self.num_envs)
        self.vector_iterations(self.lattice_budget, self.num_envs)

    def vector_iterations(self, transitions: int, num_envs: int | None = None) -> int:
        parallel = self.num_envs if num_envs is None else num_envs
        if parallel <= 0 or transitions % parallel:
            raise ValueError("transition budget must be divisible by num_envs")
        return transitions // parallel

    def to_mapping(self) -> dict[str, Any]:
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        return result

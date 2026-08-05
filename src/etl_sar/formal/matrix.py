from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable

from etl_sar.formal.config import FormalDomainConfig


class Method(str, Enum):
    ETL_NO_SAR = "etl_no_sar"
    ETL_SAR = "etl_sar"
    LATTICE = "lattice"


@dataclass(frozen=True)
class SourceStage:
    stage_id: str
    domain: str
    seed: int
    environment_id: str
    transitions: int


@dataclass(frozen=True)
class TargetJob:
    job_id: str
    domain: str
    method: Method
    seed: int
    environment_id: str
    source_stage_id: str | None
    source_transitions: int
    target_transitions: int
    total_transitions: int
    checkpoint_interval: int
    intermediate_episodes: int
    final_episodes: int
    num_envs: int

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["method"] = self.method.value
        return payload


@dataclass(frozen=True)
class ExperimentMatrix:
    configs: tuple[FormalDomainConfig, ...]
    source_stages: tuple[SourceStage, ...]
    target_jobs: tuple[TargetJob, ...]

    @classmethod
    def from_configs(
        cls, configs: Iterable[FormalDomainConfig]
    ) -> "ExperimentMatrix":
        normalized = tuple(configs)
        if {config.domain for config in normalized} != {"hand", "leg"}:
            raise ValueError("formal matrix requires one hand and one leg config")
        sources: list[SourceStage] = []
        targets: list[TargetJob] = []
        for config in sorted(normalized, key=lambda item: item.domain):
            for seed in config.seeds:
                source_id = f"{config.domain}-source-seed{seed}"
                sources.append(
                    SourceStage(
                        stage_id=source_id,
                        domain=config.domain,
                        seed=seed,
                        environment_id=config.source_env,
                        transitions=config.source_budget,
                    )
                )
                for method in Method:
                    is_lattice = method is Method.LATTICE
                    source_transitions = 0 if is_lattice else config.source_budget
                    target_transitions = (
                        config.lattice_budget if is_lattice else config.target_budget
                    )
                    targets.append(
                        TargetJob(
                            job_id=f"{config.domain}-{method.value}-seed{seed}",
                            domain=config.domain,
                            method=method,
                            seed=seed,
                            environment_id=config.target_env,
                            source_stage_id=None if is_lattice else source_id,
                            source_transitions=source_transitions,
                            target_transitions=target_transitions,
                            total_transitions=source_transitions + target_transitions,
                            checkpoint_interval=config.checkpoint_interval,
                            intermediate_episodes=config.intermediate_episodes,
                            final_episodes=config.final_episodes,
                            num_envs=config.num_envs,
                        )
                    )
        return cls(normalized, tuple(sources), tuple(targets))

    @property
    def total_declared_interactions(self) -> int:
        return sum(job.total_transitions for job in self.target_jobs)

    @staticmethod
    def seed_bank(*, domain: str, seed: int, episodes: int) -> tuple[int, ...]:
        domain_offset = 0 if domain == "hand" else 1_000_000
        start = 10_000_000 + domain_offset + seed * 10_000
        return tuple(range(start, start + episodes))

    @staticmethod
    def training_seed_bank(*, domain: str, seed: int, count: int) -> tuple[int, ...]:
        domain_offset = 0 if domain == "hand" else 1_000_000
        start = domain_offset + seed * 10_000
        return tuple(range(start, start + count))

    def to_mapping(self) -> dict[str, Any]:
        return {
            "source_stages": [asdict(stage) for stage in self.source_stages],
            "target_jobs": [job.to_mapping() for job in self.target_jobs],
            "summary": {
                "source_stage_count": len(self.source_stages),
                "target_job_count": len(self.target_jobs),
                "total_declared_interactions": self.total_declared_interactions,
            },
        }

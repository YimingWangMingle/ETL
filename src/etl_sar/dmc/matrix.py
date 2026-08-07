from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Iterable

from etl_sar.dmc.config import DMCTransferConfig


class DMCMethod(str, Enum):
    ETL_NO_SAR = "etl_no_sar"
    ETL_SAR = "etl_sar"
    LATTICE = "lattice"


@dataclass(frozen=True)
class DMCSourceStage:
    stage_id: str
    domain: str
    task: str
    seed: int
    transitions: int


@dataclass(frozen=True)
class DMCJob:
    job_id: str
    domain: str
    task: str
    method: DMCMethod
    seed: int
    source_stage_id: str | None
    source_budget: int
    target_budget: int
    total_budget: int

    def to_mapping(self) -> dict[str, Any]:
        result = asdict(self)
        result["method"] = self.method.value
        return result


@dataclass(frozen=True)
class DMCExperimentMatrix:
    configs: tuple[DMCTransferConfig, ...]
    sources: tuple[DMCSourceStage, ...]
    jobs: tuple[DMCJob, ...]

    @classmethod
    def from_configs(
        cls, configs: Iterable[DMCTransferConfig]
    ) -> "DMCExperimentMatrix":
        normalized = tuple(configs)
        if {config.domain for config in normalized} != {"humanoid", "dog"}:
            raise ValueError("DMC matrix requires one humanoid and one dog config")
        sources: list[DMCSourceStage] = []
        jobs: list[DMCJob] = []
        for config in sorted(normalized, key=lambda item: item.domain):
            source_id = f"{config.domain}-source-seed{config.seed}"
            sources.append(
                DMCSourceStage(
                    stage_id=source_id,
                    domain=config.domain,
                    task=config.source_task,
                    seed=config.seed,
                    transitions=config.source_budget,
                )
            )
            for method in DMCMethod:
                is_lattice = method is DMCMethod.LATTICE
                source_budget = 0 if is_lattice else config.source_budget
                target_budget = (
                    config.lattice_budget if is_lattice else config.target_budget
                )
                jobs.append(
                    DMCJob(
                        job_id=f"{config.domain}-{method.value}-seed{config.seed}",
                        domain=config.domain,
                        task=config.target_task,
                        method=method,
                        seed=config.seed,
                        source_stage_id=None if is_lattice else source_id,
                        source_budget=source_budget,
                        target_budget=target_budget,
                        total_budget=source_budget + target_budget,
                    )
                )
        return cls(normalized, tuple(sources), tuple(jobs))

    @property
    def total_declared_interactions(self) -> int:
        return sum(job.total_budget for job in self.jobs)

    def config_for(self, domain: str) -> DMCTransferConfig:
        return next(config for config in self.configs if config.domain == domain)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "sources": [asdict(source) for source in self.sources],
            "jobs": [job.to_mapping() for job in self.jobs],
            "summary": {
                "source_stage_count": len(self.sources),
                "target_job_count": len(self.jobs),
                "total_declared_interactions": self.total_declared_interactions,
            },
        }

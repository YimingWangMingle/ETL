from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.typing import NDArray

from etl_sar.types import Limb


FloatArray = NDArray[np.floating]


@dataclass(frozen=True)
class TrajectoryEpisode:
    limb: Limb
    source_task: str
    observations: FloatArray
    sampled_actions: FloatArray
    executed_actions: FloatArray
    rewards: FloatArray
    next_observations: FloatArray
    terminated: NDArray[np.bool_]
    truncated: NDArray[np.bool_]
    behaviors: FloatArray
    success: bool

    @property
    def length(self) -> int:
        return int(self.executed_actions.shape[0])

    def validate(self) -> None:
        lengths = {
            self.observations.shape[0],
            self.sampled_actions.shape[0],
            self.executed_actions.shape[0],
            self.rewards.shape[0],
            self.next_observations.shape[0],
            self.terminated.shape[0],
            self.truncated.shape[0],
            self.behaviors.shape[0],
        }
        if len(lengths) != 1:
            raise ValueError("all trajectory arrays must have the same leading dimension")
        if self.executed_actions.ndim != 2:
            raise ValueError("executed actions must have shape [time, action_dim]")


class TrajectoryStore:
    def __init__(
        self,
        root: str | Path,
        *,
        limb: Limb,
        source_task: str,
        action_dim: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.limb = limb
        self.source_task = source_task
        self.action_dim = action_dim

    def append_episode(self, episode: TrajectoryEpisode) -> Path:
        episode.validate()
        if episode.limb != self.limb:
            raise ValueError(
                f"episode limb {episode.limb.value!r} does not match store limb {self.limb.value!r}"
            )
        if episode.source_task != self.source_task:
            raise ValueError(
                f"episode source task {episode.source_task!r} does not match {self.source_task!r}"
            )
        episode_action_dim = int(episode.executed_actions.shape[1])
        if self.action_dim is None:
            self.action_dim = episode_action_dim
        elif episode_action_dim != self.action_dim:
            raise ValueError(
                f"episode action dimension {episode_action_dim} does not match {self.action_dim}"
            )
        path = self.root / f"episode_{len(self._paths()):06d}.npz"
        np.savez_compressed(
            path,
            limb=np.asarray(episode.limb.value),
            source_task=np.asarray(episode.source_task),
            observations=episode.observations,
            sampled_actions=episode.sampled_actions,
            executed_actions=episode.executed_actions,
            rewards=episode.rewards,
            next_observations=episode.next_observations,
            terminated=episode.terminated,
            truncated=episode.truncated,
            behaviors=episode.behaviors,
            success=np.asarray(episode.success, dtype=np.bool_),
        )
        return path

    def iter_episodes(self) -> Iterator[TrajectoryEpisode]:
        for path in self._paths():
            with np.load(path, allow_pickle=False) as data:
                yield TrajectoryEpisode(
                    limb=Limb(str(data["limb"].item())),
                    source_task=str(data["source_task"].item()),
                    observations=data["observations"],
                    sampled_actions=data["sampled_actions"],
                    executed_actions=data["executed_actions"],
                    rewards=data["rewards"],
                    next_observations=data["next_observations"],
                    terminated=data["terminated"],
                    truncated=data["truncated"],
                    behaviors=data["behaviors"],
                    success=bool(data["success"].item()),
                )

    def action_pool(self) -> FloatArray:
        return self._pool(success_only=False)

    def success_pool(self) -> FloatArray:
        return self._pool(success_only=True)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.limb.value.encode("utf-8"))
        digest.update(self.source_task.encode("utf-8"))
        for episode in self.iter_episodes():
            digest.update(str(episode.success).encode("ascii"))
            for array in (
                episode.observations,
                episode.sampled_actions,
                episode.executed_actions,
                episode.rewards,
                episode.next_observations,
                episode.terminated,
                episode.truncated,
                episode.behaviors,
            ):
                contiguous = np.ascontiguousarray(array)
                digest.update(str(contiguous.dtype).encode("ascii"))
                digest.update(str(contiguous.shape).encode("ascii"))
                digest.update(contiguous.tobytes())
        return digest.hexdigest()

    def _pool(self, *, success_only: bool) -> FloatArray:
        arrays = [
            episode.executed_actions
            for episode in self.iter_episodes()
            if not success_only or episode.success
        ]
        if arrays:
            return np.concatenate(arrays, axis=0)
        return np.empty((0, self.action_dim or 0), dtype=np.float32)

    def _paths(self) -> list[Path]:
        return sorted(self.root.glob("episode_*.npz"))

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class EpisodeMetrics:
    episode: int
    seed: int
    episode_return: float
    length: int
    success: bool
    terminated: bool
    fall: bool
    object_drop: bool
    forward_distance: float | None
    velocity_tracking_error: float | None
    muscle_effort: float

    def to_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["return"] = payload.pop("episode_return")
        return payload


def resolve_root_x_qpos_address(model: Any) -> int:
    """Return the x qpos address of the model's root free joint."""
    candidates: list[tuple[str, int]] = []
    for joint_id in range(int(model.njnt)):
        if int(model.jnt_type[joint_id]) != 0:  # mjJNT_FREE
            continue
        joint = model.joint(joint_id)
        name = str(getattr(joint, "name", "") or "")
        candidates.append((name.lower(), int(model.jnt_qposadr[joint_id])))
    if not candidates:
        raise ValueError("MuJoCo model has no free joint for root translation")
    for name, address in candidates:
        if "root" in name or "pelvis" in name:
            return address
    if len(candidates) == 1:
        return candidates[0][1]
    raise ValueError("MuJoCo model has multiple free joints and no named root")


def normalized_auc(
    *,
    transitions: Iterable[int],
    values: Iterable[float],
    total_budget: int,
) -> float:
    x = np.asarray(tuple(transitions), dtype=np.float64)
    y = np.asarray(tuple(values), dtype=np.float64)
    if total_budget <= 0:
        raise ValueError("total budget must be positive")
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("AUC requires equal one-dimensional curves with at least two points")
    if np.any(np.diff(x) <= 0):
        raise ValueError("AUC transitions must be strictly increasing")
    if x[0] < 0 or x[-1] > total_budget:
        raise ValueError("AUC transitions must lie inside the declared budget")
    return float(np.trapezoid(y, x=x / float(total_budget)))

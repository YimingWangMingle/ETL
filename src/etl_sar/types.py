from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Limb(str, Enum):
    HAND = "hand"
    LEG = "leg"


class TaskRole(str, Enum):
    SOURCE = "source"
    TARGET = "target"
    EVALUATION = "evaluation"


@dataclass(frozen=True)
class TaskMetadata:
    env_id: str
    role: TaskRole
    limb: Limb

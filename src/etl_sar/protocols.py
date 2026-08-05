from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def task_succeeded(info: Mapping[str, Any]) -> bool:
    return bool(info.get("success", info.get("solved", False)))

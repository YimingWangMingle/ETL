from __future__ import annotations

import numpy as np
import pytest

from etl_sar.data import TrajectoryEpisode, TrajectoryStore
from etl_sar.types import Limb


def make_episode(
    *,
    limb: Limb = Limb.HAND,
    success: bool = False,
    value: float = 0.1,
    length: int = 2,
) -> TrajectoryEpisode:
    return TrajectoryEpisode(
        limb=limb,
        source_task="reorient8" if limb is Limb.HAND else "flat_walk",
        observations=np.full((length, 5), value, dtype=np.float32),
        sampled_actions=np.full((length, 3), value, dtype=np.float32),
        executed_actions=np.full((length, 3), value, dtype=np.float32),
        rewards=np.full(length, value, dtype=np.float32),
        next_observations=np.full((length, 5), value + 1, dtype=np.float32),
        terminated=np.zeros(length, dtype=np.bool_),
        truncated=np.zeros(length, dtype=np.bool_),
        behaviors=np.full((length, 2), value, dtype=np.float32),
        success=success,
    )


def test_action_and_success_views_are_distinct(tmp_path) -> None:
    store = TrajectoryStore(tmp_path, limb=Limb.HAND, source_task="reorient8")
    store.append_episode(make_episode(success=False, value=0.1))
    store.append_episode(make_episode(success=True, value=0.8))

    assert store.action_pool().shape == (4, 3)
    assert store.success_pool().shape == (2, 3)
    np.testing.assert_allclose(store.success_pool(), 0.8)


def test_store_rejects_wrong_limb(tmp_path) -> None:
    store = TrajectoryStore(tmp_path, limb=Limb.HAND, source_task="reorient8")

    with pytest.raises(ValueError, match="limb"):
        store.append_episode(make_episode(limb=Limb.LEG))


def test_store_rejects_wrong_source_task(tmp_path) -> None:
    store = TrajectoryStore(tmp_path, limb=Limb.HAND, source_task="another_task")

    with pytest.raises(ValueError, match="source task"):
        store.append_episode(make_episode())


def test_fingerprint_is_stable_for_same_episodes(tmp_path) -> None:
    first = TrajectoryStore(tmp_path / "first", limb=Limb.HAND, source_task="reorient8")
    second = TrajectoryStore(tmp_path / "second", limb=Limb.HAND, source_task="reorient8")
    for store in (first, second):
        store.append_episode(make_episode(success=True, value=0.4))

    assert first.fingerprint() == second.fingerprint()


def test_empty_pool_has_known_action_dimension(tmp_path) -> None:
    store = TrajectoryStore(
        tmp_path,
        limb=Limb.HAND,
        source_task="reorient8",
        action_dim=39,
    )

    assert store.action_pool().shape == (0, 39)
    assert store.success_pool().shape == (0, 39)

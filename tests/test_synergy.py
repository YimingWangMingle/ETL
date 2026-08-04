from __future__ import annotations

import numpy as np
import pytest

from etl_sar.synergy import SynergyArtifact
from etl_sar.types import Limb


def structured_actions(samples: int = 160, action_dim: int = 39) -> np.ndarray:
    rng = np.random.default_rng(5)
    latent = rng.normal(size=(samples, 20))
    mixing = rng.normal(size=(20, action_dim))
    return np.tanh(latent @ mixing + 0.01 * rng.normal(size=(samples, action_dim))).astype(
        np.float32
    )


def test_icapca_uses_twenty_components_and_bounded_codes() -> None:
    actions = structured_actions()
    artifact = SynergyArtifact.fit(
        actions,
        components=20,
        limb=Limb.HAND,
        source_task="reorient8",
        data_fingerprint="abc123",
    )

    codes = artifact.transform(actions[:4])

    assert codes.shape == (4, 20)
    assert np.max(np.abs(codes)) <= 1.0 + 1e-6
    assert artifact.basis.shape == (39, 20)
    assert artifact.explained_variance_ratio.shape == (20,)


def test_icapca_round_trip_is_finite() -> None:
    actions = structured_actions()
    artifact = SynergyArtifact.fit(
        actions,
        components=20,
        limb=Limb.HAND,
        source_task="reorient8",
        data_fingerprint="abc123",
    )

    reconstructed = artifact.inverse_transform(artifact.transform(actions[:8]))

    assert reconstructed.shape == (8, 39)
    assert np.isfinite(reconstructed).all()


def test_artifact_round_trip_rejects_wrong_limb(tmp_path) -> None:
    artifact = SynergyArtifact.fit(
        structured_actions(),
        components=20,
        limb=Limb.HAND,
        source_task="reorient8",
        data_fingerprint="abc123",
    )
    path = tmp_path / "hand_synergy.pkl"
    artifact.save(path)

    loaded = SynergyArtifact.load(path, expected_limb=Limb.HAND)
    assert loaded.data_fingerprint == "abc123"
    with pytest.raises(ValueError, match="limb"):
        SynergyArtifact.load(path, expected_limb=Limb.LEG)


def test_fit_requires_enough_samples_and_dimensions() -> None:
    with pytest.raises(ValueError, match="20 components"):
        SynergyArtifact.fit(
            np.zeros((10, 8), dtype=np.float32),
            components=20,
            limb=Limb.LEG,
            source_task="flat_walk",
            data_fingerprint="small",
        )

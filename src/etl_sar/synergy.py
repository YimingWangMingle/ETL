from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import FastICA, PCA
from sklearn.preprocessing import StandardScaler

from etl_sar.types import Limb


@dataclass
class SynergyArtifact:
    scaler: StandardScaler
    pca: PCA
    ica: FastICA
    code_scale: NDArray[np.float64]
    basis: NDArray[np.float32]
    limb: Limb
    source_task: str
    data_fingerprint: str
    components: int

    @classmethod
    def fit(
        cls,
        actions: NDArray[np.floating],
        *,
        components: int,
        limb: Limb,
        source_task: str,
        data_fingerprint: str,
        random_state: int = 0,
    ) -> "SynergyArtifact":
        if actions.ndim != 2:
            raise ValueError("actions must have shape [samples, action_dim]")
        if actions.shape[0] < components or actions.shape[1] < components:
            raise ValueError(
                f"20 components require at least 20 samples and 20 action dimensions; "
                f"received {actions.shape}"
            )
        scaler = StandardScaler().fit(actions)
        standardized = scaler.transform(actions)
        pca = PCA(n_components=components, random_state=random_state).fit(standardized)
        pca_codes = pca.transform(standardized)
        ica = FastICA(
            n_components=components,
            whiten="unit-variance",
            random_state=random_state,
            max_iter=2000,
            tol=1e-4,
        ).fit(pca_codes)
        raw_codes = ica.transform(pca_codes)
        code_scale = np.maximum(np.max(np.abs(raw_codes), axis=0), 1e-8)

        def inverse(codes: NDArray[np.floating]) -> NDArray[np.float64]:
            raw = np.asarray(codes, dtype=np.float64) * code_scale
            pca_values = ica.inverse_transform(raw)
            standardized_values = pca.inverse_transform(pca_values)
            return scaler.inverse_transform(standardized_values)

        origin = inverse(np.zeros((1, components), dtype=np.float64))[0]
        basis = (inverse(np.eye(components, dtype=np.float64)) - origin).T.astype(np.float32)
        return cls(
            scaler=scaler,
            pca=pca,
            ica=ica,
            code_scale=code_scale,
            basis=basis,
            limb=limb,
            source_task=source_task,
            data_fingerprint=data_fingerprint,
            components=components,
        )

    @property
    def explained_variance_ratio(self) -> NDArray[np.float64]:
        return self.pca.explained_variance_ratio_

    def transform(self, actions: NDArray[np.floating]) -> NDArray[np.float32]:
        standardized = self.scaler.transform(actions)
        raw_codes = self.ica.transform(self.pca.transform(standardized))
        normalized = np.clip(raw_codes / self.code_scale, -1.0, 1.0)
        return normalized.astype(np.float32)

    def inverse_transform(self, codes: NDArray[np.floating]) -> NDArray[np.float32]:
        bounded = np.clip(np.asarray(codes), -1.0, 1.0)
        raw_codes = bounded * self.code_scale
        pca_values = self.ica.inverse_transform(raw_codes)
        standardized = self.pca.inverse_transform(pca_values)
        return self.scaler.inverse_transform(standardized).astype(np.float32)

    def projection(self) -> NDArray[np.float32]:
        return (self.basis @ np.linalg.pinv(self.basis)).astype(np.float32)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_limb: Limb,
        expected_source_task: str | None = None,
    ) -> "SynergyArtifact":
        artifact = joblib.load(Path(path))
        if not isinstance(artifact, cls):
            raise ValueError("synergy artifact has an incompatible type")
        if artifact.limb != expected_limb:
            raise ValueError(
                f"synergy limb {artifact.limb.value!r} does not match {expected_limb.value!r}"
            )
        if expected_source_task is not None and artifact.source_task != expected_source_task:
            raise ValueError("synergy artifact source task does not match the experiment")
        return artifact

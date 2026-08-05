from __future__ import annotations

import json

import pytest

from etl_sar.formal.manifest import RunManifest, finalize_manifest, verify_manifest


def make_manifest() -> RunManifest:
    return RunManifest(
        schema_version=1,
        job_id="hand-etl_sar-seed0",
        method="etl_sar",
        domain="hand",
        seed=0,
        source_transitions=1_000_000,
        target_transitions=19_000_000,
        git_commit="abc123",
        lattice_commit="846d02fa993b9b80ce5ecb806463e0a05711bad3",
        environment_id="myoHandReorient100-v0",
        environment_fingerprint="envhash",
        command=["etl-sar", "formal-run-job"],
        python_version="3.11",
        packages={"myosuite": "2.12.0"},
        hardware={"cpu": "test"},
    )


def test_manifest_finalizes_only_after_all_artifacts_hash(tmp_path) -> None:
    (tmp_path / "policy.zip").write_bytes(b"policy")
    (tmp_path / "actions.pt").write_bytes(b"actions")
    path = finalize_manifest(
        make_manifest(),
        tmp_path / "manifest.json",
        artifacts=[tmp_path / "policy.zip", tmp_path / "actions.pt"],
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "complete"
    assert set(payload["artifact_sha256"]) == {"policy.zip", "actions.pt"}
    verify_manifest(path)


def test_manifest_rejects_modified_artifact(tmp_path) -> None:
    artifact = tmp_path / "policy.zip"
    artifact.write_bytes(b"policy")
    path = finalize_manifest(
        make_manifest(), tmp_path / "manifest.json", artifacts=[artifact]
    )
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact hash"):
        verify_manifest(path)


def test_manifest_does_not_finalize_with_missing_artifact(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing artifact"):
        finalize_manifest(
            make_manifest(),
            tmp_path / "manifest.json",
            artifacts=[tmp_path / "missing.zip"],
        )
    assert not (tmp_path / "manifest.json").exists()

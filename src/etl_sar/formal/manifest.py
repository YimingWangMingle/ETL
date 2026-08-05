from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from etl_sar.checkpoints import sha256_file


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    job_id: str
    method: str
    domain: str
    seed: int
    source_transitions: int
    target_transitions: int
    git_commit: str
    lattice_commit: str
    environment_id: str
    environment_fingerprint: str
    command: list[str]
    python_version: str
    packages: dict[str, str]
    hardware: dict[str, Any]
    status: str = "running"
    artifact_sha256: dict[str, str] = field(default_factory=dict)
    resource_usage: dict[str, float] = field(default_factory=dict)


def finalize_manifest(
    manifest: RunManifest,
    path: str | Path,
    *,
    artifacts: list[str | Path],
    resource_usage: dict[str, float] | None = None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for raw_path in artifacts:
        artifact = Path(raw_path)
        if not artifact.is_file():
            raise ValueError(f"missing artifact: {artifact}")
        try:
            name = str(artifact.resolve().relative_to(destination.parent.resolve()))
        except ValueError as error:
            raise ValueError("manifest artifacts must be inside the run directory") from error
        hashes[name.replace("\\", "/")] = sha256_file(artifact)
    complete = replace(
        manifest,
        status="complete",
        artifact_sha256=hashes,
        resource_usage=dict(resource_usage or manifest.resource_usage),
    )
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(asdict(complete), indent=2), encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def verify_manifest(path: str | Path) -> RunManifest:
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = RunManifest(**payload)
    if manifest.schema_version != 1:
        raise ValueError(f"unsupported run manifest schema {manifest.schema_version}")
    if manifest.status != "complete":
        raise ValueError("run manifest is not complete")
    for relative, expected in manifest.artifact_sha256.items():
        artifact = manifest_path.parent / relative
        if not artifact.is_file():
            raise ValueError(f"manifest artifact is missing: {relative}")
        if sha256_file(artifact) != expected:
            raise ValueError(f"artifact hash does not match manifest: {relative}")
    return manifest

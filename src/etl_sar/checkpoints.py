from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from etl_sar.action_model import ETLSARActionModel


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_transition_count(directory: str | Path, transitions: int) -> Path:
    if transitions < 0:
        raise ValueError("transition count must be nonnegative")
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "transition_count.json"
    temporary = output / f".transition_count.{uuid.uuid4().hex}.tmp"
    temporary.write_text(
        json.dumps({"transitions": int(transitions)}, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination


@dataclass(frozen=True)
class CheckpointPair:
    schema_version: int
    pair_id: str
    transitions: int
    policy_file: str
    policy_sha256: str
    action_model_file: str
    action_model_sha256: str

    @classmethod
    def from_path(cls, path: str | Path) -> "CheckpointPair":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**payload)


def _safe_member(manifest_path: Path, filename: str) -> Path:
    if Path(filename).name != filename:
        raise ValueError("checkpoint manifest contains an unsafe filename")
    return manifest_path.parent / filename


def _verify(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"checkpoint {label} file is missing: {path.name}")
    if sha256_file(path) != expected:
        raise ValueError(f"checkpoint {label} hash does not match manifest")


def save_checkpoint_pair(
    *,
    policy: Any,
    action_model: ETLSARActionModel,
    directory: str | Path,
    name: str,
    transitions: int,
) -> Path:
    if not name or Path(name).name != name:
        raise ValueError("checkpoint name must be a single path component")
    if transitions < 0:
        raise ValueError("checkpoint transitions must be nonnegative")

    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    pair_id = uuid.uuid4().hex
    policy_name = f"{name}_policy.zip"
    action_name = f"{name}_action_model.pt"
    manifest_name = f"{name}_pair.json"

    token = uuid.uuid4().hex
    temporary_policy_base = output / f".{name}_{token}_policy"
    temporary_policy = temporary_policy_base.with_suffix(".zip")
    temporary_action = output / f".{name}_{token}_action_model.pt"
    temporary_manifest = output / f".{name}_{token}_pair.json"
    policy_path = output / policy_name
    action_path = output / action_name
    manifest_path = output / manifest_name

    try:
        policy.save(temporary_policy_base)
        torch.save(
            {
                "schema_version": 1,
                "pair_id": pair_id,
                "transitions": int(transitions),
                "enabled_scale": float(action_model.enabled_scale),
                "rho": float(action_model.rho),
                "state_dict": action_model.state_dict(),
            },
            temporary_action,
        )
        os.replace(temporary_policy, policy_path)
        os.replace(temporary_action, action_path)
        manifest = CheckpointPair(
            schema_version=1,
            pair_id=pair_id,
            transitions=int(transitions),
            policy_file=policy_name,
            policy_sha256=sha256_file(policy_path),
            action_model_file=action_name,
            action_model_sha256=sha256_file(action_path),
        )
        temporary_manifest.write_text(
            json.dumps(asdict(manifest), indent=2), encoding="utf-8"
        )
        os.replace(temporary_manifest, manifest_path)
    finally:
        for temporary in (temporary_policy, temporary_action, temporary_manifest):
            temporary.unlink(missing_ok=True)
    return manifest_path


def load_checkpoint_pair(
    manifest_path: str | Path,
    *,
    policy_loader: Callable[..., Any],
    action_model: ETLSARActionModel,
    device: str = "auto",
) -> tuple[Any, CheckpointPair]:
    path = Path(manifest_path)
    manifest = CheckpointPair.from_path(path)
    if manifest.schema_version != 1:
        raise ValueError(f"unsupported checkpoint schema {manifest.schema_version}")
    policy_path = _safe_member(path, manifest.policy_file)
    action_path = _safe_member(path, manifest.action_model_file)
    _verify(policy_path, manifest.policy_sha256, "policy")
    _verify(action_path, manifest.action_model_sha256, "action-model")

    payload = torch.load(action_path, map_location="cpu", weights_only=False)
    if payload.get("pair_id") != manifest.pair_id:
        raise ValueError("checkpoint action-model pair ID does not match manifest")
    if int(payload.get("transitions", -1)) != manifest.transitions:
        raise ValueError("checkpoint transition count does not match manifest")
    action_model.load_state_dict(payload["state_dict"])
    action_model.enabled_scale = float(payload["enabled_scale"])
    action_model.rho = float(payload["rho"])
    action_model.eval()
    return policy_loader(policy_path, device=device), manifest

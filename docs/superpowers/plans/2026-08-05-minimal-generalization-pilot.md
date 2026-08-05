# Minimal Hand and Leg Generalization Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a restartable, equal-budget Hand and Leg pilot that compares ETL-SB3 with SAR disabled against the ETL-dominant SAR extension and reports compatible ETL-Ray references without running full training.

**Architecture:** Keep the existing stage-oriented CLI and add only the controls required by the pilot. Normalize MyoSuite success metadata once, stop source exploration after a minimum budget plus enough successful actions, load one representation bundle with a runtime SAR-scale override, and orchestrate matched baseline/extension runs through a PowerShell script with signed completion markers.

**Tech Stack:** Python 3.11, Gymnasium 0.29-1.0, MyoSuite 2.12.2, Stable-Baselines3 2.x, PyTorch 2.x, Typer, pytest, Windows PowerShell

## Global Constraints

- Run both `myoHandReorient8-v0 -> myoHandReorient100-v0` and `myoLegWalk-v0 -> myoLegRoughTerrainWalk-v0`.
- ETL remains dominant; do not change rewards, BDR, GMVAE, decoder loss, PPO architecture, or the SAR hard cap `rho=0.20`.
- The local baseline and extension share source data and the representation bundle; only runtime SAR scale differs (`0.0` versus `1.0`).
- Use exactly 20,000 target PPO steps and 10 deterministic evaluation episodes per method and domain.
- Compare final `latest_model.zip` checkpoints with identical evaluation seeds.
- Do not fabricate legacy ETL-Ray values or calculate cross-protocol deltas.
- Do not add vectorized subprocess environments, CUDA requirements, hyperparameter sweeps, or full-training defaults.

---

### Task 1: Unify MyoSuite Success Semantics

**Files:**
- Create: `src/etl_sar/protocols.py`
- Modify: `src/etl_sar/exploration.py:139`
- Modify: `src/etl_sar/evaluation.py:54`
- Modify: `tests/test_evaluation.py`
- Test: `tests/test_explore_trainer.py`

**Interfaces:**
- Consumes: `info: Mapping[str, Any]` from a Gymnasium environment step.
- Produces: `task_succeeded(info: Mapping[str, Any]) -> bool`.

- [ ] **Step 1: Write failing evaluation protocol tests**

Add to `tests/test_evaluation.py`:

```python
class SolvedEvalEnv(EvalEnv):
    def step(self, action):
        observation, reward, terminated, truncated, _ = super().step(action)
        return observation, reward, terminated, truncated, {"solved": terminated}


def test_evaluation_accepts_myosuite_solved_flag(tmp_path) -> None:
    summary = evaluate_checkpoint(
        Predictor(),
        SolvedEvalEnv(),
        episodes=2,
        output_dir=tmp_path,
        environment_steps=64,
    )

    assert summary.success_rate == pytest.approx(1.0)
```

- [ ] **Step 2: Run the test and verify the expected failure**

Run:

```powershell
python -m pytest tests/test_evaluation.py::test_evaluation_accepts_myosuite_solved_flag -q
```

Expected: FAIL because `success_rate` is `0.0`.

- [ ] **Step 3: Add the shared protocol function and replace both call sites**

Create `src/etl_sar/protocols.py`:

```python
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def task_succeeded(info: Mapping[str, Any]) -> bool:
    return bool(info.get("success", info.get("solved", False)))
```

In `exploration.py` and `evaluation.py`, import `task_succeeded` and replace the
inline success expressions with `task_succeeded(info)`.

- [ ] **Step 4: Run focused and full tests**

Run:

```powershell
python -m pytest tests/test_evaluation.py tests/test_explore_trainer.py -q
python -m pytest -m "not myo" -q
```

Expected: all selected tests pass; the full non-MyoSuite suite passes.

- [ ] **Step 5: Commit**

```powershell
git add -- src/etl_sar/protocols.py src/etl_sar/exploration.py src/etl_sar/evaluation.py tests/test_evaluation.py tests/test_explore_trainer.py
git commit -m "fix: normalize MyoSuite success metadata"
```

---

### Task 2: Stop Source Exploration After the Success Gate

**Files:**
- Modify: `src/etl_sar/exploration.py`
- Modify: `src/etl_sar/cli.py:100-145`
- Modify: `tests/test_explore_trainer.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Extends `ExploreTrainer.__init__` with `min_timesteps: int = 0` and `min_success_actions: int = 0`.
- Extends `ExplorationArtifacts` with `environment_steps: int` and `successful_actions: int`.
- Adds `InsufficientSourceSuccessError(RuntimeError)`.
- Adds CLI options `--min-timesteps` and `--min-success-actions` to `explore`.

- [ ] **Step 1: Write failing early-stop and exhausted-budget tests**

Add to `tests/test_explore_trainer.py`:

```python
import pytest

from etl_sar.exploration import InsufficientSourceSuccessError


def make_explore_trainer(tmp_path, env_factory, **overrides) -> ExploreTrainer:
    values = {
        "env_factory": env_factory,
        "state_encoder": StateEncoder(4, 20, hidden_dims=(16,)),
        "representation": make_representation(),
        "trajectory_store": TrajectoryStore(
            tmp_path / "data",
            limb=Limb.HAND,
            source_task="reorient8",
            action_dim=6,
        ),
        "limb": Limb.HAND,
        "source_task": "reorient8",
        "run_dir": tmp_path / "run",
        "total_timesteps": 64,
        "n_steps": 16,
        "batch_size": 8,
        "representation_update_interval": 12,
        "seed": 3,
    }
    values.update(overrides)
    return ExploreTrainer(**values)


def test_explore_stops_after_minimum_and_success_gate(tmp_path) -> None:
    trainer = make_explore_trainer(
        tmp_path,
        TinySolvedSourceEnv,
        min_timesteps=16,
        min_success_actions=6,
    )

    artifacts = trainer.run()

    assert artifacts.environment_steps == 16
    assert artifacts.successful_actions >= 6


class NeverSolvedSourceEnv(TinySolvedSourceEnv):
    def step(self, action):
        observation, reward, terminated, truncated, _ = super().step(action)
        return observation, reward, terminated, truncated, {"solved": False}


def test_explore_reports_insufficient_success_at_max_budget(tmp_path) -> None:
    trainer = make_explore_trainer(
        tmp_path,
        NeverSolvedSourceEnv,
        total_timesteps=32,
        min_timesteps=16,
        min_success_actions=6,
    )

    with pytest.raises(
        InsufficientSourceSuccessError,
        match="insufficient_source_success",
    ):
        trainer.run()
```

- [ ] **Step 2: Run both tests and verify they fail for missing controls**

Run:

```powershell
python -m pytest tests/test_explore_trainer.py -q
```

Expected: FAIL because the new constructor arguments and error class do not exist.

- [ ] **Step 3: Implement the gate in `_ExploreCallback` and `ExploreTrainer`**

Add gate fields to `_ExploreCallback`, evaluate the gate only after a completed
episode has been flushed, and return `False` when both conditions hold:

```python
def _success_gate_met(self) -> bool:
    if self.min_success_actions <= 0:
        return False
    if self.num_timesteps < self.min_timesteps:
        return False
    return (
        self.trajectory_store.success_pool().shape[0]
        >= self.min_success_actions
    )
```

After saving source artifacts, raise the explicit error if the maximum budget ends
without enough successful actions:

```python
successful_actions = int(self.trajectory_store.success_pool().shape[0])
if self.min_success_actions > 0 and successful_actions < self.min_success_actions:
    raise InsufficientSourceSuccessError(
        "insufficient_source_success: "
        f"required {self.min_success_actions}, collected {successful_actions} "
        f"within {self.model.num_timesteps} steps"
    )
```

- [ ] **Step 4: Expose and validate CLI options**

Extend `explore`:

```python
min_timesteps: int = typer.Option(0, min=0),
min_success_actions: int = typer.Option(0, min=0),
```

Reject `min_timesteps > timesteps` with `typer.BadParameter`, pass both values to
`ExploreTrainer`, and print a JSON object containing checkpoint,
`environment_steps`, and `successful_actions`.

- [ ] **Step 5: Run focused and full tests**

Run:

```powershell
python -m pytest tests/test_explore_trainer.py tests/test_cli.py -q
python -m pytest -m "not myo" -q
```

Expected: all tests pass and existing fixed-budget callers remain valid.

- [ ] **Step 6: Commit**

```powershell
git add -- src/etl_sar/exploration.py src/etl_sar/cli.py tests/test_explore_trainer.py tests/test_cli.py
git commit -m "feat: stop source exploration after success gate"
```

---

### Task 3: Add Matched SAR Scale and Evaluation Controls

**Files:**
- Modify: `src/etl_sar/cli.py:45-79,228-275`
- Modify: `src/etl_sar/trainers.py:145-190`
- Modify: `src/etl_sar/evaluation.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_trainers.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Extends `_load_bundle(..., sar_scale: float | None = None)`.
- Adds CLI option `--sar-scale` to `transfer` and `evaluate`.
- Extends `TransferTrainer.__init__` with `eval_freq: int | None = None`.
- Adds CLI option `--eval-freq` to `transfer`.
- Extends `EvaluationSummary` with `environment_id`, `evaluation_seed`, and `sar_scale` metadata.

- [ ] **Step 1: Write failing runtime-control tests**

Add a pure resolver in the desired API and tests in `tests/test_cli.py`:

```python
import pytest

from etl_sar.cli import _resolve_sar_scale


def test_runtime_sar_scale_overrides_bundle_value() -> None:
    assert _resolve_sar_scale(1.0, 0.0) == 0.0
    assert _resolve_sar_scale(1.0, None) == 1.0


def test_runtime_sar_scale_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _resolve_sar_scale(1.0, 1.1)
```

In `tests/test_trainers.py`, monkeypatch `etl_sar.trainers.EvalCallback` with a
recording test double and assert that `TransferTrainer(eval_freq=5000)` passes
`eval_freq=5000` instead of `n_steps`.

- [ ] **Step 2: Run focused tests and verify missing interfaces fail**

Run:

```powershell
python -m pytest tests/test_cli.py tests/test_trainers.py -q
```

Expected: FAIL because `_resolve_sar_scale` and `eval_freq` do not exist.

- [ ] **Step 3: Implement runtime SAR-scale resolution**

Add to `cli.py`:

```python
def _resolve_sar_scale(bundle_scale: float, override: float | None) -> float:
    scale = float(bundle_scale if override is None else override)
    if not 0.0 <= scale <= 1.0:
        raise ValueError("SAR scale must be between 0 and 1")
    return scale
```

Use the resolved value when constructing `ETLSARActionModel`. Add optional
`sar_scale` Typer options to both target commands and pass them into `_load_bundle`.

- [ ] **Step 4: Implement configurable intermediate evaluation frequency**

Store this value in `TransferTrainer.__init__`:

```python
self.eval_freq = self.n_steps if eval_freq is None else int(eval_freq)
if self.eval_freq < 1:
    raise ValueError("eval_freq must be positive")
```

Pass `self.eval_freq` into `EvalCallback`, expose `--eval-freq` in `transfer`, and
keep `None` as the backward-compatible default.

- [ ] **Step 5: Record and validate matched evaluation metadata**

Extend `EvaluationSummary` with optional backward-compatible fields:

```python
environment_id: str | None = None
evaluation_seed: int | None = None
sar_scale: float | None = None
```

Pass these values from the CLI. Update `compare_runs` to reject mismatched episode
counts, environment IDs, evaluation seeds, or environment-step budgets when both
summaries contain those fields. Include the matched metadata in its result.

- [ ] **Step 6: Run focused and full tests**

Run:

```powershell
python -m pytest tests/test_cli.py tests/test_trainers.py tests/test_evaluation.py -q
python -m pytest -m "not myo" -q
```

Expected: all tests pass; existing summary JSON without optional metadata still loads.

- [ ] **Step 7: Commit**

```powershell
git add -- src/etl_sar/cli.py src/etl_sar/trainers.py src/etl_sar/evaluation.py tests/test_cli.py tests/test_trainers.py tests/test_evaluation.py
git commit -m "feat: add matched pilot runtime controls"
```

---

### Task 4: Build the Two-Domain Pilot Summary

**Files:**
- Create: `src/etl_sar/pilot.py`
- Create: `tests/test_pilot.py`
- Modify: `src/etl_sar/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces `build_pilot_summary(hand_comparison, leg_comparison, legacy_reference=None) -> dict[str, Any]`.
- Adds CLI command `pilot-summary`.
- Consumes optional legacy JSON with `method` and `domains.hand/leg` entries.

- [ ] **Step 1: Write failing decision and legacy-comparability tests**

Create `tests/test_pilot.py`:

```python
from etl_sar.pilot import build_pilot_summary


def local_comparison(env_id: str, return_delta: float, success_delta: float = 0.0) -> dict:
    return {
        "environment_steps": 20_000,
        "episodes": 10,
        "environment_id": env_id,
        "evaluation_seed": 10_007,
        "mean_return_delta": return_delta,
        "success_rate_delta": success_delta,
    }


def test_pilot_is_positive_only_when_both_domains_pass() -> None:
    result = build_pilot_summary(
        local_comparison("myoHandReorient100-v0", 1.0),
        local_comparison("myoLegRoughTerrainWalk-v0", 2.0),
    )
    assert result["pilot_positive"] is True

    failed = build_pilot_summary(
        local_comparison("myoHandReorient100-v0", 1.0),
        local_comparison("myoLegRoughTerrainWalk-v0", -0.1),
    )
    assert failed["pilot_positive"] is False


def test_legacy_reference_is_not_comparable_when_protocol_differs() -> None:
    legacy = {
        "method": "ETL-Ray",
        "domains": {
            "hand": {
                "environment_id": "different-task",
                "protocol": "legacy",
                "metric": "mean_return",
                "value": 1.0,
            },
            "leg": {
                "environment_id": "different-task",
                "protocol": "legacy",
                "metric": "mean_return",
                "value": 1.0,
            },
        },
    }

    result = build_pilot_summary(
        local_comparison("myoHandReorient100-v0", 1.0),
        local_comparison("myoLegRoughTerrainWalk-v0", 1.0),
        legacy,
    )

    assert result["legacy_reference"]["domains"]["hand"]["comparable"] is False
    assert "mean_return_delta" not in result["legacy_reference"]["domains"]["hand"]
```

- [ ] **Step 2: Run tests and verify the missing module failure**

Run:

```powershell
python -m pytest tests/test_pilot.py -q
```

Expected: collection ERROR because `etl_sar.pilot` does not exist.

- [ ] **Step 3: Implement the decision rule and legacy validation**

Use these exact local protocol requirements in `pilot.py`:

```python
LOCAL_PROTOCOL = "myosuite-2.12.2-default"
EXPECTED_TARGETS = {
    "hand": "myoHandReorient100-v0",
    "leg": "myoLegRoughTerrainWalk-v0",
}
```

The pilot is positive only when both mean-return deltas are greater than zero,
both success-rate deltas are nonnegative, both budgets are 20,000, and both
evaluation episode counts are 10. A legacy entry is comparable only when its
environment ID, protocol, and metric exactly match the local values.

- [ ] **Step 4: Add `pilot-summary` CLI command**

The command accepts:

```python
hand: Path
leg: Path
output: Path
legacy_reference: Path | None = None
```

Load both local comparison JSON files, optionally load legacy JSON, call
`build_pilot_summary`, write UTF-8 JSON to `output`, and print the same JSON.

- [ ] **Step 5: Run focused and full tests**

Run:

```powershell
python -m pytest tests/test_pilot.py tests/test_cli.py -q
python -m pytest -m "not myo" -q
```

Expected: all tests pass and CLI help includes `pilot-summary`.

- [ ] **Step 6: Commit**

```powershell
git add -- src/etl_sar/pilot.py src/etl_sar/cli.py tests/test_pilot.py tests/test_cli.py
git commit -m "feat: summarize minimal hand and leg pilot"
```

---

### Task 5: Add the Restartable PowerShell Pilot Runner

**Files:**
- Create: `scripts/run_minimal_pilot.ps1`
- Create: `tests/test_pilot_script.py`
- Modify: `README.md`

**Interfaces:**
- Script parameters: `-EtlSar`, `-RunRoot`, and optional `-LegacyReference`.
- Produces the exact `runs/minimal_pilot` layout from the design specification.
- Uses a SHA-256 stage signature plus expected artifact paths before skipping work.

- [ ] **Step 1: Write failing script contract tests**

Create `tests/test_pilot_script.py`:

```python
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_minimal_pilot.ps1"


def test_minimal_pilot_script_contains_both_matched_domains() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "configs/hand_quick.yaml" in text
    assert "configs/leg_quick.yaml" in text
    assert "--sar-scale 0.0" in text
    assert "--sar-scale 1.0" in text
    assert text.count("--timesteps 20000") >= 4
    assert text.count("--episodes 10") >= 4
    assert "latest_model.zip" in text


def test_minimal_pilot_script_uses_validated_completion_markers() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "Get-FileHash" in text
    assert "stage.complete.json" in text
    assert "$ErrorActionPreference = \"Stop\"" in text
```

- [ ] **Step 2: Run tests and verify the script is missing**

Run:

```powershell
python -m pytest tests/test_pilot_script.py -q
```

Expected: FAIL with `FileNotFoundError`.

- [ ] **Step 3: Implement the PowerShell runner**

The script must:

1. Set strict mode and stop on errors.
2. Resolve the project root from `$PSScriptRoot`.
3. Define Hand budgets `(10000, 30000)` and Leg budgets `(20000, 50000)`.
4. Hash the domain name, config file hash, all budgets, seed, and command arguments
   into each `stage.complete.json` marker.
5. Skip a stage only when the marker signature matches and every expected artifact
   exists; otherwise rerun that stage.
6. Run source exploration and representation fitting once per domain.
7. Train baseline and extension with identical 20,000-step budgets,
   `--decoder-freeze-steps 2000`, and `--eval-freq 5000`.
8. Evaluate both `latest_model.zip` files with 10 episodes and the matching SAR
   scale.
9. Run `compare` per domain and `pilot-summary` after both domains complete.

The commands for each domain must use this structure:

```powershell
& $EtlSar explore --config $config --run-dir $source `
    --timesteps $sourceMax --min-timesteps $sourceMin `
    --min-success-actions 20
& $EtlSar fit-representation --config $config --data-dir "$source/data" `
    --explore-checkpoint "$source/representation.pt" `
    --output-dir $representation --sar-steps 200
& $EtlSar transfer --config $config --bundle $bundle `
    --run-dir $baselineTarget --timesteps 20000 `
    --decoder-freeze-steps 2000 --eval-freq 5000 --sar-scale 0.0
& $EtlSar transfer --config $config --bundle $bundle `
    --run-dir $extensionTarget --timesteps 20000 `
    --decoder-freeze-steps 2000 --eval-freq 5000 --sar-scale 1.0
```

- [ ] **Step 4: Validate PowerShell syntax and static contract**

Run:

```powershell
powershell -NoProfile -Command "[scriptblock]::Create((Get-Content -Raw scripts/run_minimal_pilot.ps1)) | Out-Null"
python -m pytest tests/test_pilot_script.py -q
```

Expected: PowerShell exits 0 and both pytest tests pass.

- [ ] **Step 5: Document the pilot command and correct the Leg task name**

Append a clear `Minimal Hand + Leg pilot` section to `README.md` with:

```powershell
.\scripts\run_minimal_pilot.ps1
```

Document the output root and decision rule. Replace the stale
`myoLegUneven-v0` reference with `myoLegRoughTerrainWalk-v0`.

- [ ] **Step 6: Run the complete verification suite**

Run:

```powershell
python -m pytest -m "not myo" -q
python -m pytest -m myo -q
python -m compileall -q src tests
git diff --check
```

Expected: all tests pass, all six registered real MyoSuite environments load, and
the source tree compiles without errors.

- [ ] **Step 7: Perform a no-training runner validation**

Add `-WhatIf` support to the script and run:

```powershell
.\scripts\run_minimal_pilot.ps1 -WhatIf
```

Expected: the script prints source, representation, baseline, extension,
evaluation, comparison, and summary commands for both Hand and Leg without creating
run artifacts or starting training.

- [ ] **Step 8: Commit**

```powershell
git add -- scripts/run_minimal_pilot.ps1 tests/test_pilot_script.py README.md
git commit -m "feat: add restartable minimal pilot runner"
```

---

## Final Verification

- [ ] Confirm `git status --short` is empty.
- [ ] Confirm CLI help lists `pilot-summary` and the new runtime options.
- [ ] Confirm the dry run contains both domains and matched baseline/extension budgets.
- [ ] Confirm no actual 4-7 hour pilot training was started during implementation.
- [ ] Record that legacy ETL-Ray numerical comparison requires a real reference JSON;
  absent data must be reported as unavailable, never invented.

# Single-Seed Ten-Hour Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an isolated, restartable Hand+Leg comparison of ETL+SAR, ETL-noSAR, and official Lattice that uses one seed and is sized for 6-9 hours on one RTX 4090 plus 16 CPU cores.

**Architecture:** Reuse the existing formal matrix, server runner, exact-transition callbacks, and manifest system with two new YAML configs. Add a sequential launcher that passes those configs explicitly and writes to an isolated output root. Generalize aggregation so a configured single seed produces descriptive deltas without inferential statistics while the five-seed formal path remains unchanged.

**Tech Stack:** Python 3.11, PyYAML, NumPy, PyTorch, Stable-Baselines3, sb3-contrib, Gymnasium/MyoSuite, pytest, Bash.

## Global Constraints

- Compare exactly `etl_sar`, `etl_no_sar`, and pinned official `lattice` on both Hand and Leg.
- Use training seed `0` only.
- Hand budgets are 80,000 source + 1,520,000 ETL target, or 1,600,000 Lattice target.
- Leg budgets are 120,000 source + 1,080,000 ETL target, or 1,200,000 Lattice target.
- Use 16 Lattice vector environments, 10 intermediate evaluation episodes, and 50 final evaluation episodes.
- Use Hand checkpoint interval 80,000 and Leg checkpoint interval 120,000 so all learning curves end at the exact target budget.
- The matrix must contain 2 source stages, 6 target jobs, and 8,400,000 attributed interactions.
- Preserve `configs/formal_hand.yaml`, `configs/formal_leg.yaml`, and all five-seed inferential behavior.
- A one-seed aggregate must not emit standard errors, confidence intervals, p-values, Holm corrections, or a boolean protocol-success claim.
- Default output is `runs/single_seed_10h`; reruns must resume through existing manifests and checkpoints.
- The launcher targets one CUDA-capable RTX 4090 and runs jobs sequentially.

---

### Task 1: Lock The Short Experiment Matrix

**Files:**
- Create: `configs/single_seed_10h_hand.yaml`
- Create: `configs/single_seed_10h_leg.yaml`
- Create: `tests/test_single_seed_protocol.py`

**Interfaces:**
- Consumes: `FormalDomainConfig.from_yaml()` and `ExperimentMatrix.from_configs()`.
- Produces: two validated configs usable through existing `--hand-config` and `--leg-config` server options.

- [ ] **Step 1: Write the failing matrix tests**

```python
def test_single_seed_matrix_locks_budget_and_methods() -> None:
    hand, leg = load_short_configs()
    assert hand.seeds == leg.seeds == (0,)
    assert (hand.source_budget, hand.target_budget, hand.lattice_budget) == (
        80_000, 1_520_000, 1_600_000
    )
    assert (leg.source_budget, leg.target_budget, leg.lattice_budget) == (
        120_000, 1_080_000, 1_200_000
    )
    matrix = ExperimentMatrix.from_configs((hand, leg))
    assert len(matrix.source_stages) == 2
    assert len(matrix.target_jobs) == 6
    assert matrix.total_declared_interactions == 8_400_000
    assert {job.method for job in matrix.target_jobs} == set(Method)

def test_single_seed_checkpoints_end_at_each_budget() -> None:
    for config in load_short_configs():
        assert config.target_budget % config.checkpoint_interval == 0
        assert config.lattice_budget % config.checkpoint_interval == 0
        assert config.intermediate_episodes == 10
        assert config.final_episodes == 50
        assert config.num_envs == 16
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest -q tests/test_single_seed_protocol.py`

Expected: FAIL because both short YAML files are missing.

- [ ] **Step 3: Add the exact YAML configs**

```yaml
# configs/single_seed_10h_hand.yaml
domain: hand
source_env: myoHandReorient8-v0
target_env: myoHandReorient100-v0
action_dim: 39
source_budget: 80000
target_budget: 1520000
lattice_budget: 1600000
seeds: [0]
num_envs: 16
checkpoint_interval: 80000
intermediate_episodes: 10
final_episodes: 50
```

The Leg config uses `myoLegWalk-v0`, `myoLegRoughTerrainWalk-v0`, action dimension 80, budgets `120000/1080000/1200000`, and checkpoint interval `120000`; all other schedule values match Hand.

- [ ] **Step 4: Run the matrix tests and verify GREEN**

Run: `python -m pytest -q tests/test_single_seed_protocol.py tests/test_formal_matrix.py`

Expected: all tests pass and the original formal matrix remains 10 sources, 30 targets, and 525,000,000 attributed interactions.

- [ ] **Step 5: Commit**

```bash
git add configs/single_seed_10h_hand.yaml configs/single_seed_10h_leg.yaml tests/test_single_seed_protocol.py
git commit -m "feat: add single-seed ten-hour protocol"
```

### Task 2: Add Honest Single-Seed Aggregation

**Files:**
- Modify: `src/etl_sar/formal/aggregate.py`
- Modify: `tests/test_formal_aggregate.py`

**Interfaces:**
- Consumes: `ExperimentMatrix.configs`, `SeedResult`, and the existing five-seed aggregate functions.
- Produces: `summarize_seed_results(..., expected_seeds_by_domain=...)` and a descriptive single-seed JSON result.

- [ ] **Step 1: Write failing descriptive-output tests**

```python
def test_single_seed_aggregate_reports_deltas_without_inference() -> None:
    records = single_seed_records()
    result = summarize_seed_results(
        records,
        bootstrap_seed=123,
        expected_seeds_by_domain={"hand": (0,), "leg": (0,)},
    )
    assert result["analysis_mode"] == "descriptive_single_seed"
    assert result["protocol_success"] is None
    hand = result["domains"]["hand"]
    assert hand["comparisons"]["etl_no_sar"]["auc_delta"] == pytest.approx(0.3)
    encoded = json.dumps(result)
    for forbidden in ("raw_p", "holm_p", "ci_low", "ci_high", '"se"'):
        assert forbidden not in encoded
```

Also extend the existing five-seed test to assert it still returns a boolean `protocol_success`, paired CI fields, and Holm p-values.

- [ ] **Step 2: Run the aggregate test and verify RED**

Run: `python -m pytest -q tests/test_formal_aggregate.py`

Expected: FAIL because `expected_seeds_by_domain` is not accepted and seeds are hard-coded to `0..4`.

- [ ] **Step 3: Implement seed validation and descriptive aggregation**

Add an optional expected-seed mapping to `summarize_seed_results`. Validate that every method has exactly the configured seed list. When all domains contain one seed, return method point values and ETL+SAR deltas:

```python
{
    "analysis_mode": "descriptive_single_seed",
    "domains": {
        domain: {
            "methods": {
                method: {
                    "seed": row.seed,
                    "normalized_auc": row.normalized_auc,
                    "final_primary": row.final_primary,
                }
            },
            "comparisons": {
                baseline: {
                    "auc_delta": treatment.normalized_auc - baseline.normalized_auc,
                    "final_delta": treatment.final_primary - baseline.final_primary,
                }
            },
        }
    },
    "protocol_success": None,
}
```

For multiple seeds, retain the existing output and calculations. In `write_aggregate`, derive `{config.domain: config.seeds}` from `matrix.configs` and pass it to the summarizer.

- [ ] **Step 4: Run aggregate tests and verify GREEN**

Run: `python -m pytest -q tests/test_formal_aggregate.py tests/test_formal_statistics.py`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/etl_sar/formal/aggregate.py tests/test_formal_aggregate.py
git commit -m "feat: report single-seed results descriptively"
```

### Task 3: Add The Sequential RTX 4090 Launcher

**Files:**
- Create: `scripts/run_single_seed_10h.sh`
- Modify: `tests/test_server_scripts.py`

**Interfaces:**
- Consumes: `etl_sar.formal.server`, `etl_sar.formal.aggregate_cli`, both short configs, and existing manifests.
- Produces: `dry-run`, `run`, and `aggregate` modes with `SHORT_OUTPUT_ROOT` override.

- [ ] **Step 1: Write failing launcher tests**

Assert that the script contains both short config paths, isolated output root, CUDA preflight, source loop `0 1`, target loop `0 1 2 3 4 5`, explicit server config arguments, and aggregate CLI config arguments.

- [ ] **Step 2: Run the script tests and verify RED**

Run: `python -m pytest -q tests/test_server_scripts.py`

Expected: FAIL because `scripts/run_single_seed_10h.sh` is missing.

- [ ] **Step 3: Implement the launcher**

Use `set -euo pipefail`, default mode `dry-run`, default output root `runs/single_seed_10h`, and one helper that appends the explicit config arguments. In `run` mode, verify `torch.cuda.is_available()`, print the GPU name, run two sources and six targets sequentially, run aggregation, and print elapsed seconds. Do not invoke Slurm or launch concurrent GPU jobs.

- [ ] **Step 4: Verify launcher and dry-run**

Run: `python -m pytest -q tests/test_server_scripts.py tests/test_formal_server.py`

Run: `python -m etl_sar.formal.server dry-run --hand-config configs/single_seed_10h_hand.yaml --leg-config configs/single_seed_10h_leg.yaml`

Expected summary: 2 sources, 6 targets, and 8,400,000 attributed interactions.

- [ ] **Step 5: Mark executable and commit**

```bash
git add scripts/run_single_seed_10h.sh tests/test_server_scripts.py
git update-index --chmod=+x scripts/run_single_seed_10h.sh
git commit -m "feat: add sequential ten-hour server launcher"
```

### Task 4: Document Server Transfer And Execution

**Files:**
- Modify: `README.md`
- Test: `tests/test_server_scripts.py`

**Interfaces:**
- Consumes: the short launcher and Python 3.11 project environment.
- Produces: copy-ready Conda setup, preflight, dry-run, run, resume, aggregate, and output-location instructions.

- [ ] **Step 1: Extend the documentation assertions**

Require README references to `run_single_seed_10h.sh`, `SHORT_OUTPUT_ROOT`, `8.4M`, `seed=0`, RTX 4090, and the descriptive-only limitation.

- [ ] **Step 2: Run the documentation test and verify RED**

Run: `python -m pytest -q tests/test_server_scripts.py`

Expected: FAIL because README lacks the short-protocol section.

- [ ] **Step 3: Add concise server instructions**

Document Python 3.11 Conda creation, CUDA verification, MyoSuite smoke tests, dry-run, `SHORT_OUTPUT_ROOT=/data/... bash scripts/run_single_seed_10h.sh run`, restart behavior, and separate aggregation. State that one seed supports a budget-constrained case comparison only.

- [ ] **Step 4: Run documentation tests and commit**

Run: `python -m pytest -q tests/test_server_scripts.py`

```bash
git add README.md tests/test_server_scripts.py
git commit -m "docs: add ten-hour server workflow"
```

### Task 5: Full Verification

**Files:**
- Verify only; modify implementation files only if a failing test exposes a defect and first add a regression test.

**Interfaces:**
- Consumes: all tasks above.
- Produces: a tested, committed short experiment ready to transfer to the server.

- [ ] **Step 1: Run static and full tests**

Run: `git diff --check`

Run: `python -m pytest -q`

Expected: all tests pass; only known Gymnasium float32 precision warnings may remain.

- [ ] **Step 2: Run real MyoSuite smoke tests**

Run: `python -m pytest -m myo -q`

Expected: all registered Hand/Leg and Lattice smoke tests pass.

- [ ] **Step 3: Verify final matrix and repository state**

Run the explicit short dry-run and parse its summary. Confirm 2 sources, 6 targets, and 8,400,000 interactions. Confirm the existing formal dry-run remains 10 sources, 30 targets, and 525,000,000 interactions. Confirm `MUJOCO_LOG.TXT` remains untracked and untouched.

- [ ] **Step 4: Commit any test-only corrections and report**

If no correction was required, do not create an empty commit. Report commit IDs, test totals, short launcher command, output root, and the 6-9 hour estimate.

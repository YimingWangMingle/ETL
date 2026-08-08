# Public Reproducibility Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a clean, complete, English-documented reproducibility release to `YimingWangMingle/ETL`.

**Architecture:** Keep the implementation and matched DMC protocol unchanged. Add release documentation around existing configuration and shell entry points, preserve upstream provenance and remote license/history, then validate the exact tracked artifact before a normal push.

**Tech Stack:** Python 3.11, PyTorch, Stable-Baselines3, dm_control, MuJoCo/EGL, pytest, Bash, Git.

## Global Constraints

- README content is English-only.
- Preserve remote MIT license and history; never force-push.
- Publish no generated results, virtual environments, caches, checkpoints, datasets, or secrets.
- Describe one-seed results as descriptive and do not claim statistical significance or overall superiority.

---

### Task 1: Commit the verified Lattice compatibility fix

**Files:**
- Modify: `src/etl_sar/lattice/policies.py`
- Modify: `tests/test_lattice_policies.py`

**Interfaces:**
- Consumes: SB3's `Actor.get_std()` logging hook.
- Produces: one concatenated standard-deviation tensor for logging while preserving Lattice sampling behavior.

- [ ] **Step 1:** Run `python -m pytest tests/test_lattice_policies.py -q` and expect all policy tests to pass.
- [ ] **Step 2:** Review `git diff --check` and the two-file diff.
- [ ] **Step 3:** Commit the two files as `fix: expose lattice standard deviations to sb3`.

### Task 2: Merge the public repository history

**Files:**
- Preserve: `LICENSE`
- Replace later: `README.md`

**Interfaces:**
- Consumes: public remote `https://github.com/YimingWangMingle/ETL.git`, branch `main`.
- Produces: one non-rewritten history containing both the remote initial commit and the local implementation history.

- [ ] **Step 1:** Add `origin` and fetch `origin/main`.
- [ ] **Step 2:** Merge with `--allow-unrelated-histories`, resolving only the placeholder README while retaining `LICENSE`.
- [ ] **Step 3:** Confirm `git log --graph --oneline --all` contains both histories.

### Task 3: Write the reproducibility README

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: `configs/dmc_dog_pilot.yaml`, `configs/dmc_humanoid_pilot.yaml`, `scripts/run_dmc_transfer_pilot.sh`, and DMC aggregation outputs.
- Produces: clean-server installation and exact experiment reproduction instructions.

- [ ] **Step 1:** Replace the existing mixed/corrupted README with the approved English structure.
- [ ] **Step 2:** Include exact environment, smoke-test, dry-run, run/resume, and aggregate commands.
- [ ] **Step 3:** Include matched budgets, metric definitions, reference results, limitations, and upstream provenance.
- [ ] **Step 4:** Check every referenced command and path against the repository.

### Task 4: Validate the release artifact

**Files:**
- Inspect: all Git-tracked files.

**Interfaces:**
- Consumes: final working tree and test environment.
- Produces: evidence that code, documentation, and upload boundary are sound.

- [ ] **Step 1:** Run `python -m pytest -m "not myo" -q`; expect the full non-MyoSuite suite to pass.
- [ ] **Step 2:** Run `git diff --check`; expect no whitespace errors.
- [ ] **Step 3:** Search tracked content for tokens, passwords, API keys, and private-key headers; review every hit.
- [ ] **Step 4:** Inspect tracked count, aggregate size, largest files, and ignored runtime directories.

### Task 5: Commit and publish

**Files:**
- Add: `docs/superpowers/specs/2026-08-08-public-reproducibility-release-design.md`
- Add: `docs/superpowers/plans/2026-08-08-public-reproducibility-release.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: validated merged repository.
- Produces: public `main` branch at `https://github.com/YimingWangMingle/ETL`.

- [ ] **Step 1:** Set repository-local author identity to `Trefor.Moe <162333414+moyueming@users.noreply.github.com>`.
- [ ] **Step 2:** Commit documentation as `docs: add reproducible DMC experiment guide`.
- [ ] **Step 3:** Rename the local branch to `main` and push with upstream tracking.
- [ ] **Step 4:** Fetch and verify that local `HEAD` equals `origin/main` and the tree is clean.

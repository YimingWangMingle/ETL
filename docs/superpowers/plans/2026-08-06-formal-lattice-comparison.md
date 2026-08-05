# Formal ETL-SAR vs. Lattice Implementation Plan

> Execute this plan test-first. The approved design in
> `docs/superpowers/specs/2026-08-06-formal-lattice-comparison-design.md` is the
> source of truth. Do not change algorithm mathematics or declared budgets.

## Goal

Produce a server-ready, restartable, equal-interaction comparison of ETL-noSAR,
ETL+SAR, and official Lattice on the approved MyoSuite Hand and Leg tasks.

## 1. Fix paired ETL checkpoints

Files:

- Add `src/etl_sar/checkpoints.py`
- Modify `src/etl_sar/trainers.py`
- Modify `src/etl_sar/cli.py`
- Add `tests/test_checkpoints.py`
- Modify `tests/test_trainers.py`

Steps:

1. Write tests for a checkpoint manifest containing the policy filename,
   action-model filename, SHA-256 hashes, transition count, and pair ID.
2. Write tests that reject a missing, renamed, modified, or cross-paired file.
3. Write a round-trip test: save policy plus fine-tuned action model, reload the
   pair, and assert identical latent-policy outputs and executed muscle actions.
4. Implement `CheckpointPair`, atomic temporary-file replacement, SHA-256
   verification, and manifest loading in `checkpoints.py`.
5. Replace `EvalCallback` with a paired callback that snapshots both PPO and
   `ETLSARActionModel` whenever a latest/intermediate/best policy is saved.
6. Change `TrainingArtifacts` to expose manifest paths and make CLI evaluation
   accept a pair manifest, never an independent bundle plus policy path.
7. Run `python -m pytest tests/test_checkpoints.py tests/test_trainers.py tests/test_cli.py -q`.

## 2. Vendor and port official Lattice

Files:

- Add `third_party/lattice/LICENSE`
- Add `third_party/lattice/UPSTREAM.md`
- Add `third_party/lattice/distributions.py`
- Add `src/etl_sar/lattice/__init__.py`
- Add `src/etl_sar/lattice/distributions.py`
- Add `src/etl_sar/lattice/ppo_policy.py`
- Add `src/etl_sar/lattice/sac_policy.py`
- Add `src/etl_sar/lattice/trainers.py`
- Add `tests/test_lattice_distribution.py`
- Add `tests/test_lattice_policies.py`

Steps:

1. Vendor the pinned upstream distribution implementation from commit
   `846d02fa993b9b80ce5ecb806463e0a05711bad3` and preserve the MIT license.
2. Document source URLs, file hashes, upstream commit, and every SB3 2.x /
   Gymnasium compatibility edit in `UPSTREAM.md`.
3. Add fixed-seed parity tests for covariance construction, sampled actions,
   log probability, entropy, and deterministic mode.
4. Port the distribution to the current SB3 distribution interface without
   changing its covariance/noise equations.
5. Port official recurrent PPO and SAC policy hooks, retaining Lattice defaults:
   `use_lattice=true`, `use_sde=false`, `freq=1`, `log_std_init=0`,
   `std_reg=0`, and inactive `alpha=1`.
6. Implement Hand RecurrentPPO and Leg SAC constructors with precisely the
   approved upstream hyperparameters and 16 vector environments.
7. Ensure recurrent evaluation resets state and episode-start masks.
8. Run `python -m pytest tests/test_lattice_distribution.py tests/test_lattice_policies.py -q`.

## 3. Add formal experiment definitions

Files:

- Add `src/etl_sar/formal/__init__.py`
- Add `src/etl_sar/formal/config.py`
- Add `src/etl_sar/formal/matrix.py`
- Add `src/etl_sar/formal/seeds.py`
- Add `configs/formal_hand.yaml`
- Add `configs/formal_leg.yaml`
- Add `tests/test_formal_config.py`
- Add `tests/test_formal_matrix.py`

Steps:

1. Test strict parsing of methods, tasks, five seeds, source/target budgets,
   250k checkpoint cadence, and final episode counts.
2. Test that source plus target equals the Lattice budget for each domain.
3. Test transition accounting under 16 environments, including non-divisible
   budget rejection rather than silent overshoot.
4. Generate disjoint deterministic training and evaluation seed banks and store
   the exact seed list in every run directory.
5. Expand the two YAML files into exactly 10 source stages and 30 target jobs;
   ETL-noSAR and ETL+SAR must reference the same per-seed source artifact.
6. Add a dry-run CLI that emits the full matrix as JSON without importing
   MyoSuite or creating environments.
7. Run `python -m pytest tests/test_formal_config.py tests/test_formal_matrix.py tests/test_cli.py -q`.

## 4. Add unified metrics and statistics

Files:

- Add `src/etl_sar/formal/metrics.py`
- Add `src/etl_sar/formal/evaluate.py`
- Add `src/etl_sar/formal/statistics.py`
- Add `tests/test_formal_metrics.py`
- Add `tests/test_formal_statistics.py`

Steps:

1. Test episode rows for return, length, success, termination/fall, object drop,
   velocity error, muscle effort, and domain primary metric.
2. Resolve Leg root-x qpos through MuJoCo joint metadata; test a root free joint
   at a deliberately nonzero qpos address.
3. Support standard SB3 and recurrent predictors with identical deterministic
   seed-bank evaluation semantics.
4. Compute normalized trapezoidal learning-curve AUC from the declared budget.
5. Test per-seed mean, SE, median, paired effects, deterministic bootstrap 95%
   confidence intervals, and Holm adjustment for the two comparisons.
6. Emit episode CSV, checkpoint summaries, final summaries, and aggregate JSON /
   CSV suitable for plotting; never discard raw episode records.
7. Run `python -m pytest tests/test_formal_metrics.py tests/test_formal_statistics.py -q`.

## 5. Add restartable server orchestration

Files:

- Add `src/etl_sar/formal/manifest.py`
- Add `src/etl_sar/formal/runner.py`
- Add `scripts/run_formal_server.sh`
- Add `scripts/submit_formal_slurm.sh`
- Add `environment-server.yml`
- Add `tests/test_formal_manifest.py`
- Add `tests/test_formal_runner.py`
- Modify `README.md`

Steps:

1. Test run manifests for method/domain/seed/budgets, commands, repository and
   upstream commits, Python/packages, environment fingerprint, hardware,
   resource usage, transition counters, and artifact hashes.
2. Sign a stage complete only after all expected artifacts hash successfully.
3. Test resume behavior: completed stages skip, partial ETL PPO restores paired
   checkpoints, and SAC additionally restores replay buffer and counters.
4. Add a portable one-job runner plus a Slurm array mapping over 30 targets;
   source stages use dependency-aware submission and are shared by ETL pairs.
5. Record wall time, transitions/second, peak GPU memory, and CPU utilization.
6. Document dry-run, miniature validation, full launch, resume, aggregation,
   expected disk usage, and upload instructions.
7. Run `python -m pytest tests/test_formal_manifest.py tests/test_formal_runner.py -q`.

## 6. Validate and deliver

1. Run `python -m pytest -m "not myo" -q`.
2. Run `python -m pytest -m myo -q` on the local MyoSuite installation.
3. Run miniature Hand RecurrentPPO and Leg SAC smoke jobs and verify the final
   result schema; clearly mark them non-formal.
4. Run the dry-run command and assert 30 target jobs, 10 ETL source stages, and
   525,000,000 declared training interactions.
5. Verify all methods report the same target environment fingerprint per domain.
6. Inspect `git diff --check`, `git status --short`, all licenses/provenance, and
   README commands. Confirm no placeholders, unpaired checkpoints, hard-coded
   Leg qpos indexes, or changes to upstream Lattice mathematics remain.
7. Commit each completed stage and integrate the branch into
   `D:\学习\etl_lattice_sar` without touching `MUJOCO_LOG.TXT`.


# Public Reproducibility Release Design

## Objective

Publish the complete `etl-lattice-sar` implementation to `YimingWangMingle/ETL` with an English README that lets a new user reproduce the matched DMC pilot from a clean Ubuntu server.

## Publication Boundary

- Preserve the remote repository's MIT `LICENSE` and initial commit history.
- Publish source code, configs, scripts, tests, and the pinned Lattice provenance snapshot.
- Exclude environments, caches, training outputs, checkpoints, downloaded data, credentials, and machine-specific files.
- Describe this repository as an ETL-dominant research extension, not as an official joint release from the ETL, SAR, or Lattice authors.

## README Structure

The root README will be English-only and organized around the currently reproducible DMC experiment:

1. Scope and method summary.
2. A comparison table for ETL+SAR, ETL-noSAR, and Lattice.
3. Matched Dog Walk-to-Run and Humanoid Walk-to-Run budgets.
4. Ubuntu 22.04, Python 3.11, CUDA PyTorch, EGL, and editable-install instructions.
5. Smoke test, dry run, training, resume, and aggregation commands.
6. Output layout and definitions for return AUC and final mean return.
7. A clearly labelled single-seed reference-results table.
8. Limitations, upstream provenance, testing, and citation notes.

The README must state that ETL uses 200k source plus 800k target transitions while Lattice uses 1M target transitions. It must also state that all methods use the same SAC backbone and observation normalization in this matched pilot, and that one seed is descriptive rather than statistically conclusive.

## Git Strategy

Commit the local Lattice logging compatibility fix first. Fetch the remote `main`, merge its unrelated initial history without force-pushing, retain the remote MIT license, replace the placeholder README, rename the local branch to `main`, and push normally.

## Verification

- Run the focused Lattice policy tests and the complete non-MyoSuite test suite.
- Run `git diff --check`.
- scan tracked files for credential/private-key patterns.
- inspect tracked file count, total size, and largest files.
- verify ignored runtime artifacts are not tracked.
- verify the remote `main` points to the published commit after pushing.

## Honesty Constraints

The reference results may show that SAR improves ETL-noSAR on both domains, but they must not claim overall superiority over Lattice. The README must explicitly note that the pilot uses one seed and reduced budgets and therefore does not reproduce the papers' full multi-seed protocols or establish statistical significance.

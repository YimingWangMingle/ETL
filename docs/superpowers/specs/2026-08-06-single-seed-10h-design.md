# Single-Seed Ten-Hour Comparison Design

## Objective

Add a server-ready pilot protocol that compares exactly three methods on both
MyoSuite domains and is sized for a single RTX 4090 plus 16 CPU cores:

1. ETL with SAR enabled (`etl_sar`)
2. ETL with SAR disabled (`etl_no_sar`)
3. The pinned official Lattice method (`lattice`)

The complete sequential run should normally finish in 6-9 hours and must be
sized to remain below a ten-hour target on the specified hardware. This is a
single-seed, budget-constrained comparison, not a statistical significance
experiment.

## Non-Goals

- Do not change the existing five-seed formal protocol or its results path.
- Do not change ETL, SAR, Lattice covariance equations, policy architectures,
  optimizer settings, or task definitions.
- Do not claim statistical significance from one seed.
- Do not enforce the ten-hour target by killing an active process. A killed run
  can be resumed, but the launcher should preserve checkpoints by default.

## Protocol

Both domains use training seed `0`. ETL-noSAR and ETL+SAR share the same source
stage within each domain. Each method receives the same attributed interaction
budget within a domain.

| Domain | Method | Source | Target | Attributed total |
| --- | --- | ---: | ---: | ---: |
| Hand | ETL-noSAR | 80,000 | 1,520,000 | 1,600,000 |
| Hand | ETL+SAR | 80,000 shared | 1,520,000 | 1,600,000 |
| Hand | Lattice | 0 | 1,600,000 | 1,600,000 |
| Leg | ETL-noSAR | 120,000 | 1,080,000 | 1,200,000 |
| Leg | ETL+SAR | 120,000 shared | 1,080,000 | 1,200,000 |
| Leg | Lattice | 0 | 1,200,000 | 1,200,000 |

The matrix therefore contains two source stages, six target jobs, 8,400,000
attributed interactions, and 8,200,000 physically executed interactions. The
tasks remain:

- Hand: `myoHandReorient8-v0 -> myoHandReorient100-v0`
- Leg: `myoLegWalk-v0 -> myoLegRoughTerrainWalk-v0`

Hand uses an 80,000-transition checkpoint interval. Leg uses a
120,000-transition interval. These values divide both the ETL target budget and
the Lattice target budget, so every learning curve ends exactly at its declared
budget. Every intermediate checkpoint is evaluated for 10 episodes, and every
final policy is evaluated for 50 episodes. Training uses 16 Lattice vector
environments and the existing fixed, disjoint evaluation seed banks.

## Fairness

Wall-clock time is a deployment constraint, not the training budget. Algorithms
stop at exact transition counts. ETL source interactions are attributed to both
ETL variants because both consume the shared representation bundle. Lattice is
trained only on the complex target task and receives the same total attributed
budget as either ETL variant.

The short protocol reuses the existing official Lattice commit, distribution
equations, Hand RecurrentPPO configuration, and Leg SAC configuration. Only the
environment and interaction budget differ from upstream, as already required by
the shared MyoSuite comparison.

## Runtime And Hardware

The target server is one RTX 4090, 16 CPU cores, at least 64 GB RAM, and at least
100 GB storage. Jobs run sequentially so only one process owns the GPU at a
time. The expected wall-clock allocation is:

- shared Hand and Leg source stages: 0.5-1.0 hour
- ETL-noSAR across both domains: 1.5-2.5 hours
- ETL+SAR across both domains: 1.5-2.5 hours
- Lattice across both domains: 2.0-3.0 hours
- evaluation, aggregation, and checkpoint I/O: included in the 6-9 hour total

These are estimates rather than a portable guarantee because CPU model, storage
latency, thermal limits, and cloud contention affect MuJoCo throughput. Each
completed manifest records wall time and transitions per second.

## Files And Entry Points

Add two configs without replacing `formal_hand.yaml` or `formal_leg.yaml`:

- `configs/single_seed_10h_hand.yaml`
- `configs/single_seed_10h_leg.yaml`

Add `scripts/run_single_seed_10h.sh` with three modes:

- `dry-run`: print and validate the two-source, six-target matrix.
- `run`: verify CUDA is available, execute source indices `0..1`, execute target
  indices `0..5`, and aggregate the result.
- `aggregate`: rerun only descriptive aggregation.

The default output root is `runs/single_seed_10h`; `SHORT_OUTPUT_ROOT` may
override it. The launcher passes the short config paths explicitly to the
existing server and aggregate CLIs. Re-running `run` validates completed
manifests and resumes incomplete training from existing latest checkpoints.

## Single-Seed Reporting

The current five-seed aggregate remains unchanged. Aggregation derives expected
seeds from `ExperimentMatrix.configs` instead of hard-coding seeds `0..4`.

For one seed, `summary.json` uses `analysis_mode` value
`descriptive_single_seed`, sets `protocol_success` to JSON null, and reports:

- the seed, normalized AUC, and final primary metric for each method;
- ETL+SAR minus ETL-noSAR and ETL+SAR minus Lattice deltas;
- no standard error, confidence interval, sign-flip p-value, or Holm-adjusted
  p-value.

For two or more configured seeds, existing mean, standard error, bootstrap CI,
paired effects, sign-flip tests, Holm correction, and protocol success behavior
remain unchanged.

## Verification

Automated tests must prove that the short configs produce exactly two sources,
six targets, all three methods, seed `0`, exact equal budgets, endpoint-aligned
checkpoint schedules, and 8,400,000 attributed interactions. Script tests must
prove that all indices and config paths are passed explicitly and the short
output root is isolated. Aggregation tests must prove that one-seed output is
descriptive and contains no inferential claims while the five-seed output is
unchanged. The complete test suite and real MyoSuite smoke tests must pass.

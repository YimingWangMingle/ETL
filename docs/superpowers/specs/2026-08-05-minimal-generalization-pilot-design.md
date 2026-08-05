# Minimal Hand and Leg Generalization Pilot Design

## Objective

Produce preliminary local evidence that the ETL-dominant extension with SAR
transfer outperforms a matched ETL-SB3 baseline on both supported complex-task
generalization chains, without running a convergence-scale experiment.

This is a pilot result, not a publication-grade statistical claim. A single seed
is sufficient for the initial go/no-go decision. Additional seeds are run only if
the pilot is positive but noisy.

## Task Scope

The pilot must run both domains:

| Domain | Simple source task | Complex target task |
| --- | --- | --- |
| Hand | `myoHandReorient8-v0` | `myoHandReorient100-v0` |
| Leg | `myoLegWalk-v0` | `myoLegRoughTerrainWalk-v0` |

No other MyoSuite tasks, hyperparameter sweeps, or full convergence runs are in
scope.

## Compared Methods

### Local matched baseline

The primary baseline is ETL-SB3 with the SAR residual disabled by setting the
runtime SAR scale to `0.0`. It uses the same source trajectories, GMVAE, ETL
decoder, target PPO budget, seed, and evaluation initial states as the extension.

### Extension

The extension uses the same representation bundle with runtime SAR scale `1.0`.
The existing hard cap remains `rho=0.20`, so SAR can contribute at most 20 percent
of the ETL decoder action norm.

### Legacy ETL-Ray reference

Results reported by the earlier ETL-Ray work are included as an external reference.
A numerical delta is valid only when the legacy result uses the exact same MyoSuite
environment ID, reward definition, episode horizon, and metric. Otherwise the
legacy value is reported separately with `comparable=false`; the pilot must not
present an invalid cross-task delta.

## Minimal Budget

| Stage | Hand | Leg |
| --- | ---: | ---: |
| Source minimum steps | 10,000 | 20,000 |
| Source maximum steps | 30,000 | 50,000 |
| Required successful source actions | 20 | 20 |
| SAR head updates | 200 | 200 |
| Baseline target PPO steps | 20,000 | 20,000 |
| Extension target PPO steps | 20,000 | 20,000 |
| Decoder freeze steps | 2,000 | 2,000 |
| Intermediate evaluation frequency | 5,000 | 5,000 |
| Final deterministic episodes | 10 | 10 |
| Initial seeds | 1 | 1 |

Source exploration may stop only after both the domain-specific minimum step count
and 20 successful source actions have been reached. If the maximum budget is
reached first, that domain ends with a clear `insufficient_source_success` result;
the pipeline must not fabricate, duplicate, or relabel unsuccessful actions.

## Data Flow

1. Run ETL directional exploration on the simple source task.
2. Interleave BDR and GMVAE updates while collecting trajectories.
3. Fit one 20-component PCA+ICA SAR artifact from successful actions.
4. Save one shared representation bundle per domain.
5. Train target PPO twice from matched seeds and the same bundle:
   baseline with SAR scale `0.0`, extension with SAR scale `1.0`.
6. Evaluate both final `latest_model.zip` checkpoints with identical episode seeds.
7. Produce local paired summaries and a legacy ETL-Ray reference section.

The pilot evaluates final-step models rather than selecting the best checkpoint
from frequent one-episode evaluations. This avoids adding selection noise to a
small-budget comparison.

## Required Engineering Changes

1. Add optional source early-stop controls: minimum steps and minimum successful
   actions. Existing fixed-budget behavior remains the default.
2. Expose target intermediate evaluation frequency through the trainer and CLI;
   retain the current default for existing callers and use `5000` in the pilot.
3. Add an optional runtime SAR-scale override to `transfer` and `evaluate`, so one
   immutable representation bundle supports the matched baseline and extension.
4. Add a resumable PowerShell pilot runner for Hand and Leg. It must use separate
   output directories, skip only validated completed artifacts, and stop on the
   first failed command.
5. Add a pilot summary command that reads both evaluation summaries, verifies equal
   target budgets, and records legacy comparability rather than assuming it.

## Output Layout

```text
runs/minimal_pilot/
  hand/
    source/
    representation/
    baseline_target/
    extension_target/
    baseline_eval/
    extension_eval/
    comparison.json
  leg/
    source/
    representation/
    baseline_target/
    extension_target/
    baseline_eval/
    extension_eval/
    comparison.json
  pilot_summary.json

# Formal ETL-SAR vs. Lattice Comparison Design

## Objective

Build and run a server-scale comparison that tests whether the ETL-dominant
extension with SAR transfer is more sample-efficient and achieves better final
performance than both ETL without SAR and the official Lattice algorithm.

The primary claim is deliberately narrow:

> Under an equal total environment-interaction budget on the same MyoSuite
> Hand and Leg tasks, ETL+SAR outperforms ETL-noSAR and Lattice in learning
> efficiency and final task performance.

The experiment does not claim that all methods use the same optimizer, network,
or policy class. Lattice retains its official algorithm choices. Fairness is
provided by identical tasks, total interaction budgets, seed sets, evaluation
episodes, and reporting rules.

## Methods

### ETL-noSAR

- Use the complete ETL representation pipeline: directional exploration, BDR,
  GMVAE, latent policy, and decoder fine-tuning.
- Learn the representation from the simple source task.
- Disable only the SAR residual with `sar_scale=0.0`.
- Use the existing SB3 PPO target trainer.

### ETL+SAR

- Use the same ETL pipeline, source data, representation bundle, target PPO
  settings, and target budget as ETL-noSAR.
- Extract 20 ICAPCA synergies from successful source-task actions.
- Enable the frozen SAR residual in the target action model.
- Keep ETL as the methodological core; SAR supplies simple-to-complex synergy
  transfer.

### Lattice-official

- Pin upstream `amathislab/lattice` at commit
  `846d02fa993b9b80ce5ecb806463e0a05711bad3`.
- Preserve the upstream MIT license and record every compatibility change.
- Keep the official Lattice covariance/noise mathematics, policy type, network
  architecture, optimizer hyperparameters, and exploration parameters.
- Hand uses the official RecurrentPPO Reorient configuration.
- Leg uses the official SAC locomotion configuration because the upstream
  repository has no MyoSuite MyoLeg entry point.
- Replace only the environment construction and current SB3/Gymnasium API
  surfaces required to run the shared MyoSuite tasks.
- Report Leg as `Lattice-SAC (official locomotion configuration adapted to
  MyoLeg)`, not as an upstream-provided MyoLeg result.

## Upstream Compatibility Boundary

The original Lattice repository pins Python 3.8, SB3 1.6.1,
SB3-Contrib 1.6.0, and MyoSuite 1.2.4. The current project pins Python 3.11,
SB3 2.x, Gymnasium, and MyoSuite 2.12.x. Running different MyoSuite task
implementations would violate the task-fairness requirement.

All three methods therefore run the same MyoSuite 2.12.x task implementation.
The Lattice port may change imports, constructor signatures, recurrent state
plumbing, Gymnasium reset/step handling, serialization, and vector-environment
integration. It must not change covariance construction, noise sampling,
distribution log-probability, official policy architecture, or learning
hyperparameters. An `UPSTREAM.md` file will map vendored files to the pinned
commit and list compatibility-only edits.

## Tasks

### Hand

- Source: `myoHandReorient8-v0`
- Target: `myoHandReorient100-v0`
- Action dimension: 39 muscle activations
- ETL source budget: 1,000,000 transitions
- ETL target budget: 19,000,000 transitions
- Lattice target budget: 20,000,000 transitions

### Leg

- Source: `myoLegWalk-v0`
- Target: `myoLegRoughTerrainWalk-v0`
- Action dimension: 80 muscle activations
- ETL source budget: 1,500,000 transitions
- ETL target budget: 13,500,000 transitions
- Lattice target budget: 15,000,000 transitions

The source and target budgets count actual environment transitions. With a
vectorized environment, one vector step contributes `num_envs` transitions.
Evaluation transitions never count toward the training budget.

## Fairness Protocol

The primary comparison follows the SAR paper's equal-total-samples protocol.

| Domain | Method | Source | Target | Total |
| --- | --- | ---: | ---: | ---: |
| Hand | ETL-noSAR | 1.0M | 19.0M | 20.0M |
| Hand | ETL+SAR | 1.0M | 19.0M | 20.0M |
| Hand | Lattice | 0 | 20.0M | 20.0M |
| Leg | ETL-noSAR | 1.5M | 13.5M | 15.0M |
| Leg | ETL+SAR | 1.5M | 13.5M | 15.0M |
| Leg | Lattice | 0 | 15.0M | 15.0M |

All methods use:

- MyoSuite `>=2.12,<2.13` and the same MuJoCo version and assets.
- Identical environment IDs, reward definitions, termination conditions, and
  episode limits within each domain.
- Seeds `0, 1, 2, 3, 4`.
- The same held-out evaluation seed bank.
- The same checkpoint and final-evaluation schedule.
- No test-set-based hyperparameter selection or checkpoint cherry-picking.

The primary table compares equal total interactions. A supplementary plot also
shows Lattice at the ETL target-only budget checkpoint, but it is not the basis
of the primary conclusion.

## Official Lattice Hyperparameters

Hand retains the official `main_reorient.py` RecurrentPPO settings:

- 16 parallel environments
- batch size 32
- rollout length 128
- learning rate `2.55673e-5`
- entropy coefficient `3.62109e-6`
- clip range 0.3
- gamma 0.99
- GAE lambda 0.9
- max gradient norm 0.7
- value coefficient 0.835671
- 10 epochs
- ReLU policy/value networks with two 256-unit layers
- official launch flags: `use_lattice=True`, `use_sde=False`,
  `freq=1`, `log_std_init=0.0`, and `std_reg=0.0`

These flags select the official default Lattice distribution; no gSDE variant
is enabled for the primary baseline. The policy's `alpha=1` default is retained
but is inactive when `use_sde=False`.

Leg retains the official `main_walker.py` SAC locomotion settings:

- 16 parallel environments
- learning rate `3e-4`
- replay buffer 300,000
- learning starts at 10,000 transitions
- batch size 256
- tau 0.02
- gamma 0.98
- train frequency 8 steps and 8 gradient steps
- automatic entropy coefficient and target entropy
- GELU actor/critic networks with 400 and 300 units
- official launch flags: `use_lattice=True`, `use_sde=False`,
  `freq=1`, `log_std_init=0.0`, and `std_reg=0.0`

These flags select the official default Lattice distribution; no gSDE variant
is enabled for the primary baseline. The policy's `alpha=1` default is retained
but is inactive when `use_sde=False`.

Environment-specific reward shaping from the upstream custom Die Reorient task
or PyBullet Walker is not imported. The shared MyoSuite target environment owns
the reward for all methods.

## Training Scale

- Methods: 3
- Domains: 2
- Seeds: 5
- Formal target runs: 30
- Total formal training interactions: 525 million

Source artifacts are generated independently for each ETL seed, then shared by
the matched ETL-noSAR and ETL+SAR target runs for that seed. This pairing
isolates the SAR switch and reduces source-data variance.

The server runner supports Slurm arrays and a portable shell fallback. Runs are
restartable and write signed stage manifests only after required artifacts pass
integrity checks.

## Checkpoint Correctness

Before formal training, fix the existing checkpoint defect: target learning
fine-tunes the ETL decoder, but current evaluation reloads the original decoder
from `representation_bundle.pt`.

Every ETL checkpoint must save an atomic pair:

- PPO policy checkpoint
- Fine-tuned action-model/decoder checkpoint

The manifest records hashes for both. Evaluation refuses to combine a policy
with an unpaired decoder. Best and latest checkpoints each have their own paired
action model. Lattice checkpoints include policy, recurrent state requirements,
VecNormalize statistics when enabled, and replay-independent evaluation state.

## Evaluation Schedule

- Save and run lightweight evaluation every 250,000 training transitions.
- Use 20 fixed-seed episodes per intermediate checkpoint.
- Evaluate only the final, predeclared checkpoint for the primary final table.
- Hand final evaluation: 500 episodes per seed.
- Leg final evaluation: 100 episodes per seed.
- Use deterministic policy actions during evaluation.
- RecurrentPPO evaluation resets hidden state and episode-start masks at every
  episode boundary.

Evaluation uses a stored seed-bank file so every method sees the same sequence
of Hand objects/goals and Leg terrain randomizations. Training and evaluation
seed banks are disjoint.

## Metrics

### Primary

- Hand: success rate and normalized learning-curve AUC.
- Leg: forward distance and normalized learning-curve AUC.

### Secondary

- Episode return and median return.
- Episode length and termination/fall rate.
- Hand object-drop rate.
- Leg velocity tracking error.
- Mean squared muscle activation as an effort metric.
- Wall-clock time, transitions per second, GPU memory, and CPU utilization.

Leg forward distance is measured from the root free joint's world-frame x
translation. The MuJoCo joint address is resolved by name/model metadata and is
never assumed to be a fixed qpos index.

## Statistical Analysis

- Report per-seed results, mean, standard error, median, and 95% bootstrap
  confidence intervals.
- Pair methods by seed and evaluation seed bank.
- Compare ETL+SAR separately with ETL-noSAR and Lattice.
- Report paired effect sizes and bootstrap confidence intervals for AUC and
  final primary metrics.
- Apply Holm correction to the two primary method comparisons per domain.
- Publish all episodes, not only aggregate summaries.

The method is considered successful only if ETL+SAR exceeds both baselines in
mean normalized AUC in both domains, the paired improvement confidence interval
is above zero, and final performance is not degraded. A Hand result dominated
by isolated high-return episodes does not pass unless success rate and median
performance also improve.

## Code Organization

Planned ownership boundaries:

- `third_party/lattice/`: pinned upstream algorithm files, license, and
  provenance/diff documentation.
- `src/etl_sar/lattice/`: Gymnasium, SB3 2.x, MyoSuite, checkpoint, and metric
  adapters around the vendored algorithm.
- `src/etl_sar/formal/`: experiment matrix, manifests, seed banks, statistics,
  and aggregation.
- `configs/formal_hand.yaml`: Hand method and budget matrix.
- `configs/formal_leg.yaml`: Leg method and budget matrix.
- `scripts/run_formal_server.sh`: portable server runner.
- `scripts/submit_formal_slurm.sh`: Slurm job-array submission.
- `tests/`: upstream parity, adapter, checkpoint, budget, resume, and summary
  tests.

The CLI will expose dry-run inspection before any server job starts. Each run
manifest records git commit, upstream Lattice commit, Python/package versions,
environment fingerprint, method, domain, seed, transition budget, command,
hardware, and artifact hashes.

## Validation Gates

Implementation is not ready for server upload until all gates pass:

1. Original Lattice distribution tests match the pinned implementation on
   fixed tensors and random seeds.
2. Hand RecurrentPPO and Leg SAC complete short MyoSuite smoke runs.
3. ETL policy/decoder paired checkpoint round trips produce identical actions.
4. Transition counters remain correct with 16 vector environments.
5. All three methods receive identical target environment fingerprints.
6. Resume tests continue without resetting counters, seed sequences, or replay
   state required by the method.
7. Formal dry-run expands to exactly 30 target jobs and 10 paired ETL source
   stages.
8. A miniature end-to-end matrix produces the final schema without using its
   scores as formal results.

## Deliverables

- Locally modified repository ready for upload to the server.
- Reproducible dependency/container definitions.
- Official Lattice provenance and compatibility report.
- Formal Hand and Leg launch scripts.
- Restartable checkpoints and run manifests.
- Episode-level CSV/JSON results, aggregate tables, learning curves, and
  statistical comparison outputs.


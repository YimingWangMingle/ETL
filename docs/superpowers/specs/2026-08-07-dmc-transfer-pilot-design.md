# DMC transfer pilot design

## Goal

Replace the short Hand/Leg comparison with two DeepMind Control Suite transfer
chains on which the ETL paper reports a clearer advantage:

- `humanoid/walk -> humanoid/run`
- `dog/walk -> dog/run`

The claim is deliberately narrow: under one matched, single-seed pilot, test
whether adding SAR's source-to-target synergy transfer improves the ETL core and
whether ETL+SAR is competitive with Lattice. This is not a multi-seed
reproduction of the ETL paper.

## Compared methods

All target policies use the same Stable-Baselines3 SAC implementation, network,
optimizer, replay settings, observation normalization, random seed, evaluation
seed bank, and transition accounting.

1. `etl_sar`: ETL directional source exploration, a four-dimensional GMVAE
   action representation, decoder fine-tuning on the target, plus a frozen
   source-task ICA/PCA synergy basis constrained to 20% of the ETL action norm.
2. `etl_no_sar`: the identical source artifact and target pipeline, with only
   the SAR residual scale set to zero.
3. `lattice`: direct target-task SAC from random initialization, using the
   pinned official Lattice state-dependent exploration distribution. It does
   not receive a source artifact or pretrained weights.

Lattice therefore differs from the matched SAC control only in the policy's
exploration distribution. The bundled upstream snapshot and commit provenance
remain authoritative for that distribution.

## Budget and evaluation

- Seed: `0`.
- Each method/domain is charged exactly `1,000,000` environment transitions.
- ETL methods: `200,000` source transitions plus `800,000` target transitions.
- Lattice: `1,000,000` target transitions.
- Target evaluation: every `100,000` target transitions and at the final
  checkpoint, using a disjoint deterministic seed bank.
- Intermediate evaluation: 10 episodes; final evaluation: 50 episodes.
- Primary metric: mean undiscounted episode return.
- Learning metric: trapezoidal return AUC over the complete charged interaction
  budget. ETL curves include the source-cost offset before target evaluation.

Evaluation transitions are not added to the training budget. Observation
normalization statistics are learned only during training and frozen during
evaluation.

## ETL and SAR representation

The ETL latent dimension and GMVAE component count are both four, matching the
best setting reported by the ETL paper's component-count ablation. SAR basis
rank is an independent configuration field and is also four for this short
pilot. The implementation must not change the legacy MyoSuite default of 20.

The DMC source stage stores complete source episodes. GMVAE fitting uses the
source action pool. SAR fitting uses actions from the highest-return source
episodes, making "successful source behavior" explicit without inventing a DMC
`solved` flag. ETL+SAR and ETL-noSAR share the exact same serialized bundle.

## Environment contract

The adapter loads `dm_control.suite`, flattens ordered observation mappings in
their declared order, exposes bounded one-dimensional Gymnasium `Box` actions,
and preserves DMC termination/discount semantics. Action repeat is fixed and
identical across all methods in a domain. Reset and evaluation seeds are
explicit.

## Artifacts and restart behavior

The new protocol writes only below a dedicated output root, defaulting to
`runs/dmc_transfer_pilot`, and never reuses Hand/Leg paths. Each job stores
configuration, policy, replay buffer, observation-normalization state,
transition count, evaluation CSV/JSON files, and a completion manifest. A
repeated launcher invocation skips complete jobs and resumes compatible target
checkpoints.

The launcher runs sequentially and requires no `tmux`. The entire repository is
self-contained for upload; DMC is installed through the `dmc` optional extra.

## Limits

One seed supports a descriptive comparison only. Winning these selected tasks
does not establish universal superiority, statistical significance, or an exact
reproduction of either paper's ten-million-step curves.

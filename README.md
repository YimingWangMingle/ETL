# ETL-SAR: Source-to-Target Action Representation Transfer

This repository is an ETL-dominant reinforcement-learning research prototype. It combines:

- **ETL** for directional source-task exploration, behavior-aware representation learning, a GMVAE action representation, latent-action control, and supervised decoder fine-tuning;
- **SAR-style transfer** for extracting a low-rank PCA/ICA action-synergy basis from high-return source trajectories and applying it as a bounded residual on the target task; and
- **Lattice** as a matched exploration baseline implemented from the official Lattice covariance equations and adapted to Stable-Baselines3 2.x.

The main reproducible experiment is a single-seed DeepMind Control Suite (DMC) walk-to-run pilot on Dog and Humanoid. This is an independent extension, not an official joint release from the ETL, SAR, or Lattice authors.

## Method overview

The source-to-target pipeline is:

1. Train an ETL directional explorer on the easier `walk` source task.
2. Store complete source episodes and train a GMVAE on their actions.
3. Select actions from the highest-return 25% of source episodes.
4. Fit a PCA-to-FastICA synergy basis on those selected actions.
5. Train SAC on the harder `run` target task in the learned ETL latent action space.
6. For ETL+SAR, add a learned residual in the frozen synergy subspace. Its L2 norm is hard-capped at 20% of the ETL decoder output norm.
7. After the initial target phase, fine-tune the ETL decoder with an independent supervised update over observed latent/action pairs.

ETL+SAR and ETL-noSAR load the same serialized source bundle. Their target runs differ only in whether the SAR residual scale is `1.0` or `0.0`.

### Compared methods

| Method | Source stage | Target action space | Exploration | SAR residual |
| --- | --- | --- | --- | --- |
| `etl_sar` | ETL source exploration + GMVAE + PCA/ICA | 4-D ETL latent action | SAC with gSDE | Enabled, capped at 20% |
| `etl_no_sar` | Same source bundle as `etl_sar` | 4-D ETL latent action | SAC with gSDE | Disabled |
| `lattice` | None | Native environment action | SAC with Lattice state-dependent covariance | Not applicable |

All three target methods use the same SAC optimizer settings, two-layer 256-unit policy/value networks, observation normalization, evaluation seeds, action repeat, and total charged interaction budget. The Lattice run uses its native action space because that is part of the method, whereas ETL methods use their learned latent action space.

The DMC Lattice baseline is a controlled adaptation, not a reproduction of an experiment from the Lattice paper: the official distribution equations and policy mechanism are retained, while the environment and shared SAC training protocol are supplied by this repository. See [`third_party/lattice/UPSTREAM.md`](third_party/lattice/UPSTREAM.md) for the pinned upstream commit, file hashes, license, and compatibility boundary.

## DMC experiment protocol

Two source-to-target chains are evaluated with `seed=0`:

| Domain | Source task | Target task | ETL source | ETL target | Lattice target | Charged total per method |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Dog | `walk` | `run` | 200,000 | 800,000 | 1,000,000 | 1,000,000 |
| Humanoid | `walk` | `run` | 200,000 | 800,000 | 1,000,000 | 1,000,000 |

ETL source interactions are charged separately to both ETL methods even though the two runs share one physical source bundle. Evaluation occurs every 100,000 charged transitions using 10 fixed episodes, followed by a 50-episode final evaluation. Across the six method/domain jobs, the comparison declares 6,000,000 charged interactions; the two shared source stages are physically executed once each.

## Requirements

The tested server profile is:

- Ubuntu 22.04;
- Python 3.11;
- one CUDA-capable NVIDIA GPU (an RTX 4090 was used for the pilot);
- 16 CPU cores and 64 GB RAM are sufficient;
- at least 100 GB of free disk space is recommended for environments, replay buffers, checkpoints, TensorBoard logs, and results.

Training is GPU-enabled, but DMC simulation remains CPU-bound enough that GPU utilization may fluctuate. Runtime varies substantially by CPU, storage, driver, and evaluation speed; plan for several hours and allow more than 10 hours until timing has been measured on the target server.

## Installation on a clean Ubuntu server

Clone the repository and enter its root:

```bash
git clone https://github.com/YimingWangMingle/ETL.git
cd ETL
```

Install the headless rendering libraries:

```bash
apt-get update
apt-get install -y libegl1 libgl1 libglfw3 libglew2.2
```

Create an isolated Python 3.11 environment:

```bash
conda create -n etl-dmc python=3.11 pip -y
conda activate etl-dmc
```

Install a CUDA build of PyTorch and the project with DMC support:

```bash
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[dmc,test]"
```

Use EGL for headless MuJoCo rendering and verify the environment:

```bash
export MUJOCO_GL=egl
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python -c "from dm_control import suite; env=suite.load('humanoid', 'walk'); print(env.reset())"
```

The training script intentionally refuses to start without a CUDA device.

## Reproduce the DMC pilot

Keep all commands in the repository root and keep the same `MUJOCO_GL` value in every shell.

### 1. Inspect the job matrix without training

```bash
conda activate etl-dmc
export MUJOCO_GL=egl
bash scripts/run_dmc_transfer_pilot.sh dry-run
```

The output should list two shared source stages and six target jobs: three methods for Dog and three for Humanoid.

### 2. Start the complete sequential run

Choose a persistent output directory with sufficient free space:

```bash
export DMC_OUTPUT_ROOT=/root/autodl-tmp/dmc_transfer_pilot
bash scripts/run_dmc_transfer_pilot.sh run
```

The script runs one job at a time. Re-running the same command skips completed stages and resumes incomplete target SAC jobs from the latest policy, replay buffer, action model, and observation-normalization checkpoint. An interrupted source stage is not checkpoint-resumable; avoid stopping during either initial source stage.

### 3. Run stages manually when needed

The source and target indices are printed by `dry-run`:

```bash
bash scripts/run_dmc_transfer_pilot.sh source 0
bash scripts/run_dmc_transfer_pilot.sh target 0
```

Manual stage commands use the same `DMC_OUTPUT_ROOT`. A target ETL job requires its corresponding source bundle to be complete.

### 4. Rebuild the comparison summary

```bash
bash scripts/run_dmc_transfer_pilot.sh aggregate
```

Aggregation requires every method's final evaluation and full checkpoint curve.

## Outputs

The main output tree is:

```text
$DMC_OUTPUT_ROOT/
|-- sources/
|   |-- dog-source-seed0/
|   `-- humanoid-source-seed0/
|-- jobs/
|   |-- dog-etl_sar-seed0/
|   |-- dog-etl_no_sar-seed0/
|   |-- dog-lattice-seed0/
|   |-- humanoid-etl_sar-seed0/
|   |-- humanoid-etl_no_sar-seed0/
|   `-- humanoid-lattice-seed0/
`-- aggregate/
    |-- results.csv
    `-- summary.json
```

Each target job contains resumable `latest_*` artifacts, periodic checkpoints, TensorBoard logs, per-episode evaluation CSV files, and evaluation summaries.

The aggregate CSV reports:

- `return_auc`: trapezoidal area under the checkpoint mean-return curve divided by the total charged interaction budget. ETL curves include the 200,000-transition source cost as a zero-return prefix;
- `final_mean_return`: mean undiscounted episode return over the fixed 50-episode final evaluation.

Higher is better for both metrics. Because DMC return scales differ between Dog and Humanoid, compare methods within a domain rather than comparing raw values across domains.

## Reference single-seed result

The following values came from one completed server run of the checked-in pilot configuration. They are provided only as a pipeline reference, not as a paper-level benchmark claim.

| Domain | Method | Return AUC | Final mean return |
| --- | --- | ---: | ---: |
| Dog | `etl_no_sar` | 3.3235 | 5.4493 |
| Dog | `etl_sar` | 9.2942 | 12.6263 |
| Dog | `lattice` | 89.9359 | 10.5048 |
| Humanoid | `etl_no_sar` | 0.8764 | 0.9052 |
| Humanoid | `etl_sar` | 0.8940 | 1.0921 |
| Humanoid | `lattice` | 121.2435 | 169.8061 |

In this run, enabling SAR improved ETL-noSAR on both reported metrics in both domains. ETL+SAR did not outperform Lattice overall: it achieved a higher final Dog return, while Lattice had much higher return AUC on both domains and much higher final Humanoid return.

## Tests

The fast suite excludes tests that require local MyoSuite assets:

```bash
python -m pytest -m "not myo" -q
```

The focused DMC checks are:

```bash
python -m pytest tests/test_dmc_config.py tests/test_dmc_env.py tests/test_dmc_protocol.py tests/test_lattice_policies.py -q
```

The repository also contains MyoSuite Hand/Leg pilot and formal multi-seed infrastructure. Install `.[myosuite,test]`, then use the scripts and configs under `scripts/` and `configs/`. Those protocols are retained for research extension but are not the primary reproduction path documented here.

## Repository layout

```text
configs/                 Experiment configurations
scripts/                 DMC, MyoSuite, server, and aggregation entry points
src/etl_sar/             ETL, SAR, DMC, Lattice, and evaluation implementation
tests/                   Unit, protocol, resume, and optional environment tests
third_party/lattice/     Pinned official Lattice source snapshot and MIT license
docs/superpowers/        Design specifications and implementation plans
```

## Scope and limitations

- The DMC pilot uses only one training seed, so it provides no confidence interval or statistical-significance test.
- Its 1M-transition budget is a reduced pilot, not a reproduction of the papers' longer multi-seed experiments.
- ETL and SAR are reimplemented from their paper logic in a Stable-Baselines3 training stack; this repository does not include or wrap the original ETL Ray code.
- The Lattice equations are pinned and parity-tested against the official repository, but the DMC environment/protocol is this repository's matched adaptation.
- Hyperparameter conclusions from this pilot should not be generalized to all environments.

## Lattice provenance and licenses

Lattice is from [amathislab/lattice](https://github.com/amathislab/lattice) and accompanies the paper [*Latent Exploration for Reinforcement Learning*](https://arxiv.org/abs/2305.20065). The vendored snapshot is pinned to commit `846d02fa993b9b80ce5ecb806463e0a05711bad3`; its MIT license is preserved in [`third_party/lattice/LICENSE`](third_party/lattice/LICENSE).

The repository-level license is available in [`LICENSE`](LICENSE).

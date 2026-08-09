# ETL

This repository is an ETL-dominant reinforcement-learning research prototype. It combines:

- **ETL** for directional source-task exploration, behavior-aware representation learning, a GMVAE action representation, latent-action control, and supervised decoder fine-tuning;


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

## Licenses

The repository-level license is available in [`LICENSE`](LICENSE).

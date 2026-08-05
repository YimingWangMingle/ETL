# SAR-guided ETL

这是一个以 ETL 为算法主干的新实现：ETL 负责方向性探索、BDR、GMVAE、潜在动作策略和 Eq. 10 decoder fine-tuning；SAR 从简单任务成功轨迹提取 20 个 ICAPCA synergies，并在复杂任务中作为冻结的低秩动作修正。正式实验另包含固定 commit 的官方 Lattice 算法基线。

## 方法边界

- `ExploreTrainer`：每个 episode 采样一个 20 维单位方向，以 ETL Eq. 1 的内积 bonus 引导完整肌肉动作 PPO，训练防坍缩 BDR encoder，并从增长中的 action buffer 交替更新 GMVAE。
- `SynergyArtifact`：仅使用源任务成功轨迹，执行 StandardScaler -> PCA(20) -> FastICA(20)，将控制系数归一化到 `[-1,1]`。
- `ETLSARActionModel`：ETL decoder 是动作主输出；SAR residual 的 L2 范数被硬限制为 ETL 输出的 20%。`enabled_scale=0` 是同代码 SB3-ETL 内部基线。
- `TransferTrainer`：SB3 PPO 只在 20 维 ETL latent action space 中训练；环境 wrapper 负责解码。冻结期后，decoder 使用交互 `(z_t,a_t)` 对执行独立的 ETL Eq. 10 监督更新，不接收 PPO 梯度。

BDR 的低奖励差正则使用有界 hinge 实现论文文字中的 anti-collapse 语义，避免直接最大化无界距离。

## 安装

目标环境为 Python 3.11：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test,myosuite]"
```

## 快速协议

```powershell
etl-sar inspect --config configs/hand_quick.yaml
etl-sar explore --config configs/hand_quick.yaml --run-dir runs/hand_source --timesteps 100000
etl-sar fit-representation --config configs/hand_quick.yaml --data-dir runs/hand_source/data --explore-checkpoint runs/hand_source/representation.pt --output-dir runs/hand_representation
etl-sar transfer --config configs/hand_quick.yaml --bundle runs/hand_representation/representation_bundle.pt --run-dir runs/hand_target --timesteps 100000
etl-sar evaluate --config configs/hand_quick.yaml --bundle runs/hand_representation/representation_bundle.pt --pair-manifest runs/hand_target/best_pair.json --output-dir runs/hand_eval --episodes 20 --environment-steps 100000
```

Leg 使用 `configs/leg_quick.yaml`，泛化链为 `myoLegWalk-v0 -> myoLegRoughTerrainWalk-v0`。

## 最小 Hand + Leg 泛化试验

```powershell
.\scripts\run_minimal_pilot.ps1 -WhatIf
.\scripts\run_minimal_pilot.ps1
```

如果 `etl-sar` 不在 `PATH` 中：

```powershell
.\scripts\run_minimal_pilot.ps1 -EtlSar .\.venv\Scripts\etl-sar.exe
```

输出写入 `runs/minimal_pilot`。每个阶段使用包含配置 SHA-256、预算、种子和命令的 `stage.complete.json`，失败后可直接重新执行同一命令。基线使用 `--sar-scale 0.0`，extension 使用 `--sar-scale 1.0`。最小试验只用于初步判断，不代表完整收敛或统计显著性结论。

## 正式 ETL / SAR / Lattice 对比

正式配置固定为 5 个种子、30 个目标任务和 525M 归属交互量：

```bash
python -m etl_sar.formal.server dry-run > formal_matrix.json
```

服务器环境：

```bash
conda env create -f environment-server.yml
conda activate etl-lattice-sar
python -m pip install -e ".[myosuite,test]"
```

先确认服务器 CUDA 与 MyoSuite 冒烟测试，再提交 Slurm：

```bash
python -m pytest -m myo -q
bash scripts/run_formal_server.sh dry-run
bash scripts/submit_formal_slurm.sh
```

没有 Slurm 时按索引执行。源数组为 `0..9`，完成后运行目标数组 `0..29`：

```bash
for index in $(seq 0 9); do bash scripts/run_formal_server.sh source "$index"; done
for index in $(seq 0 29); do bash scripts/run_formal_server.sh target "$index"; done
bash scripts/aggregate_formal.sh
```

可设置 `FORMAL_OUTPUT_ROOT=/path/to/results` 改变输出位置。重复同一命令会验证完成 manifest；中断任务从 `latest` policy、VecNormalize、ETL action model 和（Leg SAC）replay buffer 恢复。

正式协议的关键约束：

- Hand：ETL 为 1M source + 19M target，Lattice 为 20M target；最终每 seed 评估 500 episodes。
- Leg：ETL 为 1.5M source + 13.5M target，Lattice 为 15M target；最终每 seed 评估 100 episodes。
- 所有方法每 250k transitions 用同一固定 seed bank 评估 20 episodes。
- `ETL-noSAR` 与 `ETL+SAR` 按 domain/seed 共享同一个源 bundle，目标阶段只改变 SAR scale。
- Hand Lattice 保留官方 RecurrentPPO Reorient 配置；Leg 标记为官方 SAC locomotion 配置适配 MyoLeg。
- 中间 SAC 快照不复制 replay buffer；只有可恢复的 `latest_replay_buffer.pkl` 保留。
- 成功要求 ETL+SAR 在 Hand 和 Leg 的 normalized AUC 上都超过两个 baseline，配对 bootstrap CI 下界大于 0，且最终主指标不退化。

官方 Lattice 源码、MIT 许可证、固定 commit 和兼容性边界记录在 `third_party/lattice/UPSTREAM.md`。

## 测试

```powershell
python -m pytest -m "not myo" -q
python -m pytest -m myo -q
```

第一条验证数学、数据隔离、GMVAE、SAR 硬门控、Lattice 对等性、配对 checkpoint、正式矩阵和统计；第二条要求本机安装 MyoSuite/MuJoCo 和对应任务资产。

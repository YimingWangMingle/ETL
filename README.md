# SAR-guided ETL

这是一个以 ETL 为算法主干的新实现：ETL 负责方向性探索、BDR、GMVAE、潜在动作策略和 Eq. 10 decoder fine-tuning；SAR 从简单任务成功轨迹提取 20 个 ICAPCA synergies，并在复杂任务中作为冻结的低秩动作修正；Lattice 只贡献 Stable-Baselines3 风格的 Trainer、callback、checkpoint 和日志结构。

## 方法边界

- `ExploreTrainer`：每个 episode 采样一个 20 维单位方向，以 ETL Eq. 1 的内积 bonus 引导完整肌肉动作 PPO，训练防坍缩 BDR encoder，并从增长中的 action buffer 交替更新 GMVAE。
- `SynergyArtifact`：仅使用源任务成功轨迹，执行 StandardScaler → PCA(20) → FastICA(20)，将控制系数归一化到 `[-1,1]`。
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
etl-sar evaluate --config configs/hand_quick.yaml --bundle runs/hand_representation/representation_bundle.pt --model-path runs/hand_target/best_model.zip --output-dir runs/hand_eval --episodes 20 --environment-steps 100000
```

Leg 使用 `configs/leg_quick.yaml`，泛化链为 `myoLegWalk-v0 -> myoLegRoughTerrainWalk-v0`。

同预算对比：

```powershell
etl-sar compare --baseline runs/baseline_eval/summary.json --extension runs/hand_eval/summary.json --output runs/comparison.json
```

## 最小 Hand + Leg 泛化试验

先查看两域所有阶段和参数，不启动训练或创建运行目录：

```powershell
.\scripts\run_minimal_pilot.ps1 -WhatIf
```

确认后运行最小试验：

```powershell
.\scripts\run_minimal_pilot.ps1
```

如果 `etl-sar` 不在 `PATH` 中，可显式指定虚拟环境入口：

```powershell
.\scripts\run_minimal_pilot.ps1 -EtlSar .\.venv\Scripts\etl-sar.exe
```

输出写入 `runs/minimal_pilot`。每个阶段使用包含配置 SHA-256、预算、种子和命令的 `stage.complete.json`；只有签名匹配且预期产物仍存在时才会跳过，因此失败后可直接重新执行同一命令。

本试验对 Hand 和 Leg 分别复用同一个源数据与 representation bundle。基线使用 `--sar-scale 0.0`，extension 使用 `--sar-scale 1.0`；两者都训练 20,000 个目标环境步，并以相同种子确定性评估 10 回合。只有两个域都满足平均回报提升且成功率不下降，`pilot_summary.json` 的 `pilot_positive` 才为 `true`。该结果只用于最小工作量的初步判断，不代表完整收敛或统计显著性结论。

旧 ETL-Ray 结果只能通过 `-LegacyReference <json>` 作为外部参考传入。环境 ID、MyoSuite 协议和指标不完全一致时，摘要会标记 `comparable=false`，不会计算跨协议差值。

## 测试

```powershell
python -m pytest -m "not myo" -v
python -m pytest -m myo -v
```

第一条验证数学、数据隔离、GMVAE、SAR 硬门控、Explore/Transfer SB3 生命周期和评估不可变性；第二条要求本机安装 MyoSuite/MuJoCo 和对应任务资产。

# ETL 主导的 SAR-Guided Extension 设计规格

## 1. 目标

实现一个以 ETL 为算法核心、使用 SAR 协同迁移机制增强、采用 Lattice 风格 Stable-Baselines3 工程结构的新扩展。项目从论文设计重新实现，不依赖旧 ETL/Ray 源代码。

方法职责边界：

- ETL 负责行为度量方向探索、BDR、动作数据收集、GMVAE 非线性表征、潜在动作控制和解码器微调。
- SAR 负责从简单任务成功轨迹中提取肌肉协同，并将固定协同基底迁移到复杂任务。
- Lattice 只贡献 Stable-Baselines3 的 Trainer、向量环境、回调、检查点、日志和配置组织方式；不使用 Lattice 的噪声模型或概率分布创新。

工作名称为 **SAR-guided ETL**。SAR 对 ETL 提供辅助先验，不是主干算法。

## 2. 范围与非目标

### 2.1 范围

- Python 3.11。
- MyoSuite 2.12.x、Gymnasium、Stable-Baselines3 2.x、PyTorch 和 scikit-learn。
- Hand 与 Leg 两条完全隔离的训练管线。
- Hand 主路径：Reorient8 到 Reorient100，并评估 ID/OOD；RealWorldObjs 为可选扩展评估。
- Leg 主路径：Flat Walk 到 Uneven；Hilly、Stair、Diagonal 作为可配置扩展任务。
- 支持短训练、断点恢复、确定性评估、CSV/TensorBoard 日志和 best checkpoint。

### 2.2 非目标

- 不复刻或维护旧 Ray 工程。
- 不进行 SAR 论文规模的全面多任务、多种子训练。
- 不将 Lattice 的算法组件作为新方法贡献。
- 不允许使用复杂任务测试数据重新拟合 PCA/ICA。
- 不把单个训练种子的最高分表述为统计显著结论。

## 3. 总体架构

简单任务上的 ETL Explorer 先生成具有行为方向和多样性的轨迹。完整有效动作形成 ETL action pool；其中成功或接近成功的高质量轨迹形成 SAR success pool。

ETL action pool 用于训练 GMVAE。SAR success pool 经 PCA 和 ICA 得到肢体专属的固定协同基底 `W_SAR`。复杂任务策略输出 ETL 潜变量，ETL 解码器产生主要动作均值，SAR 分支只产生受限的低秩修正。

动作均值定义为：

\[
\mu_a = \Pi_{\mathcal A}\left(
D_\theta^{ETL}(z)+\Delta a_{SAR}
\right)
\]

其中 `Pi_A` 使用运行环境 `action_space.low/high` 投影到合法动作域，不硬编码动作上下界。

- `D_ETL` 是完整 GMVAE 非线性解码器，是动作主输出。
- `W_SAR` 是从简单任务提取并在复杂任务中冻结的协同基底。
- `g_psi` 预测协同系数。
- `lambda` 控制 SAR 分支启用程度，默认启用值为 `1.0`，`0.0` 只用于内部消融。SAR 实际贡献由硬范数门控限制。

## 4. 组件边界

### 4.1 Environment Adapter

统一 MyoSuite/Gymnasium 接口，显式验证环境 ID、观测维度、动作维度、动作范围、success 字段、termination/truncation 语义、对象集合和地形类型。不存在或不兼容的环境必须报错，不允许静默替换。

### 4.2 ExploreTrainer

使用 SB3 PPO 实现 ETL 简单任务探索。它负责 BDR、行为描述量、方向性探索、向量环境、轨迹记录和 best checkpoint。算法和奖励语义遵守 ETL 论文；框架从 Ray 替换为 SB3。

### 4.3 Dataset Builder

每个时间步保存 observation、策略采样动作、环境实际执行动作、环境奖励、next observation、terminated、truncated、行为描述量及其变化量、success 和任务元数据。

数据构建器产生：

- ETL action pool：全部通过合法性检查的探索动作。
- SAR success pool：成功或达到配置质量阈值的完整轨迹。

Hand 和 Leg 的数据目录、归一化统计量和元数据必须物理隔离。

### 4.4 SynergyExtractor

对 SAR success pool 的肌肉激活进行训练集拟合的标准化，然后依次执行 PCA 和 FastICA。组件数依据简单任务验证集确定。输出包括 `W_SAR`、标准化统计量、解释方差、组件数、肢体类型、源任务和数据指纹。

投影矩阵使用：

\[
P_{SAR}=W_{SAR}W_{SAR}^{\dagger}
\]

其中 `dagger` 为数值稳定的 Moore-Penrose 伪逆。

### 4.5 RepresentationTrainer

使用 PyTorch 离线训练 ETL GMVAE，再训练零初始化的 SAR 低秩辅助头。保存编码器、ETL 解码器、混合先验参数、SAR 头、动作归一化统计量和训练配置。

### 4.6 TransferTrainer

使用自定义 SB3 PPO policy。策略从 observation 产生潜变量 `z`，再通过 ETL 解码器和 SAR 辅助头得到完整动作分布的均值。PPO rollout buffer 保存策略采样动作及其 log-probability；环境适配器另行记录投影到合法动作域后的实际执行动作。解码器位于 policy 均值计算图内并可按阶段微调。

### 4.7 EvaluationRunner

只加载固定 checkpoint，不允许优化器更新。输出逐 episode 结果、聚合指标、动作分支范数和任务元数据。

## 5. 数据流与迁移协议

### 5.1 Hand

1. 在 Reorient8 上进行 ETL 探索。
2. 建立 Hand ETL action pool 和 SAR success pool。
3. 训练 Hand GMVAE 并提取 Hand `W_SAR`。
4. 在 Reorient100 上训练潜在动作 PPO。
5. 使用固定 checkpoint 评估 ID/OOD；RealWorldObjs 通过独立配置启用。

### 5.2 Leg

1. 在 Flat Walk 上进行 ETL 探索。
2. 建立 Leg ETL action pool 和 SAR success pool。
3. 训练 Leg GMVAE 并提取 Leg `W_SAR`。
4. 在 Uneven 上训练潜在动作 PPO。
5. Hilly、Stair、Diagonal 使用相同接口作为后续迁移配置。

### 5.3 复杂任务冻结与微调

1. 加载源任务 GMVAE、`W_SAR` 和归一化统计量。
2. 永久冻结 `W_SAR`。
3. 初期冻结 ETL 解码器，训练新的复杂任务潜在策略。
4. 达到配置步数后，以较小学习率解冻 ETL 解码器。
5. SAR 系数头允许以较小学习率更新。
6. 性能回退时保留并恢复 best checkpoint，不覆盖最佳模型。

## 6. 损失函数

### 6.1 ETL GMVAE

基础表征先按 ETL 训练：

\[
\mathcal L_{ETL}=\mathcal L_{rec}
+\beta_{KL}\mathcal L_{KL}
+\beta_{mix}\mathcal L_{mixture}
\]

\[
\mathcal L_{rec}=\|a-D_\theta(z)\|_2^2
\]

`L_mixture` 包含 GMVAE 混合先验所需的类别/聚类项。其展开与 ETL 论文定义保持一致，并由单元测试验证各项均参与反向传播。

### 6.2 SAR 辅助分支

先计算未经门控的修正：

\[
\Delta a_{raw}=\lambda W_{SAR}g_\psi(z)
\]

再执行逐样本硬范数门控：

\[
s=\min\left(
1,
\frac{\rho\|D_\theta^{ETL}(z)\|_2}
{\|\Delta a_{raw}\|_2+\epsilon}
\right),\qquad
\Delta a_{SAR}=s\Delta a_{raw}
\]

因此 SAR 修正的范数不能超过 ETL 主输出的 `rho` 比例。SAR 分支拟合 ETL 重建残差中位于协同子空间的部分：

\[
\mathcal L_{SAR}=\left\|
P_{SAR}(a-D_\theta^{ETL}(z))-\Delta a_{SAR}
\right\|_2^2
\]

软预算损失为：

\[
\mathcal L_{budget}=\max\left(
0,
\frac{\|\Delta a_{raw}\|_2}
{\|D_\theta^{ETL}(z)\|_2+\epsilon}-\rho
\right)^2
\]

默认使用 `rho=0.20`。`rho=0.25` 只作为简单任务验证或消融候选，不根据复杂任务测试结果选择。硬门控保证上限，软损失减少长期触发门控造成的梯度饱和。

联合表征损失为：

\[
\mathcal L_{rep}=\mathcal L_{ETL}
+\alpha\mathcal L_{SAR}
+\gamma\mathcal L_{budget}
\]

### 6.3 复杂任务 PPO

PPO policy 的完整动作均值由 ETL 解码器和 SAR 修正共同产生：

\[
o\rightarrow\pi_\omega\rightarrow z
\rightarrow\mu_a
\rightarrow\mathcal N(\mu_a,\sigma_a)
\rightarrow a
\]

基础目标为标准 PPO clipped objective、value loss 和 entropy bonus。解冻解码器后加入预训练参数锚定：

\[
\mathcal L_{anchor}=\|\theta-\theta_0\|_2^2
\]

\[
\mathcal L_{transfer}=\mathcal L_{PPO}
+\eta\mathcal L_{anchor}
+\gamma\mathcal L_{budget}
\]

`W_SAR` 不进入优化器。ETL 解码器、SAR 头和其余 policy 参数使用独立 optimizer parameter groups。

## 7. 训练顺序

1. 探索简单任务并构建两种数据视图。
2. 训练纯 ETL GMVAE，保存不可变的基础 checkpoint。
3. 从成功轨迹拟合 PCA+ICA，保存固定 `W_SAR`。
4. 零初始化 SAR 辅助头，训练受预算约束的低秩修正。
5. 复杂任务 PPO 初期冻结 ETL 解码器。
6. 按配置解冻解码器并降低其学习率。
7. 定期确定性评估，分别保存 best 和 latest checkpoint。

## 8. 实验与评价

### 8.1 比较组

- ETL 论文中的 Ray 结果：只作为 reported reference。
- SB3-ETL：同一代码设置 `lambda=0`，作为同预算内部基线。
- SAR-guided ETL：启用固定 SAR 基底和低秩辅助头。

`lambda=0` 不需要旧 Ray 源码，也不维护第二套实现。

### 8.2 快速验证

- 开发阶段：Hand 和 Leg 各一个训练种子。
- 确认阶段：Reorient100 与 Uneven 各运行三个受限预算训练种子。
- 所有方法使用相同环境步数、网络规模、评估种子和 episode 数。

### 8.3 指标与判据

- episode return 与 success rate。
- 达到目标成功率所需环境步数。
- 训练曲线 AUC。
- best/final checkpoint 指标。
- Hand ID/OOD 成功率。
- Leg 行走距离、跌倒率和任务成功率。
- `norm(delta_a_SAR) / norm(D_ETL(z))`。

快速成功判据：SAR-guided ETL 在相同预算下提高 SB3-ETL 的平均 success rate 或 AUC，Hand 和 Leg 各至少一个主要指标提升，任一领域不得发生明显崩溃，且 SAR 修正保持在预算范围内。

## 9. 测试与故障处理

### 9.1 单元测试

- PCA、ICA、伪逆投影的 shape、dtype 和数值性质。
- `W_SAR` 在复杂任务训练中参数不变化。
- `lambda=0` 时 SAR 分支对动作严格无影响。
- SAR 修正满足硬贡献预算。
- GMVAE 编码/解码和混合先验 loss 可反向传播。
- policy sampled action、log-probability 和 rollout buffer 一致；executed action 单独记录且满足环境边界。
- 冻结/解冻和 optimizer parameter groups 正确。
- checkpoint round-trip 输出一致。
- Hand/Leg 数据与 checkpoint 不可交叉加载。
- 复杂任务数据不进入 PCA/ICA 拟合。

### 9.2 集成测试

- 每个源任务完成短 rollout 和数据落盘。
- 每个主复杂任务完成短 PPO learn/evaluate 循环。
- callback 正确生成 latest/best checkpoint、CSV 和 TensorBoard 数据。
- 从中断 checkpoint 恢复后继续累计环境步数。

### 9.3 硬失败条件

环境不存在、shape 不兼容、肢体类型不匹配、checkpoint 版本不兼容、NaN/Inf、评估时参数变化均必须终止并给出明确错误。不得静默裁剪维度、替换环境或忽略损坏 checkpoint。

## 10. 可复现性与产物

每次运行保存解析后的完整配置、Git commit、依赖版本、训练和评估随机种子、数据指纹、CSV、TensorBoard 日志及 checkpoint。输出目录必须包含 source task、target task、limb、method、seed 和时间戳，避免实验互相覆盖。

## 11. 验收条件

- 项目能够在干净的 Python 3.11 环境安装。
- Hand 和 Leg 主路径可分别从探索运行到评估。
- 自动测试全部通过。
- 快速实验可在相同预算下生成 SB3-ETL 与 SAR-guided ETL 对比表。
- 日志能证明 ETL 解码器是主要动作来源，`W_SAR` 来自简单任务且在复杂任务中保持冻结。
- README 明确区分 ETL 核心、SAR 迁移机制和 Lattice/SB3 工程结构。

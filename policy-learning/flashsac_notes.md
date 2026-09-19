# FlashSAC 算法笔记

Kim et al., *FlashSAC: Fast and Stable Off-Policy Reinforcement Learning for High-Dimensional Robot Control*（[arXiv:2604.04539](https://arxiv.org/abs/2604.04539), 2026）。项目页：<https://holiday-robot.github.io/FlashSAC>。  
前置：[`sac_notes.md`](sac_notes.md)（soft Bellman 备份、双 $Q$、自动温度调节）。本文不重复最大熵框架的推导，仅阐述相对 SAC 的方法改动。

---

## 一、算法定位

FlashSAC 以 Soft Actor-Critic（SAC）为骨架，面向高维机器人控制（灵巧操作、人形 locomotion、视觉策略）的 off-policy Actor-Critic 学习。PPO 在低维任务且仿真吞吐极高时通常稳定且足够；当状态–动作维数升高，on-policy 数据的支撑集过窄，样本丢弃直接体现为 wall-clock 时间的浪费。标准 off-policy 方法虽可复用经验回放，但在宽分布上拟合 bootstrapped $Q$ 需大量梯度步，逼近误差经 Bellman 备份放大，导致训练缓慢且不稳定。

FlashSAC 的核心主张：在固定算力预算下，以更大网络、更大 batch、更低更新–数据比（UTD）换取 wall-clock 效率；同时以结构约束限制权重、激活与梯度范数，使容量扩展不致破坏 critic 稳定性；并以与动作维数成比例的目标熵与噪声重复机制补充探索。

| 特性 | PPO | SAC | FlashSAC |
|------|-----|-----|----------|
| 类型 | On-policy 策略梯度 | Off-policy Actor-Critic | 同 SAC |
| 目标 | 期望回报 | 回报 + 熵正则 | 同 SAC |
| 数据 | 当前策略，样本即用即弃 | 经验回放 | 更大池 + 大规模并行仿真 |
| 网络 | 中小 MLP 常见 | 小 MLP（约 0.2–0.5M） | 约 2.5M，六层 inverted residual |
| 更新密度 | 每批 on-policy 数据多 epoch | UTD 通常较高 | GPU 设定 UTD = 2/1024 |
| 稳定手段 | Clip + GAE | 双 Q、目标网络、$\alpha$ | 另加 BN / RMSNorm / 权重归一 / 分布 Q |
| 典型场景 | 低维、高吞吐仿真 | 连续控制通解 | 高维操纵与人形 sim-to-real |

相对 FastTD3 / FastSAC：后者 wall-clock 快但网络约 0.2M，渐近回报受限。FlashSAC 将容量扩展与范数约束耦合，使大网络可训。

---

## 二、问题背景

### 2.1 On-policy 方法的局限

策略评估依赖当前 $\pi$ 的窄支撑。高维连续动作空间中，重要性采样方差过大，难以有效复用旧数据。仿真成本升高（接触动力学、视觉编码、大策略前向）时，on-policy 样本的即时丢弃转化为显著的 wall-clock 开销。

### 2.2 Off-policy critic 的困难

经验回放上的 bootstrapped 损失（标准 TD，尚未写入熵项）为

$$
\mathcal{L}_{Q}=\mathbb{E}_{(s,a,r,s')\sim\mathcal{D}}\Big[\big(Q_{\theta}(s,a)-(r+\gamma Q_{\theta}(s',a'))\big)^{2}\Big],
\quad a'\sim\pi(\cdot\mid s')
$$

宽状态–动作覆盖要求多次梯度更新方能拟合；目标 $y$ 依赖网络自身预测，逼近误差与外推误差沿备份传递（deadly triad）。网络容量越大，该放大效应越显著。

### 2.3 探索不足

仅依赖最大熵目标，往往不足以在高维动作空间形成时间上连贯的探索轨迹。

FlashSAC 分别通过：降低 UTD 并扩展数据与模型规模；约束 critic 更新动力学；统一目标熵与噪声重复，应对上述三方面问题。

---

## 三、保留的 SAC 骨架

MDP $\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma)$，连续动作空间。经验回放 $\mathcal{D}$ 存储 $(s,a,r,s')$。双 critic $Q_{\phi_1},Q_{\phi_2}$，目标网络软更新

$$
\bar\phi_j\leftarrow\tau\phi_j+(1-\tau)\bar\phi_j,\qquad j\in\{1,2\}
$$

策略损失与 soft 目标（与 [`sac_notes.md`](sac_notes.md) 第五节一致）：

$$
\mathcal{L}_{\pi}(\theta)=\mathbb{E}_{s\sim\mathcal{D},\,a\sim\pi_{\theta}}\big(\alpha\log\pi_{\theta}(a\mid s)-\min_{i}Q_{\phi_i}(s,a)\big)
$$

$$
y=r+\gamma\big(\min_{j}Q_{\bar\phi_j}(s',a')-\alpha\log\pi_{\theta}(a'\mid s')\big),\qquad a'\sim\pi_{\theta}(\cdot\mid s')
$$

标量 critic 情形下 $\mathcal{L}_{Q}(\phi_i)=\mathbb{E}[(Q_{\phi_i}(s,a)-y)^{2}]$。FlashSAC 将 $Q$ 改为分布输出，以交叉熵拟合投影后的 Bellman 目标（第五节）；$\min$ 算子与 $-\alpha\log\pi$ 的结构保持不变。

---

## 四、效率：数据吞吐、容量与低 UTD

动机来自监督学习的 scaling law：固定算力下，大模型 + 大 batch + 少步更新往往优于小模型上的高频更新。Off-policy 设定中直接放大容量易致不稳定，故必须与第五节稳定性设计联用。

**并行仿真。** 默认 1024 个环境并行采集，以覆盖高维状态–动作空间；经典 SAC/TD3 通常仅使用少量并行环境。

**大容量回放。** 池容量至 $10^7$ 量级（常见 off-policy 为 $10^6$），减轻长尾转移被覆盖及分布外推。消融表明 $10^7$ 有利于稳定；$5\times 10^7$ 会稀释近期优质样本，wall-clock 变慢，渐近性能或略升。

**大模型、大 batch、低 UTD。** Actor 与 critic 各约 2.5M 参数、六层。Batch size 2048（接近 GPU 显存上限）。更新–数据比

$$
\mathrm{UTD}=\frac{2}{1024}
$$

即每新增 1024 条转移仅作 2 次梯度更新，配合较大学习率。PyTorch JIT 与混合精度进一步降低约 5%–10% wall-clock。

CPU 单环境、样本为瓶颈时：batch 改为 512，UTD = 1，其余设计不变。

---

## 五、稳定性：约束权重、特征与梯度范数

Bellman 备份将 $s'$ 上的误差传回当前目标。FlashSAC 通过一组结构约束，使参数、激活与梯度范数在训练过程中保持有界，并降低 critic 损失的条件数（消融中逐项加入后条件数单调下降）。

### 5.1 Inverted residual 与 RMSNorm

主干为堆叠 inverted residual block（先扩张维数再投影回原维并加残差，结构类 Transformer FFN / MobileNet 瓶颈）。最后一块之后对每个样本作 RMSNorm，限制进入价值头的特征范数，避免分布外输入产生无界激活并破坏备份稳定性。

### 5.2 预激活 Batch Normalization

回放数据由演变中的行为策略混合而成，输入分布非平稳。非线性层之前使用 BN，减轻 dead ReLU 与梯度退化。相对 LayerNorm，大 batch 上 BN 的 running 统计来自多样本回放，经验上损失曲面更平滑、有效条件数更低。

### 5.3 Cross-batch 价值预测

BN 按 mini-batch 估计均值与方差。若当前 $Q(s,a)$ 与目标 $Q(s',a')$ 分两次前向，两侧归一化统计不一致。将当前转移与下一状态拼成**同一 batch** 前向（CrossQ 做法），使 Bellman 两侧共享 BN 统计量。

### 5.4 分布 critic 与奖励缩放

将 $Q$ 表示为 $[G_{\min},G_{\max}]$ 上均匀放置的 $n_{\mathrm{atom}}$ 个支撑点上的 categorical 分布，网络输出原子概率，对投影后的分布 Bellman 目标作交叉熵（C51 型）。相对标量 MSE，对随机 Bellman 目标更不敏感。

支撑区间固定，故对奖励作尺度归一而非仅中心化回报。跟踪折现回报的运行方差 $\sigma_{t,G}^{2}$ 与最大幅值 $G_{t,\max}$：

$$
\bar r_{t}=\frac{r_{t}}{\max\bigl(\sqrt{\sigma_{t,G}^{2}+\epsilon},\; G_{t,\max}/G_{\max}\bigr)}
$$

使有效回报落在分布支撑内，且训练全程尺度一致。

### 5.5 权重归一化

无约束的权重增长会抬高 $Q$ 的方差并放大备份误差。每步梯度更新后将各权重向量投影至单位球面，归一化层的 $(\gamma,\beta)$ 投影至范数 $\sqrt{d}$；信息主要编码于方向而非模长。单独使用权归一增益有限，但在样本受限时提高稳健性，故保留。

---

## 六、探索：目标熵与噪声重复

Off-policy 框架允许采集策略与优化策略分离。

### 6.1 与动作维数成比例的目标熵

自动温度调节需指定目标熵 $\bar{\mathcal{H}}$。SAC 常用 $-\lvert\mathcal{A}\rvert$，跨任务体仍常需微调。FlashSAC 指定对角高斯策略的目标标准差 $\sigma_{\mathrm{tgt}}$（实验统一取 0.15）：

$$
\bar{\mathcal{H}}=\frac{1}{2}\lvert\mathcal{A}\rvert\log\bigl(2\pi e\,\sigma_{\mathrm{tgt}}^{2}\bigr)
$$

随 $\lvert\mathcal{A}\rvert$ 线性增长，不同本体上探索强度保持同量级。消融中 $\sigma_{\mathrm{tgt}}\in\{0.05,\ldots,0.25\}$ 渐近性能接近，$\sigma_{\mathrm{tgt}}=0.15$ 附近即可。

### 6.2 Noise repetition

Ornstein–Uhlenbeck / pink noise 在数千并行环境上需为每个环境维护相关过程，内存与算力开销大。FlashSAC 改为：每隔一段重复区间采样 $\varepsilon\sim\mathcal{N}(0,I)$，在动作选择中保持 $k$ 步不变；$k$ 服从 Zeta 分布 $P(k)\propto k^{-s}$，多数为短重复、偶发长相关段。相对逐步独立噪声，时间相关的扰动不易被高维动力学平均掉。移除该机制会减慢收敛并降低总分。

---

## 七、架构示意

```text
观测 s（或视觉编码器输出）
        │
  inverted residual × L     预激活 BN + 非线性
        │                   残差连接
     RMSNorm
        │
   ┌────┴────┐
   Actor     双分布 critic（原子 logits）
   μ, σ      与目标网络；Cross-batch 与 s' 共享 BN
        │
   熵损失 + min Q     投影 Bellman + 交叉熵
```

视觉任务：三层卷积 + 线性瓶颈；堆叠最近三帧（84×84×9）；n-step 取 3；action repeat 2。稳定模块与状态设定正交，可叠加表示学习目标。

---

## 八、训练流程（GPU 默认）

```text
1. 初始化 Actor、两套分布 critic 与目标网络、容量至 10M 的回放池、log α
2. 1024 环境并行交互；动作为 π 采样，噪声 ε 按 Zeta 间隔重复
   将转移写入回放池
3. 每积累 1024 条新数据，作 2 次更新（UTD=2/1024），每次 batch=2048：
     a. 将 (s,a) 与 (s',a') 拼成同一 batch，共享 BN
        奖励按式 (6) 缩放；分布 Bellman 投影得目标
     b. 交叉熵更新两套 critic；权重向量投影至单位球
     c. 最小化 α log π − min Q，更新 Actor
     d. J(α) 更新温度（目标熵为式 (7)）
     e. 软更新目标网络
4. 回到步骤 2
```

---

## 九、损失函数（相对 SAC 的改动）

$$
\begin{aligned}
L_{Q} &= \sum_{i=1}^{2}\mathbb{E}\big[\mathrm{CE}\big(p_{\phi_i}(s,a),\;\mathcal{P}y_{\mathrm{dist}}\big)\big] \\
L_{\pi} &= \mathbb{E}\big[\alpha\log\pi_{\theta}(a\mid s)-\min_i Q_{\phi_i}(s,a)\big] \\
L_{\alpha} &= \mathbb{E}\big[-\alpha\big(\log\pi_{\theta}(a\mid s)+\bar{\mathcal{H}}\big)\big]
\end{aligned}
$$

$p_{\phi_i}$ 为原子概率，$Q_{\phi_i}$ 为其期望；$\mathcal{P}$ 为向固定原子网格的投影。$L_{\pi}$、$L_{\alpha}$ 与 SAC 同型，$\bar{\mathcal{H}}$ 采用式 (7) 而非 $-\lvert\mathcal{A}\rvert$。

---

## 十、设计要点

| 设计 | 针对问题 | 实现 |
|------|----------|------|
| 低 UTD + 大模型 + 大 batch | 高频更新 wall-clock 差、小网络上界 | 2/1024、2.5M、2048 |
| 大并行 + $10^7$ 回放 | 高维覆盖、长尾遗忘 | 1024 环境 |
| Inverted residual + RMSNorm | 深层梯度、无界特征 | 瓶颈块后 RMSNorm |
| 预激活 BN + Cross-batch | 非平稳回放、备份两侧统计不一致 | 与下一状态同 batch |
| 分布 Q + 奖励缩放 | 标量 TD 对随机目标敏感、支撑溢出 | C51 型，式 (6) |
| 权重归一 | 权重范数增长放大 $Q$ 方差 | 步后投影 |
| 目标熵 $\sigma_{\mathrm{tgt}}$ | 跨本体目标熵需逐任务调整 | 式 (7)，默认 0.15 |
| 噪声重复 | 高维需时间相关探索、并行 OU 开销大 | Zeta 间隔固定噪声 |

---

## 十一、实验结果（论文报告）

评测超过 60 个任务、10 个仿真器；wall-clock 在单卡 RTX 5090 上按「交互时间 + 算法更新时间」协议估计。

- **GPU 状态输入（IsaacLab、ManiSkill、Genesis、Playground 等）。** Off-policy 训练 50M 步；PPO 训练 200M 步（约 3 倍算力）以探渐近。FlashSAC 采用统一超参数，仅折扣因子 $\gamma$ 随仿真器默认（如 IsaacLab 0.99，Playground 0.97）。低维夹爪 / 四足与 PPO 接近或略优；高维灵巧手与人形在渐近回报与 wall-clock 上显著优于 PPO。相对 FastTD3 更稳、渐近更高。
- **CPU 单环境（MuJoCo、DMC、HumanoidBench、MyoSuite）。** 强调样本效率；对比 XQC、SimbaV2、TD-MPC2、MR.Q 等。FlashSAC 仍用统一配置（batch / UTD 按第四节缩小）。PPO 在该设定下表现欠佳。
- **视觉 DMC。** 对比 DrQ-v2、MR.Q；1M 步。FlashSAC wall-clock 与渐近性能不弱于基线，且无任务级探索或辅助动力学模块。
- **Sim-to-real（Unitree G1 盲行走）。** 地形课程与域随机；与 PPO 共用奖励、非对称 Actor–Critic、隐式系统辨识。平地约 20 分钟（PPO 约 3 小时）；真实楼梯（训练未见同尺寸）约 4 小时（PPO 约 20 小时）。

**覆盖分析。** Shadow Hand 上，$10^6$ 池内 off-policy 样本相对终策略再采集 $10^6$ on-policy 样本，在物体位置–指尖动作空间上支撑明显更宽，用以解释高维设定下 on-policy 评估的困难。

---

## 十二、方法小结

标准 SAC 可概括为：小网络、高频更新、$10^6$ 回放、按维数启发式设定目标熵。FlashSAC 保留同一 soft Actor–Critic 目标，但改变实现路径：

- **低 UTD、大 batch：** 每次更新使用覆盖更广的 mini-batch，而非对同一批误差反复反传；
- **先约束 critic、再扩展容量：** 范数与 BN 统计一致，防止 Bellman 备份随容量发散；
- **探索强度按本体维数对齐：** $\sigma_{\mathrm{tgt}}$ 固定，时间相关噪声以重复采样替代每环境一条 OU 过程。

低维、仿真近乎免费时，PPO 仍属合理选择；维数与仿真成本升高后，瓶颈从策略梯度方差转为「宽分布上 $Q$ 能否稳定、高效地拟合」。

---

## 十三、关键超参数

| 参数 | GPU 默认 | 作用 |
|------|----------|------|
| 并行环境数 | 1024 | 覆盖与吞吐 |
| 回放容量 | 至 $10^7$ | 长尾与多样性 |
| Batch size | 2048（CPU：512） | 大步长、BN 统计 |
| UTD | 2/1024（CPU：1） | wall-clock 与拟合精度 |
| 网络 | 约 2.5M，六层 | 容量 |
| $\sigma_{\mathrm{tgt}}$ | 0.15 | 目标熵，式 (7) |
| $\gamma$ | 随仿真器 | 折扣因子 |
| $\tau$ | 与 SAC 同类软更新 | 目标网络 |
| 分布支撑 | $G_{\min}, G_{\max}$，原子数 | categorical Q |

---

## 十四、与仓库内笔记的关系

- 最大熵目标、soft $V/Q$、Boltzmann Actor、温度对偶：[`sac_notes.md`](sac_notes.md)。
- Clip 与 GAE、on-policy 样本效率：[`ppo_notes.md`](ppo_notes.md)。
- FlashSAC 不改变上述目标的定义，而改变**网络规模与更新频率、Critic 约束机制、高维探索噪声的构造方式**。

局限与展望：当前重点在状态输入与中等视觉任务；触觉、演示混合、高保真仿真器是自然延伸。稳定模块可与辅助表示损失叠加。

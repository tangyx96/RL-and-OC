# World Action Model（WAM）与 Fast-WAM 算法笔记

下文以 Yuan 等（2026）的 [Fast-WAM](https://arxiv.org/abs/2603.16666) 为主线，梳理 World Action Model 的概率建模、训练目标与推理接口；与在线强化学习的对照见 [`ppo_notes.md`](ppo_notes.md)，与动作扩散模仿学习的衔接见 [`diffusion_policy_notes.md`](diffusion_policy_notes.md)。官方实现：[yuantianyuan01/FastWAM](https://github.com/yuantianyuan01/FastWAM)。

---

## 一、算法定位

World Action Model（WAM）将未来视觉预测与动作生成纳入同一生成式框架，旨在弥补标准 Vision-Language-Action（VLA）模型仅继承静态图文先验、未显式建模「动作如何改变观测」这一局限。Fast-WAM 进一步考察：WAM 的性能增益究竟来自训练阶段的视频联合建模，还是来自推理阶段对未来观测的显式生成。

| 特性 | PPO | VLA（如 $\pi_0$） | Diffusion Policy | WAM / Fast-WAM |
|------|-----|-------------------|------------------|----------------|
| 学习范式 | 在线 RL（on-policy） | 离线模仿 / 预训练微调 | 离线模仿 | 离线模仿 + 视频联合训练 |
| 训练信号 | 奖励 $r_t$、优势函数 | 专家动作 | 专家动作序列 + 噪声监督 | 专家动作 + 未来视频 latent |
| 策略接口 | $\pi_\theta(a \mid s)$ | $p(a \mid o, l)$ | $p(\mathbf{A}_t \mid \mathbf{O}_t)$ | $p(a_{1:H} \mid o, l)$ 或经 $v_{1:T}$ 的因子分解 |
| 世界模型 | 无（或外接 model-based RL） | 无 | 无 | 训练期预测未来视频；Fast-WAM 推理期不生成未来帧 |
| 推理开销 | 单次前向 | 单次前向（或少量 flow 步） | $K$ 步去噪 | 标准 WAM：视频与动作双重去噪；Fast-WAM：首帧编码 + 动作去噪 |

PPO 最大化期望回报 $J(\theta)=\mathbb{E}_\tau[\sum_t \gamma^t r_t]$，梯度经优势函数加权对数似然作用于策略参数。WAM 不利用环境奖励，而是在演示数据上拟合条件生成分布；关于环境动态的信息以辅助损失 $\mathcal{L}_{\mathrm{vid}}$ 进入表征学习，而非经 Bellman 备份进入价值函数。二者在「策略即条件分布」这一抽象层面相通，但目标函数、数据来源与推理计算图截然不同。

---

## 二、问题形式化

### 2.1 标准 visuomotor 策略

记当前观测 $o$（多相机图像，可含本体感觉）、语言指令 $l$、动作序列 $a_{1:H}$（$H$ 为 action horizon）。标准 visuomotor 策略学习

$$
p_\theta(a_{1:H} \mid o, l).
\tag{1}
$$

式 (1) 是 VLA 与 Diffusion Policy 的直接优化对象：给定当前上下文，输出一段未来动作。

### 2.2 Imagine-then-execute 的 WAM 分解

引入未来视觉序列 $v_{1:T}$（$T$ 为预测帧数），多数 WAM 采用如下因子分解：

$$
p(a_{1:H} \mid o, l)
= \int p(v_{1:T} \mid o, l)\, p(a_{1:H} \mid o, l, v_{1:T})\, \mathrm{d}v_{1:T}.
\tag{2}
$$

式 (2) 对应「先预测未来观测，再以其为条件生成动作」的两阶段决策。实现上主要有两类：

- **Joint modeling**：$v_{1:T}$ 与 $a_{1:H}$ 在同一扩散或 flow 过程中联合去噪（如 Motus、Cosmos Policy）；
- **Causal / IDM**：先完成 $v_{1:T}$ 的生成，再以其为条件预测 $a_{1:H}$（如 LingBot-VA、Vidar）。

Fast-WAM 的受控对比省略外层 chunk 级自回归 rollout，以便在固定 horizon 内隔离上述两类因素。

### 2.3 Fast-WAM 的直接策略接口

Fast-WAM 在推理阶段回到式 (1) 的直接接口，但用经视频联合训练得到的 latent 表征 $z(o,l)$ 参数化动作分布：

$$
p_\theta(a_{1:H} \mid o, l) = p_\theta(a_{1:H} \mid z(o, l)).
\tag{3}
$$

与 imagine-then-execute 范式的根本区别在于：$z(o,l)$ 由 video DiT 对当前首帧 clean latent 作**单次前向编码**得到，推理阶段不对 $v_{1:T}$ 作迭代采样。训练阶段仍保留 $\mathcal{L}_{\mathrm{vid}}$，故 $z$ 所张成的函数类在优化过程中受到未来视频监督；部署阶段则无需承担未来视频去噪的计算开销。

### 2.4 两个可分离因素

| 因素 | 训练 | 推理 | 典型实现 |
|------|------|------|----------|
| 视频联合建模 | $\mathcal{L}_{\mathrm{vid}}>0$ | — | 对未来帧 latent 作 flow matching |
| 显式未来生成 | 可选（attention mask 不同） | 对 $v_{1:T}$ 迭代去噪 | Joint / IDM 变体 |
| Fast-WAM | 保留 | 省略 | 仅首帧在 $t=0$ 通过 video backbone |

论文的受控实验表明：移除 $\mathcal{L}_{\mathrm{vid}}$ 所致的性能下降，显著大于移除推理期未来生成；即训练目标对最终控制性能的贡献，大于测试期想象机制。

---

## 三、与 PPO 的形式对照

### 3.1 目标函数

PPO 最大化

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right],
\qquad
\nabla_\theta J \propto \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, \hat{A}_t\right].
$$

WAM 在演示集 $\mathcal{D}$ 上最小化 flow matching 损失（见第五节），不涉及 $\hat{A}_t$ 或 $V(s)$。若置于同一理论框架下表述，WAM 等价于在**固定数据分布** $d^{\mathcal{D}}(o,l,a,v)$ 上作极大似然或分数匹配，而非在**策略诱导的占用测度** $d^{\pi_\theta}$ 上优化期望回报。

### 3.2 策略参数化

PPO 的 Actor 直接参数化 $\pi_\theta(a \mid s)$（如高斯或对数线性策略）。WAM 将策略嵌入连续归一化流：动作 chunk 为 flow ODE 从先验噪声到数据流形的终端状态。推理时 PPO 仅需一次采样；Fast-WAM 需 $N_{\mathrm{act}}$ 步 Euler 积分（默认 10 步），但无需 $N_{\mathrm{vid}}$ 步视频去噪。

### 3.3 辅助任务与 Critic

PPO 的 Critic $V_\phi(s)$ 用于 baseline 估计与 GAE。WAM 的辅助分支为未来视频预测：该分支不估计回报，而是通过 $\mathcal{L}_{\mathrm{vid}}$ 约束 video DiT 的表征，使 action expert 所依赖的 cross-attention 上下文编码物理动态信息。Fast-WAM 的消融实验表明，该辅助任务对下游控制的作用，大于在推理阶段完整执行同一视频分支。

---

## 四、模型结构

### 4.1 组件

Fast-WAM 以 [Wan2.2-5B](https://arxiv.org/abs/2503.20314) 视频 Diffusion Transformer 为骨干：

| 模块 | 功能 | 规模（论文） |
|------|------|-------------|
| Video DiT | 世界建模 backbone | 5B |
| Action Expert DiT | 动作 chunk 生成 | 1B（$d_a=1024$） |
| T5 文本编码器 | 指令 $l \mapsto$ context | 复用 Wan |
| Video VAE | 图像/视频 $\mapsto$ latent token | 复用 Wan |

总参数量约 6B。Action Expert 与 Video DiT 结构同构但宽度缩减（hidden 3072 $\to$ 1024），二者经 Mixture-of-Transformer（MoT）在若干层共享 attention。

### 4.2 Token 分组

每个训练样本包含三类 token：

1. **首帧 clean latent** $z_0$：当前观测经 VAE 编码，作为共享视觉锚点；
2. **未来 noisy video latent** $z_{1:T}$：仅训练期使用，参与 $\mathcal{L}_{\mathrm{vid}}$；
3. **Action tokens** $a_{1:H}$：经线性嵌入后的动作 chunk，参与 $\mathcal{L}_{\mathrm{act}}$。

多相机图像在 VAE 编码前拼接为单幅宽图；时间维 $4\times$ 下采样，每个 chunk 含 9 帧视频、$H=32$ 步动作。

### 4.3 结构化 attention mask

记 video token 序列长 $L_v$，action 序列长 $L_a$。布尔 mask $M \in \{0,1\}^{(L_v+L_a)\times(L_v+L_a)}$ 规定：

$$
M_{ij} = 1 \quad \Rightarrow \quad \text{第 } i \text{ 个 query 可 attend 至第 } j \text{ 个 key}.
$$

构造规则（与实现 `_build_mot_attention_mask` 一致）：

- **Video $\to$ Video**：由 `first_frame_causal` 模式决定（未来帧之间双向 attention，且可 attend 首帧）；
- **Action $\to$ Action**：chunk 内全连接（双向）；
- **Action $\to$ Video**：仅允许 attend 首帧 token，禁止 attend 未来 noisy video token；
- **首帧 $\to$ ***：不允许 attend 任何其他 token（纯上下文锚点，不接收 future 信息）。

形式化地，设首帧占 $L_0 = \min(\text{tokens\_per\_frame}, L_v)$ 个 video 位置，则对 action 行 $i \in \{L_v+1,\ldots,L_v+L_a\}$：

$$
M_{i,j} = \begin{cases}
1 & j \le L_0 \\
0 & L_0 < j \le L_v \\
1 & j > L_v \quad (\text{action 分支内部})
\end{cases}
$$

**设计动机**：$\mathcal{L}_{\mathrm{vid}}$ 驱动 backbone 学习环境动力学；$\mathcal{L}_{\mathrm{act}}$ 在**与部署一致的可见信息**（当前帧与语言）下预测动作，防止 action 分支在训练期访问未来 video token 而造成信息泄漏，从而掩盖表征质量不足。

### 4.4 训练与推理计算图

**训练**：三类 token 同时存在，video 与 action 分支经 MoT 联合前向；首帧 latent 可注入 clean 值（`first_frame_latents`）。

**Fast-WAM 推理**（`infer_action`）：

1. 将当前图像编码为 `first_frame_latents`；
2. Video Expert 在 $t_{\mathrm{video}}=0$（无噪声）下单次前向，得到 KV cache；
3. Action Expert 以该 cache 为 cross-attention 上下文，对动作 latent 作 $N_{\mathrm{act}}$ 步 flow 去噪；
4. 不实例化未来 video token，不执行 video 迭代。

相对 Joint（580 ms）与 IDM（810 ms），Fast-WAM 在 RTX 5090D 上约 190 ms（含文本与 VAE 编码）；主要增益来自省略 $N_{\mathrm{vid}}$ 步视频采样。

---

## 五、Flow Matching 训练目标

Fast-WAM 对动作与视频采用同一套 conditional flow matching（CFM），与 $\pi_0$ 等 VLA flow 模型同族，而非 DDPM 的 $\epsilon$-prediction（见 [`diffusion_policy_notes.md`](diffusion_policy_notes.md) 第四节）。

### 5.1 从概率路径到速度场

设数据 $y \sim q(y)$（可为动作或 video latent）。构造线性插值路径（Rectified Flow / 最优传输直线路径的特例）：

$$
y_t = (1-t)\, y + t\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I),\; t \in (0,1).
\tag{4}
$$

对 $t$ 求导，条件速度场为常数

$$
u_t(y_t \mid y) = \frac{\mathrm{d} y_t}{\mathrm{d} t} = \epsilon - y.
\tag{5}
$$

在路径 (4) 下，各样本对应的速度与 $t$ 无关，训练目标简化为回归 $f_\theta(y_t, t, o, l) \approx \epsilon - y$。

**理论依据**（Lipman et al., 2023）：边缘速度场 $u_t(y_t) = \mathbb{E}[u_t(y_t \mid y) \mid y_t]$ 难以直接估计；但条件目标

$$
\mathcal{L}_{\mathrm{CFM}}(\theta) = \mathbb{E}_{y,\epsilon,t,y_t}\big[\| v_\theta(y_t,t) - (\epsilon - y) \|^2\big]
$$

在适当正则条件下与边缘 flow matching 目标梯度一致。Fast-WAM 论文式 (5)–(6) 即此形式，网络 $f_\theta$ 同时以观测与语言为条件。

### 5.2 与 scheduler 实现的对应

官方 `WanContinuousFlowMatchScheduler` 采用离散时间步 $\sigma \in [0,1]$ 与 shift 变换 $\phi$：

$$
\phi(u; s) = \frac{s u}{1 + (s-1)u}, \qquad u \sim \mathcal{U}(0,1),\; \sigma = \phi(u; s_{\mathrm{shift}}).
$$

加噪与回归目标：

$$
\tilde{y} = (1-\sigma)\, y + \sigma\, \epsilon, \qquad
\text{target} = \epsilon - y.
\tag{6}
$$

推理阶段以 Euler 法更新 $\tilde{y} \leftarrow \tilde{y} + f_\theta(\tilde{y}, t)\, \Delta\sigma$。论文中 $t \in (0,1)$ 与实现中 $\sigma = t/T$（$T=1000$）相差一个尺度因子，不改变「回归 $\epsilon - y$」这一训练本质。

训练时对 $t$ 采用 logit-normal 采样（经 $\phi$ 变换），并对不同 $t$ 加权：

$$
w(t) \propto \exp\!\left(-2\left(\frac{t - T/2}{T}\right)^2\right),
$$

以强调中间噪声水平上的梯度贡献——与 DDPM 中丢弃时间步权重 $w_k$ 的经验类似，属于实现层面的稳定性取舍。

### 5.3 分项损失

对动作 $y = a_{1:H}$：

$$
\mathcal{L}_{\mathrm{act}} = \mathbb{E}\left[ w(t_a)\, \big\| f_\theta(a_t, t_a, o, l) - (\epsilon_a - a_{1:H}) \big\|_2^2 \right].
\tag{7}
$$

对未来 video latent $y = z_{1:T}$（首帧在 loss 中可剔除，因其保持 clean）：

$$
\mathcal{L}_{\mathrm{vid}} = \mathbb{E}\left[ w(t_v)\, \big\| f_\theta(z_t, t_v, o, l) - (\epsilon_v - z_{1:T}) \big\|_2^2 \right].
\tag{8}
$$

总损失

$$
\mathcal{L} = \lambda_{\mathrm{act}}\, \mathcal{L}_{\mathrm{act}} + \lambda_{\mathrm{vid}}\, \mathcal{L}_{\mathrm{vid}}.
\tag{9}
$$

官方默认 $\lambda_{\mathrm{act}} = \lambda_{\mathrm{vid}} = 1$（`configs/model/fastwam.yaml` 中 `lambda_action: 1.0`，`lambda_video` 缺省为 1.0）。padding 帧与步长经 `action_is_pad`、`image_is_pad` 在 batch 内作 masked mean。

**无 video co-train 变体**：令 $\lambda_{\mathrm{vid}} = 0$，架构与 Fast-WAM 推理路径不变，用于检验「无动力学监督的表征学习」之对照。

---

## 六、受控变体与推理图

| 变体 | $\mathcal{L}_{\mathrm{vid}}$ | 推理期未来生成 | Attention 差异 |
|------|------------------------------|----------------|----------------|
| **Fast-WAM** | ✓ | 无 | Action 仅见首帧；video 单次 $t=0$ |
| **Fast-WAM-Joint** | ✓ | 有 | Video–Action token 互 attend；联合去噪 |
| **Fast-WAM-IDM** | ✓ | 有 | 先 video 去噪，再条件于生成结果预测 action；训练期对 GT video token 以 $p=0.5$ 加噪增强 |
| **w.o. video co-train** | ✗ | 无 | 同 Fast-WAM |

Optional IDM checkpoint 可在同一组权重下切换 `idm` / `first_frame` 推理模式，在不重新训练的前提下对比两种推理路径。

```text
训练（Fast-WAM）:
  o, l ──► VAE ──► z_0 (clean) ──┐
  future frames ──► z_{1:T} ──noisy──► Video DiT ── MoT ──► L_vid
  a_{1:H} ──noisy──► Action DiT ────────┘              └──► L_act
  mask: action ↛ z_{1:T}^{noisy}

推理（Fast-WAM）:
  o, l ──► z_0 ──► Video DiT (t=0, 单次) ──► KV cache ──► Action flow (N_act 步) ──► a_{1:H}

推理（IDM）:
  o, l ──► ... ──► Video flow (N_vid 步) ──► ẑ_{1:T} ──► Action flow (N_act 步) ──► a_{1:H}
```

---

## 七、实验结果

### 7.1 仿真基准

**RoboTwin 2.0**（无 embodied pretraining）：Fast-WAM 达 91.8%，接近 LingBot-VA（92.2%，有预训练），显著高于同 backbone 无预训练的 LingBot-VA（80.6%）。移除 video co-train 后降至 83.8%。

**LIBERO** 四套件平均：Fast-WAM 97.6%；Joint 98.5%、IDM 98.0% 略高但差距有限；无 co-train 93.5%，Spatial 与 Long 子集降幅最为显著。

受控对比的定量关系（平均成功率）：

$$
\underbrace{|\mathrm{Fast} - \mathrm{Joint}|,\; |\mathrm{Fast} - \mathrm{IDM}|}_{\text{推理机制差异}} \;\ll\; \underbrace{|\mathrm{Fast} - \mathrm{w/o\ co\text{-}train}|}_{\text{训练目标差异}}.
$$

### 7.2 真机实验（毛巾折叠）

含 co-train 的 Fast-WAM 系列均显著优于无预训练 $\pi_{0.5}$；无 co-train 成功率约 10%，且平均完成时间最长。推理延迟：Fast-WAM 190 ms，Joint 580 ms，IDM 810 ms。真机任务上 IDM 成功率可略高于 Fast-WAM，精度–延迟权衡取决于部署约束。

### 7.3 结果解读

实验支持如下结论，而非否定世界模型的作用：

> 视频预测作为训练期表征学习信号，是 WAM 性能增益的主要来源；测试期显式生成未来视频在多数基准上仅带来边际提升，却引入数倍推理延迟。

Joint 与 IDM 在部分设定下仍略优，表明 future imagination 并非严格为零贡献；但在论文所考察的单 chunk、固定 horizon 设定下，其边际收益小于 video co-training。

---

## 八、与 Diffusion Policy、$\pi_0$ 的关系

| 方法 | 生成对象 | 条件 | 世界建模 |
|------|----------|------|----------|
| Diffusion Policy | 动作序列 | 观测历史 | 无 |
| $\pi_0$ / $\pi_{0.5}$ | 动作 flow | 图像 + 语言 | 无显式 future video |
| Fast-WAM | 动作 flow +（训练期）video flow | 首帧 + 语言 | $\mathcal{L}_{\mathrm{vid}}$ 约束 video backbone |

Diffusion Policy 的 $\epsilon$-prediction 与 Fast-WAM 的 velocity matching 同属生成式策略参数化；WAM 额外对 $p(z_{1:T} \mid z_0, l)$ 施加监督，且经 MoT 与 action 分支共享参数。Fast-WAM 的推理计算图接近 $\pi_0$：不生成未来视觉，仅保留经 co-train 强化的 encoder 表征。

---

## 九、待研究问题

1. **Horizon 与外层自回归**：论文省略 chunk 级自回归；更长任务中，显式 future imagination 是否因误差累积而成为必要？
2. **表征分析**：$z(o,l)$ 编码了哪些动力学因素（接触、刚体、可变形体）？需结合 probing 与反事实干预，而非仅依赖成功率。
3. **$\lambda_{\mathrm{vid}}$ 与 mask 设计**：默认等权；更弱的 video 分支或更严格的 causal mask 是否改变「co-train 优于 imagination」的结论？
4. **与 RL 微调**：当前为纯模仿学习；若在 PPO 微调阶段保留 $\mathcal{L}_{\mathrm{vid}}$ 作正则，能否缓解 on-policy 更新中的灾难性遗忘——此为 WAM 与 PPO 的可能交汇，Fast-WAM 原文未涉及。

---

## 参考文献

- Yuan, T., Dong, Z., Liu, Y., & Zhao, H. (2026). Fast-WAM: Do World Action Models Need Test-time Future Imagination? [arXiv:2603.16666](https://arxiv.org/abs/2603.16666).
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling. ICLR.
- Wan Team (2025). Wan: Open and Advanced Large-Scale Video Generative Models. [arXiv:2503.20314](https://arxiv.org/abs/2503.20314).
- Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347).
- Chi, C., et al. (2023). Diffusion Policy. RSS. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137).

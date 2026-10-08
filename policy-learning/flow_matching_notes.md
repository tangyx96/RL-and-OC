# Flow Matching 与机器人策略

下文以 Physical Intelligence 的 [openpi](https://github.com/Physical-Intelligence/openpi) 为主线，梳理 conditional flow matching（CFM）在 visuomotor 策略中的形式化、训练目标与推理流程。论文：[π₀: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)（Black et al., 2024）。动作序列的 receding horizon 设定见 [`diffusion_policy_notes.md`](diffusion_policy_notes.md)；DDPM 的 $\epsilon$-prediction 见同文第四节；离散 VLA 见 [`vla_notes.md`](vla_notes.md)。

---

## 一、算法定位

Flow matching（FM）是一类**连续时间生成模型**的训练目标：学习把简单源分布（通常为 $\mathcal{N}(0,I)$）沿 ODE 轨迹推至数据分布。将其接入机器人策略时，生成对象通常是**动作序列 chunk** $\mathbf{A}\in\mathbb{R}^{H\times D_a}$：$H$ 为一次预测的步数，$D_a$ 为单步动作维数。条件 $\mathbf{O}$ 包含图像、语言与本体感觉。记 $q_0(\mathbf{A}\mid\mathbf{O})$ 为专家动作在给定观测下的条件分布。优化仍属**离线模仿学习**，拟合 $q_0$，不使用环境奖励。

相对 Diffusion Policy（DDPM 的噪声预测），FM 在机器人社区中逐渐成为 VLA 动作头的默认选择：π₀、π₀.₅、GR00T、SmolVLA 等均回归路径速度，而非多步 DDPM 去噪。记 $\boldsymbol{\epsilon}\sim\mathcal{N}(0,I)$ 为标准正态噪声，$\epsilon_\theta$ 为对它的预测。条件路径上的 $\mathrm{d}\mathbf{x}/\mathrm{d}t$ 记为 $\mathbf{u}$，网络对速度的预测记为 $\mathbf{v}_\theta$。$K$ 为 DDPM 的离散步数，$N$ 为 Euler 步数。openpi 仓库中 π₀ 与 π₀.₅ 的 flow head 为当前最有影响力的开源实现；π₀-FAST 走自回归离散动作，不在本文范围内。

| 特性 | Diffusion Policy | Flow Matching（π₀） | OpenVLA |
|------|------------------|---------------------|---------|
| 生成对象 | 动作序列 | 动作 chunk | 离散动作 token |
| 训练目标 | $\|\epsilon-\epsilon_\theta\|^2$ | $\|\mathbf{u}-\mathbf{v}_\theta\|^2$ | 下一 token 交叉熵 |
| 推理 | $K$ 步去噪（DDIM 可减至 $\sim 10$） | $N$ 步 Euler 积分（默认 $\sim 10$） | 自回归解码 |
| 骨干 | 1D U-Net + 视觉编码器 | 预训练 VLM + action expert | 预训练 VLM |
| 学习范式 | 离线模仿 | 离线模仿 | 离线模仿 |

FM 本身不是模仿学习算法。记条件策略为 $\pi_\theta(\mathbf{A}\mid\mathbf{O})$。与行为克隆的关系是：把 $\pi_\theta$ 取成流映射诱导的分布，再用条件速度回归去拟合它。显式似然的做法见第四节。

---

## 二、与行为克隆的关系

设演示 $\mathcal{D}=\{(\mathbf{O}_j,\mathbf{A}_j)\}$，专家条件分布为 $q_0(\mathbf{A}\mid\mathbf{O})$。行为克隆选取 $\pi_\theta$ 使

$$
\hat\theta=\arg\min_\theta\;
\mathbb{E}_{(\mathbf{O},\mathbf{A})\sim\mathcal{D}}
\big[-\log\pi_\theta(\mathbf{A}\mid\mathbf{O})\big],
$$

等价于最小化经验分布上的正向 KL（推导见 [`diffusion_policy_notes.md`](diffusion_policy_notes.md) 第 2.2 节）。Diffusion Policy 通过 DDPM 参数化 $\pi_\theta$；FM 策略通过 ODE 定义的映射参数化同一对象。二者共享动作序列接口：预测 $H$ 步（`action_horizon`），每轮执行其中 $T_a$ 步后按新观测重规划，$T_a\le H$（见 DP 笔记第三节）。

---

## 三、Flow Matching 的形式化

### 3.1 问题设定与两条计算链

目标是拟合 $q_0(\mathbf{A}\mid\mathbf{O})$ 并从其抽样。$q_0$ 无闭式密度，可用的只是演示中的有限样本。Flow matching 把生成写成一个常微分方程的初值问题：源分布取 $p_1=\mathcal{N}(0,I)$，待学习的是右端的速度场。训练与推理用的是两条不同的计算链。

```text
训练（一次插值，不对 ODE 做积分）：
  (A, ε, t) → x_t = (1-t) A + t ε → 回归 v_θ(x_t, t, O) ≈ u = ε − A

推理（积分学习到的 ODE）：
  x_1 ~ p_1，直线路径上 x_1 = ε，t = 1
  → 重复 N 次：x ← x − v_θ(x, t, O)/N，t ← t − 1/N
  → t ≈ 0 时的 x 作为动作 chunk
```

训练链的角色与 Diffusion Policy 的前向边际相同：已知专家动作，造一个带噪状态，并给出回归标签。DP 的闭式（[`diffusion_policy_notes.md`](diffusion_policy_notes.md) 第 2.4.2 节）是

$$
\mathbf{A}^{k}
= \sqrt{\bar\alpha_k}\,\mathbf{A}
+ \sqrt{1-\bar\alpha_k}\,\boldsymbol{\epsilon}.
$$

其中 $\boldsymbol{\epsilon}\sim\mathcal{N}(0,I)$，$\alpha_s=1-\beta_s$，$\bar\alpha_k=\prod_{s=1}^{k}\alpha_s$，$\beta_s\in(0,1)$ 为给定的噪声幅度，$k=1,\ldots,K$。$\bar\alpha_k$ 从近 $1$ 降到近 $0$。FM 在训练时同样一次重参数化，系数换成直线插值 $(1-t)$ 与 $t$。DP 的 $\{\beta_k\}$ 只用来推出上述边际；FM 的插值本身就是定义。

推理链的角色与 DP 的反向采样相同：从纯噪声出发，逐步走到数据。记一步的均值为 $\mu_\theta(\mathbf{A}^{k},k)$，方差 $\sigma_k^{2}$ 由 $\{\beta_s\}$ 给出、不由网络学习；$z\sim\mathcal{N}(0,I)$。反向一步为 $\mathbf{A}^{k-1}=\mu_\theta+\sigma_k z$。$\mu_\theta$ 由 $\epsilon_\theta$ 与 $\alpha_k,\beta_k$ 写成，系数见 DP 笔记第 2.4.4 节。令 $z=0$ 得到的是该后验均值。DDIM（$\eta=0$）将 $\epsilon_\theta$ 反解出的干净动作与 $\epsilon_\theta$ 分别乘以 $\sqrt{\bar\alpha_{k-1}}$ 与 $\sqrt{1-\bar\alpha_{k-1}}$ 后相加，系数与 $\mu_\theta$ 不同。FM 的网络输出是速度 $\mathbf{v}_\theta$，更新只乘步长 $-1/N$，积分过程中不再抽噪声。

构造 $\mathbf{x}_t$ 对应 DP 的加噪边际，Euler 循环对应 DP 的去噪。`sample_actions` 里的积分沿 $t:1\to 0$，状态从噪声走到动作。

| 对象 | 何时使用 | 公式 |
|------|----------|------|
| 直线插值 | 训练，构造样本与标签 | $\mathbf{x}_t=(1-t)\mathbf{A}+t\boldsymbol{\epsilon}$，$\mathbf{u}=\boldsymbol{\epsilon}-\mathbf{A}$ |
| DP 前向边际 | 训练，同一角色的另一条路径 | $\mathbf{A}^{k}=\sqrt{\bar\alpha_k}\mathbf{A}+\sqrt{1-\bar\alpha_k}\boldsymbol{\epsilon}$，标签 $\boldsymbol{\epsilon}$ |
| Euler 积分 | 推理 | $\mathbf{x}\leftarrow\mathbf{x}-(1/N)\mathbf{v}_\theta$，$t:1\to 0$ |
| DP 反向一步 | 推理，同一角色 | $\mathbf{A}^{k-1}=\mu_\theta(\mathbf{A}^{k},k)+\sigma_k z$ |

### 3.2 连续归一化流

记 $\mathbf{x}_0$ 为干净样本，分布为 $q_0$；给定观测时即 $q_0(\cdot\mid\mathbf{O})$。动作策略中 $\mathbf{x}_0=\mathbf{A}$。观测 $\mathbf{O}$ 先略去，第 3.4 节加回。源分布为 $p_1$。速度场 $\mathbf{v}_t(\mathbf{x})$ 定义 ODE

$$
\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t}=\mathbf{v}_t(\mathbf{x}_t),
\qquad t\in[0,1].
\tag{1}
$$

设 $\varphi_{s\to t}$ 为 (1) 从时刻 $s$ 积到时刻 $t$ 的映射，并记 $T_\theta=\varphi_{1\to 0}$。生成时抽 $\mathbf{x}_1\sim p_1$，输出 $\mathbf{x}_0=T_\theta(\mathbf{x}_1)$。该映射诱导的条件分布就是 $\pi_\theta(\cdot\mid\mathbf{O})$。行为克隆要求它接近 $q_0(\cdot\mid\mathbf{O})$。

### 3.3 高斯条件路径

Lipman et al. (2023) 取一族以干净样本为条件的高斯路径。给定 $\mathbf{x}_0$，令

$$
\mathbf{x}_t=\mu_t(\mathbf{x}_0)+\sigma_t\boldsymbol{\epsilon},
\qquad
\boldsymbol{\epsilon}\sim p_1,
$$

其中标量 $\sigma_t>0$，$\mu_t$ 取值于与 $\mathbf{x}$ 相同的空间。则

$$
q_t(\mathbf{x}\mid\mathbf{x}_0)
=\mathcal{N}\big(\mathbf{x};\,\mu_t(\mathbf{x}_0),\,\sigma_t^{2} I\big).
$$

固定端点样本 $(\mathbf{x}_0,\boldsymbol{\epsilon})$，对 $t$ 求导：

$$
\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t}
=\mu_t'(\mathbf{x}_0)+\sigma_t'\boldsymbol{\epsilon}.
$$

由 $\boldsymbol{\epsilon}=(\mathbf{x}_t-\mu_t(\mathbf{x}_0))/\sigma_t$，条件速度可写成只依赖当前状态的形式

$$
\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_0)
=\frac{\sigma_t'}{\sigma_t}\big(\mathbf{x}-\mu_t(\mathbf{x}_0)\big)+\mu_t'(\mathbf{x}_0).
\tag{2}
$$

固定 $(\mathbf{x}_0,\boldsymbol{\epsilon})$ 时，(2) 与上式相等：$\mathbf{u}_t(\mathbf{x}_t\mid\mathbf{x}_0)=\mathrm{d}\mathbf{x}_t/\mathrm{d}t$，只是用当前位置 $\mathbf{x}$ 换掉了 $\boldsymbol{\epsilon}$。$q_t(\mathbf{x}\mid\mathbf{x}_0)$ 是同一条路径上粒子的密度，$\mathbf{u}_t(\cdot\mid\mathbf{x}_0)$ 是这些粒子的速度；二者的方程见第 3.5 节。常用的三条路径都是 (2) 的特例，差别只在 $\mu_t,\sigma_t$。条件速度由 $\mu_t',\sigma_t'$ 决定，因此可能与 $t$ 无关，也可能随 $t$ 变化。

| 路径 | $\mu_t$ | $\sigma_t$ | 条件速度 | 端点 |
|------|---------|------------|----------|------|
| 直线（rectified flow） | $(1-t)\mathbf{x}_0$ | $t$ | $\boldsymbol{\epsilon}-\mathbf{x}_0$，与 $t$ 无关 | $t=0$ 数据，$t=1$ 噪声 |
| 方差保持（variance preserving, VP） | $\alpha_t\mathbf{x}_0$ | $\sqrt{1-\alpha_t^{2}}$ | $\alpha_t'\mathbf{x}_0+\sigma_t'\boldsymbol{\epsilon}$，随 $t$ 弯曲 | $\alpha_0=1$，$\alpha_1\approx 0$ |
| 方差爆炸（variance exploding, VE） | $\mathbf{x}_0$ | $\sigma_t$，自近 $0$ 增至远大于 $1$ | $\sigma_t'\boldsymbol{\epsilon}=(\sigma_t'/\sigma_t)(\mathbf{x}-\mathbf{x}_0)$ | 均值停在数据，方差膨胀 |

方差保持路径的 $\alpha_t$ 与 DDPM 的 $\sqrt{\bar\alpha}$ 是同一族边际：把离散指标 $k$ 看成连续时间后，$\mathbf{x}_t=\alpha_t\mathbf{x}_0+\sqrt{1-\alpha_t^{2}}\boldsymbol{\epsilon}$。其条件速度一般不是常向量。用条件 score 重写它，得到的仍是以 $\mathbf{x}_0$ 为条件的场。Song et al. (2021) 的 probability-flow ODE 用边缘 score $\nabla\log q_t(\mathbf{x})$，等于这个条件场的后验平均。推理时仍然要知道 $\alpha_t$ 的日程。方差爆炸路径保持条件均值等于 $\mathbf{x}_0$，只把噪声标准差放大，状态的尺度在路径上变化很大。

给定端点 $(\mathbf{x}_0,\boldsymbol{\epsilon})$，线段 $(1-t)\mathbf{x}_0+t\boldsymbol{\epsilon}$ 是两点间的最短路径，速度为常向量 $\boldsymbol{\epsilon}-\mathbf{x}_0$。训练将 $q_0$ 与 $p_1$ 独立配对，记作 $q_0\otimes p_1$。在此乘积测度下，同一时空位置 $(\mathbf{x},t)$ 可被多个不同来源的直线穿过，且各直线的条件速度 $\boldsymbol{\epsilon}-\mathbf{x}_0$ 方向互异。网络 $\mathbf{v}_\theta(\mathbf{x},t)$ 在每点只能输出单个向量，最小二乘回归迫使它收敛于这些条件速度的后验期望 $\mathbf{u}_t(\mathbf{x})$（定义见第 3.5 节）。推理时沿 $\mathrm{d}\mathbf{x}/\mathrm{d}t=\mathbf{v}_\theta$ 积分，粒子在每一位置走的是平均方向而非任一原始直线的方向，轨道因此偏离直线而弯曲。以动作 $\pm 1$ 为例：$-1$ 配 $+1$、$+1$ 配 $-1$ 时，两线在 $t=1/2$ 相交于 $0$，条件速度为 $+2$ 与 $-2$，后验平均为 $0$；同号配对时线段不相交，各段长度之和更小。一般地，该配对的传输代价是地面代价取 $\|\cdot\|$ 时的 $\mathbb{E}[\|\boldsymbol{\epsilon}-\mathbf{x}_0\|]$。$W_2$ 的配对代价是 $\mathbb{E}[\|\boldsymbol{\epsilon}-\mathbf{x}_0\|^2]$，Wasserstein 距离还要对全部配对取下确界。独立配对只给出其中一个，并不达到下确界。Liu et al. (2023) 的 rectify 先按独立配对训练，再利用学到的映射将各噪声推至对应样本，以所得新端点重新连线并再次回归，使边缘轨道更直。下文仅涉及独立配对这一轮。

π₀、π₀.₅、GR00T、SmolVLA 以及 Fast-WAM 的动作头都取这条直线（Fast-WAM 的视频项与动作项见 [`wam_notes.md`](wam_notes.md) 第三节）。条件速度 $\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_0)$ 是常向量，标签的代数形式不随 $t$ 改变。网络 $\mathbf{v}_\theta$ 拟合的是这些常向量的后验平均，即边缘速度 $\mathbf{u}_t(\mathbf{x})$；推理时对 $\mathbf{v}_\theta$ 作 Euler 积分，不使用 $\{\alpha_k,\beta_k\}$。截断误差与步数见第 5.3 节。

### 3.4 直线路径

取 $\mu_t(\mathbf{x}_0)=(1-t)\mathbf{x}_0$，$\sigma_t=t$。代入上一节的构造，

$$
\mathbf{x}_t=(1-t)\,\mathbf{x}_0+t\,\boldsymbol{\epsilon},
\qquad
\mathbf{x}_0\sim q_0,\;
\boldsymbol{\epsilon}\sim p_1,\;
t\in(0,1).
\tag{3}
$$

因此 $t=1$ 时 $\mathbf{x}_1=\boldsymbol{\epsilon}$。对固定 $(\mathbf{x}_0,\boldsymbol{\epsilon})$ 求导。$\mathbf{x}_0$ 与 $\boldsymbol{\epsilon}$ 都不依赖 $t$，故

$$
\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t}
=-\mathbf{x}_0+\boldsymbol{\epsilon}
=\boldsymbol{\epsilon}-\mathbf{x}_0.
\tag{4}
$$

(4) 与 $t$ 无关，也与 $\mathbf{x}_t$ 在路径上的位置无关：同一对端点在整段时间里共用一个常向量。这就是条件速度

$$
\mathbf{u}_t(\mathbf{x}_t\mid\mathbf{x}_0)=\boldsymbol{\epsilon}-\mathbf{x}_0.
$$

用 (2) 复核。$\mu_t'=-\mathbf{x}_0$，$\sigma_t'=1$，且 $\mathbf{x}_t-\mu_t=t\boldsymbol{\epsilon}$，于是

$$
\mathbf{u}_t(\mathbf{x}_t\mid\mathbf{x}_0)
=\frac{1}{t}\cdot t\boldsymbol{\epsilon}+(-\mathbf{x}_0)
=\boldsymbol{\epsilon}-\mathbf{x}_0.
$$

因为 $\boldsymbol{\epsilon}$ 是标准正态，$\mathbf{x}_t$ 在给定 $\mathbf{x}_0$ 时为高斯，均值 $(1-t)\mathbf{x}_0$，协方差 $t^{2}I$：

$$
q_t(\mathbf{x}\mid\mathbf{x}_0)
=\mathcal{N}\big(\mathbf{x};\,(1-t)\mathbf{x}_0,\,t^{2} I\big).
\tag{5}
$$

$\boldsymbol{\epsilon}\perp\!\!\!\perp\mathbf{x}_0$ 保证交叉协方差为零。由协方差的线性性质，

$$
\begin{aligned}
\mathrm{Cov}(\mathbf{x}_t)
&=\mathrm{Cov}\big((1-t)\mathbf{x}_0+t\boldsymbol{\epsilon}\big)\\
&=(1-t)^{2}\,\mathrm{Cov}(\mathbf{x}_0)+t^{2}\,\mathrm{Cov}(\boldsymbol{\epsilon})\\
&=(1-t)^{2}\,\mathrm{Cov}(\mathbf{x}_0)+t^{2} I.
\end{aligned}
$$

动作若已标准化到 $\mathrm{Cov}(\mathbf{x}_0)=I$，则系数是 $(1-t)^{2}+t^{2}$。$t=0$ 与 $t=1$ 时该系数为 $1$，$t=1/2$ 时为 $1/2$。方差保持路径在同样的标准化下边缘协方差全程为 $I$：信号系数的平方与噪声系数的平方相加为 $1$。直线插值没有这个约束。这是它和 DP 前向边际在公式上的差别；二者在训练中的角色仍都是「一次造出带噪样本」。

加入观测后，路径只在动作上插值，条件 $\mathbf{O}$ 不进入插值。演示中的一对 $(\mathbf{O},\mathbf{A})$ 给出

$$
\mathbf{x}_t=(1-t)\,\mathbf{A}+t\,\boldsymbol{\epsilon},
\qquad
\mathbf{u}_t(\mathbf{x}_t\mid\mathbf{A})=\boldsymbol{\epsilon}-\mathbf{A}.
$$

网络 $\mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{O})$ 拟合这个条件速度。图像与语言属于 $\mathbf{O}$，既不加噪，也不被生成。

### 3.5 边缘密度与边缘速度

条件路径同时给出条件密度 $q_t(\mathbf{x}\mid\mathbf{x}_0)$ 与条件速度 $\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_0)$。边缘密度是前者对数据分布的混合：

$$
q_t(\mathbf{x})
=\int q_t(\mathbf{x}\mid\mathbf{x}_0)\,q_0(\mathbf{x}_0)\,\mathrm{d}\mathbf{x}_0.
$$

同一位置 $\mathbf{x}$ 可以来自不同的 $\mathbf{x}_0$。直线路径上，这些样本的条件速度 $\boldsymbol{\epsilon}-\mathbf{x}_0$ 一般不同，把条件速度对 $q_0$ 直接积分并不得到推动 $q_t$ 的场。与 $q_t$ 配套、并满足连续性方程的速度，是条件速度关于后验的平均。后验由贝叶斯公式给出

$$
q(\mathbf{x}_0\mid\mathbf{x})
=\frac{q_t(\mathbf{x}\mid\mathbf{x}_0)\,q_0(\mathbf{x}_0)}{q_t(\mathbf{x})},
$$

边缘速度取为

$$
\mathbf{u}_t(\mathbf{x})
=\int\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_0)\,q(\mathbf{x}_0\mid\mathbf{x})\,\mathrm{d}\mathbf{x}_0
=\mathbb{E}\big[\mathbf{u}_t(\mathbf{x}_t\mid\mathbf{x}_0)\mid\mathbf{x}_t=\mathbf{x}\big].
$$

$q_t$ 只依赖条件密度与 $q_0$；$\mathbf{u}_t$ 还依赖条件速度。二者由同一条条件路径产生，公式不同。

这一对满足连续性方程。沿条件路径有 $\mathbf{u}_t(\mathbf{x}_t\mid\mathbf{x}_0)=\mathrm{d}\mathbf{x}_t/\mathrm{d}t$，故

$$
\partial_t q_t(\mathbf{x}\mid\mathbf{x}_0)
+\nabla\cdot\big(q_t(\mathbf{x}\mid\mathbf{x}_0)\,\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_0)\big)
=0.
$$

乘以 $q_0(\mathbf{x}_0)$ 并对 $\mathbf{x}_0$ 积分。第一项变为 $\partial_t q_t(\mathbf{x})$。第二项化为

$$
\nabla\cdot\left(
\int q_t(\mathbf{x}\mid\mathbf{x}_0)\,\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_0)\,q_0(\mathbf{x}_0)\,\mathrm{d}\mathbf{x}_0
\right).
$$

括号内的向量等于 $q_t(\mathbf{x})\,\mathbf{u}_t(\mathbf{x})$：$\mathbf{u}_t(\mathbf{x})$ 的定义恰好乘上了后验权重。因此

$$
\partial_t q_t(\mathbf{x})+\nabla\cdot\big(q_t(\mathbf{x})\,\mathbf{u}_t(\mathbf{x})\big)=0.
$$

$\mathbf{u}_t(\mathbf{x})$ 不能当作训练标签。后验的分子含有 $q_0$，演示只提供有限样本，没有这个密度的闭式。条件速度在抽样时已知：抽到 $(\mathbf{A},\boldsymbol{\epsilon})$ 后，直线路径的标签就是 $\boldsymbol{\epsilon}-\mathbf{A}$，不必知道还有哪些动作也能到达当前的 $\mathbf{x}_t$。

网络 $\mathbf{v}_\theta(\mathbf{x},t)$ 只以 $\mathbf{x}$ 与 $t$ 为自变量，读不到 $\mathbf{x}_0$。对随机标签 $\mathbf{u}_t(\mathbf{x}_t\mid\mathbf{x}_0)$ 作最小二乘时，给定 $\mathbf{x}_t=\mathbf{x}$ 的最优预测就是条件期望 $\mathbf{u}_t(\mathbf{x})$。回归条件速度与回归边缘速度因此指向同一个场；二者对 $\theta$ 的梯度相同，证明见第 4.2 节。

### 3.6 时间约定

直线路径有两种互为 $t\leftarrow 1-t$ 的写法。换元后速度反号，因为 $\mathrm{d}t$ 的方向反了。

| 约定 | $t=0$ | $t=1$ | 路径 | 条件速度 | 生成 |
|------|-------|-------|------|----------|------|
| Lipman；π₀ 论文 | 噪声 $\boldsymbol{\epsilon}$ | 数据 $\mathbf{A}$ | $(1-t)\boldsymbol{\epsilon}+t\mathbf{A}$ | $\mathbf{A}-\boldsymbol{\epsilon}$ | $t:0\to 1$ |
| openpi | 数据 $\mathbf{A}$ | 噪声 $\boldsymbol{\epsilon}$ | $(1-t)\mathbf{A}+t\boldsymbol{\epsilon}$ | $\boldsymbol{\epsilon}-\mathbf{A}$ | $t:1\to 0$ |

openpi 的 `pi0.py` 采用后者，注释写明与 π₀ 论文相反。下文公式与实现均按 openpi：$t=1$ 为纯噪声，$t=0$ 为目标动作。论文中的 $t$ 与 DDPM 的 $k=0$ 干净、$k=K$ 最噪也不对齐，阅读时先核对端点再套公式。

---

## 四、训练目标推导

### 4.1 从极大似然到散度项

行为克隆要求 $\pi_\theta(\mathbf{A}\mid\mathbf{O})\approx q_0(\mathbf{A}\mid\mathbf{O})$，即最小化

$$
\mathbb{E}_{(\mathbf{O},\mathbf{A})\sim\mathcal{D}}\big[-\log\pi_\theta(\mathbf{A}\mid\mathbf{O})\big].
$$

$\pi_\theta(\cdot\mid\mathbf{O})$ 是 $T_\theta$ 把 $p_1$ 推到 $t=0$ 所得的分布。似然可以由沿轨道的对数密度写出。设 $\mathbf{x}_t$ 服从 ODE (1)，$q_t$ 为边缘密度。这里 (1) 中的 $\mathbf{v}_t$ 即第 3.5 节定义的边缘速度 $\mathbf{u}_t$。沿轨道展开对数密度：

$$
\begin{aligned}
\frac{\mathrm{d}}{\mathrm{d}t}\log q_t(\mathbf{x}_t)
&=\frac{1}{q_t}\Big(\partial_t q_t+\nabla q_t\cdot\mathbf{v}_t\Big)\\
&=\frac{1}{q_t}\Big(\partial_t q_t+\nabla\cdot(q_t\mathbf{v}_t)-q_t\nabla\cdot\mathbf{v}_t\Big)\\
&=-\nabla\cdot\mathbf{v}_t(\mathbf{x}_t),
\end{aligned}
$$

其中第二步用了 $\nabla\cdot(q\mathbf{v})=\nabla q\cdot\mathbf{v}+q\nabla\cdot\mathbf{v}$，第三步用了连续性方程。记 $\mathbf{x}_1=\varphi_{0\to 1}(\mathbf{x}_0)$，$q_1$ 为 $q_0$ 被 $\mathbf{v}_t$ 推到 $t=1$ 的密度。从 $t=0$ 积到 $t=1$，

$$
\log q_0(\mathbf{x}_0)
=\log q_1(\mathbf{x}_1)+\int_0^{1}\nabla\cdot\mathbf{v}_t(\mathbf{x}_t)\,\mathrm{d}t.
$$

模型不估计 $q_1$。它把 $t=1$ 的密度取为源密度 $p_1$，速度取为 $\mathbf{v}_\theta$，于是 $\mathbf{x}_0$ 处的模型密度为

$$
\log\pi_\theta(\mathbf{x}_0\mid\mathbf{O})
=\log p_1(\mathbf{x}_1)+\int_0^{1}\nabla\cdot\mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{O})\,\mathrm{d}t.
$$

右端第一项是 $p_1$ 的对数密度，有闭式；第二项是速度场雅可比的迹，维数等于动作维 $H\cdot D_a$。每个似然评估都要沿 ODE 再积一次，迹的计算与维度成正比。FFJORD 一类方法用 Hutchinson 估计把迹换成随机投影，方差和求解步数仍然可观。机器人动作头不走这条路。

### 4.2 条件目标与边缘目标的梯度

第 3.5 节已将边缘速度写成条件速度的后验平均。记条件速度为 $\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_0)$，边缘速度为

$$
\mathbf{u}_t(\mathbf{x})
=\int
\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_0)\,
\frac{q_t(\mathbf{x}\mid\mathbf{x}_0)\,q_0(\mathbf{x}_0)}{q_t(\mathbf{x})}
\,\mathrm{d}\mathbf{x}_0
=\mathbb{E}\big[\mathbf{u}_t(\mathbf{x}_t\mid\mathbf{x}_0)\mid\mathbf{x}_t=\mathbf{x}\big].
$$

边缘 flow matching 直接回归这个不可得的标签：

$$
\mathcal{L}_{\mathrm{FM}}(\theta)
=\mathbb{E}_{t,\,\mathbf{x}\sim q_t}
\big\|\mathbf{v}_\theta(\mathbf{x},t)-\mathbf{u}_t(\mathbf{x})\big\|^2.
$$

条件 flow matching 改为回归样本上的 $\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_0)$。直线路径下 $\mathbf{u}_t(\mathbf{x}\mid\mathbf{A})=\boldsymbol{\epsilon}-\mathbf{A}$，并加上观测条件，即

$$
\boxed{
\mathcal{L}_{\mathrm{CFM}}(\theta)
=\mathbb{E}_{\mathbf{A},\boldsymbol{\epsilon},t,\mathbf{O}}
\left[
\big\|
\mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{O})
-(\boldsymbol{\epsilon}-\mathbf{A})
\big\|^2
\right]
}
\tag{6}
$$

其中 $\mathbf{x}_t$ 由 (3) 构造。下面证明：在同一个 $t$ 的分布下，$\mathcal{L}_{\mathrm{CFM}}$ 与 $\mathcal{L}_{\mathrm{FM}}$ 对 $\theta$ 的梯度相同。论证对每条高斯路径都成立，不限于直线；观测 $\mathbf{O}$ 作为两边共同的条件，略去不影响等式。

把平方展开。对固定的 $t$，简记 $\mathbf{u}_c=\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_0)$，$\mathbf{u}_m=\mathbf{u}_t(\mathbf{x})$，

$$
\begin{aligned}
\mathcal{L}_{\mathrm{CFM}}(t)
&=\mathbb{E}\|\mathbf{v}_\theta\|^2
-2\,\mathbb{E}[\mathbf{v}_\theta\cdot\mathbf{u}_c]
+\mathbb{E}\|\mathbf{u}_c\|^2,\\[4pt]
\mathcal{L}_{\mathrm{FM}}(t)
&=\mathbb{E}\|\mathbf{v}_\theta\|^2
-2\,\mathbb{E}[\mathbf{v}_\theta\cdot\mathbf{u}_m]
+\mathbb{E}\|\mathbf{u}_m\|^2.
\end{aligned}
$$

**第一项。** CFM 中的期望取在 $(x_0,x)$ 联合分布 $q_0(x_0)\,q_t(x\mid x_0)$ 上；对 $x_0$ 积分后，$x$ 的边缘即为 $q_t(x)$，因此

$$
\mathbb{E}_{(x_0,x)}\|\mathbf{v}_\theta(x,t)\|^2
=\mathbb{E}_{x\sim q_t}\|\mathbf{v}_\theta(x,t)\|^2,
$$

与 $\mathcal{L}_{\mathrm{FM}}(t)$ 的第一项相同，相减时抵消。

**交叉项。** CFM 的交叉项对后验取内层期望：

$$
\begin{aligned}
\mathbb{E}[\mathbf{v}_\theta\cdot\mathbf{u}_c]
&=\mathbb{E}_{x\sim q_t}
\Big[\mathbf{v}_\theta(x,t)\cdot
\mathbb{E}_{x_0\mid x}[\mathbf{u}_t(x\mid x_0)]\Big]\\
&=\mathbb{E}_{x\sim q_t}[\mathbf{v}_\theta(x,t)\cdot\mathbf{u}_m(x)].
\end{aligned}
$$

这正是 $\mathcal{L}_{\mathrm{FM}}(t)$ 的交叉项，同样抵消。

**平方项之差。** 前两步抵消后：

$$
\mathcal{L}_{\mathrm{CFM}}(t)-\mathcal{L}_{\mathrm{FM}}(t)
=\mathbb{E}\|\mathbf{u}_c\|^2-\mathbb{E}\|\mathbf{u}_m\|^2.
$$

将右侧化为方差形式。展开 $\mathbb{E}\|\mathbf{u}_c-\mathbf{u}_m\|^2$：

$$
\mathbb{E}\|\mathbf{u}_c-\mathbf{u}_m\|^2
=\mathbb{E}\|\mathbf{u}_c\|^2
-2\,\mathbb{E}[\mathbf{u}_c\cdot\mathbf{u}_m]
+\mathbb{E}\|\mathbf{u}_m\|^2.
$$

交叉项 $\mathbb{E}[\mathbf{u}_c\cdot\mathbf{u}_m]$ 同样对后验取条件期望，由 $\mathbf{u}_m$ 的定义立得 $\mathbb{E}_{x_0\mid x}[\mathbf{u}_c]=\mathbf{u}_m$，故

$$
\mathbb{E}[\mathbf{u}_c\cdot\mathbf{u}_m]
=\mathbb{E}_{x\sim q_t}\big[\mathbf{u}_m(x)\cdot\mathbb{E}_{x_0\mid x}[\mathbf{u}_c]\big]
=\mathbb{E}\|\mathbf{u}_m\|^2.
$$

代回即得 $\mathbb{E}\|\mathbf{u}_c-\mathbf{u}_m\|^2
=\mathbb{E}\|\mathbf{u}_c\|^2-\mathbb{E}\|\mathbf{u}_m\|^2$。

**结论。** 综合以上，

$$
\boxed{
\mathcal{L}_{\mathrm{CFM}}(t)-\mathcal{L}_{\mathrm{FM}}(t)
=\mathbb{E}\big\|\mathbf{u}_t(\mathbf{x}\mid\mathbf{x}_0)-\mathbf{u}_t(\mathbf{x})\big\|^2
\ge 0
},
$$

即条件速度的后验方差，与 $\theta$ 无关。对 $t$ 取任何采样密度，梯度仍然相等：

$$
\nabla_\theta\mathcal{L}_{\mathrm{CFM}}(\theta)=\nabla_\theta\mathcal{L}_{\mathrm{FM}}(\theta).
$$

$t$ 可以取均匀分布，也可以取第 6.3 节的 Beta 分布。等价关系是逐个 $t$ 成立的，更换 $t$ 的采样密度只改变各时刻在总损失中的权重，不改变该时刻上的最优场。最小化 (6) 得到的 $\mathbf{v}_\theta$ 因而生成边缘路径 $q_t$，其 $t=0$ 端就是数据分布。

直线路径把标签进一步简化为与 $t$ 无关的 $\boldsymbol{\epsilon}-\mathbf{A}$。网络仍然接收 $t$，因为边缘场 $\mathbf{u}_t(\mathbf{x})=\mathbb{E}[\boldsymbol{\epsilon}-\mathbf{A}\mid\mathbf{x}_t=\mathbf{x}]$ 依赖 $t$：不同时刻、到达同一 $\mathbf{x}$ 的样本来自不同的后验。

### 4.3 与 score、噪声回归的关系

Song et al. 指出，DDPM 的 $\epsilon$-prediction 与 score matching 差一个由噪声日程决定的因子。直线路径上，速度回归与这二者也差一个由 $(\mathbf{x}_t,t)$ 决定的仿射变换。本小节把这个变换写出来。结论是：三者的总体最优预测互相决定；损失在 $t$ 上的权重不同；推理时速度场可以直接积分，噪声预测还要折回后验均值。

(5) 是各向同性高斯。$\mathcal{N}(\mu,\sigma^{2}I)$ 的对数密度对 $\mathbf{x}$ 的梯度为 $-(\mathbf{x}-\mu)/\sigma^{2}$。代入 $\mu=(1-t)\mathbf{x}_0$、$\sigma=t$，

$$
\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t\mid\mathbf{x}_0)
=-\frac{\mathbf{x}_t-(1-t)\mathbf{x}_0}{t^{2}}.
\tag{7}
$$

由 (3)，$\mathbf{x}_t-(1-t)\mathbf{x}_0=t\boldsymbol{\epsilon}$，故条件 score 就是

$$
\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t\mid\mathbf{x}_0)=-\frac{\boldsymbol{\epsilon}}{t}.
$$

这与 DP 笔记第五节的方差保持公式同一结构。那里 $\nabla_{\mathbf{x}_k}\log q(\mathbf{x}_k\mid\mathbf{x}_0)=-\boldsymbol{\epsilon}/\sqrt{1-\bar\alpha_k}$，因子来自噪声标准差；这里标准差是 $t$，因子就是 $t$。

再把速度标签代进去。令 $\mathbf{u}=\boldsymbol{\epsilon}-\mathbf{x}_0$，则 $\boldsymbol{\epsilon}=\mathbf{u}+\mathbf{x}_0$，

$$
\begin{aligned}
\mathbf{x}_t
&=(1-t)\mathbf{x}_0+t(\mathbf{u}+\mathbf{x}_0)\\
&=\mathbf{x}_0+t\mathbf{u}.
\end{aligned}
$$

解出端点：

$$
\mathbf{x}_0=\mathbf{x}_t-t\mathbf{u},
\qquad
\boldsymbol{\epsilon}=\mathbf{x}_t+(1-t)\mathbf{u}.
\tag{8}
$$

代回条件 score，

$$
\nabla_{\mathbf{x}_t}\log q_t(\mathbf{x}_t\mid\mathbf{x}_0)
=-\frac{\boldsymbol{\epsilon}}{t}
=-\frac{\mathbf{x}_t}{t}-\frac{1-t}{t}\,\mathbf{u}.
\tag{9}
$$

(9) 对 $\mathbf{u}$ 是仿射的。斜率 $-(1-t)/t$ 与截距 $-\mathbf{x}_t/t$ 只依赖网络已经看见的 $(\mathbf{x}_t,t)$。反方向同样是仿射的：由 $\boldsymbol{\epsilon}=-t\nabla\log q_t(\cdot\mid\mathbf{x}_0)$ 与 (8)，

$$
\mathbf{u}
=\frac{\boldsymbol{\epsilon}-\mathbf{x}_t}{1-t}
=\frac{-t\,\nabla\log q_t(\mathbf{x}_t\mid\mathbf{x}_0)-\mathbf{x}_t}{1-t}.
$$

这些等式是样本层面的。回归得到的是条件期望。对 (8) 的第二式在给定 $\mathbf{x}_t$ 时取期望，$\mathbf{x}_t$ 可提出来，

$$
\mathbb{E}[\boldsymbol{\epsilon}\mid\mathbf{x}_t]
=\mathbf{x}_t+(1-t)\,\mathbb{E}[\boldsymbol{\epsilon}-\mathbf{x}_0\mid\mathbf{x}_t].
$$

因此边缘速度 $\mathbf{u}_t(\mathbf{x}_t)=\mathbb{E}[\boldsymbol{\epsilon}-\mathbf{A}\mid\mathbf{x}_t]$ 与 $\mathbb{E}[\boldsymbol{\epsilon}\mid\mathbf{x}_t]$ 由同一个仿射关系联系；边缘 score 是条件 score 的后验平均，等于 $-\mathbb{E}[\boldsymbol{\epsilon}\mid\mathbf{x}_t]/t$。在模型容量足以表示该函数时，速度回归、噪声回归与 score 回归的总体最小化解互相决定。这就是线性路径上的信息等价。

损失的数值权重并不相同。若由速度预测按 (8) 换算噪声预测 $\hat{\boldsymbol{\epsilon}}=\mathbf{x}_t+(1-t)\mathbf{v}_\theta$，残差差一个因子：

$$
\begin{aligned}
\boldsymbol{\epsilon}-\mathbf{x}_t
&=(1-t)(\boldsymbol{\epsilon}-\mathbf{x}_0),\\
\boldsymbol{\epsilon}-\hat{\boldsymbol{\epsilon}}
&=(1-t)\big((\boldsymbol{\epsilon}-\mathbf{x}_0)-\mathbf{v}_\theta\big),
\end{aligned}
$$

故

$$
\|\boldsymbol{\epsilon}-\hat{\boldsymbol{\epsilon}}\|^2
=(1-t)^{2}\,\|\mathbf{u}-\mathbf{v}_\theta\|^2.
$$

固定 $t$ 时 $(1-t)^{2}$ 是常数，两个最小化解相同。对 $t$ 积分之后，噪声回归相对速度回归多乘了 $(1-t)^{2}$。$t=1$ 时 $\mathbf{x}_t=\boldsymbol{\epsilon}$，该因子为 $0$，噪声损失对任何有限的 $\mathbf{v}_\theta$ 都是 $0$。速度标签 $\boldsymbol{\epsilon}-\mathbf{A}$ 仍然依赖 $\mathbf{A}$，纯噪声端的监督由速度回归承担。

DP 的推理不能停在 $\epsilon_\theta$ 上。反向一步的均值是

$$
\mu_\theta
=\frac{1}{\sqrt{\alpha_k}}
\left(
\mathbf{x}_k-\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\,\epsilon_\theta
\right),
$$

系数来自前向后验，必须与训练用的日程一致。FM 的网络输出就是 (1) 的右端，离散化时只乘 $\Delta t$（第五节）。训练时 $t$ 在 $(0,1)$ 上按某个密度采样，推理步数 $N$ 另行选取，二者不必共用一套网格。

### 4.4 推导链

```text
min E[−log π_θ(A | O)]                         行为克隆，目标分布 q_0
   ↓  π_θ 为 T_θ 推动 p_1 的分布，T_θ = φ_{1→0}
log q_0(x_0) = log q_1(x_1) + ∫ ∇·v_t dt       q_1 是 q_0 的推前向
   ↓  模型把 t=1 的密度取为 p_1，速度取为 v_θ
log π_θ(x_0) = log p_1(x_1) + ∫ ∇·v_θ dt       迹与动作维成正比
   ↓  改为回归条件速度 u；4.2 节证明梯度与边缘速度 u_t(x) 一致
L_CFM = E[ ||v_θ(x_t, t, O) − (ε − A)||² ]
   ↓  直线路径上 u = ε − A 为常向量，且 x_1 = ε
   ↓  4.3 节：u 与条件 score、与 ε 相差 (x_t, t) 的仿射变换
训练 = 回归条件速度；学到的 v_θ 等于边缘速度 u_t(x)
推理 = 从 x_1 ~ p_1 对 v_θ 作 Euler 积分，不再使用 α_k, β_k
```

---

## 五、推理：ODE 数值积分

### 5.1 积分形式

训练结束后 $\mathbf{v}_\theta$ 固定。给定观测 $\mathbf{O}$，抽一次 $\mathbf{x}_1\sim p_1$。直线路径上 $\mathbf{x}_1=\boldsymbol{\epsilon}$。求初值问题

$$
\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t}=\mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{O}),
\qquad
\mathbf{x}\big|_{t=1}=\mathbf{x}_1,
$$

在 $t=0$ 的值。写成积分，

$$
\mathbf{x}_0
=\mathbf{x}_1+\int_1^{0}\mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{O})\,\mathrm{d}t.
$$

$\mathbf{O}$ 在整个积分中保持不变。高斯随机性只出现在 $\mathbf{x}_1$；给定 $\mathbf{x}_1$ 与 $\mathbf{O}$，轨道由 ODE 唯一确定。DDPM 反向核在每一步另抽 $z\sim\mathcal{N}(0,I)$，那一项在这里不存在。

### 5.2 前向 Euler

把 $[0,1]$ 分成 $N$ 段，步长 $\Delta t=-1/N$。前向 Euler 为

$$
\mathbf{x}_{t+\Delta t}
=\mathbf{x}_t+\Delta t\,\mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{O}).
\tag{10}
$$

$N$ 即 `num_steps`，openpi 默认 $10$。循环写成

$$
\mathbf{x}\leftarrow\mathbf{x}_1,\qquad t\leftarrow 1,
$$

然后重复 $N$ 次

$$
\mathbf{x}\leftarrow\mathbf{x}-\frac{1}{N}\mathbf{v}_\theta(\mathbf{x},t,\mathbf{O}),
\qquad
t\leftarrow t-\frac{1}{N}.
$$

$t$ 从 $1$ 减到 $0$，状态从噪声走到动作。每一步的网络输入是当前的 $(\mathbf{x},t)$ 和一开始就固定的 $\mathbf{O}$，不是重新采样的高斯。$N=1$ 时循环只跑一次，

$$
\mathbf{x}\leftarrow\mathbf{x}_1-\mathbf{v}_\theta(\mathbf{x}_1,1,\mathbf{O}),
$$

退化为单步映射。π₀ 里图像与语言的前向只做一次并写入 KV cache，每个 Euler 步只重算动作专家；这一实现细节放在第 6.4 节，不改变上述积分。

### 5.3 截断误差与步数

设 $\mathbf{a}_t=\mathrm{d}\mathbf{v}_\theta/\mathrm{d}t$ 为速度沿轨道的全导数，$\mathbf{a}_t=\partial_t\mathbf{v}_\theta+(\nabla_{\mathbf{x}}\mathbf{v}_\theta)\mathbf{v}_\theta$。Taylor 展开在 $t$ 与 $t+\Delta t$ 之间的某点 $\xi$ 取值：

$$
\mathbf{x}(t+\Delta t)
=\mathbf{x}(t)+\Delta t\,\mathbf{v}_\theta+\frac{(\Delta t)^{2}}{2}\mathbf{a}_{\xi}.
$$

局部截断为 $O((\Delta t)^{2})$。在 Lipschitz 条件下，$N$ 步的全局误差为 $O(|\Delta t|)=O(1/N)$。

若速度在 $(\mathbf{x},t)$ 上为常向量，则 $\mathbf{a}_t=0$，一步即精确。条件速度 (4) 正是这种情形。验证：取 $\mathbf{u}_t=\boldsymbol{\epsilon}-\mathbf{x}_0$，从 $\mathbf{x}(1)=\boldsymbol{\epsilon}$ 出发以 $\Delta t=-1$ 更新，

$$
\mathbf{x}(0)
=\boldsymbol{\epsilon}-(\boldsymbol{\epsilon}-\mathbf{x}_0)
=\mathbf{x}_0.
$$

单条条件线段上，Euler 没有离散化误差。网络看不到 $(\mathbf{x}_0,\boldsymbol{\epsilon})$，它拟合的是边缘速度 $\mathbf{u}_t(\mathbf{x})=\mathbb{E}[\boldsymbol{\epsilon}-\mathbf{A}\mid\mathbf{x}_t]$。不同的 $\mathbf{x}_0$ 经过同一 $\mathbf{x}_t$ 时 $\mathbf{u}_t(\cdot\mid\mathbf{x}_0)$ 不同，平均后的 $\mathbf{u}_t(\mathbf{x})$ 依赖 $\mathbf{x}$ 与 $t$，$\mathbf{a}_t$ 一般不为零，轨道是弯的。$N=1$ 只沿初始点的速度走完全程，$N$ 增大则跟着 $\mathbf{u}_t(\mathbf{x})$ 的弯曲修正方向。Liu et al. (2023) 说明，独立配对下的直线插值已经使边缘轨迹比扩散概率流更直；π₀ 取 $N=10$。步数再增会线性增加动作专家的前向次数。训练时的 $t$ 采样与这个 $N$ 无关。

---

## 六、π₀ 中的 Flow Matching

### 6.1 问题形式化

π₀ 将预训练 VLM 扩展为 vision-language-action（VLA）模型。记多视角图像为 $o$、语言指令为 $l$、本体感觉为 $\mathbf{s}$，三者合起来就是条件 $\mathbf{O}$。输出长度 $H$ 的动作 chunk $\mathbf{A}\in\mathbb{R}^{H\times D_a}$。策略为

$$
\mathbf{A} = T_\theta(\boldsymbol{\epsilon};\, o, l, \mathbf{s}),
\qquad
\boldsymbol{\epsilon}\sim p_1,
$$

其中 $T_\theta=\varphi_{1\to 0}$（第 3.2 节），直线路径上 $\boldsymbol{\epsilon}=\mathbf{x}_1$。$(o,l,\mathbf{s})$ 经 VLM 编码后作为 $\mathbf{v}_\theta$ 的条件。控制频率可达 $50\,\mathrm{Hz}$，依赖 chunk 执行与 receding horizon，而非逐步自回归。

### 6.2 架构

π₀ 采用 **双专家** Transformer（Transfusion 风格）：

```text
Prefix（一次前向，KV cache）
  图像 token ──→ SigLIP 编码 ──┐
  语言 token ──→ PaliGemma embed ──┴──→ 共享注意力

Suffix（每 Euler 步更新）
  [state token]（π₀；π₀.₅ 省略，改 adaRMS 注入时间）
  带噪动作 token × H ──→ action expert（Gemma 小专家）
  时间 t ──→ sin-cos 嵌入 ──→ 与动作 token 融合 / adaRMS 条件
                              ↓
                         action_out_proj → v_θ
```

- **PaliGemma**：预训练 VLM，提供 Internet 规模视觉–语言先验；
- **action expert**：较小 Gemma 变体，专责动作 token 的自注意力与对 prefix 的 cross-attention；
- **attention mask**（`make_attn_mask`）：图像与语言同一块，内部双向；state 单独成块，可见 prefix，不可见动作；动作 chunk 内部双向，可见 prefix 与 state。prefix 不可见 suffix。

π₀.₅ 相对 π₀ 的主要改动：去掉显式 state token，时间嵌入经 MLP 后以 adaRMS 注入 action expert；训练采用 knowledge insulation 以改善开集泛化。openpi 仓库对 π₀.₅ 目前仅开放 flow matching 头。

### 6.3 训练（`Pi0.compute_loss`）

从 openpi 实现，单步损失构造如下：

1. 采样 $\boldsymbol{\epsilon}\sim p_1$，与 $\mathbf{A}$ 同形；
2. 采样 $t\sim\mathrm{Beta}(1.5,1)\times 0.999+0.001$；
3. 构造 $\mathbf{x}_t = t\boldsymbol{\epsilon}+(1-t)\mathbf{A}$；
4. 目标为条件速度 $\mathbf{u}_t(\mathbf{x}_t\mid\mathbf{A})=\boldsymbol{\epsilon}-\mathbf{A}$；
5. 单次前向：prefix + suffix，输出 $\mathbf{v}_\theta$；
6. 标量损失对 $H\times D_a$ 个分量取均方：
$$\mathcal{L}=\frac{1}{H D_a}\big\|\mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{O})-\mathbf{u}_t(\mathbf{x}_t\mid\mathbf{A})\big\|^2.$$
`compute_loss` 先做 `mean(square(v_t - u_t), axis=-1)`，只平均动作维 $D_a$，返回形状为 `(batch, H)`。训练步再对 batch 与 $H$ 取平均。各步的 $D_a$ 相同，两次平均等于对全部 $H\times D_a$ 个分量取平均。

(6) 式是平方范数的期望，即各分量平方之和。上式与它相差正常数 $1/(H D_a)$。$H$、$D_a$ 固定时最小化解相同，梯度相差同一个正倍数。$\mathrm{Beta}(1.5,1)$ 的密度为 $f(t)=\tfrac{3}{2}\sqrt{t}$，在 $(0,1)$ 上递增，均值 $1.5/2.5=0.6$，相对均匀分布把质量移向较大的 $t$（更接近纯噪声的一段）。仿射 $0.999\,t+0.001$ 把样本限制在 $[0.001,1]$，避开 $t=0$。第 4.2 节已说明，更换 $t$ 的密度不改变各时刻的最优速度场，只改变各时刻损失的权重。

### 6.4 推理（`Pi0.sample_actions`）

1. **Prefix 编码**：图像与语言 token 一次前向，填充 KV cache；
2. **初始化**：$\mathbf{x}\leftarrow\mathbf{x}_1$，$t\leftarrow 1$，直线路径上 $\mathbf{x}_1=\boldsymbol{\epsilon}$；
3. **Euler 循环**（$N$ 步，`while_loop`）：
   - 以当前 $(\mathbf{x}, t)$ 构造 suffix token；
   - action expert 在 cached prefix 条件下前向，得 $\mathbf{v}_\theta$；
   - 更新 $\mathbf{x}\leftarrow\mathbf{x}+(\Delta t)\mathbf{v}_\theta$，$t\leftarrow t+\Delta t$，$\Delta t=-1/N$；
4. **输出**：$t\approx 0$ 时的 $\mathbf{x}$ 作为动作 chunk。

Prefix 只编码一次、suffix 每步重算是实机低延迟的关键：$N=10$ 时 VLM 主干仅运行 $1$ 次，其余为 action expert 上的轻量迭代。

---

## 七、与 Diffusion Policy 的对照

| 维度 | Diffusion Policy | π₀（FM） |
|------|------------------|----------|
| 条件分布 | $q_0(\mathbf{A}\mid\mathbf{O})$ | $q_0(\mathbf{A}\mid o,l,\mathbf{s})$ |
| 参数化 | $\epsilon_\theta(\mathbf{A}^k,k,\mathbf{O})$ | $\mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{O})$ |
| 路径 | $\sqrt{\bar\alpha_k}\mathbf{A}+\sqrt{1-\bar\alpha_k}\boldsymbol{\epsilon}$，边缘方差约保持为 $1$ | $(1-t)\mathbf{A}+t\boldsymbol{\epsilon}$，标准化后 $t=1/2$ 时方差为 $1/2$ |
| 训练监督 | 预测噪声 $\boldsymbol{\epsilon}$ | 预测速度 $\boldsymbol{\epsilon}-\mathbf{A}$ |
| 推理 | $k=K,\ldots,1$，每步可加 $z$；DDIM（$\eta=0$）的系数不同于 $\mu_\theta$ | Euler $N$ 步，$t:1\to 0$；噪声只在初始 $\mathbf{x}_1$ |
| 骨干 | 1D U-Net | VLM + action expert |
| 预训练 | 通常无 | 10k+ 小时机器人数据 + VLM |

路径公式与边缘协方差见第 3.4 节，边缘密度和边缘速度的配对见第 3.5 节，速度与 $\epsilon$、score 的仿射关系见第 4.3 节。Euler 与 DP 反向一步的对应见第 3.1 与第五节。二者都在做生成式模仿，对象都是多模态的动作 chunk。生成头与是否使用 VLM 先验是其余的主要差别。LeRobot 的 `MultiTaskDiT` 在同一 Diffusion Transformer 上切换 `objective=diffusion` 与 `objective=flow_matching`。其流匹配取 $t=0$ 为噪声，条件速度为 $\mathbf{A}-\boldsymbol{\epsilon}$，端点与 openpi 相反。

---

## 八、训练与推理流程

### 8.1 训练

```text
1. 演示转为 LeRobot 格式（openpi 微调入口）

2. 每个 step：
   a. 采样 batch (O, A)
   b. ε ~ p_1，t ~ Beta(1.5,1)（或 Uniform）
   c. x_t = t·ε + (1-t)·A
   d. v_θ = network(O, x_t, t)
   e. u = ε − A；L = mean (v_θ − u)²
   f. 反向传播；可选 EMA、LoRA

3. 定期在仿真/实机评估
```

### 8.2 推理

```text
1. 读取图像、语言 prompt、本体感觉

2. Prefix 前向 → KV cache

3. x_1 ~ p_1，直线路径上 x_1 = ε，t = 1

4. repeat N times:
       x ← x − v_θ(O, x, t)/N
       t ← t − 1/N

5. 输出 x 为 action chunk；执行前 T_a 步

6. T_a 步后回到步骤 1（可选 RTC 平滑 chunk 衔接，见 LeRobot 文档）
```

---

## 九、关键超参数

| 参数 | openpi / π₀ 典型值 | 说明 |
|------|-------------------|------|
| `action_horizon` ($H$) | 50（π₀ 高频任务） | 单次预测动作步数 |
| `action_dim` ($D_a$) | 本体相关 | 关节 / 末端 / 夹爪 |
| `num_steps` ($N$) | 10 | Euler 积分步数 |
| $t$ 采样 | Beta$(1.5,1)$，密度 $\tfrac{3}{2}\sqrt{t}$ | 均值 $0.6$，侧重较大的 $t$ |
| VLM | PaliGemma | 预训练视觉–语言骨干 |
| action expert | Gemma 300M 级 | 动作专用 Transformer |
| 微调 | LoRA / 全参 | 全参需 $>70\,\mathrm{GB}$ GPU |

---

## 十、参考文献与延伸阅读

1. Black, K., et al. *π₀: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164, 2024.
2. Lipman, Y., et al. *Flow Matching for Generative Modeling*. ICLR, 2023.
3. Liu, Q., et al. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR, 2023.
4. Song, Y., et al. *Score-Based Generative Modeling through Stochastic Differential Equations*. ICLR, 2021.
5. Ho, J., Jain, A., Abbeel, P. *Denoising Diffusion Probabilistic Models*. NeurIPS, 2020.
6. Chi, C., et al. *Diffusion Policy*. RSS, 2023. [`diffusion_policy_notes.md`](diffusion_policy_notes.md)
7. Grathwohl, W., et al. *FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models*. ICLR, 2019.
8. Physical Intelligence. [openpi](https://github.com/Physical-Intelligence/openpi) — π₀ / π₀.₅ 官方实现。
9. 视频–动作联合 CFM：[`wam_notes.md`](wam_notes.md) 第三节。

建议阅读顺序：第三节的路径与第 4.2–4.3 节的梯度、score 推导 → Lipman et al. 2023 → Black et al. 2024 与 openpi `src/openpi/models/pi0.py` 中的 `compute_loss`、`sample_actions` → 与 [`diffusion_policy_notes.md`](diffusion_policy_notes.md) 第 2.4 节、第五节对照前向边际与 $\epsilon$-prediction。
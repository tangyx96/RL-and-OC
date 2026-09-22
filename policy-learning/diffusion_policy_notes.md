# Diffusion Policy 算法笔记

下文公式与流程对应的实现，见官方仓库 [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)（Chi et al., RSS 2023；[arXiv:2303.04137](https://arxiv.org/pdf/2303.04137v5)）。  
同一动作序列接口下的 flow matching 头见 [`flow_matching_notes.md`](flow_matching_notes.md)（π₀ / openpi）。

---

## 一、算法定位

Diffusion Policy 将 **visuomotor 策略**（视觉–运动策略）参数化为**动作空间上的条件去噪扩散过程**（conditional denoising diffusion process）。Visuomotor 指以相机图像（常辅以本体感觉）为输入、以机器人动作为输出的闭环映射，与以低维状态为观测的策略相对；论文标题中的 *Visuomotor Policy Learning* 即这一设定。给定最近若干步观测 $\mathbf{O}_t$，策略不直接回归单步动作，而是对长度为 $T_p$ 的动作序列 $\mathbf{A}_t$ 执行迭代去噪，从中取出 $T_a$ 步执行，并在下一控制周期重新规划。

该方法属于**离线模仿学习**：训练数据为专家演示轨迹，不使用环境奖励，也不维护价值函数。优化目标是拟合条件数据分布 $p(\mathbf{A}_t \mid \mathbf{O}_t)$。专家分布无闭式密度，演示集仅提供有限样本；策略以生成过程参数化该条件分布并从中抽样。观测 $\mathbf{O}_t$ 可以是图像、低维状态或二者拼接，作为条件，不参与扩散。

| 特性 | 行为克隆（MSE / 高斯） | Diffusion Policy |
|------|------------------------|------------------|
| 学习范式 | 离线模仿 | 离线模仿 |
| 训练信号 | 专家动作 | 专家动作序列 + 噪声监督 |
| 策略输出 | 单步点估计或对角高斯 | 动作序列 $\mathbf{A}_t \in \mathbb{R}^{T_p \times D_a}$ |
| 多模态 | 回归收敛到条件均值，通常单峰 | 生成式，可表达多峰 |
| 核心网络 | MLP / RNN | 1D U-Net 或 Transformer + 视觉编码器 |
| 推理 | 一次前向 | $K$ 步（或 DDIM 的 $N<K$ 步）迭代去噪 |

经典 MDP 的单步观测–单步动作形式是上述框架的特例：$T_o = T_a = T_p = 1$。

---

## 二、前置基础

### 2.1 Visuomotor 策略

策略是从观测到动作的条件分布。按观测类型可区分为

$$
\pi(\mathbf{a}_t \mid \mathbf{s}_t)
\qquad\text{与}\qquad
\pi(\mathbf{a}_t \mid \mathbf{o}_t^{\mathrm{img}}, \mathbf{o}_t^{\mathrm{proprio}}),
$$

前者以仿真器或状态估计给出的低维向量 $\mathbf{s}_t$（如关节角、物体位姿）为输入，后者以图像 $\mathbf{o}_t^{\mathrm{img}}$ 为主、常拼接本体感觉 $\mathbf{o}_t^{\mathrm{proprio}}$。后者即 visuomotor 策略；写成动作序列形式则为 $\pi(\mathbf{A}_t \mid \mathbf{O}_t)$。

二者的差别在观测，不在优化算法。实机通常无法获得仿真中那种完整状态，策略须从像素中提取任务相关信息。端到端 visuomotor 指视觉编码器与动作头联合训练，而不是先做独立的检测或位姿估计，再交给另一套控制器。仓库同时提供状态观测与图像观测两种实现；第七节的视觉编码器对应图像分支。

### 2.2 行为克隆

设演示 $\mathcal{D} = \{(\mathbf{o}_i, \mathbf{a}_i)\}_{i=1}^N$ 独立取自专家联合分布 $p^{\ast}(\mathbf{o},\mathbf{a})$。将策略参数化为条件密度 $\pi_\theta(\mathbf{a}\mid\mathbf{o})$。观测的边缘 $p^{\ast}(\mathbf{o})$ 不依赖 $\theta$，故 $\mathcal{D}$ 在模型下的似然为

$$
p_\theta(\mathcal{D})=\prod_{i=1}^N\pi_\theta(\mathbf{a}_i\mid\mathbf{o}_i).
$$

极大似然估计选取使该似然最大的参数，

$$
\hat\theta_{\mathrm{MLE}}=\arg\max_\theta\, p_\theta(\mathcal{D}).
$$

$\log$ 在 $(0,\infty)$ 上严格递增，因而 $p_\theta(\mathcal{D})$ 与 $\log p_\theta(\mathcal{D})$ 的最大值点相同：

$$
\hat\theta_{\mathrm{MLE}}=\arg\max_\theta\log p_\theta(\mathcal{D}).
$$

由独立抽样，

$$
\log p_\theta(\mathcal{D})
=\sum_{i=1}^N\log\pi_\theta(\mathbf{a}_i\mid\mathbf{o}_i).
$$

对任意 $c>0$，$\arg\max_\theta f(\theta)=\arg\max_\theta\, c\,f(\theta)$。取 $c=1/N$，上式与样本均值同最优。将 $\mathcal{D}$ 视为经验分布 $\hat p_{\mathcal{D}}=\frac{1}{N}\sum_{i=1}^{N}\delta_{(\mathbf{o}_i,\mathbf{a}_i)}$，该均值即经验期望，故

$$
\hat\theta_{\mathrm{MLE}}
=\arg\max_\theta\sum_{i=1}^N\log\pi_\theta(\mathbf{a}_i\mid\mathbf{o}_i)
=\arg\max_\theta\;
\mathbb{E}_{(\mathbf{o},\mathbf{a})\sim\mathcal{D}}\big[\log\pi_\theta(\mathbf{a}\mid\mathbf{o})\big].
$$

记号 $\mathbb{E}_{(\mathbf{o},\mathbf{a})\sim\mathcal{D}}$ 表示对 $\mathcal{D}$ 中 $N$ 个样本的算术平均；$N\to\infty$ 时依大数定律收敛到 $\mathbb{E}_{p^{\ast}}[\log\pi_\theta]$。训练中通常改为最小化负对数似然

$$
\mathcal{L}_{\mathrm{NLL}}(\theta)
=\mathbb{E}_{(\mathbf{o},\mathbf{a})\sim\mathcal{D}}\big[-\log\pi_\theta(\mathbf{a}\mid\mathbf{o})\big],
$$

它与 $\max_\theta p_\theta(\mathcal{D})$ 仅差符号。

同一目标可写成条件正向 KL。对固定 $\mathbf{o}$，

$$
D_{\mathrm{KL}}\big(p^{\ast}(\cdot\mid\mathbf{o})\,\big\|\,\pi_\theta(\cdot\mid\mathbf{o})\big)
=\int p^{\ast}(\mathbf{a}\mid\mathbf{o})\,
\log\frac{p^{\ast}(\mathbf{a}\mid\mathbf{o})}{\pi_\theta(\mathbf{a}\mid\mathbf{o})}\,
\mathrm{d}\mathbf{a}.
$$

由 $\log(u/v)=\log u-\log v$，

$$
\begin{aligned}
D_{\mathrm{KL}}
&=\int p^{\ast}(\mathbf{a}\mid\mathbf{o})\log p^{\ast}(\mathbf{a}\mid\mathbf{o})\,\mathrm{d}\mathbf{a}
-\int p^{\ast}(\mathbf{a}\mid\mathbf{o})\log\pi_{\theta}(\mathbf{a}\mid\mathbf{o})\,\mathrm{d}\mathbf{a}\\
&=\mathbb{E}_{p^{\ast}(\cdot\mid\mathbf{o})}\big[\log p^{\ast}(\mathbf{a}\mid\mathbf{o})\big]
-\mathbb{E}_{p^{\ast}(\cdot\mid\mathbf{o})}\big[\log\pi_{\theta}(\mathbf{a}\mid\mathbf{o})\big].
\end{aligned}
$$

第一项不依赖 $\theta$。因此

$$
\arg\min_\theta\,
D_{\mathrm{KL}}\big(p^{\ast}(\cdot\mid\mathbf{o})\,\big\|\,\pi_\theta(\cdot\mid\mathbf{o})\big)
=\arg\max_\theta\,
\mathbb{E}_{p^{\ast}(\cdot\mid\mathbf{o})}\big[\log\pi_\theta(\mathbf{a}\mid\mathbf{o})\big].
$$

再对 $p^{\ast}(\mathbf{o})$ 取期望，

$$
\mathbb{E}_{\mathbf{o}\sim p^{\ast}}\Big[
D_{\mathrm{KL}}\big(p^{\ast}(\cdot\mid\mathbf{o})\,\big\|\,\pi_\theta(\cdot\mid\mathbf{o})\big)
\Big]
=C+\mathbb{E}_{(\mathbf{o},\mathbf{a})\sim p^{\ast}}\big[-\log\pi_\theta(\mathbf{a}\mid\mathbf{o})\big],
$$

其中 $C$ 与 $\theta$ 无关。用经验分布代替 $p^{\ast}$ 即得 $\mathcal{L}_{\mathrm{NLL}}$。故极大似然等价于在专家占用上最小化 $D_{\mathrm{KL}}(p^{\ast}(\cdot\mid\mathbf{o})\,\|\,\pi_\theta(\cdot\mid\mathbf{o}))$。

若进一步取 $\pi_\theta(\mathbf{a}\mid\mathbf{o})=\mathcal{N}(\mu_\theta(\mathbf{o}),\sigma^{2}I)$ 且 $\sigma$ 不作为可学习参数，则

$$
-\log\pi_\theta(\mathbf{a}\mid\mathbf{o})
=\frac{1}{2\sigma^{2}}\|\mathbf{a}-\mu_\theta(\mathbf{o})\|^{2}
+\frac{d}{2}\log(2\pi\sigma^{2}).
$$

第二项与 $\theta$ 无关，于是 $\mathcal{L}_{\mathrm{NLL}}$ 与最小二乘

$$
\mathcal{L}_{\mathrm{BC}}(\theta)
=\mathbb{E}_{(\mathbf{o},\mathbf{a})\sim\mathcal{D}}\big[\|\mu_\theta(\mathbf{o})-\mathbf{a}\|^{2}\big]
$$

同最优。确定性回归 BC 即此特例：网络输出条件均值。GMM、能量模型与扩散策略不满足该高斯假设，仍以 $\mathcal{L}_{\mathrm{NLL}}$ 或其等价形式（去噪、score matching）为训练目标。

该方法实现简单，但训练分布为专家占用，部署时策略误差使状态分布偏移，误差沿时间累积（covariate shift）。若同一 $\mathbf{o}$ 对应多种合理动作，$\mathcal{L}_{\mathrm{BC}}$ 收敛到 $\mathbb{E}[\mathbf{a}\mid\mathbf{o}]$，动作被平均化，多模态任务上性能下降。

### 2.3 现有模仿学习方法的局限

| 方法 | 多模态 | 主要困难 |
|------|--------|----------|
| MLP 回归 BC | 否 | 动作模糊 |
| BC-GMM | 需指定模态数 $K$ | $K$ 难定；模态坍缩 |
| BC-RNN | 弱 | 误差累积；仍偏单峰 |
| BET（离散化 + 自回归） | 有 | 动作空间离散化损失精度 |
| IBC（隐式 BC / 能量模型） | 有 | 高维动作空间采样困难；训练需负样本 |

Diffusion Policy 的出发点，是将 DDPM 在图像生成中已验证的性质——**高维输出、多峰分布、训练稳定**——迁移到机器人动作序列上。

### 2.4 DDPM 的基本框架

目标是拟合高维、多峰的数据分布 $q(x_0)$ 并从中抽样。$q$ 无闭式密度，可用的只是有限样本，经验测度为 $\frac{1}{N}\sum_{i=1}^{N}\delta_{x_0^{(i)}}$。直接学习单步映射 $x=f_{\theta}(z)$、$z\sim\mathcal{N}(0,I)$ 在多峰上易收敛到条件均值，且训练不稳定。DDPM（Ho et al., 2020）将生成分解为多步：以不含可学习参数的前向核将边缘 $q(x_0)$ 逐步变换为 $q(x_K)\approx\mathcal{N}(0,I)$，再学习其逐步逆。前向核固定，$x_k$ 中的噪声由重参数化显式给出，训练目标化为对 $\epsilon$ 的回归；生成分布由反向核 $p_{\theta}$ 定义。$\beta_k$ 较小时每步扰动很小，逆核可用高斯逼近。

原论文中 $x_0$ 为图像。Diffusion Policy 将其换为动作序列并加入观测条件，拟合 $q(\mathbf{A}_t\mid\mathbf{O}_t)$；图像属于 $\mathbf{O}_t$，既不加噪也不被生成。同一批专家样本亦可用 GMM、能量模型或 flow matching 拟合，DDPM 是其中一种在高维、多峰上训练较稳定的参数化。

记 $x_0$ 为数据，$x_k$ 为第 $k$ 步状态（$k=1,\ldots,K$）。$k$ 增大则噪声增强，$k$ 减小则噪声减弱：

```text
前向（加噪，核 q）：  x_0 → x_1 → ⋯ → x_K ≈ N(0, I)
反向（去噪，核 p_θ）： x_K → x_{K-1} → ⋯ → x_0
```

$q$ 由数据与固定前向核诱导，训练时用于构造加噪样本；$p_\theta$ 为学习到的逆过程，推理时从 $x_K$ 迭代采样。二者不是同一条计算链。

#### 前向过程

每步将 $x_{k-1}$ 收缩并加入高斯噪声：

$$
q(x_k \mid x_{k-1}) = \mathcal{N}\!\left(x_k;\, \sqrt{\alpha_{k}}\, x_{k-1},\, \beta_{k} \mathbf{I}\right),
\qquad \alpha_{k} = 1 - \beta_{k},
$$

其中 $\{\beta_{k}\}_{k=1}^{K}$ 为预先给定的噪声日程。均值 $\sqrt{\alpha_{k}}\,x_{k-1}$ 将信号按 $\sqrt{\alpha_{k}}<1$ 收缩，协方差 $\beta_{k}I$ 注入噪声。由附录 A.1，

$$
x_k=\sqrt{\alpha_{k}}\,x_{k-1}+\sqrt{\beta_{k}}\,\epsilon_{k},
\qquad \epsilon_{k}\sim\mathcal{N}(0,I).
$$

定义 $\bar{\alpha}_{k}=\prod_{s=1}^{k}\alpha_{s}$。下面对 $k$ 归纳证明

$$
q(x_k \mid x_0) = \mathcal{N}\!\left(x_k;\, \sqrt{\bar{\alpha}_{k}}\, x_0,\, (1 - \bar{\alpha}_{k})\mathbf{I}\right).
$$

$k=1$ 时 $x_1=\sqrt{\alpha_1}\,x_0+\sqrt{\beta_1}\,\epsilon_1$，方差 $\beta_1=1-\alpha_1=1-\bar\alpha_1$，命题成立。

设 $x_{k-1}=\sqrt{\bar\alpha_{k-1}}\,x_0+\sqrt{1-\bar\alpha_{k-1}}\,\epsilon'$，其中 $\epsilon'\sim\mathcal{N}(0,I)$ 与 $\epsilon_k$ 独立。代入一步核，并注意 $\bar\alpha_k=\alpha_k\bar\alpha_{k-1}$，

$$
x_k=\sqrt{\alpha_k}\,x_{k-1}+\sqrt{\beta_k}\,\epsilon_k
=\sqrt{\bar\alpha_k}\,x_0+\sqrt{\alpha_k(1-\bar\alpha_{k-1})}\,\epsilon'+\sqrt{\beta_k}\,\epsilon_k.
$$

两项噪声独立、均值为零，方差分别为 $\alpha_k(1-\bar\alpha_{k-1})$ 与 $\beta_k$。由附录 A.2，和仍为高斯，均值 $\sqrt{\bar\alpha_k}\,x_0$，方差

$$
\alpha_k(1-\bar\alpha_{k-1})+\beta_k
=\alpha_k-\bar\alpha_k+(1-\alpha_k)
=1-\bar\alpha_k,
$$

其中用了 $\bar\alpha_k=\alpha_k\bar\alpha_{k-1}$ 与 $\beta_k=1-\alpha_k$。故

$$
x_k = \sqrt{\bar{\alpha}_{k}}\, x_0 + \sqrt{1 - \bar{\alpha}_{k}}\, \epsilon,
\qquad \epsilon \sim \mathcal{N}(0, \mathbf{I}).
$$

$k=K$ 且 $\bar\alpha_K\approx 0$ 时，$x_K$ 近似 $\mathcal{N}(0,I)$。

#### 反向过程

生成从 $x_K\sim\mathcal{N}(0,I)$ 出发，沿 $k=K,\ldots,1$ 逐步降低噪声。理想的一步转移是真反向核 $q(x_{k-1}\mid x_k)$。由全概率公式对未知的干净数据边缘化：

$$
q(x_{k-1}\mid x_k)
=\int q(x_{k-1}\mid x_k,x_0)\,q(x_0\mid x_k)\,\mathrm{d}x_0.
$$

给定 $x_0$ 与 $x_k$ 时，$q(x_{k-1}\mid x_k,x_0)$ 为高斯且有闭式（4.2 节）。$q(x_0\mid x_k)$ 是仅观测到加噪样本时干净数据的后验，即未知的数据分布，故该积分无闭式，也不进入实现。$\beta_k$ 较小时 $x_k$ 相对 $x_{k-1}$ 仅有微小扰动，$q(x_{k-1}\mid x_k)$ 仍接近高斯，故以参数化高斯逼近：

$$
p_{\theta}(x_{k-1}\mid x_k)
=\mathcal{N}\!\big(x_{k-1};\,\mu_{\theta}(x_k,k),\,\sigma_k^{2}I\big).
$$

方差 $\sigma_k^{2}$ 取 $\beta_k$ 或后验方差 $\tilde\beta_k$，由噪声日程给出，不由网络输出。网络只决定均值。整条生成分布为

$$
p_{\theta}(x_{0:K})
=p(x_K)\prod_{k=1}^{K}p_{\theta}(x_{k-1}\mid x_k),
\qquad
p(x_K)=\mathcal{N}(0,I),
$$

即先抽 $x_K$，再依次抽 $x_{K-1}\mid x_K,\ldots,x_0\mid x_1$。乘积中每一项对应推理循环的一次迭代。

前向一步边际为 $x_k=\sqrt{\bar\alpha_k}\,x_0+\sqrt{1-\bar\alpha_k}\,\epsilon$。解出

$$
x_0=\frac{1}{\sqrt{\bar\alpha_k}}\big(x_k-\sqrt{1-\bar\alpha_k}\,\epsilon\big).
$$

若 $\epsilon$ 已知，即可从 $x_k$ 还原 $x_0$。将此式代入 $q(x_{k-1}\mid x_k,x_0)$ 的均值 $\tilde\mu_k(x_k,x_0)$，$x_0$ 消去后得到仅依赖 $(x_k,\epsilon)$ 的中心（推导见 4.3 节）：

$$
\tilde\mu_k(x_k,\epsilon)
=\frac{1}{\sqrt{\alpha_k}}\left(x_k-\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\,\epsilon\right).
$$

这是噪声已知时一步去噪的最优中心，推理时 $\epsilon$ 未知。以网络 $\epsilon_{\theta}(x_k,k)$ 替代，均值取同一形式：

$$
\mu_{\theta}(x_k,k)
=\frac{1}{\sqrt{\alpha_k}}\left(x_k-\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\,\epsilon_{\theta}(x_k,k)\right).
$$

$\tilde\mu_k$ 与 $\mu_\theta$ 仅差 $\epsilon$ 与 $\epsilon_\theta$。一步采样即从 $p_\theta$ 抽点。由附录 A.1，

$$
x_{k-1}=\mu_{\theta}(x_k,k)+\sigma_k z,
\qquad z\sim\mathcal{N}(0,I).
$$

将 $\mu_\theta$ 代入后，形式为从 $x_k$ 减去一块噪声再除以 $\sqrt{\alpha_k}$。系数是后验中的 $\beta_k/\sqrt{1-\bar\alpha_k}$，不是单步前向求逆的 $\sqrt{\beta_k}$；二者仅在 $k=1$（$1-\bar\alpha_1=\beta_1$）时相同。$\sigma_k^{2}$ 取 $\beta_k$ 或 $\tilde\beta_k$，与前向一步方差同源；$k=1$ 时常取 $z=0$。去掉随机项即 DDIM（第八节）。对 $k=K,\ldots,1$ 重复，输出 $x_0$。

训练时 $x_0$ 已知，无需展开反向链。由前向边际从 $(x_0,\epsilon,k)$ 一次构造 $x_k$，监督 $\epsilon_{\theta}(x_k,k)\approx\epsilon$。推理将 $\epsilon_\theta$ 代入 $\mu_\theta$ 后迭代。二者分别对应前向闭式与反向采样。

| 对象 | 出现位置 | 作用 |
|------|----------|------|
| $q(x_{k-1}\mid x_k)$ 的积分 | 推导 | 标明真逆无闭式 |
| $x_k=\sqrt{\bar\alpha_k}\,x_0+\sqrt{1-\bar\alpha_k}\,\epsilon$ | 训练 | 一次构造加噪样本 |
| $\epsilon_\theta(x_k,k)\approx\epsilon$ | 训练 | 去噪 MSE |
| $\tilde\mu_k(x_k,\epsilon)$ | 推导 | 噪声已知时的理想均值 |
| $\mu_\theta(x_k,k)$ | 推理 | 以 $\epsilon_\theta$ 代入后的去噪中心 |
| $x_{k-1}=\mu_\theta+\sigma_k z$ | 推理 | 反向循环的一步更新 |
| $p_\theta(x_{0:K})=p(x_K)\prod_k p_\theta$ | 推导 | 反向循环的联合分布 |

---

## 三、Diffusion Policy 的 formulation

### 3.1 符号与 horizon

| 符号 | 含义 | 代码变量 |
|------|------|----------|
| $T_o$ | 观测历史长度 | `n_obs_steps` |
| $T_p$ | 动作预测长度 | `horizon` |
| $T_a$ | 每轮实际执行步数 | `n_action_steps` |
| $\mathbf{O}_t$ | 时刻 $t$ 起最近 $T_o$ 步观测 | `obs` |
| $\mathbf{A}_t$ | 动作序列 $[\mathbf{a}_t, \ldots, \mathbf{a}_{t+T_p-1}]$ | `action` |
| $D_a$ | 单步动作维度 | — |
| $k$ | 去噪步索引（$k=K$ 最噪，$k=0$ 为干净数据） | `timesteps` |

关系：$T_a \leq T_p$。每 $T_a$ 个环境步重新推理一次。

时间轴示意（$T_o=3, T_a=4, T_p=6$）：

```text
|o|o|o|
| | |a|a|a|a|
```

### 3.2 条件分布

Diffusion Policy 建模

$$
p_\theta(\mathbf{A}_t \mid \mathbf{O}_t),
$$

而非 Janner et al. (2022) 规划框架中的联合分布 $p(\mathbf{A}_t, \mathbf{O}_t)$。观测仅作为条件，不参与扩散，从而避免在推理时推断未来状态，并允许视觉编码器在 $K$ 步去噪中**只运行一次**。第二节的 $x_0$ 在此对应 $\mathbf{A}_t$，数据边缘 $q(x_0)$ 换为条件 $q(\mathbf{A}_t\mid\mathbf{O}_t)$。

### 3.3 反向去噪

记 $\mathbf{A}^{0}\equiv\mathbf{A}_t$ 为干净动作序列，$\mathbf{A}^{k}$ 为第 $k$ 步加噪版本。反向核仍取 2.4 节的高斯 $p_\theta$，噪声网络以观测为条件：$\epsilon_{\theta}(\mathbf{O}_t,\mathbf{A}^{k},k)$。一步为

$$
\mathbf{A}^{k-1}
=\frac{1}{\sqrt{\alpha_k}}\left(\mathbf{A}^{k}-\frac{\beta_k}{\sqrt{1-\bar\alpha_k}}\,\epsilon_{\theta}(\mathbf{O}_t,\mathbf{A}^{k},k)\right)
+\sigma_k z,
\qquad z\sim\mathcal{N}(0,I),
$$

其中 $\alpha_k=1-\beta_k$ 与 2.4 节相同。从 $\mathbf{A}^{K}\sim\mathcal{N}(0,I)$ 迭代 $k=K,\ldots,1$ 得到 $\mathbf{A}^{0}$，取 $\mathbf{A}^{0}[0:T_a]$ 执行。该循环实现 $p_\theta(\mathbf{A}^{0:K}\mid\mathbf{O}_t)$。

### 3.4 Receding horizon 与 warm-start

每轮预测 $T_p$ 步、执行 $T_a$ 步，构成 receding horizon control。当 $T_p > T_a$ 时，上一轮预测中尚未执行的后缀动作，可作为下一轮去噪的初始化（warm-start），而非从纯噪声重新开始。这在 $T_p - T_a$ 步重叠区间内保持动作连续性，减轻重规划带来的抖动。

---

## 四、训练目标：从 ELBO 到 $\epsilon$-prediction

### 4.1 变分下界

无条件 DDPM 的最大似然目标不可解析。对数据 $x_0$，

$$
\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:K} \mid x_0)} \left[ \log \frac{p_\theta(x_{0:K})}{q(x_{1:K} \mid x_0)} \right].
$$

Ho et al. (2020, Eq. 5) 最小化该下界的相反数

$$
L = \underbrace{D_{\mathrm{KL}}\!\left(q(x_K \mid x_0) \,\|\, p(x_K)\right)}_{\text{先验匹配}}
+ \sum_{k=2}^{K} \underbrace{\mathbb{E}_{q}\!\left[D_{\mathrm{KL}}\!\left(q(x_{k-1} \mid x_k, x_0) \,\|\, p_\theta(x_{k-1} \mid x_k)\right)\right]}_{\text{去噪匹配}}
- \underbrace{\mathbb{E}_{q}\!\left[\log p_\theta(x_0 \mid x_1)\right]}_{\text{重建项}}.
$$

最小化 $L$ 与最大化下界同向。前向后验 $q(x_{k-1} \mid x_k, x_0)$ 有闭式高斯解（见 4.2 节）。Diffusion Policy 在每一项中加入观测条件 $\mathbf{O}_t$，即对 $p_\theta(\mathbf{A}^{k-1} \mid \mathbf{A}^k, \mathbf{O}_t)$ 进行同样的变分推断。

### 4.2 前向后验的闭式

前向为马尔可夫链，故 $q(x_k\mid x_{k-1},x_0)=q(x_k\mid x_{k-1})$，从而

$$
q(x_{k-1}\mid x_k,x_0)
\propto
q(x_k\mid x_{k-1})\,q(x_{k-1}\mid x_0).
$$

第二项已是 $x_{k-1}$ 上的高斯 $\mathcal{N}(\sqrt{\bar\alpha_{k-1}}\,x_0,\,(1-\bar\alpha_{k-1})I)$。第一项是 $x_k$ 的密度

$$
q(x_k\mid x_{k-1})
=(2\pi\beta_k)^{-d/2}
\exp\Bigl(-\frac{\|x_k-\sqrt{\alpha_k}\,x_{k-1}\|^2}{2\beta_k}\Bigr).
$$

前置因子不含 $x_{k-1}$。固定 $x_k$ 后

$$
\|x_k-\sqrt{\alpha_k}\,x_{k-1}\|^2
=\alpha_k\Bigl\|x_{k-1}-\frac{x_k}{\sqrt{\alpha_k}}\Bigr\|^2,
$$

指数化为

$$
-\frac{\alpha_k}{2\beta_k}\Bigl\|x_{k-1}-\frac{x_k}{\sqrt{\alpha_k}}\Bigr\|^2
=-\frac{1}{2\sigma^2}\Bigl\|x_{k-1}-\frac{x_k}{\sqrt{\alpha_k}}\Bigr\|^2,
\qquad
\sigma^2=\frac{\beta_k}{\alpha_k}.
$$

这与 $\mathcal{N}(x_k/\sqrt{\alpha_k},\,(\beta_k/\alpha_k)I)$ 的指数相同，归一化常数不同，故只是正比。由附录 A.3，精度 $\tau_1=\alpha_k/\beta_k$，$\tau_2=1/(1-\bar\alpha_{k-1})$，后验协方差为

$$
\tilde\beta_k=\frac{1}{\tau_1+\tau_2}
=\frac{\beta_k(1-\bar\alpha_{k-1})}{\alpha_k(1-\bar\alpha_{k-1})+\beta_k}.
$$

分母用 $\bar\alpha_k=\alpha_k\bar\alpha_{k-1}$ 与 $\beta_k=1-\alpha_k$ 化为 $1-\bar\alpha_k$，故

$$
\tilde{\beta}_k = \frac{1 - \bar{\alpha}_{k-1}}{1 - \bar{\alpha}_k}\, \beta_k.
$$

均值按精度加权：

$$
\begin{aligned}
\tilde{\mu}_k(x_k, x_0)
&=\tilde\beta_k\Bigl(\tau_1\cdot\frac{x_k}{\sqrt{\alpha_k}}+\tau_2\cdot\sqrt{\bar\alpha_{k-1}}\,x_0\Bigr)\\
&=\frac{\sqrt{\alpha_k}\,(1-\bar\alpha_{k-1})}{1-\bar\alpha_k}\,x_k
+\frac{\sqrt{\bar\alpha_{k-1}}\,\beta_k}{1-\bar\alpha_k}\,x_0.
\end{aligned}
$$

因此

$$
q(x_{k-1} \mid x_k, x_0) = \mathcal{N}\!\left(x_{k-1};\, \tilde{\mu}_k(x_k, x_0),\, \tilde{\beta}_k \mathbf{I}\right).
$$

### 4.3 用 $x_0$ 重参数化后验均值

由 $x_k = \sqrt{\bar{\alpha}_k}\, x_0 + \sqrt{1 - \bar{\alpha}_k}\, \epsilon$ 解出

$$
x_0 = \frac{1}{\sqrt{\bar{\alpha}_k}}\left(x_k - \sqrt{1 - \bar{\alpha}_k}\, \epsilon\right).
$$

代入 $\tilde{\mu}_k$，整理得

$$
\tilde{\mu}_k(x_k, \epsilon) = \frac{1}{\sqrt{\alpha_k}}\left(x_k - \frac{\beta_k}{\sqrt{1 - \bar{\alpha}_k}}\, \epsilon\right).
$$

反向模型 $p_\theta(x_{k-1} \mid x_k)$ 的参数化均值取同一形式，以网络输出 $\epsilon_\theta(x_k, k)$ 替代真实噪声 $\epsilon$：

$$
\mu_\theta(x_k, k) = \frac{1}{\sqrt{\alpha_k}}\left(x_k - \frac{\beta_k}{\sqrt{1 - \bar{\alpha}_k}}\, \epsilon_\theta(x_k, k)\right).
$$

该代入只用于导出 $\mu_\theta$ 的参数化。推理并不先恢复 $x_0$，而是直接用 $\epsilon_\theta$ 计算 $\mu_\theta$；$\tilde\mu_k$ 本身不出现在实现中。

### 4.4 去噪项与 MSE 损失

对固定 $k$，KL 项 $D_{\mathrm{KL}}(q(x_{k-1} \mid x_k, x_0) \,\|\, p_\theta(x_{k-1} \mid x_k))$ 在两个高斯之间，等价于

$$
\mathbb{E}_{q(x_k \mid x_0)} \left[ \frac{1}{2\tilde{\beta}_k} \left\| \tilde{\mu}_k(x_k, x_0) - \mu_\theta(x_k, k) \right\|^2 \right] + \text{const}.
$$

将 $\tilde{\mu}_k$ 与 $\mu_\theta$ 的表达式代入，$\tilde{\mu}_k - \mu_\theta$ 仅差在 $\epsilon$ 与 $\epsilon_\theta$ 之间，系数为 $\beta_k / (\sqrt{\alpha_k(1-\bar{\alpha}_k)})$。因此去噪匹配等价于加权 MSE：

$$
\mathbb{E}_{x_0, \epsilon, k} \left[ w_k \left\| \epsilon - \epsilon_\theta(x_k, k) \right\|^2 \right],
\qquad
w_k = \frac{(1 - \alpha_k)^2}{2\alpha_k(1 - \bar{\alpha}_k)\tilde{\beta}_k}.
$$

Ho et al. 发现丢弃权重 $w_k$、对所有 $k$ 使用均匀 MSE（simple loss）在实践中效果更好：

$$
\mathcal{L}_{\mathrm{simple}} = \mathbb{E}_{x_0, \epsilon, k} \left[ \left\| \epsilon - \epsilon_\theta(x_k, k) \right\|^2 \right],
\qquad
x_k = \sqrt{\bar{\alpha}_k}\, x_0 + \sqrt{1 - \bar{\alpha}_k}\, \epsilon.
$$

### 4.5 Diffusion Policy 的条件损失

将 $x_0$ 换为动作序列 $\mathbf{A}^0$，$x_k$ 换为 $\mathbf{A}^k$，并加入观测条件 $\mathbf{O}_t$，得到论文 Eq. (5)：

$$
\boxed{
\mathcal{L}(\theta) = \mathbb{E}_{(\mathbf{O}_t, \mathbf{A}^0) \sim \mathcal{D},\, \epsilon,\, k}
\left[ \left\| \epsilon - \epsilon_\theta(\mathbf{O}_t, \mathbf{A}^k, k) \right\|^2 \right]
}
$$

$$
\mathbf{A}^k = \sqrt{\bar{\alpha}_k}\, \mathbf{A}^0 + \sqrt{1 - \bar{\alpha}_k}\, \epsilon,
\qquad
\epsilon \sim \mathcal{N}(0, \mathbf{I}),
\qquad
k \sim \mathrm{Uniform}\{1, \ldots, K\}.
$$

训练时随机采样 $k$ 与 $\epsilon$，对任意噪声水平同时监督，无需运行完整 $K$ 步去噪链。

**推导链总结**：

```text
max log p(A^0 | O)
   ↓ 变分下界；最小化其相反数 L（Ho et al. Eq. 5）
L = KL(q(A^K|A^0) || p(A^K)) + Σ_k KL(q(A^{k-1}|A^k,A^0) || p_θ(A^{k-1}|A^k)) - log p_θ(A^0|A^1)
   ↓ 前向后验 μ̃_k 用 ε 表示
μ̃_k = (1/√α_k)(A^k - (β_k/√(1-ᾱ_k)) ε)
   ↓ 反向模型 μ_θ 以 ε_θ 参数化
去噪 KL ∝ E[||ε - ε_θ(A^k, O, k)||²]
   ↓ 去掉时间步权重 w_k
L_simple = MSE(ε, ε_θ)   ← Diffusion Policy 实际优化
```

---

## 五、Score function 视角与 IBC 的对比

Song et al. 指出，DDPM 的去噪方向与 score function $\nabla_x \log q(x)$ 相关。对加噪分布 $q(x_k \mid x_0)$，有

$$
\nabla_{x_k} \log q(x_k \mid x_0) = -\frac{1}{\sqrt{1 - \bar{\alpha}_k}}\, \epsilon.
$$

因此 $\epsilon_\theta(x_k, k) \approx -\sqrt{1 - \bar{\alpha}_k}\, \nabla_{x_k} \log p(x_k \mid \mathbf{O}_t)$：网络学习的是**条件 score function**，即对数条件密度的梯度方向。

隐式行为克隆（IBC）建模 $p(\mathbf{a} \mid \mathbf{o}) \propto \exp(-E_\theta(\mathbf{a}, \mathbf{o}))$，训练需估计配分函数 $Z(\mathbf{o}) = \int \exp(-E_\theta(\mathbf{a}, \mathbf{o}))\, d\mathbf{a}$ 或采样负样本。Score matching 形式下，

$$
\nabla_\mathbf{a} \log p(\mathbf{a} \mid \mathbf{o}) = -\nabla_\mathbf{a} E_\theta(\mathbf{a}, \mathbf{o}) - \nabla_\mathbf{a} \log Z(\mathbf{o}),
$$

而 $\nabla_\mathbf{a} \log Z(\mathbf{o}) = 0$，故配分函数对梯度无贡献。Diffusion Policy 的 $\epsilon$-prediction 训练不涉及 $Z$，这是其相对 IBC 训练更稳定的原因之一。

---

## 六、动作序列预测的必要性

单步 BC 将 $(\mathbf{o}_t, \mathbf{a}_t)$ 独立处理，无法显式约束 $\mathbf{a}_t, \mathbf{a}_{t+1}, \ldots$ 之间的时序关系。将输出扩展为 $\mathbf{A}_t \in \mathbb{R}^{T_p \times D_a}$ 后：

1. **时序一致性**：模型联合生成一段轨迹，动作过渡更平滑；
2. **有效维度**：$T_p \times D_a$ 可能远大于 $D_a$，GMM 与 IBC 在此维度上采样困难，扩散模型在图像等高维生成中已有成熟实践；
3. **短 horizon 规划**：预测未来 $T_p$ 步动作，等价于在动作空间做有限步前瞻。

执行时只取前 $T_a$ 步，使策略保持闭环：每 $T_a$ 步重新观测并规划，对扰动仍具备响应能力。$T_a$ 减小则重规划更频繁、响应更快；$T_a$ 增大则动作更平滑、但对开环预测依赖更强。论文 Sec. 4.3 对此作了实验分析。

---

## 七、网络结构

### 7.1 总体数据流

```text
观测 O_t ──→ Visual Encoder（ResNet-18 等）──→ 条件向量 c
                                                    │
带噪动作 A^k ──→ 1D Conditional U-Net ──→ ε_θ       │
                 （FiLM 注入 c 与 k）               │
                                                    ↓
                              迭代 K 步去噪 → A^0 → 取前 T_a 步
```

### 7.2 1D Temporal U-Net

骨干 adapted from Janner et al. (2022) 的 1D temporal CNN，三处修改：

1. 仅对动作序列 $\mathbf{A}^k$ 扩散；观测 $\mathbf{O}_t$ 经 FiLM 注入，不参与加噪；
2. 不拼接观测–动作联合轨迹（与 Diffuser 的 inpainting 方案不同）；
3. 移除基于 inpainting 的目标状态条件（与 receding horizon 不兼容）；目标条件仍可通过 FiLM 实现。

**FiLM（Feature-wise Linear Modulation）**：对特征 $h$，条件 $c$（观测嵌入与时间步嵌入的拼接）产生仿射变换

$$
\mathrm{FiLM}(h \mid c) = \gamma(c) \odot h + \beta(c),
$$

其中 $\gamma, \beta$ 为 $c$ 的 MLP 输出。

### 7.3 视觉编码

图像观测对应 visuomotor 设定中的 $\mathbf{o}^{\mathrm{img}}$。编码器采用 ResNet-18（无预训练），两处改动：

- 全局平均池化替换为 **spatial softmax pooling**，保留空间信息；
- BatchNorm 替换为 **GroupNorm**，与 DDPM 常用的 EMA 权重更新兼容。

多相机各自独立编码后拼接；多时刻观测逐帧编码后拼接为 $\mathbf{O}_t$ 的条件向量。该编码在每次推理中只执行一次，$K$ 步去噪共享同一条件。

### 7.4 Transformer 变体

论文另给出 Diffusion Transformer，以 Transformer 替代 U-Net 处理动作序列。在长 horizon 任务上有时优于 CNN 版本。官方仓库提供 `diffusion_transformer_*` 实现。

---

## 八、噪声 schedule 与 DDIM 加速

### 8.1 噪声 schedule

$\{\beta_k\}$ 控制各步加噪幅度。控制任务中，论文采用 iDDPM 提出的 **square cosine schedule**，对动作信号的高低频成分匹配较好。schedule 影响训练时各 $k$ 对应的信噪比，进而影响 $\epsilon_\theta$ 在不同频率上的学习难度。

### 8.2 DDIM

标准 DDPM 推理需 $K$ 步去噪（如 $K=100$），难以满足实时控制。DDIM（Song et al., 2021）将训练步数与推理步数解耦：训练仍用 $K$ 步前向过程，推理可跳步至 $N \ll K$。论文实机配置为 $K=100$ 训练、$N=10$ 推理，在 Nvidia 3080 上约 0.1 s 延迟。

DDIM 走确定性 ODE 轨迹，去除 DDPM 反向过程中的随机项，以少量多样性换取速度。对机器人控制，动作精度通常优先于生成多样性。

---

## 九、训练流程

```text
1. 收集专家演示，存入 ReplayBuffer（zarr / numpy）

2. SequenceSampler 滑窗采样：
   - 观测窗口长度 T_o，动作窗口长度 T_p
   - 处理 episode 首尾 padding

3. LinearNormalizer 对 obs / action 做仿射归一化

4. 每个 training step：
   a. 采样 batch (O_t, A^0)
   b. k ~ Uniform{1, ..., K}，ε ~ N(0, I)
   c. A^k = sqrt(ᾱ_k) A^0 + sqrt(1-ᾱ_k) ε
   d. ε_pred = ε_θ(O_t, A^k, k)
   e. L = ||ε - ε_pred||²，对 padding 位置 mask
   f. 反向传播；可选 EMA 更新

5. 定期评估 success rate
```

数据来自固定演示集，可跨 epoch 反复使用。损失为去噪 MSE：标签是前向注入的 $\epsilon$，不是专家动作 $\mathbf{A}^{0}$。该步骤对应 2.4 节的前向边际，不展开反向链。

---

## 十、推理流程

```text
1. 读取最近 T_o 步观测 O_t

2. 编码 O_t → 条件向量 c（一次）

3. 初始化 A^K ~ N(0, I)，shape (T_p, D_a)
   （可选：warm-start 用上一轮未执行动作初始化）

4. for k = K, K-1, ..., 1:
       ε_pred = ε_θ(O_t, A^k, k)
       μ = (1/√α_k) (A^k - (β_k/√(1-ᾱ_k)) ε_pred)
       A^{k-1} = μ + σ_k z              # DDPM；DDIM 则去掉随机项
       # k=1 时常取 z=0

5. 执行 A^0[0], ..., A^0[T_a - 1]

6. 等待 T_a 步后回到步骤 1
```

该循环实现 2.4 节的反向过程；$\mathbf{O}_t$ 的编码在 $K$ 步内保持不变。实机部署中，策略与环境异步交互：`get_obs` 读取最新观测，`exec_actions` 将动作序列及时间戳送入插值控制器，不阻塞等待执行完成。

---

## 十一、关键超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `n_obs_steps` ($T_o$) | 2–3 | 观测历史 |
| `horizon` ($T_p$) | 8–16 | 预测动作长度 |
| `n_action_steps` ($T_a$) | 4–8 | 每轮执行步数 |
| `num_train_timesteps` ($K$) | 100 | 训练扩散步数 |
| `num_inference_steps` ($N$) | 10 | DDIM 推理步数 |
| noise schedule | square cosine | 加噪曲线 |
| U-Net `down_dims` | [256, 512, 1024] | 通道宽度 |
| learning rate | $10^{-4}$ | Adam |
| EMA | 启用 | 稳定生成质量 |

---

## 十二、参考文献与延伸阅读

1. Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023. [arXiv:2303.04137](https://arxiv.org/pdf/2303.04137v5)
2. Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
3. Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021.
4. Janner et al., *Planning with Diffusion for Flexible Behavior Synthesis*, ICML 2022.
5. Lipman et al., *Flow Matching for Generative Modeling*, ICLR 2023. [`flow_matching_notes.md`](flow_matching_notes.md)
6. Black et al., *π₀: A Vision-Language-Action Flow Model*, arXiv:2410.24164, 2024.

建议阅读顺序：Ho et al. 2020（DDPM 与 $\epsilon$-prediction）→ 论文 Sec. III–IV（动作序列 formulation 与架构）→ 官方 `diffusion_unet_image_policy.py` 中的 `compute_loss` 与 `predict_action` → [`flow_matching_notes.md`](flow_matching_notes.md)（CFM 与 π₀）。

---

## 附录：高斯的若干运算

下文协方差均为单位阵的倍数。证明对每一坐标相同，故写成各向同性形式。

### A.1 仿射

若 $z\sim\mathcal{N}(0,I)$，$\sigma>0$，则

$$
\mu+\sigma z\sim\mathcal{N}(\mu,\sigma^{2}I).
$$

$z$ 的密度为 $(2\pi)^{-d/2}\exp(-\|z\|^{2}/2)$。令 $x=\mu+\sigma z$，则 $z=(x-\mu)/\sigma$，雅可比为 $\sigma^{-d}$，密度为

$$
(2\pi\sigma^{2})^{-d/2}\exp\Bigl(-\frac{\|x-\mu\|^{2}}{2\sigma^{2}}\Bigr),
$$

即 $\mathcal{N}(\mu,\sigma^{2}I)$。

### A.2 独立和

设 $X\sim\mathcal{N}(\mu_1,\sigma_1^{2}I)$、$Y\sim\mathcal{N}(\mu_2,\sigma_2^{2}I)$ 相互独立。则

$$
X+Y\sim\mathcal{N}\bigl(\mu_1+\mu_2,\,(\sigma_1^{2}+\sigma_2^{2})I\bigr).
$$

由 A.1 写 $X=\mu_1+\sigma_1\epsilon_1$、$Y=\mu_2+\sigma_2\epsilon_2$，$\epsilon_1,\epsilon_2$ 独立且服从 $\mathcal{N}(0,I)$。和为 $\mu_1+\mu_2+\sigma_1\epsilon_1+\sigma_2\epsilon_2$。噪声项的交叉矩

$$
\mathbb{E}\bigl[(\sigma_1\epsilon_1)(\sigma_2\epsilon_2)^{\top}\bigr]
=\sigma_1\sigma_2\,\mathbb{E}[\epsilon_1]\,\mathbb{E}[\epsilon_2]^{\top}=0,
$$

故协方差相加，为 $(\sigma_1^{2}+\sigma_2^{2})I$。各坐标独立，化为一维：$\mathcal{N}(\mu,\sigma^{2})$ 的特征函数为 $\exp(it\mu-\sigma^{2}t^{2}/2)$，独立则特征函数相乘，得到 $\mathcal{N}(\mu_1+\mu_2,\sigma_1^{2}+\sigma_2^{2})$。

### A.3 同一变量上的乘积

设 $p_i(x)=\mathcal{N}(x;\mu_i,\sigma_i^{2}I)$，$i=1,2$。记精度 $\tau_i=1/\sigma_i^{2}$。则

$$
p_1(x)\,p_2(x)\propto\mathcal{N}(x;\mu,\sigma^{2}I),
$$

其中

$$
\sigma^{2}=\frac{1}{\tau_1+\tau_2}=\frac{\sigma_1^{2}\sigma_2^{2}}{\sigma_1^{2}+\sigma_2^{2}},
\qquad
\mu=\frac{\tau_1\mu_1+\tau_2\mu_2}{\tau_1+\tau_2}.
$$

乘积的积分一般不为 $1$，故它是高斯函数而非密度；归一化后即上述 $\mathcal{N}(\mu,\sigma^{2}I)$。

略去 $p_i$ 的归一化常数，负对数为

$$
\frac{1}{2}\sum_{i=1}^{2}\tau_i\|x-\mu_i\|^{2}
=\frac{\tau_1+\tau_2}{2}\|x\|^{2}-(\tau_1\mu_1+\tau_2\mu_2)\cdot x+\mathrm{const}.
$$

右端等于 $\frac{\tau_1+\tau_2}{2}\|x-\mu\|^{2}$ 加上与 $x$ 无关的项，其中 $\mu$ 如上。指数因此为 $\exp(-\|x-\mu\|^{2}/(2\sigma^{2}))$。

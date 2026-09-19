# Diffusion Policy 算法笔记

下文公式与流程对应的实现，见官方仓库 [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy)（Chi et al., RSS 2023；[arXiv:2303.04137](https://arxiv.org/pdf/2303.04137v5)）。  
与在线 RL 方法的对比，见 [`ppo_notes.md`](ppo_notes.md)。

---

## 一、算法定位

Diffusion Policy 将 **visuomotor 策略**（视觉–运动策略）参数化为**动作空间上的条件去噪扩散过程**（conditional denoising diffusion process）。Visuomotor 指以相机图像（常辅以本体感觉）为输入、以机器人动作为输出的闭环映射，与以低维状态为观测的策略相对；论文标题中的 *Visuomotor Policy Learning* 即这一设定。给定最近若干步观测 $\mathbf{O}_t$，策略不直接回归单步动作，而是对长度为 $T_p$ 的动作序列 $\mathbf{A}_t$ 执行迭代去噪，从中取出 $T_a$ 步执行，并在下一控制周期重新规划。

该方法属于**离线模仿学习**：训练数据为专家演示轨迹，不使用环境奖励，也不维护价值函数。与 PPO 等策略梯度方法的差别在于优化目标——Diffusion Policy 拟合条件数据分布 $p(\mathbf{A}_t \mid \mathbf{O}_t)$，而非最大化期望累积回报。观测 $\mathbf{O}_t$ 可以是图像、低维状态或二者拼接；扩散模型只规定如何参数化该条件分布。

| 特性 | PPO | 行为克隆 (BC) | Diffusion Policy |
|------|-----|---------------|------------------|
| 学习范式 | 在线 RL（on-policy） | 离线模仿 | 离线模仿 |
| 训练信号 | 奖励 $r_t$、Advantage | 专家动作 | 专家动作序列 + 噪声监督 |
| 策略输出 | 单步分布 $\pi(a \mid s)$ | 单步点估计或 GMM | 动作序列 $\mathbf{A}_t \in \mathbb{R}^{T_p \times D_a}$ |
| 多模态 | 对角高斯通常单峰 | MLP 回归为单峰 | 生成式，可表达多峰 |
| 核心网络 | Actor + Critic | MLP / RNN | 1D U-Net 或 Transformer + 视觉编码器 |
| 推理 | 一次前向采样 | 一次前向 | $K$ 步（或 DDIM 的 $N<K$ 步）迭代去噪 |

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

设演示数据集 $\mathcal{D} = \{(\mathbf{o}_i, \mathbf{a}_i)\}_{i=1}^N$。确定性行为克隆最小化

$$
\mathcal{L}_{\mathrm{BC}}(\theta) = \mathbb{E}_{(\mathbf{o}, \mathbf{a}) \sim \mathcal{D}} \left[ \left\| \pi_\theta(\mathbf{o}) - \mathbf{a} \right\|^2 \right].
$$

概率形式为最大化对数似然 $\mathbb{E}[-\log \pi_\theta(\mathbf{a} \mid \mathbf{o})]$。BC 实现简单，但存在**协变量偏移**（covariate shift）：训练分布为专家访问的状态，部署时策略自身的误差使状态分布偏移，误差随时间累积。

对同一观测存在多种合理动作时（例如绕障路径），平方损失收敛到条件期望 $\mathbb{E}[\mathbf{a} \mid \mathbf{o}]$，产生动作平均化（averaging），在多模态任务上性能下降明显。

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

Denoising Diffusion Probabilistic Model（Ho et al., 2020）包含两个过程：

1. **前向扩散**：向数据逐步加噪，直至近似标准高斯；
2. **反向去噪**：学习从噪声恢复数据的马尔可夫链。

设 $x_0$ 为数据，$x_k$ 为第 $k$ 步加噪结果（$k = 1, \ldots, K$）。前向过程定义为

$$
q(x_k \mid x_{k-1}) = \mathcal{N}\!\left(x_k;\, \sqrt{\alpha_k}\, x_{k-1},\, \beta_k \mathbf{I}\right),
\qquad \alpha_k = 1 - \beta_k,
$$

其中 $\{\beta_k\}_{k=1}^K$ 为预先给定的噪声日程（noise schedule）。定义 $\bar{\alpha}_k = \prod_{s=1}^{k} \alpha_s$，则 $x_k$ 可由 $x_0$ 一步采样：

$$
q(x_k \mid x_0) = \mathcal{N}\!\left(x_k;\, \sqrt{\bar{\alpha}_k}\, x_0,\, (1 - \bar{\alpha}_k)\mathbf{I}\right).
$$

等价地，重参数化形式为

$$
x_k = \sqrt{\bar{\alpha}_k}\, x_0 + \sqrt{1 - \bar{\alpha}_k}\, \epsilon,
\qquad \epsilon \sim \mathcal{N}(0, \mathbf{I}).
$$

反向过程学习 $p_\theta(x_{k-1} \mid x_k)$，从 $x_K \sim \mathcal{N}(0, \mathbf{I})$ 出发逐步去噪得到 $x_0$。

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

而非 Janner et al. (2022) 规划框架中的联合分布 $p(\mathbf{A}_t, \mathbf{O}_t)$。观测仅作为条件，不参与扩散，从而避免在推理时推断未来状态，并允许视觉编码器在 $K$ 步去噪中**只运行一次**。

### 3.3 反向去噪

记 $\mathbf{A}^0 \equiv \mathbf{A}_t$ 为干净动作序列，$\mathbf{A}^k$ 为第 $k$ 步加噪版本。反向一步写为

$$
\mathbf{A}^{k-1} = \alpha_k \left(\mathbf{A}^k - \gamma_k\, \epsilon_\theta(\mathbf{O}_t, \mathbf{A}^k, k)\right) + \mathcal{N}(0, \sigma_k^2 \mathbf{I}),
$$

其中 $\epsilon_\theta$ 为噪声预测网络，$\alpha_k, \gamma_k, \sigma_k$ 由噪声 schedule 确定（与 DDPM 一致）。推理时从 $\mathbf{A}^K \sim \mathcal{N}(0, \mathbf{I})$ 出发，迭代 $k = K, K-1, \ldots, 1$，得到 $\mathbf{A}^0$，取 $\mathbf{A}^0[0:T_a]$ 执行。

### 3.4 Receding horizon 与 warm-start

每轮预测 $T_p$ 步、执行 $T_a$ 步，构成 receding horizon control。当 $T_p > T_a$ 时，上一轮预测中尚未执行的后缀动作，可作为下一轮去噪的初始化（warm-start），而非从纯噪声重新开始。这在 $T_p - T_a$ 步重叠区间内保持动作连续性，减轻重规划带来的抖动。

---

## 四、训练目标：从 ELBO 到 $\epsilon$-prediction

### 4.1 变分下界

无条件 DDPM 的最大似然目标不可解析，转而优化 ELBO。对数据 $x_0$，有

$$
\log p_\theta(x_0) \geq \mathbb{E}_{q(x_{1:K} \mid x_0)} \left[ \log \frac{p_\theta(x_{0:K})}{q(x_{1:K} \mid x_0)} \right] \equiv \mathcal{L}_{\mathrm{ELBO}}.
$$

展开后（Ho et al., 2020, Eq. 5）：

$$
\mathcal{L}_{\mathrm{ELBO}} = \underbrace{D_{\mathrm{KL}}\!\left(q(x_K \mid x_0) \,\|\, p(x_K)\right)}_{\text{先验匹配}}
+ \sum_{k=2}^{K} \underbrace{\mathbb{E}_{q}\!\left[D_{\mathrm{KL}}\!\left(q(x_{k-1} \mid x_k, x_0) \,\|\, p_\theta(x_{k-1} \mid x_k)\right)\right]}_{\text{去噪匹配}}
- \underbrace{\mathbb{E}_{q}\!\left[\log p_\theta(x_0 \mid x_1)\right]}_{\text{重建项}}.
$$

其中前向后验 $q(x_{k-1} \mid x_k, x_0)$ 有闭式高斯解（见 4.2 节）。Diffusion Policy 在每一项中加入观测条件 $\mathbf{O}_t$，即对 $p_\theta(\mathbf{A}^{k-1} \mid \mathbf{A}^k, \mathbf{O}_t)$ 进行同样的变分推断。

### 4.2 前向后验的闭式

由 Bayes 公式，在已知 $x_0$ 和 $x_k$ 时，$x_{k-1}$ 的后验为

$$
q(x_{k-1} \mid x_k, x_0) = \mathcal{N}\!\left(x_{k-1};\, \tilde{\mu}_k(x_k, x_0),\, \tilde{\beta}_k \mathbf{I}\right),
$$

其中

$$
\tilde{\mu}_k(x_k, x_0) = \frac{\sqrt{\bar{\alpha}_{k-1}}\,\beta_k}{1 - \bar{\alpha}_k}\, x_0 + \frac{\sqrt{\alpha_k}\,(1 - \bar{\alpha}_{k-1})}{1 - \bar{\alpha}_k}\, x_k,
\qquad
\tilde{\beta}_k = \frac{1 - \bar{\alpha}_{k-1}}{1 - \bar{\alpha}_k}\, \beta_k.
$$

**推导要点**：前向链满足 $q(x_k \mid x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_k} x_0, (1-\bar{\alpha}_k)\mathbf{I})$ 且 $q(x_k \mid x_{k-1}) = \mathcal{N}(\sqrt{\alpha_k} x_{k-1}, \beta_k \mathbf{I})$。将 $x_k$ 对 $x_{k-1}$ 和 $x_0$ 做条件，利用高斯分布的共轭性即可得到上述后验均值。

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
   ↓ 变分下界
L_ELBO = KL(q(A^K|A^0) || p(A^K)) + Σ_k KL(q(A^{k-1}|A^k,A^0) || p_θ(A^{k-1}|A^k)) - log p_θ(A^0|A^1)
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

与 PPO 的主要差异：数据来自固定演示集，可跨 epoch 反复使用；无 Critic、无 Advantage、无 clipping；损失为去噪 MSE 而非策略梯度。

---

## 十、推理流程

```text
1. 读取最近 T_o 步观测 O_t

2. 编码 O_t → 条件向量 c（一次）

3. 初始化 A^K ~ N(0, I)，shape (T_p, D_a)
   （可选：warm-start 用上一轮未执行动作初始化）

4. for k = K, K-1, ..., 1:
       ε_pred = ε_θ(O_t, A^k, k)
       A^{k-1} = DenoiseStep(A^k, ε_pred, k)    # DDPM 或 DDIM

5. 执行 A^0[0], ..., A^0[T_a - 1]

6. 等待 T_a 步后回到步骤 1
```

实机部署中，策略与环境异步交互：`get_obs` 读取最新观测，`exec_actions` 将动作序列及时间戳送入插值控制器，不阻塞等待执行完成。

---

## 十一、与 PPO 的对照

| 概念 | PPO | Diffusion Policy |
|------|-----|------------------|
| 优化目标 | $\mathbb{E}[\sum \gamma^t r_t]$ | $\log p(\mathbf{A}^0 \mid \mathbf{O}_t)$（via ELBO） |
| 策略梯度 | $\nabla_\theta \log \pi_\theta(a \mid s) \cdot \hat{A}_t$ | $\nabla_\theta \|\epsilon - \epsilon_\theta\|^2$ |
| 训练数据 | 当前策略 rollout | 专家演示 |
| 价值函数 | 需要 Critic 估计 $V(s)$ | 不需要 |
| 多模态 | 对角高斯难表达 | 生成式，不同初始噪声对应不同模式 |
| 时序 | 逐步 MDP；RNN 可选 | 原生动作序列 |
| 推理代价 | 一次前向 | $N$ 步去噪（DDIM 可压缩） |

两者解决不同问题：PPO 适用于有奖励信号、需超越演示的在线学习；Diffusion Policy 适用于有高质量演示、奖励难以设计的模仿场景。实际系统中，常见做法是以 Diffusion Policy 初始化，再用 RL 微调，或将其作为 action proposal。

---

## 十二、关键超参数

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

## 十三、参考文献与延伸阅读

1. Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023. [arXiv:2303.04137](https://arxiv.org/pdf/2303.04137v5)
2. Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
3. Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021.
4. Janner et al., *Planning with Diffusion for Flexible Behavior Synthesis*, ICML 2022.
5. 本仓库 PPO 笔记：[`ppo_notes.md`](ppo_notes.md)

建议阅读顺序：Ho et al. 2020（DDPM 与 $\epsilon$-prediction）→ 论文 Sec. III–IV（动作序列 formulation 与架构）→ 官方 `diffusion_unet_image_policy.py` 中的 `compute_loss` 与 `predict_action`。

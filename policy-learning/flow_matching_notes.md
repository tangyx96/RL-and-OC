# Flow Matching 与机器人策略

下文以 Physical Intelligence 的 [openpi](https://github.com/Physical-Intelligence/openpi) 为主线，梳理 conditional flow matching（CFM）在 visuomotor 策略中的形式化、训练目标与推理流程。论文：[π₀: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)（Black et al., 2024）。动作序列的 receding horizon 设定见 [`diffusion_policy_notes.md`](diffusion_policy_notes.md)；DDPM 的 $\epsilon$-prediction 见同文第四节；离散 VLA 见 [`openvla_notes.md`](openvla_notes.md)。

---

## 一、算法定位

Flow matching（FM）是一类**连续时间生成模型**的训练目标：学习把简单源分布（通常为 $\mathcal{N}(0,I)$）沿 ODE 轨迹推至数据分布。将其接入机器人策略时，生成对象通常是**动作序列 chunk** $\mathbf{A}\in\mathbb{R}^{H\times D_a}$，条件为图像、语言与本体感觉 $\mathbf{O}$；优化仍属**离线模仿学习**，拟合 $p(\mathbf{A}\mid\mathbf{O})$，不使用环境奖励。

相对 Diffusion Policy（DDPM 的 $\epsilon$-prediction），FM 在机器人社区中逐渐成为 VLA 动作头的默认选择：π₀、π₀.₅、GR00T、SmolVLA 等均采用 velocity matching，而非多步 DDPM 去噪。openpi 仓库中 π₀ 与 π₀.₅ 的 flow head 为当前最有影响力的开源实现；π₀-FAST 走自回归离散动作，不在本文范围内。

| 特性 | Diffusion Policy | Flow Matching（π₀） | OpenVLA |
|------|------------------|---------------------|---------|
| 生成对象 | 动作序列 | 动作 chunk | 离散动作 token |
| 训练目标 | $\|\epsilon-\epsilon_\theta\|^2$ | $\|\mathbf{v}-\mathbf{v}_\theta\|^2$ | 下一 token 交叉熵 |
| 推理 | $K$ 步去噪（DDIM 可减至 $\sim 10$） | $N$ 步 Euler 积分（默认 $\sim 10$） | 自回归解码 |
| 骨干 | 1D U-Net + 视觉编码器 | 预训练 VLM + action expert | 预训练 VLM |
| 学习范式 | 离线模仿 | 离线模仿 | 离线模仿 |

FM 本身不是模仿学习算法；与 BC 的关系在于：将 $\pi_\theta(\mathbf{A}\mid\mathbf{O})$ 参数化为 flow 所诱导的 pushforward 分布，并以条件速度场回归实现极大似然（见第二节）。

---

## 二、与行为克隆的关系

设演示 $\mathcal{D}=\{(\mathbf{O}_j,\mathbf{A}_j)\}$，专家条件分布 $p^{\ast}(\mathbf{A}\mid\mathbf{O})$。行为克隆选取 $\pi_\theta$ 使

$$
\hat\theta=\arg\min_\theta\;
\mathbb{E}_{(\mathbf{O},\mathbf{A})\sim\mathcal{D}}
\big[-\log\pi_\theta(\mathbf{A}\mid\mathbf{O})\big],
$$

等价于最小化经验分布上的正向 KL（推导见 [`diffusion_policy_notes.md`](diffusion_policy_notes.md) 第 2.2 节）。Diffusion Policy 通过 DDPM 参数化 $\pi_\theta$；FM 策略通过 ODE 定义的 pushforward 参数化同一对象。二者共享动作序列接口：预测长度 $H$（`action_horizon`），执行其中 $T_a$ 步后重规划（见 DP 笔记第三节）。

---

## 三、Flow Matching 的形式化

### 3.1 连续归一化流

设 $\mathbf{x}_0\sim q_0$ 为数据（此处 $\mathbf{x}_0=\mathbf{A}$），$\mathbf{x}_1\sim p_1=\mathcal{N}(0,I)$ 为噪声。连续归一化流（CNF）由速度场 $\mathbf{v}_t(\mathbf{x})$ 定义 ODE

$$
\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t}=\mathbf{v}_t(\mathbf{x}_t),
\qquad t\in[0,1].
$$

若 $\mathbf{x}_0\sim q_0$，则 $t=1$ 时 $\mathbf{x}_1$ 的边缘分布 $q_1$ 由 pushforward 确定。生成时从 $\mathbf{x}_1\sim\mathcal{N}(0,I)$ 出发，沿 $t:1\to 0$ 积分得到 $\mathbf{x}_0$。

直接回归边缘速度场 $\mathbf{v}_t(\mathbf{x})$ 困难：需对未知边缘 $q_t(\mathbf{x})$ 求期望。Conditional Flow Matching（CFM, Lipman et al., 2023）转而监督**条件路径**上的速度，其期望与边缘场一致。

### 3.2 线性概率路径

取条件路径（rectified flow / 最优传输直线的特例）

$$
\mathbf{x}_t = (1-t)\,\mathbf{x}_0 + t\,\boldsymbol{\epsilon},
\qquad
\mathbf{x}_0\sim q_0,\;
\boldsymbol{\epsilon}\sim\mathcal{N}(0,I),\;
t\in(0,1).
\tag{1}
$$

对固定 $(\mathbf{x}_0,\boldsymbol{\epsilon})$，$\mathbf{x}_t$ 关于 $t$ 求导：

$$
\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = \boldsymbol{\epsilon} - \mathbf{x}_0.
\tag{2}
$$

条件速度场 $\mathbf{u}_t(\mathbf{x}_t\mid\mathbf{x}_0)= \boldsymbol{\epsilon}-\mathbf{x}_0$ 与 $t$、$\mathbf{x}_t$ 无关（在直线路径下为常向量）。这是 CFM 训练目标简洁的原因：网络只需回归 $\boldsymbol{\epsilon}-\mathbf{x}_0$，而非依赖 $t$ 的复杂 schedule。

加入观测条件 $\mathbf{O}$，路径与目标变为

$$
\mathbf{x}_t = (1-t)\,\mathbf{A} + t\,\boldsymbol{\epsilon},
\qquad
\mathbf{u} = \boldsymbol{\epsilon} - \mathbf{A},
\qquad
(\mathbf{O},\mathbf{A})\sim\mathcal{D}.
$$

### 3.3 条件 Flow Matching 损失

设 $\mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{O})$ 为待学习速度场。CFM 目标为

$$
\boxed{
\mathcal{L}_{\mathrm{CFM}}(\theta)
= \mathbb{E}_{\mathbf{A},\boldsymbol{\epsilon},t,\mathbf{O}}
\left[
\left\|
\mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{O})
- (\boldsymbol{\epsilon} - \mathbf{A})
\right\|^2
\right]
}
\tag{3}
$$

其中 $\mathbf{x}_t$ 由 (1) 构造，$t$ 从 $(0,1)$ 上采样。Lipman et al. 证明：在适当正则条件下，(3) 的梯度与边缘 flow matching 目标一致，故最小化 (3) 可学到正确的 pushforward 分布。

### 3.4 时间约定

直线路径有两种互为 $t\leftarrow 1-t$ 的写法：

| 约定 | $t=0$ | $t=1$ | 路径 | 条件速度 | 生成 |
|------|-------|-------|------|----------|------|
| Lipman；π₀ 论文 | 噪声 $\boldsymbol{\epsilon}$ | 数据 $\mathbf{A}$ | $(1-t)\boldsymbol{\epsilon}+t\mathbf{A}$ | $\mathbf{A}-\boldsymbol{\epsilon}$ | $t:0\to 1$ |
| openpi | 数据 $\mathbf{A}$ | 噪声 $\boldsymbol{\epsilon}$ | $(1-t)\mathbf{A}+t\boldsymbol{\epsilon}$ | $\boldsymbol{\epsilon}-\mathbf{A}$ | $t:1\to 0$ |

openpi 的 `pi0.py` 采用后者，注释写明与 π₀ 论文相反。下文公式与实现均按 openpi：$t=1$ 为纯噪声，$t=0$ 为目标动作。阅读代码或论文时须先核对 $t$ 的端点，不可直接套用 DDPM 的 $k=K$ 最噪写法。

---

## 四、训练目标推导

### Step 1：从极大似然到流模型

行为克隆要求 $\pi_\theta(\mathbf{A}\mid\mathbf{O})\approx p^{\ast}(\mathbf{A}\mid\mathbf{O})$。CNF 定义

$$
\mathbf{A} = T_\theta(\boldsymbol{\epsilon};\mathbf{O}),
\qquad
\boldsymbol{\epsilon}\sim\mathcal{N}(0,I),
$$

其中 $T_\theta$ 为 ODE $\mathrm{d}\mathbf{x}/\mathrm{d}t=\mathbf{v}_\theta(\mathbf{x},t,\mathbf{O})$ 从 $t=1$ 到 $t=0$ 的流映射。直接对 $\log\pi_\theta(\mathbf{A}\mid\mathbf{O})$ 求梯度需计算 Jacobian 行列式（连续情形为 trace 项），代价高。

### Step 2：条件路径上的回归

CFM 避开显式似然，改为：对每条样本 $(\mathbf{A},\boldsymbol{\epsilon})$ 构造路径 (1)，要求 $\mathbf{v}_\theta$ 在该路径切向方向上匹配 $\boldsymbol{\epsilon}-\mathbf{A}$。因 (2) 与 $t$ 无关，监督信号不随 $t$ 变化，训练方差低于 DDPM 中对不同 $k$ 预测不同 $\epsilon$ 的加权 ELBO 项。

### Step 3：与 score matching 的关系

Song et al. 指出扩散模型的 $\epsilon$-prediction 等价于 score matching。对高斯路径 $q(\mathbf{x}_t\mid\mathbf{x}_0)=\mathcal{N}((1-t)\mathbf{x}_0,\,t^2 I)$，有

$$
\nabla_{\mathbf{x}_t}\log q(\mathbf{x}_t\mid\mathbf{x}_0)
= -\frac{\mathbf{x}_t-(1-t)\mathbf{x}_0}{t^2}.
$$

由 $\mathbf{x}_t=(1-t)\mathbf{x}_0+t\boldsymbol{\epsilon}$ 可改写为 $\boldsymbol{\epsilon}-\mathbf{x}_0$ 的仿射函数。故 FM 的 velocity 回归与 DDPM 的 noise 回归在**线性路径**下信息等价，但 FM 直接参数化 ODE，推理时无需 DDPM 的 $\alpha_k,\beta_k$ schedule，积分步数 $N$ 与训练时 $t$ 的离散化解耦。

**推导链总结**：

```text
min KL(p* || π_θ)  （行为克隆）
   ↓ 用 CNF pushforward 定义 π_θ
需优化 log |det ∂T/∂ε| + ...
   ↓ CFM：改监督条件路径上的速度
L_CFM = E[||v_θ(x_t,t,O) - (ε - A)||²]
   ↓ 线性路径下 dx/dt = ε - A 为常数
训练 = 回归速度场；推理 = Euler 积分 ODE
```

---

## 五、推理：ODE 数值积分

训练完成后，从 $\mathbf{x}_1=\boldsymbol{\epsilon}\sim\mathcal{N}(0,I)$ 出发，沿 $t:1\to 0$ 积分。最常用为前向 Euler：

$$
\mathbf{x}_{t+\Delta t} = \mathbf{x}_t + \Delta t\,\mathbf{v}_\theta(\mathbf{x}_t, t, \mathbf{O}),
\qquad
\Delta t = -\frac{1}{N}.
\tag{4}
$$

$N$ 为 `num_steps`（openpi 默认 $10$）。$N=1$ 时退化为单步映射；$N$ 增大通常改善样本质量，代价是延迟。与 DDIM 类似，FM 允许训练时在连续 $t$ 上采样、推理时仅用少量离散步——但 FM 的 ODE 为确定性（不含 DDPM 反向过程中的随机项），实现更简单。

---

## 六、π₀ 中的 Flow Matching

### 6.1 问题形式化

π₀ 将预训练 VLM 扩展为 vision-language-action（VLA）模型：输入多视角图像、语言指令与本体感觉 $\mathbf{s}$，输出长度 $H$ 的动作 chunk $\mathbf{A}\in\mathbb{R}^{H\times D_a}$。策略接口为

$$
\mathbf{A} = T_\theta(\boldsymbol{\epsilon};\, o, l, \mathbf{s}),
\qquad
\boldsymbol{\epsilon}\sim\mathcal{N}(0,I),
$$

其中 $(o,l,\mathbf{s})$ 经 VLM 编码为条件。控制频率可达 $50\,\mathrm{Hz}$，依赖 chunk 执行与 receding horizon，而非逐步自回归。

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
- **attention mask**：图像与语言 token 双向可见；动作 token 对 prefix 可见，动作序列内部为因果或块因果（见 `make_attn_mask`）。

π₀.₅ 相对 π₀ 的主要改动：去掉显式 state token，时间嵌入经 MLP 后以 adaRMS 注入 action expert；训练采用 knowledge insulation 以改善开集泛化。openpi 仓库对 π₀.₅ 目前仅开放 flow matching 头。

### 6.3 训练（`Pi0.compute_loss`）

从 openpi 实现，单步损失构造如下：

1. 采样 $\boldsymbol{\epsilon}\sim\mathcal{N}(0,I)$，与 $\mathbf{A}$ 同形；
2. 采样 $t\sim\mathrm{Beta}(1.5,1)\times 0.999+0.001$（偏向小 $t$，强调接近数据的路径段）；
3. 构造 $\mathbf{x}_t = t\boldsymbol{\epsilon}+(1-t)\mathbf{A}$；
4. 目标速度 $\mathbf{u}=\boldsymbol{\epsilon}-\mathbf{A}$；
5. 单次前向：prefix + suffix，输出 $\mathbf{v}_\theta$；
6. $\mathcal{L}=\mathrm{mean}\|\mathbf{v}_\theta-\mathbf{u}\|^2$，对 $H\times D_a$ 维求平均。

与 (3) 式一致；Beta 采样替代均匀 $t\sim\mathcal{U}(0,1)$ 属于实现层面的稳定性取舍，不改变 CFM 目标的形式。

### 6.4 推理（`Pi0.sample_actions`）

1. **Prefix 编码**：图像与语言 token 一次前向，填充 KV cache；
2. **初始化**：$\mathbf{x}\leftarrow\boldsymbol{\epsilon}$，$t\leftarrow 1$；
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
| 条件分布 | $p(\mathbf{A}\mid\mathbf{O})$ | $p(\mathbf{A}\mid o,l,\mathbf{s})$ |
| 参数化 | $\epsilon_\theta(\mathbf{A}^k,k,\mathbf{O})$ | $\mathbf{v}_\theta(\mathbf{x}_t,t,\mathbf{O})$ |
| 路径 | DDPM 前向 $q(\mathbf{A}^k\mid\mathbf{A}^0)$ | 直线 $(1-t)\mathbf{A}+t\boldsymbol{\epsilon}$ |
| 训练监督 | 预测噪声 $\boldsymbol{\epsilon}$ | 预测速度 $\boldsymbol{\epsilon}-\mathbf{A}$ |
| 推理 | 离散去噪 $k=K,\ldots,1$ | 连续 ODE，Euler $N$ 步 |
| 骨干 | 1D U-Net | VLM + action expert |
| 预训练 | 通常无 | 10k+ 小时机器人数据 + VLM |

二者在「生成式模仿、动作 chunk、多模态动作分布」上一致；差别主要在生成头与是否利用 VLM 先验。LeRobot 的 `MultiTaskDiT` 可在同一 U-Net 骨干上切换 `objective=diffusion` 与 `objective=flow_matching`，便于对照实验。

---

## 八、训练与推理流程

### 8.1 训练

```text
1. 演示转为 LeRobot 格式（openpi 微调入口）

2. 每个 step：
   a. 采样 batch (O, A)
   b. ε ~ N(0,I)，t ~ Beta(1.5,1)（或 Uniform）
   c. x_t = t·ε + (1-t)·A
   d. v_θ = network(O, x_t, t)
   e. L = ||v_θ - (ε - A)||²
   f. 反向传播；可选 EMA、LoRA

3. 定期在仿真/实机评估
```

### 8.2 推理

```text
1. 读取图像、语言 prompt、本体感觉

2. Prefix 前向 → KV cache

3. x ~ N(0,I)，t = 1

4. repeat N times:
       v = v_θ(O, x, t)
       x ← x - (1/N)·v
       t ← t - 1/N

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
| $t$ 采样 | Beta$(1.5,1)$ | 训练时时间分布 |
| VLM | PaliGemma | 预训练视觉–语言骨干 |
| action expert | Gemma 300M 级 | 动作专用 Transformer |
| 微调 | LoRA / 全参 | 全参需 $>70\,\mathrm{GB}$ GPU |

---

## 十、参考文献与延伸阅读

1. Black, K., et al. *π₀: A Vision-Language-Action Flow Model for General Robot Control*. arXiv:2410.24164, 2024.
2. Lipman, Y., et al. *Flow Matching for Generative Modeling*. ICLR, 2023.
3. Liu, Q., et al. *Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow*. ICLR, 2023.
4. Chi, C., et al. *Diffusion Policy*. RSS, 2023. [`diffusion_policy_notes.md`](diffusion_policy_notes.md)
5. Physical Intelligence. [openpi](https://github.com/Physical-Intelligence/openpi) — π₀ / π₀.₅ 官方实现。
6. 视频–动作联合 CFM：[`wam_notes.md`](wam_notes.md) 第四节。

建议阅读顺序：Lipman et al. 2023（CFM 目标）→ Black et al. 2024（π₀ 架构与 VLA 设定）→ openpi `src/openpi/models/pi0.py` 中 `compute_loss` 与 `sample_actions` → 与 [`diffusion_policy_notes.md`](diffusion_policy_notes.md) 对照 DDPM 路径。

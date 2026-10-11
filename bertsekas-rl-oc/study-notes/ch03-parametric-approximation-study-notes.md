# 第 3 章 Parametric Approximation — 分节笔记

> **文献**：Bertsekas, *RL and Optimal Control* Ch.3（2019 draft）。  
> **文本**：书稿第 3 章。  
> **位置**：第 2 章给出 $\tilde J$ 的构造方式；本章固定**架构** $\tilde J_k(x_k,r_k)$ / $\tilde Q_k(x_k,u_k,r_k)$ 与**训练算法**（LS、增量梯度、FVI、拟合 Q），并讨论与 Bellman 目标的偏差。

---

## 章首

参数化方法将“选前瞻函数”化为：

1. **架构设计**——特征、线性/非线性、神经网络；  
2. **监督目标**——$(x^s,\beta^s)$ 或 FVI 中的 Bellman 目标 $\beta_k^s$；  
3. **优化**——(3.2) 最小二乘及 (3.4)–(3.16) 增量/Newton 法。

除半穷举搜索 $r$ 外，本书强调**数值优化 + 采样**。

---

## §3.1 Approximation Architectures

### 总述

**架构** $\tilde J_k(x_k,r_k)$：**训练** = 选 $r_k$ 使前瞻或 FVI 有效。架构类记为 approximation architecture；$r_k$ 称 weights / 参数向量。

---

### §3.1.1 Linear and Nonlinear Feature-Based Architectures

#### 主要内容

**一般形式**：

```math
\tilde J_k(x_k,r_k)=\hat J_k(\phi_k(x_k),r_k),
```

$\phi_k(x_k)=(\phi_{1,k},\ldots,\phi_{m,k})'$ 为**特征**；$\hat J_k$ 为读出头（常为线性）。

**线性架构 (3.1)**：

```math
\tilde J_k(x_k,r_k)=r_k'\phi_k(x_k)=\sum_{\ell=1}^m r_{\ell,k}\,\phi_{\ell,k}(x_k).
```

几何上：$\tilde J_k$ 属于 $\{\phi_{\ell,k}\}$ 张成的子空间；$\phi_{\ell,k}$ 亦称 **basis functions**。

| 例 | 特征类型 | 调参局部性 |
|----|----------|------------|
| **3.1.1** 分段常数 | $\phi_\ell(x)=\mathbf{1}_{x\in S_\ell}$ | 改 $r_\ell$ 仅影响 $S_\ell$ |
| **3.1.2** 多项式 | $1,x_i,x_ix_j,\ldots$ | 全局 |
| **3.1.3** Tetris | 22 维手工特征 | 应用驱动 |
| **3.1.4** 国际象棋 | Shannon 型评估 + 线性权 | 经典 |
| **3.1.5** 部分信息 | $\phi(S_k(I_k))$ 非 $\phi(I_k)$ | 维数控制 |

**非平稳 / 有限时域**：$\phi_k,r_k$ 可随 $k$ 变（阶段、时变状态空间）。**平稳长/无限时域**：常共用同一 $(\phi,r)$（Ch.4–5）。

**NN 角色**：学习 $\phi(x,v)$ 与 $r$  simultaneously（§3.2），替代手工特征。

#### 要点

- 最优 $J_k^{*}$ 可能高度非线性；特征应编码主要非线性，使 $\hat J_k$ 保持简单（常线性）。  
- AlphaZero 等：**发现**特征而非 Tetris 式手工。

#### 注意点

- 特征不足 → 系统性偏差，非增量法 alone 可修复。

---

### §3.1.2 Training of Linear and Nonlinear Architectures

#### 主要内容

训练集 $(x^s,\beta^s)$，$s=1,\ldots,q$；$\beta^s\approx J(x^s)$（带噪）：

```math
\min_r \sum_{s=1}^q\big(\tilde J(x^s,r)-\beta^s\big)^2. \qquad \text{(3.2)}
```

**线性闭式 (3.3)**：

```math
r=\Big(\sum_s \phi(x^s)\phi(x^s)'\Big)^{-1}\sum_s \phi(x^s)\beta^s.
```

（设计矩阵满秩时。）

**非线性**：(3.2) 非凸 → §3.1.3 或通用 NLP。

**$\beta^s$ 来源**：蒙特卡洛 rollout、专家标注、Bellman 目标（FVI）、$J_{\mu}$ 样本等。

#### 要点

- 线性：一次 LS 得全局最优（该凸问题内）。  
- 正则化：$+\lambda\|r-r_0\|^2$ 改善病态与过拟合。

#### 注意点

- $\beta^s$ 偏差直接传递为策略偏差；噪声大时需更多样本或更平滑架构。

---

### §3.1.3 Incremental Gradient and Newton Methods

面向 (3.2) 等**可微参数化**训练；属非线性规划标准内容，可独立于 DP 阅读。本节侧重**实现直觉**；收敛理论见 [BeT96]、[Ber16] 及章末文献。

---

#### 问题形式 (3.4)

将训练化为分量之和（用 $y$ 代 $r$，$m$ 代 $q$）：

```math
\min_y f(y)=\sum_{i=1}^m f_i(y),
```

每个 $f_i:\mathbb{R}^n\to\mathbb{R}$ 可微；最小二乘中 $f_i(y)=\big(\tilde J(x^i,y)-\beta^i\big)^2$。

---

#### 全梯度法 (3.5)

```math
y^{k+1}=y^k-\gamma^k\nabla f(y^k)=y^k-\gamma^k\sum_{i=1}^m\nabla f_i(y^k).
```

每步需**全部** $m$ 个分量梯度——$m$ 大时（NN 训练）代价高。

---

#### 增量梯度法 (3.6)

每步只用**一个**分量：

```math
y^{k+1}=y^k-\gamma^k\nabla f_{i_k}(y^k),
```

$i_k\in\{1,\ldots,m\}$ 由选取规则决定。**计算**：每步 1 次梯度 vs 全梯度 $m$ 次。

**分量选取规则**（影响性能）：

| 规则 | 定义 | 注意 |
|------|------|------|
| **Cyclic** | $i_k=(k\bmod m)+1$；连续 $m$ 步遍历 $1,\ldots,m$ 为一 **cycle** | 必须每 cycle 含全部分量，否则偏置 |
| **Uniform random** | 每步独立均匀抽 $i_k$ | 方差：某分量一 cycle 内出现次数随机 |
| **Shuffle（实践常用）** | 每 cycle 内随机排列后依次处理 | [GOP15c]：$m$ 大时常优于 uniform |

---

#### 远/近最优：两种互补行为

| 阶段 | 增量 vs 全梯度 |
|------|----------------|
| **(a) 远离最优** | 单分量梯度常指向「大致正确」方向；极端地 $f_i\equiv f$ 时，增量步长取 $\gamma^k m$ 等价全梯度一步 → **可快 ~$m$ 倍** |
| **(b) 接近最优** | 全梯度在合理假设下可**常数步长**收敛；增量法一般需 $\gamma^k\to 0$，极限速率可更慢 |

**步长尺度**：为保持远离最优时的优势，增量法常用比全梯度**大 ~$m$ 倍**的 $\gamma^k$（使单步增量与全梯度步幅相当）。

---

#### 例 3.1.6（标量最小二乘，Fig. 3.1.4）

```math
\min_y f(y)=\sum_{i=1}^m (c_i y-b_i)^2,\quad c_i\neq 0.
```

各分量最小值 $y_i^{*}=b_i/c_i$；总最小值

```math
y^{*}=\frac{\sum_{i=1}^m c_i b_i}{\sum_{i=1}^m c_i^2}.
```

**Region of confusion**：$R=[\min_i y_i^{*},\max_i y_i^{*}]$。在 $R$ **外**（far-out region），各 $\nabla f_i(y)=c_i(c_i y-b_i)$ 与 $\nabla f(y)$ **同号** → 小步长下单步增量朝 $y^{*}$ 前进。

**$R$ 内**：第 $i$ 步仅当 $y^k$ 不在 $[y_i^{*},y^{*}]$ 上才朝 $y^{*}$ 前进 → **振荡**；增量法需 $\gamma^k\to 0$ 才收敛。

**全梯度**（常数步长）：

```math
y^{k+1}=y^k-\gamma\sum_{i=1}^m c_i(c_i y^k-b_i),\quad 0<\gamma\le\frac{1}{\sum_{i=1}^m c_i^2}
```

可收敛至 $y^{*}$；但在 $R$ 外，**一整轮** $m$ 步增量 ≈ **一步**全梯度（步长匹配时）。

**多维推广**：若 $\nabla f_{i_k}(y^k)$ 与 $\nabla f(y^k)$ 夹角 $<90°$，增量步仍可能下降——组件「不太相异」且离最优较远时常见。

---

#### 步长选择与对角缩放

**收敛要求**：增量方向与真梯度差 $\propto\gamma^k$ → 收敛到局部极小需 $\gamma^k\to 0$（非凸时仅能期望局部极小）。

**常数小步长**：迭代进入 **limit cycle**——cycle 内第 $i$ 步子序列 $\psi_i$ 各收敛到**不同**极限；cycle 末的 $y^k$ 序列收敛，但极限**未必**是 $f$ 的极小（凸 $f$ 亦如此），通常接近最优。实践：**分段衰减**（常数若干 iter → 乘因子降步长 → 重复至 floor）。

**递减步长例**：

```math
\gamma^k=\min\Big\{\bar\gamma,\,\frac{\beta_1}{k+\beta_2}\Big\},\quad \bar\gamma,\beta_1,\beta_2>0.
```

**自适应**：检测振荡（在 confusion region）时减小 $\gamma^k$；见 [Tse98], [MYF03], [GOP15a]。

**对角缩放**：各坐标 $\gamma_j^k$ 不同；一般 NLP 用 $\gamma_j^k=\gamma\big/\frac{\partial^2 f}{\partial y_j^2}(y^k)$。但对**加性** $f=\sum f_i$，二阶导不便于逐分量累加——见下 **增量 Newton** 的对角缩放。

---

#### 随机梯度下降 SGD (3.7) vs 增量梯度 (3.8)

**SGD**（最小化期望）：

```math
\min_y f(y)=\mathbb{E}_w\big[F(y,w)\big],\qquad y^{k+1}=y^k-\gamma^k\nabla_y F(y^k,w^k),
```

$w^k$ 为 $w$ 的样本；与随机逼近、机器学习「SGD/backprop」同源 [BeT96], [KuY03], [Spa03] 等。

**与 (3.6) 的关系**：uniform random 选 $i_k$ 时，(3.6) 可**视为** SGD——但将有限和 **硬套** 为随机问题会：

- 无法利用 **aggregated gradient** 等有限和专用法；  
- 掩盖 $m$、**处理顺序**等确定性结构（shuffle cyclic 常优于 blind uniform）。

**Shuffle 优于 uniform 的直觉**：shuffle 每 cycle **恰好**每分量 1 次；uniform 仅**平均** 1 次，方差更大 → 梯度估计误差更大 → 下降方向更差 [GOP15c]。

---

#### 聚合增量梯度 (3.10)

用最近 $m$ 次分量梯度之和近似全梯度：

```math
y^{k+1}=y^k-\gamma^k\sum_{\ell=0}^{m-1}\nabla f_{i_{k-\ell}}(y^{k-\ell}).
```

（$k<m$ 时求和至 $\ell=k$，步长相应放大。）

**思想**：比单分量 $\nabla f_{i_k}$ 更接近 $\nabla f(y^k)$ → **渐近**可像全梯度一样**线性收敛**（强凸 + 足够小常数步长 [Ber16] §2.4.2），而每步仍只算 1 个新梯度。

**代价**：须存最近 $m$ 个分量梯度；**变体**用周期全梯度刷新，

```math
\tilde s^k=\nabla f_{i_k}(y^k)-\nabla f_{i_k}(\bar y^k)+\sum_{\ell=0}^{m-1}\nabla f_{i_{k-\ell}}(\bar y^k),
```

$\bar y^k$ 为最近一次算全梯度的点——省内存，$m$ 极大时周期全梯度仍贵。

**局限**：希望**第一轮 cycle** 内就收敛时，聚合旧梯度收益有限。

---

#### 增量 Newton 法 (3.11)–(3.16)

设 $f=\sum_{i=1}^m f_i$，各 $f_i$ 凸、二阶可微。分量 $f_i$ 在 $\psi$ 处的二次近似：

```math
\tilde f_i(y;\psi)=\nabla f_i(\psi)'(y-\psi)+\tfrac12(y-\psi)'\nabla^2 f_i(\psi)(y-\psi). \qquad \text{(3.11)}
```

**一 cycle**（从 $y^k$ 出发，$\psi_{0,k}=y^k$）：

```math
\psi_{i,k}\in\arg\min_{y\in\mathbb{R}^n}\sum_{\ell=1}^{i}\tilde f_\ell(y;\psi_{\ell-1,k}),\quad i=1,\ldots,m;\qquad y^{k+1}=\psi_{m,k}. \qquad \text{(3.12)}
```

**高效实现**（(3.13)–(3.15)）：

```math
\psi_{i,k}=\psi_{i-1,k}-D_{i,k}\nabla f_i(\psi_{i-1,k}),
```

```math
D_{i,k}=\Big(\sum_{\ell=1}^{i}\nabla^2 f_\ell(\psi_{\ell-1,k})\Big)^{-1},
```

```math
D_{i,k}^{-1}=D_{i-1,k}^{-1}+\nabla^2 f_i(\psi_{i-1,k}). \qquad \text{(3.15)}
```

**特殊情形**：各 $f_i$ 二次 → **一 cycle 即得精确解**（$\tilde f_i$ 与 $f_i$ 仅差常数）。

**最小二乘结构** $f_i(y)=h_i(a_i'y-b_i)$：Hessian 为秩 1，(3.15) 用 **Sherman–Morrison** 公式高效更新 $D_{i,k}$。

**多 cycle 渐近**：$D_{i,k}\sim O(1/k)$ → 渐近像 $O(1/k)$ 步长的增量梯度，**慢于线性**。**Restart**：每 cycle 初重置/放大 $D$；或修正 (3.16)：

```math
D_{i,k}^{-1}=\beta_k D_{i-1,k}^{-1}+\nabla^2 f_i(\psi_{i-1,k}),\quad \beta_k\in(0,1),
```

可加速；在 $\nabla^2 f(y^{*})$ 非奇异且 restart 设计下可达线性（甚至超线性）邻域收敛。

**对角近似**：$\nabla^2 f_i$ 取对角 → 退化为**对角缩放增量梯度**，开销接近 (3.6)，适合 $n$ 大。

**适用**：迭代次数常少于增量梯度，但每步含 Hessian + 矩阵运算 → 仅当 **$n$ 较小** 时划算。

---

#### 与 NN 训练 (3.18)

(3.18) 的代价 = **大量**样本平方和 → 几乎必用增量法；**minibatch** = 每步聚合一小批分量梯度（(3.10) 的特例/近似）。正则化、过拟合等见 §3.2。

---

## §3.2 Neural Networks

NN 用于分类、识别等广泛任务；本书聚焦**有限时域 DP** 中逼近 $J_k^{*}$。典型流程（§3.3 FVI）：先用 NN 逼近 $J_{N-1}^{*}$，再反向逐阶段逼近 $J_{N-2}^{*},\ldots,J_0^{*}$。

---

### 双参数架构 (3.17) 与单层感知机（Fig. 3.2.1）

```math
\tilde J(x,v,r)=r'\phi(x,v)=\sum_{\ell=1}^m r_\ell\,\phi_\ell(x,v).
```

| 参数 | 角色 |
|------|------|
| **$v=(A,b)$** | 定**特征映射** $\phi(x,v)$（线性层 + 非线性层） |
| **$r$** | 对特征 $\phi_\ell$ 的**线性权重** |

训练集 $(x^s,\beta^s)$，$s=1,\ldots,q$；$\beta^s$ 为 cost 样本（带噪）。目标：选 $(v,r)$ 使 $\tilde J$ 在最小二乘意义下拟合——$(v,r)$ 的生成与求解见 §3.2.1；**如何产生** $(x^s,\beta^s)$ 见 §3.3。

**状态编码**：$x\mapsto y(x)=(y_1(x),\ldots,y_n(x))$（可为 $x$ 分量、定性编码、或问题相关手工特征）。

**线性层**（$m\times n$ 矩阵 $A$，偏置 $b\in\mathbb{R}^m$）：

```math
(Ay(x)+b)_\ell,\quad \ell=1,\ldots,m.
```

**非线性层**：每标量输出经可微单调 $\sigma:\mathbb{R}\to\mathbb{R}$，得特征

```math
\phi_\ell(x,v)=\sigma\big((Ay(x)+b)_\ell\big),\quad \ell=1,\ldots,m.
```

**输出**：

```math
\tilde J(x,v,r)=\sum_{\ell=1}^m r_\ell\,\sigma\big((Ay(x)+b)_\ell\big).
```

注意 $\phi_\ell$ 只依赖 $A$ 的**第 $\ell$ 行**与 $b_\ell$——可对单行/单分量加约束以实现问题相关「手工」效应。

---

#### 常用非线性单元（Fig. 3.2.2–3.2.3）

| 类型 | 形式 | 特点 |
|------|------|------|
| **ReLU（整流）** | $\max\{0,\xi\}$；可微近似 $\sigma(\xi)=\ln(1+e^\xi)$，$\sigma'(\xi)=e^\xi/(1+e^\xi)$ | 现代默认；角点平滑 |
| **Sigmoid 族** | $\tanh(\xi)$；logistic $\sigma(\xi)=1/(1+e^{-\xi})$ | 有界：$\lim_{\xi\to-\infty}\sigma(\xi)<\lim_{\xi\to\infty}\sigma(\xi)<\infty$ |

下文统称 **nonlinear unit** / **nonlinear layer**，只假设可微。

---

#### 状态编码 = 特征提取（Fig. 3.2.4）

可将流程视为：**特征提取** $x\mapsto y(x)$（或更复杂映射）→ NN 输入 → cost 输出。

**直觉**：好的 $y(x)$ 捕获 $J^{*}$ 的主要非线性 → NN 可**更简单**（少隐层/少单元）、**更易训**。经验支持，但难严格量化。

与 §3.1.1 手工 $\phi_k(x_k)$ 的关系：NN **同时学习** $\phi$ 与 $r$；AlphaZero 等是**发现特征**而非 Tetris 式手工。

---

### §3.2.1 Training of Neural Networks

给定 $(x^s,\beta^s)$，解**非凸**最小二乘 (3.18)：

```math
\min_{A,b,r}\sum_{s=1}^q\Big(\sum_{\ell=1}^m r_\ell\,\sigma\big((Ay(x^s)+b)_\ell\big)-\beta^s\Big)^2. \qquad \text{(3.18)}
```

可能存在**多个局部极小**；无约束可微 → 标准梯度法；**结构**为 $q$ 项之和 → **必用** §3.1.3 增量法（backprop / SGD / minibatch）。

---

#### 正则化与过拟合

常加二次正则 $\lambda(\|A\|^2+\|b\|^2+\|r\|^2)$：

- 算法上：改善病态、易求解；  
- **主要**：防 **overfitting**——参数量 $\approx$ 训练集大小时，训练误差低但**泛化**差。

实践还涉及：初始化、缩放、dropout、早停、验证集等（[Bis95], [GBC16], [Hay09]）。

---

#### 训练三问题（书中 (a)–(c)）

**(a) 解法选择**

- 代价 = $\sum_{s=1}^q f_s(A,b,r)$，全梯度/全代价 $O(q)$ 太贵；  
- **增量梯度 + shuffle cyclic + minibatch** 为标配；不必极高精度——训练通常**离线**，近似即可。

**(b) 万能逼近定理（Universal Approximation）**

设 $x\in\mathbb{R}^n$，取 $y(x)\equiv x$。对闭有界 $S\subset X$，任意**分段连续** $J:S\to\mathbb{R}$ 可被 (3.17) 型 NN **任意精度逼近**（适当范数意义下），只要隐层单元数 $m$ **足够大** [Cyb89], [Fun89], [HSW89], [LLP93]。

**不回答**：给定问题需多少 $m$——实践中逐步增大 $m$ 直到满意；$m$ 大 → 训练更难 → 促发**多层/深度**结构（§3.2.2）。

**(c) NN 能产出什么特征？**

**短答**：实践中感兴趣的特征，**单隐层 + 充分多单元 + 前后线性层** 即可产生或逼近（万能逼近推论）；**不必**多隐层，但深度网络可能用**更少**单元。

---

#### ReLU 特征构造（(3.19)–(3.20)，Fig. 3.2.6–3.2.7）

标量 $x$，整流 $\sigma(\xi)=\max\{0,\xi\}$。线性 $L(x)=\gamma(x-\beta)$ 后接 ReLU：

```math
\phi_{\beta,\gamma}(x)=\max\{0,\,\gamma(x-\beta)\}. \qquad \text{(3.19)}
```

**两 ReLU 之差**（同斜率 $\gamma$，不同移位 $\beta_1,\beta_2$）：

```math
\phi_{\beta_1,\beta_2,\gamma}(x)=\phi_{\beta_1,\gamma}(x)-\phi_{\beta_2,\gamma}(x),
```

得**斜坡/门形**特征（Fig. 3.2.7(a)）。

**再差一次** → **脉冲（pulse）** 特征：

```math
\phi_{\beta_1,\beta_2,\beta_3,\beta_4,\gamma}(x)=\phi_{\beta_1,\beta_2,\gamma}(x)-\phi_{\beta_3,\beta_4,\gamma}(x). \qquad \text{(3.20)}
```

（Fig. 3.2.7(b)）。脉冲的线性组合可逼近**任意形状**特征——标量 $x$ 上特征形成的机制，也是万能逼近证明核心 [Cyb89]；可推广到多维 $x$。

---

### §3.2.2 Multilayer and Deep Neural Networks

**多层结构**（Fig. 3.2.5）：非线性层输出 → 下一线性层输入；可额外接入 $x$ 或 $y(x)$ 的分量（skip）。

```math
F(L_1,\ldots,L_{m+1},x)=L_{m+1}\Sigma_m L_m\cdots\Sigma_1 L_1 x,
```

$L_k$ = 第 $k$ 线性层矩阵；$\Sigma_k$ = 对向量逐分量施 $\sigma$。

---

#### 为何多层？（单隐层已万能逼近）

| 理由 | 说明 |
|------|------|
| **(a) 特征层次** | 每层非线性输出 = 下一层输入；**层级特征**——早期层可特化（边缘/局部），后期层组合（Go/图像） |
| **(b) 结构先验** | 稀疏 $A$、**卷积**（局部连接 + 权共享）→ 参数量大减、训练易；国际象棋/围棋棋盘、图像 |

**深度实践**：早期常用 1–3 隐层；**deep NN**（很多层）在图像/语音/近似 DP 成功。**AlphaGo/AlphaZero** [SHM16], [SHS17]：深度 NN + MCTS（Ch.2 §2.4.2）。对比：**TD-Gammon** 系双陆棋程序**不需**多非线性层仍表现良好。

---

#### 训练与反向传播（Backpropagation）

多层训练目标同 (3.18)，$\phi_\ell(x,v)$ 为**最后**非线性层输出；增量法对单项

```math
\Big(\sum_{\ell=1}^m r_\ell\,\phi_\ell(x^s,v)-\beta^s\Big)^2
```

求梯度。

**误差**（单样本，期望输出 $y=\beta^s$，网络输出 $F$）：

```math
E(L_1,\ldots,L_{m+1})=\big(y-F(L_1,\ldots,L_{m+1},x)\big)^2,\quad e=y-F(\cdot).
```

**对线性层权重** $L_k(i,j)$ 的偏导 (3.21)：

```math
\frac{\partial E}{\partial L_k(i,j)}=-e'\,L_{m+1}\Sigma_m L_m\cdots L_{k+1}\Sigma_k\,I_{ij}\,\Sigma_{k-1}L_{k-1}\cdots\Sigma_1 L_1 x,
```

$\Sigma_n$ = 第 $n$ 隐层 $\sigma$ 在对应点处导数的**对角阵**；$I_{ij}$ = $L_k$ 全 0 仅 $(i,j)$ 为 1。

**两趟计算**：

1. **Forward pass**：顺序算 $L_1 x,\,L_2\Sigma_1 L_1 x,\,\ldots,\,F$——得各层 $\Sigma_n$ 求导点与 $e$；  
2. **Backward pass**：从 $e'L_{m+1}\Sigma_m$ 起，反向累乘 $L_k,\Sigma_k$，得 (3.21) 中各项。

名称「backpropagation」在文献中多种用法；此处 = **链式法则**的高效实现 ([BeT96] §3.1.1)。

**Minibatch**：每步对若干样本平均梯度——减振荡、可用更大步长（§3.1.3 聚合梯度思想）。

---

#### 其他多层架构

**GMDH**（Ivakhnenko，60 年代起）：**多项式**非线性多层，非 sigmoid/ReLU；推理应用多，**近似 DP 尚无**应用——某些问题多项式可能比 sigmoid 更合适 [Iva68], [Iva71]。

**Schmidhuber [Sch15]** 综述：其他多层架构 + 变体 backprop。

---

#### 与 DP 流水线衔接

| 环节 | 本章位置 |
|------|----------|
| 架构 $(v,r)$、训练 (3.18) | §3.2 |
| 增量/backprop 求解 | §3.1.3 |
| 反向阶段 FVI、$\beta_k^s$ 来源 | §3.3 |
| 在线控 / $\tilde Q$ 架构 | §3.4 |
| 无限时域 Actor–Critic | Ch.4–5 |

---

## §3.3 Sequential Dynamic Programming Approximation（FVI）

**Fitted Value Iteration（FVI）**：为有限时域问题训练 $\tilde J_k(x_k,r_k)$；$r_k$ **按阶段反向**确定，与精确 DP 同序——先 $r_{N-1}$，再 $r_{N-2}$，…，$r_0$（Fig. 3.3.1）。

**思想**：每阶段 $k$ 对大量样本态 $x_k^s$ 做 LS，使 $\tilde J_k$ **拟合 DP 方程**在样本上的值——用监督回归逼近 Bellman 备份。

---

#### 算法流程（Fig. 3.3.1）

**输入**：每阶段 $k$ 的样本态 $x_k^s$，$s=1,\ldots,q$（MC 仿真等生成）。

**已知** $r_{k+1}$ 时，Bellman **目标**（标量）：

```math
\beta_k^s=\min_{u\in U_k(x_k^s)}\mathbb{E}\Big\{g_k(x_k^s,u,w_k)+\tilde J_{k+1}\big(f_k(x_k^s,u,w_k),r_{k+1}\big)\Big\}.
```

**LS 求** $r_k$（(3.22)）：

```math
r_k\in\arg\min_r\sum_{s=1}^q\Big(\tilde J_k(x_k^s,r)-\beta_k^s\Big)^2.
```

**语义**：$\beta_k^s$ = 在 $x_k^s$ 上、用**已训好**的 $\tilde J_{k+1}(\cdot,r_{k+1})$ 做一步 min + 期望后的「真值」标签；$r_k$ 使 $\tilde J_k$ 在样本上逼近该标签——**值空间**上的 Bellman 拟合。

**边界 $k=N-1$**：无 $\tilde J_N$，右端用终端代价 $g_N$：

```math
\beta_{N-1}^s=\min_{u\in U_{N-1}(x_{N-1}^s)}\mathbb{E}\Big\{g_{N-1}(x_{N-1}^s,u,w_{N-1})+g_N\big(f_{N-1}(x_{N-1}^s,u,w_{N-1})\big)\Big\}.
```

对应 LS：

```math
r_{N-1}\in\arg\min_r\sum_{s=1}^q\Big(\tilde J_{N-1}(x_{N-1}^s,r)-\beta_{N-1}^s\Big)^2.
```

**输出**：$r_0,\ldots,r_{N-1}$ → 在线用 $\tilde J_k(x_k,r_k)$ 作前瞻余值（§2.1）或 $\arg\min_u[\cdots]$ 得控。

---

#### 线性架构闭式解

$\tilde J_k(x_k,r_k)=r_k'\phi_k(x_k)$ 时，(3.22) 为线性 LS，同 (3.3)：

```math
r_k=\Big(\sum_{s=1}^q \phi_k(x_k^s)\phi_k(x_k^s)'\Big)^{-1}\sum_{s=1}^q \phi_k(x_k^s)\,\beta_k^s.
```

（设计矩阵满秩；否则加正则化。）

**非线性**（NN 等）：对 (3.22) 用 §3.1.3 增量梯度 / backprop；每阶段 $k$ 独立训 $r_k$（或 $(v_k,r_k)$）。

---

#### 与精确 DP / 第 2 章的关系

| 精确 DP | FVI |
|---------|-----|
| $J_k^{*}=\min_u\mathbb{E}[g_k+J_{k+1}^{*}]$ | $\tilde J_k\approx\min_u\mathbb{E}[g_k+\tilde J_{k+1}]$ 在样本上 |
| 每步**精确**备份 | 备份目标 $\beta_k^s$ + **回归** $\tilde J_k$ |
| 需遍历全体 $x_k$ | 仅样本 $x_k^s$ + 参数化 |

FVI **不保证**最优；误差沿 $k=N-1\to 0$ **累积**（Ch.5 §5.2 界与病理，例 5.2.1）。

---

#### 样本态 $x_k^s$ 的选取（关键实现问题）

- 通常 **Monte Carlo 仿真**生成；  
- 须 **representative**：样本中各态出现频率 $\approx$ **近最优策略**（理想为最优策略）下的占用概率；  
- 罕见态权重过低 → $\tilde J_k$ 在该区外推差；过多 off-policy 态 → 标签 $\beta_k^s$ 与部署分布不匹配。

Ch.4 在无限时域下用**平稳分布 / 占用测度**严格化「代表性」（§4.x 策略评估采样）。

**探索**：近似 PI 中常需刻意采样未充分访问的态（Ch.4–5）。

---

#### 长时域与平稳问题

$N$ 很大时：反向 $N$ 阶段 FVI 代价高、误差沿 horizon **复合**。

若问题**平稳**（$f_k,g_k,X_k,U_k$ 不随 $k$ 变）：

- 可共用同一 $(\phi,r)$，视为**无限时域** FVI（Ch.4–5）；  
- 单次备份 + 重复，而非 $N$ 组不同 $r_k$。

Ch.4 例 4.4.1、Ch.5 例 5.2.1：长时域/不当加权下的 FVI **病理**。

---

#### 与 Q-FVI 的选择（预告 §3.4）

| 路径 | 拟合对象 | 在线 min |
|------|----------|----------|
| **FVI (3.22)** | $\tilde J_k(x_k)$ | 每步需 $\min_u\mathbb{E}[g+\tilde J_{k+1}]$ |
| **拟合 Q (3.28)** | $\tilde Q_k(x_k,u)$ | 仅 $\min_u \tilde Q_k(x_k,u)$，**无期望** |

---

## §3.4 Q-Factor Parametric Approximation

**值空间另一路径**：直接逼近 **Q 因子** $Q_k^{*}(x_k,u_k)$，**不**先逼近 $J_k^{*}$ 再 min——与 §2.1 中 J 路径 vs Q 路径一致。

---

#### 定义与 DP 重写 (3.23)–(3.25)

**Q 因子**（在 $x_k$ 取 $u_k$，之后按最优策略）：

```math
Q_k^{*}(x_k,u_k)=\mathbb{E}\big\{g_k(x_k,u_k,w_k)+J_{k+1}^{*}(f_k(x_k,u_k,w_k))\big\},\quad k=0,\ldots,N-1. \qquad \text{(3.23)}
```

**$J$ 由 Q 恢复**（§1.2）：

```math
J_k^{*}(x_k)=\min_{u_k\in U_k(x_k)} Q_k^{*}(x_k,u_k). \qquad \text{(3.24)}
```

**Q-Bellman**（Q 空间备份，无需显式 $J_k$）：

```math
Q_k^{*}(x_k,u_k)=\mathbb{E}\Big\{g_k(x_k,u_k,w_k)+\min_{u'\in U_{k+1}(f_k(\cdot))} Q_{k+1}^{*}(f_k(x_k,u_k,w_k),u')\Big\}. \qquad \text{(3.25)}
```

子优控制：用 $\tilde Q_k$ 替 $Q_k^{*}$，以 (3.25) 为 FVI 目标——**直接在 Q 空间**做 Bellman 拟合。

**其他构造 $\tilde Q$ 的手段**：与 Ch.2 相同族——问题分解、确定性等价、CEG 等（§2.3）。

---

#### 参数化架构 (3.26)–(3.27)

| 架构 | 公式 | 适用 |
|------|------|------|
| **(3.26) 状态特征** | $\tilde Q_k(x_k,u_k,r_k)=r_k(u_k)'\phi_k(x_k)$ | $\|U_k\|$ **小**——每控单独权向量 |
| **(3.27) 状态–控特征** | $\tilde Q_k(x_k,u_k,r_k)=r_k'\phi_k(x_k,u_k)$ | 一般；下文默认 (3.27)，(3.26) 可平行推广 |

可为 NN（§3.2）或线性特征；$\phi_k$ 可仅依赖 $x_k$ 或同时依赖 $(x_k,u_k)$。

---

#### 拟合 Q 迭代 / Fitted Q-Iteration (3.28)–(3.29)

与 FVI 同构：**$k=N-1\to 0$** 反向。每阶段收集样本 **$(x_k^s,u_k^s)$**，$s=1,\ldots,q$。

**Bellman 目标**（(3.25) 右端，$\tilde Q_{k+1}(\cdot,r_{k+1})$ 已知）：

```math
\beta_k^s=\mathbb{E}\Big\{g_k(x_k^s,u_k^s,w_k)+\min_{u\in U_{k+1}(f_k(x_k^s,u_k^s,w_k))} r_{k+1}'\phi_{k+1}\big(f_k(x_k^s,u_k^s,w_k),u\big)\Big\}. \qquad \text{(3.29)}
```

**LS**（线性 $\tilde Q_k=r_k'\phi_k(x_k,u_k)$，(3.28)）：

```math
r_k\in\arg\min_r\sum_{s=1}^q\Big(r'\phi_k(x_k^s,u_k^s)-\beta_k^s\Big)^2.
```

**闭式解**（同 (3.3)）：

```math
r_k=\Big(\sum_{s=1}^q \phi_k(x_k^s,u_k^s)\phi_k(x_k^s,u_k^s)'\Big)^{-1}\sum_{s=1}^q \phi_k(x_k^s,u_k^s)\,\beta_k^s.
```

**语义**：在固定 $(x_k^s,u_k^s)$ 上拟合 Q-Bellman；$u_k^s$ 可来自当前策略、均匀探索或混合——影响覆盖（同 FVI 采样问题）。

**非线性** $\tilde Q$：对 (3.28) 用增量法；Deep Q-Network 等为同族（离线与在线变体见 Ch.4–5）。

---

#### 在线控 (3.30)——Q 路径的核心优势

```math
\tilde\mu_k(x_k)\in\arg\min_{u\in U_k(x_k)} \tilde Q_k(x_k,u,r_k). \qquad \text{(3.30)}
```

**执行时不算期望**——$\mathbb{E}_w[\cdot]$ 已在训练阶段进入 $\beta_k^s$。在线仅需 **$\arg\min_u$** 一次（离散控穷举或连续控 NLP）。

**对比 FVI 在线**：每步需 $\min_u\mathbb{E}[g_k+\tilde J_{k+1}(f_k)]$，实时算期望或 MC 平均 → Q 架构在**在线计算预算紧**时更有吸引力（§2.1.4 model-free 动机一致）。

---

#### $\beta_k^s$ 的 MC 实现与模拟器

(3.29) 含期望 → 可用 **少量样本**（甚至单样本 $w_k$）估计：

```math
\hat\beta_k^s=g_k(x_k^s,u_k^s,w_k)+\min_{u\in U_{k+1}(f_k(x_k^s,u_k^s,w_k))} r_{k+1}'\phi_{k+1}\big(f_k(x_k^s,u_k^s,w_k),u\big).
```

**Model-free**：只需模拟器对 $(x_k,u_k)$ 输出 $(g_k,w_k)$ 与 $x_{k+1}=f_k(\cdot)$——与 §2.1.4 一致；不必显式知 $p(w_k|\cdot)$。

内层 $\min_u$ 在 $|U_{k+1}|$ 小时可穷举；大时用 $\tilde Q_{k+1}$ 作 rollout 终端等（Ch.2）。

---

#### 策略再近似（值空间 + 策略空间，§2.1.5）

得 $\tilde\pi=\{\tilde\mu_0,\ldots,\tilde\mu_{N-1}\}$ 后，可再回归 **参数化策略** $\mu(x,r)$：

- 离线：对 FVI/Q-FVI 产生的 $(x_k,\tilde\mu_k(x_k))$ 做 LS / 分类；  
- 在线：**直接** $\mu(x,r)$，避免 (3.30) 的 $\arg\min$。

两层近似：Q/J → 控 → 策略网（Actor–Critic 中 actor 同族，Ch.4–5）。

---

#### Advantage Updating [Bai93/94]

**动机**：$Q_k^{*}(x_k,u_k)$ 常含与 $u_k$ **无关的大常数**（如 $\min_u Q_k^{*}$ 的偏移），干扰 LS / FVI (3.28)–(3.29) 的数值与泛化——与 §2.4.2 比较 **Q 差**、§3.1.3 学 **advantage** 同族。

**定义**：

```math
A_k(x_k,u_k)=Q_k^{*}(x_k,u_k)-\min_{u\in U_k(x_k)} Q_k^{*}(x_k,u).
```

- 无近似时：比 $A_k$ 与比 $Q_k^{*}$ **等价**选控；  
- 有近似时：$A_k$ **值域更小**，回归更稳；可拟合 $\tilde A_k$ 再 $\arg\min_u \tilde A_k$（或 $\tilde Q_k=\tilde A_k+\text{const}$）。

**Q-Bellman 的 advantage 形式**：$A_k^{*}=0$ 于最优控；备份可在 $A$ 空间写（[BeT96] §6.6.2 有例）。Ch.4–5 无限时域扩展。

---

#### FVI vs 拟合 Q 小结

| | **FVI (3.22)** | **拟合 Q (3.28)** |
|--|----------------|-------------------|
| 拟合 | $\tilde J_k(x_k)$ | $\tilde Q_k(x_k,u_k)$ |
| 样本 | $x_k^s$ | $(x_k^s,u_k^s)$ |
| 标签 | $\min_u\mathbb{E}[g+\tilde J_{k+1}]$ | $\mathbb{E}[g+\min_u \tilde Q_{k+1}]$ |
| 在线 | 需期望或 MC | 仅 $\min_u \tilde Q_k$ |
| 后续 | Ch.5 界 (5.19) | Q-learning、SARSA、DQN |

---

## §3.5 Notes and Sources

- 架构 vs 训练两条研究线；[BeT96]、[BCN18]、[Ber16a] 增量法理论。  
- FVI 长时域/加权病理 → Ch.4 例 4.4.1、Ch.5 例 5.2.1。  
- Advantage → [Bai93/94]、[BeT96] §6.6.2。

---

## 与后续章节接口

| 本章 | 后续 |
|------|------|
| (3.22) FVI | Ch.5 §5.2 + 界 (5.19) |
| (3.6) 增量 | Actor–Critic critic |
| (3.30) $\tilde Q$ | Q-learning、SARSA、DQN |

---

*个人学习笔记；原著 Copyright Bertsekas / Athena Scientific。*
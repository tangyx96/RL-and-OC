# Zero-Order Optimization Techniques for Robotics — 分节读书笔记

> **文献**：Armand Jordana, Jianghan Zhang, Joseph Amigo, Ludovic Righetti, *An Introduction to Zero-Order Optimization Techniques for Robotics*（[arXiv:2506.22087](https://arxiv.org/abs/2506.22087) v2, Oct 2025）。  
> **定位**：以随机搜索统一理解机器人中的零阶优化。轨迹优化与策略优化都写成 $\min_x f(x)$。  
> **式号**：本笔记的编号。自式 (7) 起与论文不完全相同。文献号与论文参考文献一致。

---

## 全文结构与阅读地图

| 节 | 主题 | 核心算法 / 概念 |
|----|------|-----------------|
| **§I** | 动机与统一问题 (1) | TO vs RL、在线/离线、MPPI / CMA / Reinforce 谱系 |
| **§II** | 随机搜索基础 | 纯随机 / 贪心局部搜索；有限差分、RCD、SPSA；**Gaussian smoothing (RS)**；模拟退火 |
| **§III** | 轨迹优化 | Predictive Sampling → **MPPI (LSE)** → **CMA / MPPI-CMA**；块对角协方差 |
| **§IV** | 策略优化 | **DPG**；RS/LSE Actor-Critic；与 **Reinforce** 公式对照 |
| **§V** | 种群算法 | Random Restart、$(N+\lambda)$-ES、SVGD |
| **§VI** | 结论 | 约束零阶优化、全局解 |

**统一问题**（式 (1)，贯穿全文）：

```math
\min_{x \in \mathbb{R}^n} f(x).
```

- **TO**：$x = (u_0,\ldots,u_{T-1})$，$n = T n_u$（single shooting，动力学约束隐式消去状态）。  
- **RL**：$x = \theta$（策略参数）；目标 $F(\theta) = \mathbb{E}_s[J(\theta,s)]$，即标准折扣 MDP 上的期望回报。

**阅读顺序**：§II 给出 RS 与 LSE；§III、§IV 分别用于轨迹优化与策略优化；§V 讨论种群。并行实现不是论文的一节，见文末附记。

---

## §I Introduction（引言）

### 主要内容

#### 为何零阶方法在机器人里流行

| 动机 | 说明 |
|------|------|
| **非光滑目标** | 接触、摩擦、碰撞使 $f$ 不可微或梯度无意义；一阶方法需额外平滑/近似 |
| **实现成本** | 对复杂仿真器手写高效 $\nabla f$（含 contact implicit）成本高昂 |
| **局部极小** | 确定性梯度下降/牛顿法易陷局部极小；零阶方法多为**随机**的，有逃逸可能 |
| **并行硬件** | GPU 上每次迭代并行执行 $K$ 次 rollout，使高维采样下的 TO/MPC 变得可行 [2,3] |

#### 历史与本文谱系

- **ES**：70 年代演化策略 [7] → **CMA-ES** [8]。  
- **TO**：Predictive Sampling [9]；**MPPI** [10]（最初从信息论/路径积分推导）。  
- **RL**：**Reinforce** [11] 等策略梯度。  
- **本文统一工具**：[12] 的 **Gaussian smoothing (RS)**；[15] 的 **log-sum-exp (LSE) 平滑**——把 MPPI、CMA、策略梯度都看成 Algorithm 3 的 $g$ 估计变体，并导出 **RS/LSE Actor-Critic**。

#### 与一阶方法的对比（概念层）

| | 零阶（本文） | 一阶梯度 |
|---|-------------|----------|
| 每步信息 | $f(x)$ 值（可批量） | $\nabla f(x)$ |
| 非光滑 | 自然适用 | 需平滑/次梯度 |
| 维数 $n$ 大 | 靠随机方向/SPSA/RS，$O(1)$ 次评估/方向 | 通常需 $O(n)$ 或自动微分 |
| 机器人瓶颈 | **仿真 rollout 次数**，而非矩阵求导运算 |

### TO vs RL（问题空间与部署）

| 维度 | 轨迹优化 TO | 策略优化 RL |
|------|-------------|-------------|
| 优化变量 | 有限维控制序列 $u_{0:T-1}$ | 策略 $\pi_\theta$ → 有限参数 $\theta$（如 NN 权重） |
| 目标 | $\sum_{t=0}^{T-1} c_t(x_t,u_t)+c_T(x_T)$（式 (17)） | $F(\theta)=\mathbb{E}_s\bigl[\sum_{k=0}^{\infty}\gamma^k r(s_k,a_k)\bigr]$（式 (31)–(32)） |
| 时域 | 有限 $T$，MPC 下滚动 | 无限 + 折扣 $\gamma$ |
| 部署 | **在线** MPC 循环，或离线生成轨迹数据 | 多为**离线**训 $\pi_\theta$，在线只前向 $a=\pi_\theta(s)$ |
| 共同性 | 都是 $\min_x f(x)$（式 (1)）；多为局部随机搜索，每步批量评估 $f$ |

**在线 vs 离线（文中强调）**：TO 可在运行时每步解式 (17)；RL 近期 SOTA [4,5] 多在仿真里离线搜 $\theta$，运行时计算量小。二者算法形态仍相似（采样 → 加权/选择 → 更新）。

### 与已有工作的关系

- **TO 侧**：[13] 将 MPPI 与 CMA-ES 视为同一类加权样本更新；[14] 将 MPPI 解释为近似梯度步；[3] 将 MPPI 与扩散退火相联系。本文用 **LSE 平滑**解释 MPPI 的指数权重，并讨论其与 risk-seeking 控制的关系。  
- **RL 侧**：[16, 17] 用随机平滑处理接触与非光滑动力学下的轨迹优化。本文沿相反方向，由同一框架导出 actor 更新。参数空间的进化策略见 [35]。

### 要点

- §II 的 **Algorithm 3**（近似梯度下降）是 MPPI 加权平均、CMA 协方差更新、RS Actor-Critic 的母版。  
- 「局部搜索」指采样分布依赖当前 $x$（算法 2）。其收敛条件比算法 1 更严，仍可收敛到全局极小 [18]。维数灾难仍然存在。

### 注意点

- 默认**确定性动力学**（足式/操作常见）；随机动力学扩展留作 future work。  
- RL 首先介绍**确定性策略 + DPG**，再通过 Reinforce 解释随机策略 = 动作空间平滑 + 额外探索噪声。

---

## §II Random Search（随机搜索）

§II 建立后文 TO/RL 算法的共同语言：样本在 TO 里是**控制序列/轨迹**，在 RL 里是**策略参数**或（经 Q）**动作扰动**。

---

### §II.A Simple random search（简单随机搜索）

虽很少单独用于高维机器人问题，但收敛性质澄清了后文「局部 vs 全局算法」术语。

#### §II.A.1 Pure (Global) Random Search — Algorithm 1

**流程**：初始化 $x \leftarrow x_0$；循环：采样 $\tilde x$（分布**不依赖**历史），若 $f(\tilde x)<f(x)$ 则 $x\leftarrow\tilde x$。

| 属性 | 内容 |
|------|------|
| 「Global」含义 | 采样分布忽略过去迭代，非指一定找到全局最优 |
| 理论 | 温和条件下全局收敛 [18] |
| 实践 | **维数灾难**——所需 $f$ 评估次数随 $n$ 指数增长，实际中很少使用 |

#### §II.A.2 Greedy Local Search — Algorithm 2

**流程**：$d \sim \mathcal{N}(0,\Sigma)$；仅当 $f(x+d)<f(x)$ 时 $x\leftarrow x+d$（严格下降才接受）。

| 属性 | 内容 |
|------|------|
| 「Local」 | 采样中心随当前 $x$ 移动。收敛条件严于算法 1，仍可证全局收敛 [18] |
| 与 ES | 即 $(1+1)$-ES：保留一个当前点，每步一个子代，更优才替换。一次采 $K$ 个子代再留最优，即 Predictive Sampling，也就是 $(1+\lambda)$-ES（$\lambda=K$） |

---

### §II.B Random search via gradient approximation（梯度近似）

#### 母算法 Algorithm 3

```math
x \leftarrow x - \alpha g, \quad g \approx \nabla f(x)\ \text{且仅用}\ f\ \text{的取值构造}.
```

$\alpha$ 为步长。MPPI 与 CMA 的实现写成加权平均或分布参数更新；推导上仍是 $x\leftarrow x-\alpha g$，其中 $g$ 取自然梯度。RS 与 LSE 直接沿这一形式推导。

#### 1) 完整有限差分 — 式 (2)

```math
g = \sum_{j=1}^n \frac{f(x+\mu e_j) - f(x)}{\mu} e_j.
```

- $\mu>0$，$e_j$ 为标准基。前向差分共用一次 $f(x)$，每步 $n+1$ 次评估。中心差分取 $\bigl(f(x+\mu e_j)-f(x-\mu e_j)\bigr)/(2\mu)$，每方向两侧各一次，共 $2n$ 次。  
- 长视界 TO 中 $n=T n_u$ 很大，二者都不适用。

#### 2) Random Coordinate Descent (RCD) — 式 (3)

```math
g = \frac{f(x+\mu e_j) - f(x)}{\mu} e_j, \quad j \sim \mathrm{Uniform}\{1,\ldots,n\}.
```

- 每步新增一次 $f$ 评估（$f(x)$ 可缓存）。  
- 凸情形下，若使用精确坐标导数而不是有限差分，期望迭代次数不超过梯度下降的 $n$ 倍；对一类函数，迭代数可与梯度下降同级，函数评估少 $n$ 倍 [20]。式 (3) 含有限差分，不在该结论内。  
- 方向不必沿坐标轴，可取任意随机方向，由此连接到 SPSA 与 RS。

#### 3) SPSA — 式 (4)

```math
g = \frac{f(x+\mu\Delta) - f(x-\mu\Delta)}{2\mu} \Delta, \quad \Delta_i \in \{+1,-1\}\ \text{i.i.d.}
```

- 两次评估得到 $n$ 维随机梯度；Spall [18] 采用中心差分。  
- 在合理假设下，性能可接近完整有限差分，而样本数少 $n$ 倍。  
- 全局收敛 [25]：不另加高斯噪声时，估计噪声 $\xi_k$ 已足以充当 Langevin 扰动，离散格式 (16) 渐近收敛到全局极小。

#### 4) Gaussian Smoothing（RS）— 核心

**平滑目标**（式 5）：

```math
f_\mu(x) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,\Sigma)}[f(x + \mu\epsilon)].
```

- $\mu\to 0$ 时 $f_\mu\to f$。$\mu$ 较小时，$\nabla f_\mu$ 接近 $\nabla f$ [12]。$\nabla f_\mu$ 可由 $f$ 的函数值平均得到。

i) 记 $\kappa=\sqrt{(2\pi)^n\det\Sigma}$，$\epsilon\sim\mathcal{N}(0,\Sigma)$。换元 $z=x+\mu\epsilon$ 得式 (6)：

```math
f_\mu(x)=\frac{1}{\kappa\mu^n}\int f(z)\exp\Bigl(-\frac{1}{2\mu^2}(z-x)^\top\Sigma^{-1}(z-x)\Bigr)\,dz.
```

对 $x$ 求导只作用在指数上，得到因子 $\Sigma^{-1}(z-x)/\mu^2$。与分母中的 $\mu^n$ 合在一起，即式 (7)：

```math
\nabla f_\mu(x)=\frac{1}{\kappa\mu^{n+2}}\int f(z)\exp\Bigl(-\frac{1}{2\mu^2}(z-x)^\top\Sigma^{-1}(z-x)\Bigr)\Sigma^{-1}(z-x)\,dz.
```

再令 $\epsilon=(z-x)/\mu$，回到式 (8)：

```math
\nabla f_\mu(x)=\mathbb{E}\left[\frac{1}{\mu}f(x+\mu\epsilon)\,\Sigma^{-1}\epsilon\right].
```

ii) 记 $p_m$ 为 $\mathcal{N}(m,\mu^2\Sigma)$ 的密度。对均值求导，

```math
\nabla_m\log p_m(z)=(\mu^2\Sigma)^{-1}(z-m).
```

取 $m=x$，$\epsilon=(z-x)/\mu$，得 $\nabla_m\log p_{m=x}(z)=\Sigma^{-1}\epsilon/\mu$。于是式 (8) 写成

```math
\nabla f_\mu(x)=\mathbb{E}\bigl[f(z)\,\nabla_m\log p_{m=x}(z)\bigr].
```

这是式 (9)，即 log-likelihood trick。将得分代回即得式 (8)，两种写法相等。论文将该高斯的协方差写成 $\Sigma$，对应于 $\mu=1$。这一形式与 Reinforce 同型，后文的 CMA 也由此出发。

iii) $\mathbb{E}[\Sigma^{-1}\epsilon]=0$，减去 $f(x)$ 不改变期望。未减的估计仍然无偏，但幅度可以任意大；减去之后对 $f$ 的平移不变，方差更小（式 10）：

```math
\nabla f_\mu(x)=\mathbb{E}\left[\frac{f(x+\mu\epsilon)-f(x)}{\mu}\Sigma^{-1}\epsilon\right].
```

| 要点 | 说明 |
|------|------|
| Baseline $f(x)$ | 减去 $f(x)$ 后估计平移不变，方差更小 |
| 实用 $g$（式 11–12） | 前向：$(f(x+\mu\epsilon)-f(x))\Sigma^{-1}\epsilon/\mu$；中心差分用 $(f(x+\mu\epsilon)-f(x-\mu\epsilon))$ |
| Remark 2 | 方向也可取自单位球面 [21]，从而 $\|\epsilon\|$ 有界 |
| 凸情形 [12] | RS 前向/中心估计的迭代复杂度 ≤ 标准梯度法的 $n$ 倍 |

#### §II.B 小结

| 方法 | 每步 $f$ 评估（量级） | 机器人相关性 |
|------|----------------------|--------------|
| 有限差分 | $O(n)$ | 维数太高 |
| RCD | $O(1)$ | 稀疏方向探索 |
| SPSA | $O(1)$ | 2 次评估，有全局收敛理论 |
| RS | $O(1)$ 方向 + 可批量 | **MPPI / CMA / RL actor 的理论枢纽** |

[12,22,24] 假设不同，但思想一致：**沿随机方向采样，用 $O(1)$ 次评估逼近梯度下降的行为**。

---

### §II.C Simulated Annealing（模拟退火）

#### 动机与 Langevin 框架

- RS/SPSA 在**凸**情形样本效率高；**非凸**时纯梯度下降陷局部极小。  
- **Langevin SDE**（式 13）：$dX_t = \nabla_x \log p(X_t)\,dt + dW_t$，平稳分布为 $p$ [22]。标准过阻尼朗之万的扩散系数为 $\sqrt{2}$；式 (13) 将其吸收进 $W_t$ 的标度。离散式 (16) 再用 $\alpha_k$、$\gamma_k$ 分别调节漂移与噪声。  
取 Boltzmann 型（式 14–15）：

```math
p(x) \propto \exp(-f(x)/\lambda), \quad \nabla_x \log p(x) = -\lambda^{-1}\nabla f(x).
```

$\lambda\to 0$ 时 $p$ 集中在 $f$ 的全局极小集上 [23]。

#### 离散算法（式 16）

```math
x_{k+1} = x_k - \alpha_k g_k + \gamma_k \epsilon_k, \quad g_k = \nabla f(x_k) + \xi_k.
```

- $\epsilon_k$：显式高斯噪声；$\xi_k$：梯度估计噪声（如 SPSA）。  
- 适当调度 $\alpha_k,\gamma_k$，使有效温度 $\lambda$ 缓慢趋于 0，即模拟退火 [24]。  
- 以 SPSA 估计代入式 (16) 可证全局收敛 [25]。随机平滑的相应结果，论文仅作猜想。

#### 实践与 Remark 3

- 上述全局收敛均为渐近结果，有限步的温度调度并不由此确定。  
- **Remark 3**：Alg.2 在 $f$ 下降时接受、上升时拒绝。若上升步以概率 $e^{-(f(x')-f(x))}$ 接受，则得到 Metropolis–Hastings [26]（附录 A）。  
- **与 §IV Remark 9**：$F(\theta)=\mathbb{E}_s[J(\theta,s)]$ 须经采样估计，梯度因而带噪声。按 §II.C，该噪声有助于离开较差的局部极小。

---

## §III Trajectory Optimization（轨迹优化）

### 问题形式 — 式 (17)

```math
\min_{u_0,\ldots,u_{T-1}} \sum_{t=0}^{T-1} c_t(x_t, u_t) + c_T(x_T), \quad x_{t+1} = f_{\mathrm{dyn}}(x_t, u_t),\ x_0 = x.
```

| 要素 | 说明 |
|------|------|
| **Single shooting** | 只优化 $u$；状态由动力学递推，约束隐式消去 |
| 记号 | 式 (17) 里的 $x$ 是状态。优化的是控制序列 $u_{0:T-1}$，拼成一条向量后就是式 (1) 的 $x$ |
| 控制约束 | 控制上下界写进 $c_t$ 的罚项。Hydrax 实验也这样处理 |
| 维数 | $n = T n_u$ |
| 黑盒 | $f(x)$ = 沿控制序列 rollout 的总代价，不要求 $\nabla_u$ |

**MPC 用法**：每控制周期解式 (17)，只执行 $u_0$，再滚动——零阶 MPC 即每步用 MPPI/CMA 等更新整条 $u$ 序列 [2,3]。

---

### §III.A Predictive Sampling — Algorithm 4

**相对 Alg.2 的改动**：每迭代采 **$K$ 个**独立高斯方向 $d_k$，构造集合 $D=\{x\}\cup\{x+d_k\}$，令 $x\leftarrow\arg\min_{\tilde x\in D} f(\tilde x)$；若无改进则保持 $x$。

| 要点 | 说明 |
|------|------|
| 并行 | $K$ 次 rollout **完全独立**，适合 GPU 批量仿真 [9] |
| 与 MPPI | 只留下代价最小的一条；MPPI 则对所有样本做指数加权平均 |
| 与 ES | 一个当前点，一次采 $K$ 个扰动并只留最优，即 §II 的 $(1+\lambda)$-ES |
| 局限 | 简单但在 Hydrax 实验中仍具竞争力（Fig.2） |

---

### §III.B Log-sum-exp transform & MPPI — Algorithm 5

#### 原始 MPPI 更新（式 18–19）

采样 $x_k \sim \mathcal{N}(x,\Sigma)$，指数权重：

```math
w_k = \frac{\exp(-\frac{1}{\lambda}(f(x_k)-\rho))}{\sum_{j=1}^K \exp(-\frac{1}{\lambda}(f(x_j)-\rho))}, \quad \rho = \min_j f(x_j),
```

```math
x \leftarrow \sum_{k=1}^K w_k x_k.
```

- $\rho$ 仅防数值溢出；权重对 $f$ 的**加法常数**不变。  
- 权重称为 **Exponential Average weights**；更新是**软选择**，而非 Alg.2 的 hard accept/reject。

#### LSE 平滑目标（式 20–21）

引入 log-sum-exp，是为了把 MPPI 的更新看成高斯平滑的一种变体。代理由式 (5) 换成

```math
f_{\mu,\lambda}(x) = -\lambda \log \mathbb{E}\left[\exp\left(-\frac{1}{\lambda} f(x+\mu\epsilon)\right)\right].
```

- $\mu\to 0$：恢复原 $f$；$\lambda\to\infty$：$f_{\mu,\lambda}\to f_\mu$（RS）[15]。  
- LSE 可看作 RS 与 **Moreau envelope** 之间的插值：Moreau 近似更好但更贵；LSE 用样本换近似质量。

对 $x$ 求导同样只作用在高斯密度上。记 $w(\epsilon)=\exp(-f(x+\mu\epsilon)/\lambda)$，

```math
\nabla f_{\mu,\lambda}(x)=-\frac{\lambda}{\mu}\frac{\mathbb{E}\bigl[w(\epsilon)\,\Sigma^{-1}\epsilon\bigr]}{\mathbb{E}[w(\epsilon)]}.
```

分母中的期望使比值的样本估计需要较大的 $K$。并行仿真使这样的样本量可以接受。$K$ 很大时，加权平均几乎不随样本波动，更新接近确定性梯度步，容易停在局部极小。§II.C 依赖 SPSA 的估计噪声或显式朗之万噪声离开局部极小。

#### Natural gradient 视角（式 23–24）

取 $\mu=1$。均值参数的 Fisher 信息为 $F=\Sigma^{-1}$ [27]，自然梯度不依赖参数化（附录 B）。步长 $\alpha=1/\lambda$ 时，$x\leftarrow x-\alpha F^{-1}\nabla f_{\mu,\lambda}$ 消去 $\Sigma^{-1}$ 与 $\lambda$。记 $\bar w_k=w_k/\sum_j w_j$，则

```math
x\leftarrow x+\frac{\sum_k w_k\epsilon_k}{\sum_j w_j}=\sum_k \bar w_k(x+\epsilon_k).
```

第二个等号用到 $\sum_k\bar w_k=1$。若 $w_k$ 已是式 (19) 的归一化权重，分母为 1，右端即 $\sum_k w_k(x+\epsilon_k)$，与式 (18) 相同。位移 $\sum_k\bar w_k\epsilon_k$ 等于 $-\Sigma\nabla f_{\mu,\lambda}/\lambda$。$\Sigma=\sigma^2 I$ 时，$\sigma^2/\lambda\le 1/L$，其中 $L$ 是 $f_{\mu,\lambda}$ 的梯度 Lipschitz 常数，对应于 $L$-光滑凸问题中的步长 $1/L$ [15, 28]。

#### Fig.1（RS 与 LSE）

一维双阱：全局极小窄而深，局部极小宽而浅。  
- **RS**：两阱的深度与宽度变得接近。  
- **LSE**：全局窄阱在平滑后更宽，局部宽阱被压低。

#### Remark 4–5

- **Remark 4**：LSE 与 risk-seeking 控制 [29] 相关（附录 C）。  
- **Remark 5**：$\lambda\to 0$ 时，$f_{\mu,\lambda}$ 集中于全局极小附近，与模拟退火的 $p\propto\exp(-f/\lambda)$ 同族。MPPI 下降 $f_{\mu,\lambda}$。另一条路先平滑 $h(x)=\exp(-f(x)/\lambda)$：

```math
h_\mu(x)=\mathbb{E}\bigl[\exp(-f(x+\mu\epsilon)/\lambda)\bigr].
```

$\exp(-f_{\mu,\lambda}/\lambda)=h_\mu$。因此温度为 $\lambda$、势能为 $f_{\mu,\lambda}$ 的朗之万动力学，平稳分布正比于 $h_\mu$，漂移为 $\nabla\log h_\mu$。温度取 1 时，平稳分布正比于 $h_\mu^\lambda$。取 $\mu=1$，确定性漂移为

```math
x\leftarrow x+\Sigma\nabla\log h_\mu(x)=x+\frac{\mathbb{E}[w\epsilon]}{\mathbb{E}[w]}.
```

略去扩散项后，该步与式 (18) 相同。

#### 与经典 MPC 的区别

| | 经典 MPC | MPPI |
|---|----------|------|
| 优化对象 | 常逐段 Bellman / 二次规划 | 整条 $u$ 序列作 $x$ |
| 更新 | 梯度/QP 或终端 $\tilde J$ | 指数加权粒子平均 |
| 并行 | 依实现 | 天然 $K$ 路 rollout |
| 共同点 | **Receding horizon**，每步只执行首控 |

---

### §III.C Covariance Matrix Adaptation (CMA) — Algorithm 6

#### 固定 $\Sigma$ 的问题

MPPI/RS 常取固定 $\Sigma$；实际中搜索尺度与病态程度随迭代变，需**自适应**搜索几何。

#### 分布空间视角（式 25–27）

```math
\min_{\theta=(x,\Sigma)} J(\theta), \quad J(\theta) = \mathbb{E}_{z \sim \mathcal{N}(x,\Sigma)}[f(z)].
```

- 优化 RS 的均值 $x$ = 在 Gaussian **信念** $N(x,\Sigma)$ 下最小化期望代价；$\Sigma$ 描述「不确定性椭圆」。  
- $J$ 最小时 $\mathcal{N}(x,\Sigma)$ 集中在 $f$ 的极小附近 [30]。

**Log-likelihood**（式 28）。推导同式 (9)：参数导数只作用于密度，

```math
\nabla J(\theta)=\mathbb{E}[f(z)\nabla_\theta\log p_\theta(z)].
```

$\theta=(x,\Sigma)$，对应于式 (9) 中 $\mu=1$。记 $v=z-x$，得分

```math
\nabla_x\log p_\theta(z)=\Sigma^{-1}v,\qquad
\nabla_\Sigma\log p_\theta(z)=\frac12\bigl(\Sigma^{-1}vv^\top\Sigma^{-1}-\Sigma^{-1}\bigr).
```

欧氏梯度为 $\nabla_x J=\mathbb{E}[f(z)\Sigma^{-1}v]$，$\nabla_\Sigma J=\frac12\mathbb{E}[f(z)(\Sigma^{-1}vv^\top\Sigma^{-1}-\Sigma^{-1})]$。均值的 Fisher 信息为 $\Sigma^{-1}$，自然梯度 $\Sigma\nabla_x J=\mathbb{E}[f(z)v]$。协方差的自然梯度为 $2\Sigma(\nabla_\Sigma J)\Sigma$；系数 $\tfrac12$ 与预因子 $2$ 相消后得 $\mathbb{E}[f(z)(vv^\top-\Sigma)]$。二者合为式 (29) [31]：

```math
\Delta x = \mathbb{E}[f(z)(z-x)], \quad \Delta \Sigma = \mathbb{E}[f(z)((z-x)(z-x)^\top - \Sigma)].
```

$\Delta$ 为 $J$ 的上升方向：最大化取 $+\Delta$，最小化取 $-\Delta$。目标为 $\min J$ 时，令 $w_k=f(x_k)$ 或 $w_k=f(x_k)-f(x)$，步长 $\alpha$ 吸收 $1/K$，

```math
\Sigma \leftarrow \Sigma-\alpha\sum_k w_k\bigl((x_k-x)(x_k-x)^\top-\Sigma\bigr),
```

```math
x \leftarrow x-\alpha\sum_k w_k(x_k-x).
```

$\Sigma$ 先于均值更新，中心为采样时的均值。沿 $+\Delta$ 展开得到式 (30)。论文 Algorithm 6 采用这一形式；$w_k=f(x_k)$ 时它上升 $J$：

```math
\begin{aligned}
\Sigma &\leftarrow \Bigl(1-\alpha\sum_k w_k\Bigr)\Sigma+\alpha\sum_k w_k(x_k-x)(x_k-x)^\top,\\
x &\leftarrow \Bigl(1-\alpha\sum_k w_k\Bigr)x+\alpha\sum_k w_k x_k.
\end{aligned}
```

若 $w_k\ge 0$、$\sum_k w_k=1$，且较优样本权重更大，则下降步为高权样本的凸组合

```math
x\leftarrow(1-\alpha)x+\alpha\sum_k w_k x_k,
```

$\alpha=1$ 时即为加权均值。指数权 $w\propto\exp(-f/\lambda)$ 对应目标 $f_{\mu,\lambda}=-\lambda\log\mathbb{E}[w]$。均值的自然梯度为 $-\lambda\mathbb{E}[w(z-x)]/\mathbb{E}[w]$；步长取 $1/\lambda$ 时，下降步为 $\mathbb{E}[w(z-x)]/\mathbb{E}[w]$，对 $f$ 平移不变。在 $\mathbb{E}[f]$ 的得分中以 $\exp(-f/\lambda)$ 替换 $f$ 并沿 $+\Delta$ 更新，得到 $\mathbb{E}[w(z-x)]$，目标为 $\mathbb{E}[\exp(-f/\lambda)]$，步长依赖于 $f$ 的加法常数。

#### Table I：权重变体 → 不同名算法

| 算法 | 权重 $w_k$ | 备注 |
|------|-----------|------|
| **CMA** | $f(x_k)$ 或 $f(x_k)-f(x)$ | 沿 $-\Delta$ 下降；减 $f(x)$ 后平移不变 |
| **CMA-ES** | 按代价排序，较优样本权更大 [8, 30] | 无 evolution path；向高权样本移动 |
| **MPPI-CMA** | 指数权 (19) | 权非负且和为 1 |
| **CEM** | 精英权：最好 $K_e$ 个为 $1/K_e$，其余 0 | Remark 6：先更新均值，协方差绕新均值（附录 D） |

#### 正定性与 xNES / MPPI-CMA

$w_k=f(x_k)$ 时，$\sum_k w_k=1$ 与 $w_k\ge 0$ 均无保证，沿 $-\Delta$ 的协方差更新也不是凸组合，$\Sigma$ 可能失去正定性。若写成 $(1-\eta)\Sigma+\eta\sum_k w_k(x_k-x)(x_k-x)^\top$，且 $0\le\eta<1$、$w_k\ge 0$、$\sum_k w_k=1$，则正定性保持 [31]。**xNES** [32] 在指数坐标中更新协方差，正定性自动保持；步长趋于零时与自然梯度一致。MPPI 的指数权非负且和为 1，**MPPI-CMA** 满足该条件。

#### 块对角 $\Sigma$ 与长视界 $T$

- 全矩阵 CMA：$O(n^2)$ 存储/更新，$n=Tn_u$ 时不可行。  
- 设 $\Sigma=\mathrm{blockdiag}(\Sigma_1,\ldots,\Sigma_T)$，每块 $n_u\times n_u$：对**每块**应用 natural gradient → 保持块对角，**$O(T)$** 复杂度；Predictive Sampling / MPPI 同理。  
- 对全矩阵 $\Sigma$ 套用式 (29) 不一定保持块对角。应对每个 $\Sigma_t$ 单独做自然梯度更新。论文称其为 block-diagonal CMA，形式证明留作后续工作。  
- **Remark 7**：块对角 MPPI-CMA 即不含时间平均的 **PI²-CMA** [13]。

---

### §III.D Numerical experiments

| 设置 | 内容 |
|------|------|
| 平台 | Hydrax [33] |
| 任务 | Cartpole, DoubleCartPole, PushT, Humanoid |
| 降维 | PushT、Humanoid 用 zero-order spline 降低控制维数 |
| 样本数 | 每迭代 **2048** |
| MPPI / MPPI-CMA | $\lambda=0.1$ |
| CMA | 实验用**全协方差**（未成为瓶颈）；块对角结果见附录 |
| MPPI-CMA 步长 | 均值与协方差**分开** lr，如 (1.0, 0.1) 明显优于共享步长 |

**Fig.2 趋势**（6 seeds 中位数）：  
- **MPPI-CMA** 稳定优于 MPPI。  
- **Predictive Sampling** 简单但强。  
- **RS**（Algorithm 3 + 式 11）**步长敏感**，不同任务需不同 lr。

---

## §IV Policy Optimization（策略优化）

### 为何从 TO 转到 RL

在线 MPC 每步解 TO，算力/时限压力大 [2,3]；RL 把搜索移到**离线**，运行时仅 $\pi_\theta(s)$ 前向，在复杂任务上更易达到 SOTA [4,5]——但优化结构仍可与 TO 侧零阶方法对照。

### 问题 — 式 (31)–(32)

```math
F(\theta) = \mathbb{E}_s[J(\theta,s)], \quad J(\theta,s) = \sum_{k=0}^\infty \gamma^k r(s_k, a_k),
```

约束：$s_0=s$，$a_k=\pi_\theta(s_k)$，$s_{k+1}=f_{\mathrm{dyn}}(s_k,a_k)$。**最大化回报**（RL 符号），与 TO 最小化代价相对。

| 对比 TO | 说明 |
|---------|------|
| 时域 | 无限 + $\gamma$ |
| 目标 | 对**初态分布**期望，非单次 rollout |
| 策略 | 先设确定性 $\pi_\theta$，即 DPG [34] |
| Remark 8 | 再对环境参数取期望，即 domain randomization |

### 两种采样哲学

1. **参数空间**：扰动 $\theta$ [35]。本文不采用这一路线。  
2. **动作空间**：DPG 将 $\nabla_\theta F$ 写成 $\partial_a Q$，再用 RS 估计 $\partial_a Q$。Actor-Critic 的其余部分不变，只更换 actor 的梯度估计。

---

### §IV.A Deterministic Policy Gradient

**Q 定义**（式 34）：$s_0=s$，$a_0=a$，其后 $a_k=\pi_\theta(s_k)$，$s_{k+1}=f_{\mathrm{dyn}}(s_k,a_k)$，

```math
Q(s,a)=\sum_{k=0}^{\infty}\gamma^k r(s_k,a_k).
```

**DPG 定理**（式 33, 35）。记 $\partial\pi_\theta(s)/\partial\theta\in\mathbb{R}^{n_\theta\times n_a}$，以下各式按此维数相乘：

```math
\frac{\partial J(\theta,s_0)}{\partial \theta} = \sum_{k=0}^\infty \gamma^k \frac{\partial \pi_\theta(s_k)}{\partial \theta} \frac{\partial Q(s_k,a_k)}{\partial a}\bigg|_{a=\pi_\theta(s_k)},
```

```math
\nabla_\theta F(\theta) = \mathbb{E}_s\left[\sum_{k=0}^\infty \gamma^k \frac{\partial \pi_\theta(s_k)}{\partial \theta} \frac{\partial Q(s_k,a_k)}{\partial a}\right].
```

- 梯度写成整条轨迹上的期望，不显式写出折扣状态分布 [36]。  
- $Q$ 无闭式。Actor-Critic 中，critic 由 Bellman 方程拟合 $Q$，actor 按式 (35) 更新 [11]。  
- **Remark 9**：$F(\theta)$ 须经采样估计，梯度带有噪声。按 §II.C，该噪声有助于离开较差的局部极小。

---

### §IV.B Gaussian smoothing on Q — Algorithm 7

**动机**：DDPG [38] 对 critic 网络自动微分得到 $\partial_a Q$；TD3 [39] 在此基础上改进。RS 用 $Q$ 的差分估计 $\partial_a Q$，不需要 $\nabla_a Q$ 的闭式。

**RS 梯度估计**（式 36）：

```math
\nabla_\theta F(\theta) \approx \mathbb{E}\left[\sum_{k=0}^\infty \gamma^k \frac{\partial \pi_\theta(s_k)}{\partial \theta} (Q(s_k, a_k+\epsilon_k) - Q(s_k, a_k)) \Sigma^{-1} \epsilon_k\right].
```

**Algorithm 7（单步示意）**：

1. $a \leftarrow \pi_\theta(s)$  
2. 采 $\epsilon \sim \mathcal{N}(0,\Sigma)$  
3. Actor：$\theta \leftarrow \theta + \alpha (Q(s,a+\epsilon)-Q(s,a)) \frac{\partial \pi_\theta(s)}{\partial \theta} \Sigma^{-1}\epsilon$  
4. Critic：按原 DDPG/TD3 逻辑更新 $Q$（正文省略 rollout 与 actor/critic 交替细节）

| 变体 | 说明 |
|------|------|
| **LSE Actor-Critic** | actor 更新用指数权（类比 MPPI） |
| **Remark 10** | $\partial\pi_\theta(s)/\partial\theta$ 把动作空间的扰动映到参数更新；ES 则直接扰动 $\theta$ |

#### 与 DDPG / TD3

| | DDPG / TD3 | RS-DDPG / LSE-DDPG |
|---|------------|---------------------|
| $\partial_a Q$ | Critic 网络 autograd | $Q(s,a+\epsilon)-Q(s,a)$ 有限差分 |
| 适用 | Q 光滑、可微 | Q 不可微、或不想反传仿真器 |
| Critic | 仍 NN + TD | 不变 |

---

### §IV.C Numerical experiments

| 设置 | 内容 |
|------|------|
| 基座 | CleanRL [40]，只改 actor 更新 |
| 环境 | MuJoCo ×7 |
| 对比 | DDPG, TD3, RS-DDPG, LSE-DDPG, RS-TD3, LSE-TD3 |
| RS 采样 | 每更新 **10** 个 $\epsilon$；$\Sigma=0.1^2 I$ |
| 重复 | 每任务 5 runs |

**Fig.3**：RS-DDPG、LSE-DDPG **显著优于** DDPG；TD3 已强，改进边际小（未全面调优 actor-critic 超参数）。

---

### §IV.D Connection to Reinforce

**随机策略目标**（式 37）：$A_k \sim p_\theta(\cdot|S_k)$，优化 $\mathbb{E}[\sum_k \gamma^k r(S_k,A_k)]$。

高斯策略（式 38–39）：$A_k \sim \mathcal{N}(\pi_\theta(S_k),\Sigma)$，

```math
\frac{\partial}{\partial \theta}\log p_\theta(a|s) = \frac{\partial \pi_\theta(s)}{\partial \theta}\,\Sigma^{-1}(a - \pi_\theta(s)).
```

$\partial\pi_\theta/\partial\theta$ 的维数与式 (35) 相同，故此处不再转置。

**策略梯度**（式 40，带 baseline $V$）：

```math
\frac{\partial J}{\partial \theta} = \mathbb{E}\left[\sum_k \gamma^k (Q(S_k,A_k)-V(S_k)) \frac{\partial \pi_\theta(S_k)}{\partial \theta}\,\Sigma^{-1}(A_k - \pi_\theta(S_k))\right].
```

| | 式 (36) RS-AC | Reinforce (40) |
|--|--|--|
| Rollout | 确定性 $s_k$，$a_k=\pi_\theta(s_k)$，另加 $\epsilon_k$ | 随机 $S_k,A_k$ |
| $Q$ | 确定性 rollout | 随机策略下的期望回报 |
| Baseline | 减去 $Q(s,a)$，对应 RS 中减去 $f(x)$ | 减去 $V(S_k)$ |
| 探索 | 差分噪声 $\epsilon_k$ | 动作分布本身 |

Reinforce 在随机策略上使用 log-likelihood。相对式 (36)，轨迹本身随机。与 TRPO、PPO 的关系，论文留作后续问题。

#### 三式对照表

| 公式 | 策略 | Q 的定义 | 噪声来源 |
|------|------|----------|----------|
| (35) DPG | 确定性 | 确定性 rollout Q | 初态/轨迹 MC |
| (36) RS-AC | 确定性 | 确定性 Q | + 动作 $\epsilon_k$ |
| (40) Reinforce | 随机 | 随机 rollout Q | 动作分布 + 轨迹 |

---

## §V Population Based Algorithms（种群算法）

### 动机

单点 Langevin / 局部搜索在多模态分布上**混合慢、自相关高**；需要**多个起点**或**粒子协调**。

### Algorithm 8：Random Restart

1. 维护当前最优 $x$  
2. 采样全局起点 $\bar x$  
3. 从 $\bar x$ 运行 Greedy Local Search 得 $\tilde x$  
4. 若 $f(\tilde x)<f(x)$ 则更新 $x$  

→ 结合 Alg.1（全局采起点）与 Alg.2（局部改进）。

### Algorithm 9：$(N+\lambda)$-ES

1. 维护种群 $\{x_1,\ldots,x_N\}$  
2. 构造 $D=\{x_1,\ldots,x_N\}$  
3. 重复 $\lambda$ 次：从种群均匀抽 $x$，加高斯扰动 $d$，加入 $D$  
4. 保留 $D$ 中 $f$ 最小的 **$N$** 个作为新种群  

| ES 记号 | 等价算法 |
|---------|----------|
| $(1+1)$-ES | Greedy Local Search (Alg.2) |
| $(1+\lambda)$-ES | Predictive Sampling (Alg.4) |

**局限**：多个局部搜索可能收敛到**同一**极小——粒子缺乏排斥。

### SVGD 与零阶扩展

**Stein Variational Gradient Descent** [41] 在粒子间加入排斥项。原版需要梯度。[42] 给出基于 CMA-ES 的无梯度形式，可与 §II–III 的得分估计衔接。

### 与 ES-RL

Salimans 等 [35] 在参数空间使用进化策略。本文的随机搜索作用在动作空间，并与 DPG 结合。

---

## §VI Conclusion（结论）

### 主要结论

- 随机搜索统一 MPPI、CMA 与策略梯度。同为 $\min f(x)$，$x$ 分别是控制序列、策略参数或动作扰动。  
- LSE 与 RS 的差别解释 MPPI 在非凸轨迹优化中的表现。CMA 是高斯族上的自然梯度。  
- RS/LSE Actor-Critic 表明这一框架可以导出新的 RL 算法。

### 开放问题

- 约束零阶优化，包括控制界与碰撞约束  
- 随机平滑的全局收敛，目前仅有 SPSA 的 Langevin 型结果  
- 与 TRPO、PPO 的关系  
- 块对角 CMA 的证明，以及块结构的选取  
- Reinforce 中额外随机性对探索的作用

---

## 算法速查（与附录）

| # | 名称 | 节 | 一句话 |
|---|------|-----|--------|
| 1 | Pure Random Search | §II.A | 独立采样，更好则替换 |
| 2 | Greedy Local Search | §II.A | 高斯扰动，仅接受下降 |
| 3 | Approximate Gradient Descent | §II.B | $x-\alpha g$，$g$=FD/RCD/SPSA/RS |
| 4 | Predictive Sampling | §III.A | $K$ 候选取 min |
| 5 | MPPI | §III.B | 指数权加权平均 |
| 6 | CMA | §III.C | 更新 $(x,\Sigma)$ |
| 7 | RS Actor-Critic | §IV.B | Q 上 RS 的 DPG |
| 8 | Random Restarts | §V | 多起点局部搜索 |
| 9 | $(N+\lambda)$-ES | §V | 种群 + 精英选择 |
| 10 | CEM | 附录 | 精英权 CMA 变体 |

论文附录未单独笔记：A 为 Metropolis–Hastings 与 Alg.2；B 为 natural gradient 与 Fisher 信息；C 为 risk-seeking 与 LSE；D 为 CEM（先更新均值，协方差绕新均值）。

---

## 附记：并行实现

论文没有单独的并行计算一节，只在引言中把 GPU 上的批量仿真作为零阶方法得以使用的条件。$f(x)$ 的一次评估是整条动力学前向，并行的是仿真器。

| 层次 | 工具 | 作用 |
|------|------|------|
| 仿真 | MJX、Isaac Sim | GPU 上的批量环境 |
| 轨迹优化 | Hydrax [33]、Evosax | 零阶 MPC 与进化策略 |
| 策略优化 | CleanRL [40] | DDPG、TD3 的实现 |

实验中每迭代使用 2048 个样本。没有相应的仿真吞吐时，这一样本量不代表文中的计算时间。

---

## 阅读注记

1. RS 平滑 $f$。LSE 是另一代理，$\lambda\to\infty$ 时回到 $f_\mu$。Reinforce 平滑的是策略分布。三者对象不同，梯度都经由 log-likelihood。  
2. MPPI 的 $K$ 很大时，比值估计的方差下降，更新接近确定性梯度步，容易停在局部极小。退火、随机重启或对 $\Sigma$ 的 CMA 用来保持探索。  
3. DPG 给出 $\nabla_\theta F$。RS/LSE Actor-Critic 只在 $\partial_a Q$ 不可用时替换 actor 的估计；critic 仍由 Bellman 方程更新。  
4. 长视界下协方差取块对角，或采用 PI² 的结构。$Tn_u$ 维的全协方差在存储和更新上不可行。  
5. 论文的轨迹优化实验依赖批量仿真。没有这一条件时，每迭代 2048 次 rollout 不能代表文中的计算代价。

---

## 参考文献（文中高频）

- [8] Hansen & Ostermeier — CMA-ES  
- [9] Howell et al. — Predictive Sampling  
- [10] Williams et al. — MPPI  
- [12] Nesterov & Spokoiny — Gaussian smoothing  
- [15] Scaman et al. — log-sum-exp 与 MPPI  
- [18] Spall — 随机搜索与 SPSA  
- [30] Ollivier et al. — information-geometric optimization  
- [31] Akimoto et al. — CMA 与自然梯度  
- [34] Silver et al. — DPG  
- [35] Salimans et al. — 参数空间进化策略  
- 代码与复现实验：[zoo-rob](https://github.com/ajordana/zoo-rob)
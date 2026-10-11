# 第 5 章 Infinite Horizon Approximate Methods — 分节笔记

> **文献**：Bertsekas, *RL and Optimal Control* Ch.5（2019 draft，2019-04）。  
> **文本**：书稿第 5 章。  
> **位置**：Ch.4 给出 SSP/折扣下的精确 VI/PI/Q-learning；本章在**大状态空间**下讨论参数化 FVI、仿真 PI、性能界、探索与振荡，并与 Ch.2–3 的实现细节衔接。本 PDF 版无独立 Aggregation 章，相关内容见 [Ber12] 与书中片段。

---

## 章首

无限时域近似方法的共同模板：

1. 用 $\tilde J(i,r)$ 或 $\tilde Q(i,u,r)$ 近似 Bellman 对象；  
2. 用**模拟器**或已知 $p_{ij}(u)$ 生成训练目标；  
3. 在**理想逐点误差**假设下建立界（§5.1），并承认**最小二乘 FVI** 可能违背该假设（例 5.2.1）。

各节在全书算法索引中的位置（VI/PI、值/策略空间、FVI vs AC vs PG）见 [`00-algorithm-taxonomy.md`](00-algorithm-taxonomy.md) §5、§8。

### 算子记号（与 Ch.4 对照）

**SSP**（(5.1)–(5.2)，同 Ch.4 (4.4)–(4.5) 型）：

```math
(TJ)(i)=\min_{u\in U(i)}\Big[p_{it}(u)g(i,u,t)+\sum_{j=1}^n p_{ij}(u)\big(g(i,u,j)+J(j)\big)\Big],
```

```math
(T_\mu J)(i)=p_{it}(\mu(i))g(i,\mu(i),t)+\sum_{j=1}^n p_{ij}(\mu(i))\big(g(i,\mu(i),j)+J(j)\big).
```

**折扣**（(5.3)–(5.4)，同 Ch.4 (4.12)–(4.13)）：

```math
(TJ)(i)=\min_{u\in U(i)}\sum_{j=1}^n p_{ij}(u)\big[g(i,u,j)+\alpha J(j)\big],
```

```math
(T_\mu J)(i)=\sum_{j=1}^n p_{ij}(\mu(i))\big[g(i,\mu(i),j)+\alpha J(j)\big].
```

**一步前瞻**（(5.5)）：在 $i$ 上取使下式最小的 $u$：

```math
\min_{u\in U(i)}\sum_{j=1}^n p_{ij}(u)\big[g(i,u,j)+\alpha\tilde J(j)\big].
```

Fig 5.1.1 与 Ch.2 Fig 2.1.1 同构：$\tilde J$ 可来自问题近似、Rollout、参数化 FVI、聚合、MCTS 等。

### 近似 PI 总流程

(a) 生成 $\mu_0,\ldots,\mu_m$；  
(b) 近似评估 $\tilde J_{\mu_k}$（常参数化/NN）；  
(c) 基于 $\tilde J_{\mu_k}$ 做一步或多步改进得 $\mu_{k+1}$；  
(d) 末策略评估 $\tilde J_{\mu_m}$ 作前瞻终端 $\tilde J$。**Rollout** = $m=0$ 的单次评估+改进特例。

---

## §5.1 Approximation in Value Space — Performance Bounds

无限时域值空间近似：先得 $\tilde J\approx J^{*}$，再一步或多步前瞻实现 $\tilde\mu$。$\ell$ 步前瞻的有效终端余值为 $T^{\ell-1}\hat J$（$\hat J$ 为初始猜测）；等价地，$\ell$ 步前瞻 = 一步前瞻，终端为 $T^{\ell-1}\hat J$。

---

### §5.1.1 Limited Lookahead

#### 主要内容

**$\ell$ 步前瞻策略** $\tilde\mu=\hat\mu_0$：在 $i$ 上最小化

```math
\min_{\mu_0,\ldots,\mu_{\ell-1}}\mathbb{E}\Big[\sum_{k=0}^{\ell-1}\alpha^k g(i_k,\mu_k(i_k),j_k)+\alpha^\ell\tilde J(i_\ell)\Big],
```

仅执行首控 $\hat\mu_0(i)$。算子形式：

```math
T_{\tilde\mu}(T^{\ell-1}\tilde J)=\min_u\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha(T^{\ell-1}\tilde J)(j)\big].
```

**Prop 5.1.1(a) / Ch.4 Prop 4.6.1(a)**：

```math
\|J_{\tilde\mu}-J^{*}\|_\infty\le\frac{2\alpha^\ell}{1-\alpha}\|\tilde J-J^{*}\|_\infty. \qquad \text{(5.6 / 4.38)}
```

**常数偏移不变性**：$\tilde J$ 加常数 $\beta$ 不改变 $J_{\tilde\mu}$，故可用更紧界

```math
\|J_{\tilde\mu}-J^{*}\|_\infty\le\frac{2\alpha^\ell}{1-\alpha}\min_{\beta\in\mathbb{R}}\max_i\big|\tilde J(i)+\beta-J^{*}(i)\big|.
```

**受限控集前瞻** (5.7)：在 $U(i)\subset U(i)$ 上 min 得 $\hat J(i)$；若

```math
\hat J(i)\le\tilde J(i)+c,\quad\forall i,
```

则 **Prop 5.1.1(b) / 4.6.1(b)**：

```math
J_{\tilde\mu}(i)\le\hat J(i)+\frac{c}{1-\alpha},\quad\forall i. \qquad \text{(5.9)}
```

$c\le 0$ 时 $J_{\tilde\mu}\le\tilde J$；$c=0$ 且 $\tilde J=J_\mu$ 时 $J_{\tilde\mu}\le J_\mu$（纯 rollout 改进）。

#### 例 5.1.1（界紧性）

两态、$\alpha\in[0,1)$、$\epsilon>0$（Fig 5.1.2）：最优 $\mu^{*}$ 为 $1\to2$；$J^{*}=(0,0)$。取 $\tilde J(1)=-\epsilon$，$\tilde J(2)=\epsilon$，则 $\|\tilde J-J^{*}\|=\epsilon$。停留策略 $\mu$ 满足一步前瞻等式，且

```math
J_\mu(1)=\frac{2\alpha\epsilon}{1-\alpha}=\frac{2\alpha}{1-\alpha}\|\tilde J-J^{*}\|,
```

(5.6) 在 $\ell=1$ 时取等——单步 $O(\epsilon)$ 代价差可放大为 $O(\epsilon/(1-\alpha))$ 策略代价差。

#### 要点

- 界随 $\ell$ 增大因子 $2\alpha^\ell/(1-\alpha)$ 减小，但**不保证**多步前瞻实际优于一步（cf. Ch.2 例 2.2.1）；保证的是**界**更紧。  
- $\alpha\to 1$ 时 (5.6) 不 reassuring；例 5.1.1 说明界在最坏情形紧。

#### 注意点

- 多步界较一步多因子 $2\alpha^\ell/(1-\alpha)$；$\ell$ 大时更依赖内层精确优化。  
- 与 Ch.2 Q 常数偏移讨论一致。

---

### §5.1.2 Rollout

#### 主要内容

**纯 rollout**：$\tilde J=J_\mu$（基策略 $\bar\mu$），一步 PI。**Prop 5.1.2 / 4.6.3**：$J_{\tilde\mu}\le J_\mu$。

**并行 rollout**（例 5.1.2）：多基策略 $\mu_1,\ldots,\mu_M$，

```math
\tilde J(i)=\min_m J_{\mu_m}(i),
```

在含各 $\mu_m(i)$ 的 $U(i)\subset U(i)$ 上一步前瞻 → 同时改进所有 $\mu_m$。

**截断 rollout**（Fig 5.1.3）：$\ell$ 步前瞻 + 基策略 $\mu$ 仿真 $m$ 步 + 终端 $\tilde J$。视为**单次乐观 PI + 多步前瞻**。

**Prop 5.1.3**（与 4.13.4 平行）：

| 部分 | 结论 |
|------|------|
| (a) | $\|J_{\tilde\mu}-J^{*}\|_\infty\le\frac{2\alpha^\ell}{1-\alpha}\|T_\mu^m\tilde J-J^{*}\|_\infty$ |
| (b) | 若 $\hat J(i)\le\tilde J(i)+c$（(5.10) 型），则 $J_{\tilde\mu}(i)\le\tilde J(i)+c/(1-\alpha)$ |
| (c) | 若 $\|\tilde J-J_\mu\|_\infty\le c/(1+\alpha)$，则 $J_{\tilde\mu}(i)\le J_\mu(i)+2c/(1-\alpha^2)$ |

(c) 说明：$\tilde J\approx J_\mu$ 时 $\tilde\mu$ **近似**改进 $\mu$；终端为启发式/回归时严格改进性可丧失。

#### 应用

| 系统 | 基策略 $\mu$ | 终端 $\tilde J$ | 多步前瞻 |
|------|-------------|----------------|----------|
| TD-Gammon rollout [TeG96] | TD-Gammon 策略 | NN 评估 | 有限深度 |
| AlphaGo [SHM16] | 深度策略网络 | 价值网络 | MCTS |

终端 $\tilde J$：启发式、问题近似、或离线对代表态仿真 $J_\mu$ 再 LS 回归——SSP 中“晚期大代价”时终端选择尤为关键。

#### 注意点

- 确定性问题：从 $j$ 出发单条轨迹即可得 $J_\mu(j)$，rollout 计算量小。  
- 折扣问题可截断仿真；剩余代价由 $\alpha^N$ 控制。

---

### §5.1.3 Approximate Policy Iteration

#### 主要内容

**近似评估** (5.12) / Ch.4 (4.44)：

```math
\|\tilde J_{\mu_k}-J_{\mu_k}\|_\infty\le\delta.
```

**近似改进** (5.13) / (4.45)：

```math
\max_i\Big[T_{\mu_{k+1}}\tilde J_{\mu_k}(i)-\min_u\sum_j p_{ij}(u)\big(g+\alpha\tilde J_{\mu_k}(j)\big)\Big]\le\epsilon.
```

**Prop 5.1.4 / 4.6.4**（折扣）：

```math
\limsup_{k\to\infty}\|J_{\mu_k}-J^{*}\|_\infty\le\frac{\epsilon+2\alpha\delta}{(1-\alpha)^2}. \qquad \text{(5.x)}
```

**典型行为**（Fig 5.1.4 / 4.6.4）：早期单调快速下降 → 进入宽度 $\approx(\epsilon+2\alpha\delta)/(1-\alpha)^2$ 的**误差带** → $J_{\mu_k}$ 随机振荡；界 pessimistic，实际带常更窄。

**策略收敛** (5.14)：$\mu_{k+1}=\mu_k=\tilde\mu$ 时（聚合等），**Prop 5.1.5 / 4.6.5**：

```math
\max_i\big(J_{\tilde\mu}(i)-J^{*}(i)\big)\le\frac{\epsilon+2\alpha\delta}{1-\alpha}.
```

界收紧因子 $1-\alpha$（Fig 5.1.5）。

**证明要点**（附录 5.9.3 / Lemma 4.13.1）：在 $\|J-J_\mu\|\le\delta$，$\|T_{\tilde\mu}J-TJ\|\le\epsilon$ 下

```math
\|J_{\tilde\mu}-J^{*}\|_\infty\le\alpha\|J_\mu-J^{*}\|_\infty+\frac{\epsilon+2\alpha\delta}{1-\alpha}.
```

#### 要点

- 有限策略集 $\Rightarrow$ $\{J_{\mu_k}\}$ 有界；**近似 PI 不受** Ch.4 例 4.4.1 类 $\tilde J_k\to\infty$ 不稳定性（与近似 VI 对比）。  
- **乐观 PI**：评估仅少量 VI + 回归；界结构类似但推导更繁（[Ber12] Ch.2）；仍受例 5.2.1 误差放大影响——近似 VI = 单次评估的乐观 PI 特例。  
- SSP 有平行界（[BeT96] §6.2.2）。

#### 注意点

- Prop 5.1.4 在无限策略空间也成立（[Ber18a] Prop 2.4.3）。  
- 界在最坏情形可紧（[BeT96] §6.2.3）。

---

## §5.2 Fitted Value Iteration

### 算法流程

与 Ch.3 §3.3 有限时域 FVI 同型；平稳问题各阶段共用 $(\phi,r)$。

**折扣 VI** (5.16) / Ch.4 (4.14)：

```math
J_{k+1}(i)=\min_{u\in U(i)}\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha J_k(j)\big].
```

**FVI 迭代**：

1. 初始化 $\tilde J_0$（参数 $r_0$）。  
2. $k=0,1,\ldots$：  
   - 对样本态 $i^s$ 计算 $\beta^s=(T\tilde J_k)(i^s)$（需 $p_{ij}$ 或仿真）；  
   - 回归 $\tilde J_{k+1}$ 使 $\|\tilde J_{k+1}-T\tilde J_k\|$ 在样本意义下小：

```math
r_{k+1}\in\arg\min_r\sum_{s=1}^q\big(\tilde J(i^s,r)-\beta^s\big)^2. \qquad \text{(5.21)}
```

3. 用最终 $\tilde J$ 作前瞻 (5.5) 式 min；或对 $\tilde J_k$ 贪心得 $\tilde\mu_k$。

**SSP** 型见 (5.15)，结构相同。

### 理想误差界 (5.17)–(5.19)

若对所有 $k,i$：

```math
\max_i\Big|\tilde J_{k+1}(i)-\min_u\sum_j p_{ij}(u)\big[g+\alpha\tilde J_k(j)\big]\Big|\le\delta, \qquad \text{(5.19)}
```

则渐近（[BeT96] §6.5.3 / [Ber12] Prop 2.5.3）：

```math
\limsup_{k\to\infty}\|\tilde J_k-J^{*}\|_\infty\le\frac{\delta}{1-\alpha}, \qquad \text{(5.17)}
```

```math
\limsup_{k\to\infty}\|J_{\tilde\mu_k}-J^{*}\|_\infty\le\frac{2\delta}{(1-\alpha)^2}, \qquad \text{(5.18)}
```

其中 $\tilde\mu_k(i)\in\arg\min_u\sum_j p_{ij}(u)[g+\alpha\tilde J_k(j)]$。

### 例 5.2.1（与 Ch.4 例 4.4.1 同型）

两状态、单策略、零阶段代价：$1\to2\to2\to\cdots$，折扣 $\alpha$。Bellman：$J(1)=\alpha J(2)$，$J(2)=\alpha J(2)$，$J^{*}=(0,0)$。

近似子空间 $S=\{(r,2r)\mid r\in\mathbb{R}\}$（$J^{*}\in S$，看似有利）。给定 $\tilde J_k=(r_k,2r_k)$：

```math
T\tilde J_k=\big(\alpha\tilde J_k(2),\alpha\tilde J_k(2)\big)=(2\alpha r_k,2\alpha r_k).
```

**加权 LS**（权重 $\xi_1,\xi_2>0$）：

```math
r_{k+1}\in\arg\min_r\Big[\xi_1(r-2\alpha r_k)^2+\xi_2(2r-2\alpha r_k)^2\Big]
\;\Rightarrow\;
r_{k+1}=\alpha\zeta r_k,
```

```math
\zeta=\frac{2(\xi_1+2\xi_2)}{\xi_1+4\xi_2}>1. \qquad \text{(5.20)}
```

**脚注（加权与长期重要性）**：

- $\xi_1=\xi_2=1$（等权）→ $\zeta=6/5$；$\alpha\in(5/6,1)$ 时 $\tilde J_k$ **无界发散**。  
- 权重 $\xi_i$ = 样本中态 $i$ 出现比例；等权 = 两态等样本 → 违背“长期重要性”。  
- 令 $\xi_2\gg\xi_1$（态 2 在轨迹中几乎独占）→ $\zeta\to 1$，若 $\alpha\zeta<1$ 则 $\tilde J_k\to J^{*}$。  
- **结论**：VI + 加权投影**不必**为收缩；**样本权重/分布**对 FVI 稳定性至关重要。

Fig 5.2.2：加权投影值域；自然 LS 可能把迭代推离 $J^{*}$。

### 实践要点

| 议题 | 建议 |
|------|------|
| 样本分布 | 接近**最优或近最优策略**的稳态分布（Ch.4 脚注；[Ber12] §6.3） |
| 实现 | 好启发式策略长轨迹 → 稳态后记录 $i^s$ → (5.21) 回归 |
| 与 PI | 一次 $T\tilde J$ + LS ≈ 一轮乐观 PI 评估 |
| 理论 | 无已知通用保证 $\{r_k\}$ 有界；需问题特定条件 |

#### 注意点

- (5.19) 是**假设**，自然 LS FVI **不保证**满足——例 5.2.1 为反例。  
- 与 Ch.3 FVI 相同架构 $(\phi,r)$，但无限时域误差可累积放大。

---

## §5.3 Simulation-Based Policy Iteration with Parametric Approximation

折扣问题为主；SSP 可类比。核心：**Critic**（评估 $\tilde J_{\mu_k}$ 或 $\tilde Q_{\mu_k}$）+ **Actor**（改进 $\mu_{k+1}$）。Fig 5.3.1（model-based）/ 5.3.2（model-free）。

---

### §5.3.1 Self-Learning and Actor–Critic

#### 主要内容

**Self-learning**：仿真 PI + 参数化评估；系统通过**观察自身在 $\mu_k$ 下产生的轨迹**更新 critic，actor 做 Bellman 贪心或样本上 min + 策略回归。

**两步骤**（每轮 PI）：

**(a) Critic**：在 $\mu_k$ 下采样代价，增量/LS 拟合 $\tilde J_{\mu_k}(i,r)$。

**(b) Actor**（model-based，需 $p_{ij}$）：

```math
\mu_{k+1}(i)\in\arg\min_{u\in U(i)}\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha\tilde J_{\mu_k}(j,r)\big],
```

或在样本态 $i^s$ 上算

```math
u^s\in\arg\min_{u\in U(i^s)}\sum_j p_{i^s j}(u)\big[g(i^s,u,j)+\alpha\tilde J_{\mu_k}(j,r)\big],
```

再 $\mu_{k+1}(i,r)$ 回归（Ch.2 §2.1.5 策略空间近似）。

**Actor–Critic 术语**：评估网络 = critic；改进网络 = actor。NN 时称 critic/actor network。

#### 区分

- 学习的是**更好策略**，非系统辨识——不学 $p_{ij}$ 显式模型。  
- 两阶段替代：先系统辨识再 model-based PI（本书不展开）。

#### 终止

收敛于误差带（Fig 5.1.4–5.1.5），或 **策略振荡** 时学习停滞（§5.3.5）。

---

### §5.3.2 A Model-Based Variant

#### 评估 (5.22)

样本 $(i^s,\beta^s)$；$\beta^s$ = 从 $i^s$ 起用 $\mu$ 仿真 $N$ 步折扣代价 + $\alpha^N\hat J(i_N)$：

```math
r\in\arg\min_r\sum_{s=1}^q\big(\tilde J_\mu(i^s,r)-\beta^s\big)^2. \qquad \text{(5.22)}
```

$\hat J$ 可取上一策略评估、零、或问题近似——类似**乐观 PI** 终端。

**增量梯度**（Ch.3 (3.6) 型）：

```math
r_{k+1}=r_k-\gamma_k\nabla_r\big(\tilde J(i^s_k,r_k)-\beta^{s_k}\big)^2.
```

线性架构 → (5.22) 闭式解（Ch.3 (3.3)）。

#### 改进 (5.23)

```math
\tilde\mu(i)\in\arg\min_u\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha\tilde J(j,r)\big].
```

#### 投影方程解释 (5.45)

抽象地，(5.22) 近似求解

```math
\tilde J_\mu\approx\Pi(T_\mu^N\hat J),
```

$\Pi$ 为加权投影；$N$ 步轨迹 + 终端 $\hat J$ 定义 $T_\mu^N\hat J$ 的样本。$N$ 大且 $q$ 大时 $\tilde J_\mu\approx\Pi(J_\mu)$。

#### 轨迹复用与 bias–variance

长轨迹 $(i_0,\ldots,i_N)$ 可从 $i_0,i_1,\ldots$ 各起算 tail 代价 → 省采样。

| $N$ | 偏差 | 方差 | 备注 |
|-----|------|------|------|
| 小 | 大（$\alpha^N\hat J$ 主导尾部） | 小 | 采样快 |
| 大 | 小 | 大、成本高 | 接近 $J_\mu$ |

多短轨迹 + 多初态 → 更好**探索**。TD($\lambda$)、LSTD($\lambda$)、LSPE($\lambda$) 由此动机（§5.5）。

---

### §5.3.3 A Model-Free Variant

#### Q 架构 (5.24)–(5.25)

- $\tilde Q_\mu(i,u,r)=r(u)'\phi(i)$——控少时；  
- $\tilde Q_\mu(i,u,r)=r'\phi(i,u)$——一般。

#### 评估 (5.26)

三元组 $(i^s,u^s,\beta^s)$：首步 $u^s$，其后 $\mu$ 共 $N$ 步：

```math
r\in\arg\min_r\sum_s\big(\tilde Q_\mu(i^s,u^s,r)-\beta^s\big)^2. \qquad \text{(5.26)}
```

$\beta^s$ 估计 $N$ 阶段 Q：

```math
Q^N_\mu(i,u)=\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha J^{N-1}_\mu(j)\big].
```

#### 改进 (5.27)

```math
\tilde\mu(i)=\arg\min_u\tilde Q_\mu(i,u,r).
```

#### 探索缺陷

轨迹复用后样本多为 $(i,\mu(i))$，**$u\neq\mu(i)$** 的 $(i,u)$ 覆盖不足。

#### 两阶段替代

1. model-free 得 $\tilde J_\mu$（(5.22)）；  
2. 再回归 $\tilde Q$ 近似 $\sum_j p_{ij}(u)[g+\alpha\tilde J_\mu(j)]$ + 策略近似。  

更复杂，但利于轨迹复用与 $(i,u)$ 探索。

---

### §5.3.4 Implementation Issues

#### 架构与 cost shaping

| 架构 | 优势 |
|------|------|
| 线性 | (5.22)(5.26) 闭式；TD/LSTD/LSPE 理论保证 |
| NN | 表达力强；训练非凸 |

**Cost shaping**（Ch.4 §4.2）：  
SSP：$\hat g(i,u,j)=g(i,u,j)+V(j)-V(i)$；  
折扣：$\hat g=g+\alpha V(j)-V(i)$。  

最优策略不变，**次优策略**会变；$V$ 应接近 $J^{*}$ 或 $J_{\mu_k}$。可 NN 学 $V$ + 局部修正；每轮 PI 可换 $V$。

#### 探索

**轨迹复用偏置**：常访问态过代表 → 罕见态评估差 → 改进步大错。

**记忆缓冲**：初态集 $I\cup I_0\cup\cdots\cup I_k$；$I_m$ 为评估 $\mu_m$ 时产生的态；评估 $\mu_k$ 时从各集按**偏近期**概率抽初态；短轨迹 + 准终端 $\hat J$。

**SSP 深探索**：晚期大代价时需长轨迹接近终止。

**Q 评估**：更需在 $(i,u)$ 空间探索；从多样 $(i,u)$ 起仿真。

**Off-policy / behavior policy**：混合 target 与探索 policy；$\epsilon$-greedy 等。**偏差校正**需特殊修改（[Ber12] §6.4.2）。探索–利用仍为开放问题（[RuV16], [RVK18], [OVR19] deep exploration）。

---

### §5.3.5 Oscillations

#### 机制：Greedy partition $\mathcal{R}_\mu$

对架构 $\tilde J(\cdot,r)$：

```math
\mathcal{R}_\mu=\Big\{r\;\Big|\;\mu(i)=\arg\min_u\sum_j p_{ij}(u)\big[g+\alpha\tilde J(j,r)\big],\;\forall i\Big\}.
```

等价地 $r\in\mathcal{R}_\mu\Leftrightarrow T_\mu\tilde J(\cdot,r)=T\tilde J(\cdot,r)$。$\mathcal{R}_\mu$ 只依赖架构，不依赖评估方法。

**非乐观 PI**：每 $\mu$ 得唯一 $r_\mu$。若 $r_{\mu_k}\in\mathcal{R}_{\mu_{k+1}}$ 且永不出现 $r_\mu\in\mathcal{R}_\mu$，则策略在有限环上**循环**（Fig 5.3.3）。

**查表**：$r_\mu=J_\mu$，$r_\mu\in\mathcal{R}_\mu\Leftrightarrow J_\mu=TJ_\mu\Leftrightarrow\mu$ 最优。有函数逼近时不再等价。

**收敛判据（非乐观）**：$r_{\mu_k}\in\mathcal{R}_{\mu_k}$ 时策略固定。

#### 乐观 PI 与 chattering

参数 $r$ 向 $r_\mu$ 移动，直至 $r$ 越界进入另一 $\mathcal{R}_{\mu'}$ → 策略切换。若步长递减，$r$ 可收敛于多个 $\mathcal{R}_\mu$ 的**公共边界**——**chattering**（Fig 5.3.4）：策略空间振荡、参数空间收敛。极限 $r$ 可能不对应任一策略的有意义 cost 近似 → 结束时需**仿真筛选**多策略。

#### 经验

- [BeT96] §6.4.2：振荡未必严重损害**平均**性能。  
- 不同评估/探索方法改 $r_\mu$，**不改变** $\mathcal{R}_\mu$——“在同一水域钓鱼”。  
- 聚合等特殊结构可证收敛（Ch.6）；Prop 5.1.5 更紧界。  
- “无 critical 态”时振荡策略代价可能相近（至多 $s=\dim r$ 个 ambiguous 态）。

---

## §5.4 Q-Learning

### 基本思想

在 Q 空间做 VI/PI；$F$ 为 $\alpha$-收缩（Ch.4 (4.51)）。随机更新 (4.52) 为 $FQ$ 的 SA。

**最优 Q**：

```math
Q^{*}(i,u)=\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha J^{*}(j)\big],
```

```math
Q^{*}(i,u)=\sum_j p_{ij}(u)\Big[g(i,u,j)+\alpha\min_{v\in U(j)}Q^{*}(j,v)\Big].
```

**VI**：$Q_{k+1}=FQ_k$，(5.28)：

```math
(FQ)(i,u)=\sum_j p_{ij}(u)\Big[g(i,u,j)+\alpha\min_{v\in U(j)}Q(j,v)\Big].
```

**原始 Q-learning** (5.29)–(5.30)：采样 $(i_k,u_k)$，后继 $j_k\sim p_{i_k j}(u_k)$：

```math
Q_{k+1}(i,u)=(1-\gamma_k)Q_k(i,u)+\gamma_k(F_k Q_k)(i,u),
```

```math
(F_k Q_k)(i,u)=
\begin{cases}
g(i_k,u_k,j_k)+\alpha\min_v Q_k(j_k,v), & (i,u)=(i_k,u_k),\\
Q_k(i,u), & \text{否则}.
\end{cases}
```

收敛需：各 $(i,u)$ 无限次访问；$\gamma_k>0$，$\sum_k\gamma_k=\infty$，$\sum_k\gamma_k^2<\infty$（[Tsi94]）。

**实践困难**：$(i,u)$ 对数可能极大 → 需 Q 逼近。

---

### §5.4.1 Optimistic PI + Parametric Q — SARSA & DQN

将 Q 视为“增广态”$(i,u)$ 上的 cost；用 §5.3 型近似 PI / 极端乐观更新。

**单样本乐观更新**（迭代 $k$，参数 $r_k$，态 $i_k$，控 $u_k$）：

1. 仿真 $(i_k,i_{k+1})$，$p_{i_k j}(u_k)$。  
2. $u_{k+1}=\arg\min_u\tilde Q(i_{k+1},u,r_k)$（或 $\epsilon$-探索）。  
3.

```math
q_k=\tilde Q(i_k,u_k,r_k)-\alpha\tilde Q(i_{k+1},u_{k+1},r_k)-g(i_k,u_k,i_{k+1}),
```

```math
r_{k+1}=r_k-\gamma_k\nabla_r\tilde Q(i_k,u_k,r_k)\,q_k.
```

$q_k$ 为 **TD 残差**（§5.5）。

**SARSA（on-policy）**：TD 目标用**实际**下一动作 $u_{k+1}$（上式含 $\tilde Q(i_{k+1},u_{k+1},r_k)$）。

**Q-learning（off-policy 常见写法）**：目标用 $\min_v\tilde Q(j,v,r)$ 替代 $\tilde Q(j,u',r)$——与 SARSA 采样目标不同；收敛需 on-policy 或特殊 off-policy 修正。

**DQN** [GBC16]：$\tilde Q(i,u,\theta)$ 为深度网络；  
- **经验回放**：缓冲存 $(i,u,g,j)$，随机 minibatch 打破时间相关；  
- **目标网络** $\theta^{-}$：延迟更新 stabilizing，目标 $\min_v\tilde Q(j,v,\theta^{-})$。

**乐观 PI + 参数 Q**：Q 评估仅若干步 + 回归 + 改进；理论界少，实践常用。

#### 要点

- 极端单样本方案 = SARSA 族；多样本平均后更新更稳。  
- 与 §5.3 model-free PI (5.26)–(5.27) 同框架，增量形式不同。

---

## §5.5 Additional Methods — Temporal Differences

本节概要 TD 族；细节见 [Ber12] §6.3、[SuB18]。动机：§5.3.2 短轨迹 **bias–variance** 折中。

### 投影视角

Bellman 方程 $J_\mu=T_\mu J_\mu$ (5.31)。线性架构 $\tilde J(i,r)=\phi(i)'r$，流形 $M=\{\Phi r\}$。

加权范数 (5.34)：

```math
\|J\|_\xi^2=\sum_i\xi_i J(i)^2.
```

投影 (5.35)：$\Pi(J)\in\arg\min_{V\in M}\|J-V\|_\xi$。线性时闭式 (5.38)：

```math
r^{*}=\Big(\sum_i\xi_i\phi(i)\phi(i)'\Big)^{-1}\sum_i\xi_i\phi(i)J(i).
```

**三种抽象途径**：（a）$\Pi(J_\mu)$；（b）$\Pi(T_\mu^N\hat J)$；（c）解投影方程 $J=\Pi(T_\mu J)$。均可 Monte Carlo 近似（(5.42)(5.43)）。

§5.3.2 的 (5.22) ≈ (b)：(5.45) $\tilde J_\mu\approx\Pi(T_\mu^N\hat J)$。

### 投影方程与 $T_\mu^{(\lambda)}$

TD($\lambda$)/LSTD/LSPE 瞄准 (5.46)：

```math
\Phi r=\Pi\big(T_\mu^{(\lambda)}\Phi r\big),
```

```math
(T_\mu^{(\lambda)}J)(i)=(1-\lambda)\sum_{\ell=0}^\infty\lambda^\ell(T_\mu^{\ell+1}J)(i),\quad 0\le\lambda<1.
```

$\lambda=0$：$T_\mu^{(0)}=T_\mu$（一步 Bellman）；$\lambda\to 1$：解趋 $\Pi(J_\mu)$（**TD(1)** 视角）。

仿真近似 (5.47)：$\Phi r=\tilde\Pi(T_\mu^{(\lambda)}\Phi r)$ → LS (5.48)。

### TD(0)、LSTD(0)、LSPE(0)

轨迹样本 $(i_1,i_2),\ldots,(i_q,i_{q+1})$，代价 $g(i_s,i_{s+1})$（$\mu$ 固定）。

**TD 残差** (5.54)：

```math
d_s(r)=\phi(i_s)'r-g(i_s,i_{s+1})-\alpha\phi(i_{s+1})'r.
```

**LSTD(0)**：解 (5.51)–(5.53)

```math
r=\Big(\sum_{s=1}^q\phi(i_s)\big(\phi(i_s)-\alpha\phi(i_{s+1})\big)'\Big)^{-1}\sum_{s=1}^q\phi(i_s)g(i_s,i_{s+1}).
```

**LSPE(0)**：增量投影 VI (5.55)

```math
r_{k+1}=r_k-\Big(\sum_{s=1}^k\phi(i_s)\phi(i_s)'\Big)^{-1}\sum_{s=1}^k\phi(i_s)d_s(r_k).
```

$k=q$ 时与 LSTD(0) 工作量相当；**热启动**下一策略评估。

**TD(0)** (5.56)：

```math
r_{k+1}=r_k-\gamma_k\phi(i_k)d_k(r_k).
```

单样本、无矩阵求逆；通常慢于 LSPE/LSTD。

### TD($\lambda$)、LSTD($\lambda$)、LSPE($\lambda$)

- **LSTD($\lambda$)**：收集 $q$ 样本后矩阵求逆解 (5.49) $Cr=d$。  
- **LSPE($\lambda$)**：(5.50) $J_{k+1}=\tilde\Pi(T_\mu^{(\lambda)}J_k)$ 的仿真增量版。  
- **TD($\lambda$)**：(5.49) 的 SA / 近端算法随机版（[Ber16c], [Ber18d]）；eligibility trace 实现 $n$-step 加权。

### Bias–variance（Fig 5.5.1）

间接法解 $\Phi r_\lambda^{*}$，一般 $\Phi r_\lambda^{*}\neq\Pi(J_\mu)$。误差界 (5.57)：

```math
\|J_\mu-\Phi r_\lambda^{*}\|_\xi\le p\|J_\mu-\Pi J_\mu\|_\xi,\quad p=\frac{\sqrt{1-\alpha^2\lambda}}{\alpha(1-\lambda)/(1-\alpha\lambda)}.
```

$\lambda\downarrow$：bias 增（相对 $\Pi J_\mu$），样本噪声减；$\lambda\uparrow$：bias 减，仿真方差增——与 §5.3.2 短/长轨迹对偶。

| 方法 | 类型 | 收敛/数值 |
|------|------|-----------|
| LSTD($\lambda$) | 间接 | 无迭代收敛问题；矩阵可能奇异/near-singular |
| LSPE($\lambda$) | 间接 | $\Pi T_\mu^{(\lambda)}$ 未必收缩；$\lambda$ 接近 1 时可收缩 |
| TD(0) | 间接 | on-policy 或特殊 off-policy；步长敏感 |

#### 注意点

- 半梯度 TD off-policy 需修正（[Ber12] §6.3、[SuB18]）。  
- 探索在近似 PI 中仍关键。  
- §5.3.2 直接法 $N\to\infty$ 得 $\Pi(J_\mu)$；TD($\lambda$) 间接法 $\lambda\to 1$ 同极限。

---

## §5.6 Exact and Approximate Linear Programming

### 精确 LP

$J^{*}$ 为满足 Bellman **不等式**（分量）的最大向量：

```math
J(i)\le\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha J(j)\big],\quad\forall i,u. \qquad \text{(5.58)}
```

线性规划 (5.59)：

```math
\max\sum_{i=1}^n J(i)\quad\text{s.t. (5.58)}.
```

$\beta_i>0$ 的任意线性目标 $\sum_i\beta_i J(i)$ 亦在 $J^{*}$ 处最优（Fig 5.6.1）。由 VI 单调性：任意可行 $J\le J^{*}$ 分量wise。

### 近似 LP

$\tilde J(i,r)=\sum_{\ell=1}^m r_\ell\phi_\ell(i)$ 代入 (5.58)：

```math
\max_{r}\sum_{i\in\tilde I}\tilde J(i,r)\quad\text{s.t.}\;
\tilde J(i,r)\le\sum_j p_{ij}(u)\big[g+\alpha\tilde J(j,r)\big],\;
i\in I,\;u\in\tilde U(i).
```

$\tilde I\subseteq\{1,\ldots,n\}$，$\tilde U(i)\subseteq U(i)$ 为约束子集。

| 困难 | 说明 |
|------|------|
| 约束数 | 可达 $n\times|U(i)|_{\max}$，$n$ 大时不可行 |
| 对策 | 随机采样约束子集；随策略丰富逐步加约束 [DFV03][DFV04] |
| 策略评估 LP | 固定 $\mu$ 时约束更少——近似 PI 中可用 |

与某些对偶 RL 方法相关；实现需 sophistication（[DeF04]）。

---

## §5.7 Approximation in Policy Space

直接在策略参数 $r$ 上优化，$\tilde\mu(i,r)$；可与值空间近似并用（actor + critic 双网络）。

---

### §5.7.1 Policy Gradient、CEM、Random Search

**目标** (5.60)：

```math
\min_r\;\mathbb{E}_{i_0}\big[J_{\tilde\mu(r)}(i_0)\big].
```

Fig 5.7.2：每个 $r$ 定策略与代价曲面。

**梯度法** (5.61)（$i_0$ 已知）：

```math
r_{k+1}=r_k-\gamma_k\nabla_r J_{\tilde\mu(r_k)}(i_0).
```

随机问题：梯度需 MC 估计 → 噪声大。

**随机化策略 + log-likelihood trick**：轨迹 $z=\{i_0,u_0,i_1,\ldots\}$，$F(z)=\sum_m\alpha^m g(i_m,u_m,i_{m+1})$。样本梯度迭代 (5.67)：

```math
r_{k+1}=r_k-\gamma_k\nabla_r\log p(z^k;r_k)\,F(z^k).
```

**折扣 DP 例 5.7.6**：(5.69)

```math
\nabla_r\log p(z;r)=\sum_{m=0}^\infty\nabla_r\log p(u_m\mid i_m;r).
```

不含 stage cost——**model-free** 只需仿真 $F(z)$ 与策略导数。REINFORCE 类方法由此。

**随机方向搜索** (5.64)–(5.66)：每步仅 2 次代价评估；Example 5.7.5。

**交叉熵法 (CEM)**（Fig 5.7.4）：在 $r_k$ 周围采样 → 保留低代价精英 → $r_{k+1}$ = 接受样本均值，更新协方差椭球。无梯度、无随机策略要求；Tetris [SzL06][ThS09] 成功案例。与进化编程、策略梯度同属“沿改进方向移动”。

**策略参数化例**：

| 例 | 内容 |
|----|------|
| 5.7.1 | 供应链：库存 $<r_1$ 则订货至 $r_2$ |
| 5.7.2 | PID：$(r_p,r_i,r_d)$ 为策略参数 |
| 5.7.3 | 通过 $\tilde J(j,r)$ 一步前瞻定义 $\tilde\mu(i,r)$ |
| 5.7.4 | 多智能体：无标准 DP，仍可策略参数化 |

**缺点**：梯度法慢、局部极小；仿真噪声；CEM/随机搜索收敛保证有限。

---

### §5.7.2 Expert Supervised Training

样本 $(i^s,u^s)$，$u^s$ 为专家（人或软件）的“好”控：

```math
\min_r\sum_{s=1}^q\big\|u^s-\tilde\mu(i^s,r)\big\|^2. \qquad \text{(5.70)}
```

**软件专家**：(5.71) 一步前瞻

```math
u^s=\arg\min_{u\in U(i^s)}\sum_j p_{i^s j}(u)\big[g(i^s,u,j)+\alpha\tilde J(j)\big],
```

或 Q 型 (5.72)。值空间生成标签 → 策略空间拟合。

**插值** (5.73)：非样本态 $i$ 用 $\phi_{is}$ 凸组合专家控——Ch.6 聚合核心。

**covariate shift**：测试态偏离专家分布 → 性能下降。行为克隆 / imitation learning。

**局限**：策略性能不超过专家（或前瞻基策略）。**优势**：在线执行快，无需每步 min。

历史：Tesauro 比较学习 [Tes89] → TD [Tes92] 更好 → rollout [TeG96] 更好。

---

### §5.7.3 Approximate PI + Rollout + Policy Approx

与 §5.3 **对偶**：精确 rollout 评估 + **策略空间**近似改进（Fig 5.7.5）。

基策略 $\mu$，rollout 得 $(i^s,u^s)$，回归：

```math
r\in\arg\min_r\sum_{s=1}^q\big\|u^s-\tilde\mu(i^s,r)\big\|^2. \qquad \text{(5.74)}
```

$\tilde\mu(\cdot,r)$ 作新基策略 → 迭代。Rollout = “专家”。

**极端乐观增量**（类 SARSA）：每得 $(i_k,u_k)$（$u_k$ 为 rollout 控）：

```math
r_{k+1}=r_k-\gamma_k\nabla_r\tilde\mu(i_k,r_k)\big(\tilde\mu(i_k,r_k)-u_k\big).
```

离线 PI 计算重；最终策略在线可作 rollout 基策略。**探索**：$i^s$ 选择仍关键。

---

## §5.8 Notes and Sources

| 节 | 要点 |
|----|------|
| 5.1 | Props 5.1.1–5.1.3 为 [Ber17] Prop 6.1.1 等 sharpened 版 |
| 5.2 | FVI：[Gor95][LoS01][AMS07] 等；例 5.2.1 [TsV96] |
| 5.3 | Q-PI [FYG06]；Tetris [Sch13][SGG15] |
| 5.4 | Q-learning [Wat89]，收敛 [Tsi94]；异步 乐观 PI [BeY10][BeY12] |
| 5.5 | TD [Sut88]；LSTD [BrB96][Boy02]；LSPE [BeI96] Tetris |
| 5.6 | LP [D'Ep60]；近似 [DFV03][DFV04] |
| 5.7 | PG [Wil92][SuB18]；CEM [MRG03][SzL06] |

与 Ch.4 §4.12 互补；TD 细节 Ch.4 §4.9 入门、本章 §5.5 连接 §5.3 bias–variance。

---

## §5.9 Appendix（证明导读）

| 节 | 内容 | Ch.4 对照 |
|----|------|-----------|
| 5.9.1 | Prop 5.1.1：$\ell$ 步界 (5.75)；用 $T,T_\mu$ 收缩 + $\hat J=T^{\ell-1}\tilde J$ | 4.13.4 一步特化 |
| 5.9.2 | Prop 5.1.2 rollout 改进；Prop 5.1.3 截断 rollout (a)–(c) | 4.6.3, 4.13.4 |
| 5.9.3 | Prop 5.1.4–5.1.5；Lemma 4.13.1 递推 | 4.13.1–4.13.2 |

**5.9.1 证明梗概**：$kT_\mu^k J^{*}-J^{*}\|\le\frac{1}{1-\alpha}\|T_\tilde\mu J^{*}-J^{*}\|$；$\|T_\tilde\mu J^{*}-J^{*}\|\le 2\alpha^\ell\|\tilde J-J^{*}\|$。

**算子性质**（全文用）：单调性 $J\ge J'\Rightarrow TJ\ge TJ'$；折扣**常数平移**：$J+c e\Rightarrow TJ+\alpha c e$。

---

## Ch.4 与 Ch.5 分工（扩展对照）

| 维度 | Ch.4 精确无限时域 | Ch.5 近似无限时域 |
|------|-------------------|-------------------|
| **问题** | SSP + 折扣；$n$ 态可枚举 | 同问题，$n$ 极大/连续 |
| **算子** | $T,T_\mu,F$；(4.1)–(4.16), (4.51) | 重述 (5.1)–(5.4)；+FVI/仿真 |
| **VI/PI** | 精确 VI (4.14)、PI (4.17) 收敛 | FVI (5.21)、仿真 Actor–Critic |
| **性能界** | 精确最优性 Prop 4.2.x, 4.3.x | 前瞻 (5.6)、rollout (5.1.3)、近似 PI (5.12)–(5.13) |
| **Q-learning** | 表格 (4.52) 收敛 [Tsi94] | 参数 Q、SARSA、DQN (5.4.1) |
| **TD/LP/PG** | §4.7 TD、§4.10 LP、§4.11 PG 概要 | §5.5 TD 投影式详述；§5.6 近似 LP；§5.7 策略空间训练 |
| **主要风险** | 例 4.4.1 近似 VI 发散 | 例 5.2.1 FVI 发散；§5.3.5 $\mathcal{R}_\mu$ 振荡 |
| **Rollout** | §4.6 界与截断 | §5.1.2 + 应用 (AlphaGo 等) |
| **实现** | 表格存储 | Ch.3 架构 + 采样 + 探索缓冲 |
| **Ch.2 衔接** | 有限时域极限 | 值空间前瞻 Fig 5.1.1 = Fig 2.1.1 |
| **Ch.3 衔接** | — | (3.2) LS、(3.6) 增量梯度用于 (5.22)(5.26) |

**阅读顺序建议**：Ch.4 算子与界 → Ch.3 架构/LS → Ch.5 FVI 与仿真 PI → §5.5 TD 深化评估 → §5.7 策略空间替代。

---

## 本章小结

无限时域近似 RL = **Bellman 收缩**（理想界 (5.6)(5.17)–(5.19)）+ **函数逼近与采样**（可能破坏 (5.19)）。核心张力：

1. **值空间**：FVI 简单但例 5.2.1 警告权重；近似 PI 有误差带但可能 $\mathcal{R}_\mu$ 振荡。  
2. **Actor–Critic**：Critic (5.22)/(5.26) + Actor (5.23)/(5.27)；探索与 bias–variance 贯穿 §5.3–5.5。  
3. **Q 空间**：Q-learning / SARSA / DQN 避免每轮解 $J_\mu$ 线性系统。  
4. **TD 族**：投影方程 (5.46) 统一 LSTD/LSPE/TD($\lambda$)；$\lambda$ 调 bias–variance。  
5. **策略空间**：PG/CEM 直接优化 (5.60)；专家 (5.70)；rollout+ (5.74) 与 §5.3 对偶。

性能界在 $\delta,\epsilon$ 小且 $\alpha$ 不极接近 1 时定性可靠；工程上须监测振荡、样本分布与终端 $\hat J$。

---

*个人学习笔记；原著 Copyright Bertsekas / Athena Scientific。*

# 第 4 章 Infinite Horizon Reinforcement Learning — 分节笔记

> **文献**：Bertsekas, *RL and Optimal Control* Ch.4（2019 draft，修订 2019-02）。  
> **文本**：书稿第 4 章。  
> **位置**：有限状态、**平稳** MDP 的精确理论（SSP、折扣）与 VI/PI/Q-learning/TD/LP/策略梯度；为 Ch.5 参数化与仿真实现提供定理与界。记号：$i,j\in\{1,\ldots,n\}$，终止态 $t$，$p_{ij}(u)$。

---

## 章首

无限时域问题可视为有限时域代价的极限（Fig 4.1.1）。与 Ch.1 相比有两处根本差异：

| 维度 | 有限时域 | 无限时域 |
|------|----------|----------|
| 阶段数 | $N$ 固定 | 形式为 $\infty$（实践中「很大」的近似） |
| 平稳性 | $f_k,g_k$ 可随 $k$ 变 | 系统方程、阶段代价、随机扰动统计**不随阶段变** |

**直觉**：最优策略常为**平稳**——从任意态出发的「未来子问题」结构相同，不必依赖绝对时间索引。

**本章结构**：§4.1 给出 SSP 与折扣两类问题及猜想 (4.2)–(4.3)；§4.2–4.3 证明 $T$ 的压缩性（Prop 4.2.5、4.3.5），从而无近似时 VI/PI 收敛；§4.4–4.5 讨论 fitted VI 与 PI，并指出近似下复合映射未必压缩（例 4.4.1）。

**范围**：有限状态离散时间系统；证明见 §4.13；实现衔接 Ch.3、Ch.5。

---

## §4.1 An Overview of Infinite Horizon Problems

### 总述

目标：最小化无限阶段总代价

```math
J_\pi(x_0)=\lim_{N\to\infty}\mathbb{E}\Big\{\sum_{k=0}^{N-1}\alpha^k g(x_k,\mu_k(x_k),w_k)\Big\}. \qquad \text{(4.1 系)}
```

$\alpha>0$：$\alpha<1$ 表示未来代价权重低于当前；$\alpha=1$ 时通常需终止态保证和有限。

---

### 两类问题（§4.1）

| 类型 | $\alpha$ | 终止 | 有限总代价的机制 |
|------|----------|------|------------------|
| **SSP** | 1 | 吸收态 $t$，$J^{*}(t)=0$ | 终止几乎必然（Assumption 4.2.1）→ 随机但期望有限的 horizon |
| **折扣** | $(0,1)$ | 可无物理 $t$ | $\sum_{k=0}^\infty \alpha^k g_k$ 在 $\|g\|$ 有界时收敛；亦可引入人工 $t$ |

**折扣 → SSP（Fig 4.3.1）**：每步以概率 $1-\alpha$ 进入零代价终止态 $t$，否则按 $\alpha p_{ij}(u)$ 转移；故折扣问题可归入 SSP 框架（§4.1 末段）。§4.3 单独列出是因为 Prop 4.3.5 直接在 sup-范数下给出模 $\alpha$ 的压缩，证明较 SSP 的加权范数（Prop 4.2.5）简洁。

---

### 猜想 (4.2)–(4.3) 与 VI

设 $J_N(x)$ 为 $N$ 阶段、零终端代价的最优值。**反向** VI（与 Ch.1 时间索引相反）：

```math
J_{k+1}(x)=\min_{u\in U(x)}\mathbb{E}_w\big[g(x,u,w)+J_k(f(x,u,w))\big],\quad J_0(x)\equiv 0. \qquad \text{(4.1)}
```

在适当假设下（§4.2–4.3 成立）：

1. **极限**：$J^{*}(x)=\lim_{N\to\infty}J_N(x)$；  
2. **Bellman 方程**：$J^{*}(x)=\min_u\mathbb{E}[g+J^{*}(f)]$；  
3. **平稳最优**：逐点取 min 得 $\mu^{*}$，策略 $\{\mu^{*},\mu^{*},\ldots\}$ 最优。

---

### 转移概率记号

- 非终止态 $1,\ldots,n$；终止 $t$。  
- 控 $u\in U(i)$ → 转移 $p_{ij}(u)$，代价 $g(i,u,j)$。  
- **系统等价**：$x_{k+1}=w_k$，$w_k$ 按 $p_{x_k w_k}(u_k)$ 抽样（与 Ch.1 的 $f_k$ 表述等价）。

| 记号 | 含义 |
|------|------|
| $J_\pi(i)$ | 初始态 $i$、策略 $\pi=\{\mu_0,\mu_1,\ldots\}$ 的期望总代价 |
| $J_\mu(i)$ | **平稳**策略 $\{\mu,\mu,\ldots\}$ 的代价 |
| $J^{*}(i)=\min_\pi J_\pi(i)$ | 最优代价 |

---

#### 要点

- 从任意态出发的子问题同构 → 最优性可在平稳策略类中讨论。  
- Q 因子、算子 $T,T_\mu,F$ 为 §4.8–4.9 采样算法奠基。

#### 注意点

- $J_\pi(i)$ 极限存在性在非标准问题中需单独验证（SSP 靠 Assumption 4.2.1；折扣靠 $\alpha<1$）。

---

#### §4.1–§4.5 结构

| 节 | 内容 |
|----|------|
| §4.1 | 问题类、猜想 (4.2)–(4.3)、平稳最优策略 |
| §4.2–4.3 | $T$ 的压缩性；VI 收敛、Bellman 唯一、策略评估、最优性条件 |
| §4.4 | 精确 VI；(4.24) fitted VI；例 4.4.1 误差放大 |
| §4.5 | 精确 PI；乐观 PI（$m_k$ 步评估）；multistep 改进 |

算法 (4.1) 称为 **value iteration（VI）**；**policy iteration（PI）** 交替做策略评估与策略改进（Fig 4.5.1）。

---

## §4.2 Stochastic Shortest Path Problems

> 与 §4.3 平行：结论对应 Prop 4.2.1–4.2.4 与 Prop 4.3.1–4.3.5。SSP 中 $T$ 在 $\|\cdot\|_\infty$ 下未必压缩，需 Assumption 4.2.1 与 Prop 4.2.5 的加权范数；折扣中 Prop 4.3.5 给出 $\|TJ_1-TJ_2\|_\infty\le\alpha\|J_1-J_2\|_\infty$。

### Bellman 方程 (4.4)

对 $i=1,\ldots,n$：

```math
J^{*}(i)=\min_{u\in U(i)}\Big[p_{it}(u)g(i,u,t)+\sum_{j=1}^n p_{ij}(u)\big(g(i,u,j)+J^{*}(j)\big)\Big]. \qquad \text{(4.4)/(4.7)}
```

**三项解释**：

| 项 | 含义 |
|----|------|
| $p_{it}(u)g(i,u,t)$ | 本步直达 $t$ 的期望代价 |
| $\sum_j p_{ij}(u)g(i,u,j)$ | 本步到非终止 $j$ 的阶段代价 |
| $\sum_j p_{ij}(u)J^{*}(j)$ | 下一状态余值（$J^{*}(t)=0$ 不入和） |

---

### VI (4.5)

```math
J_{k+1}(i)=\min_{u\in U(i)}\Big[p_{it}(u)g(i,u,t)+\sum_{j=1}^n p_{ij}(u)\big(g(i,u,j)+J_k(j)\big)\Big]. \qquad \text{(4.5)}
```

---

### Assumption 4.2.1（几乎必然终止）

存在整数 $m$，使**任意**策略下 $m$ 步内未到 $t$ 的概率 $\le\rho<1$：

```math
\rho=\max_\pi\max_{i=1,\ldots,n}P\{x_m\neq t\mid x_0=i,\pi\}<1. \qquad \text{(4.6 系)}
```

则 $P\{x_{km}\neq t\}\le\rho^k\to 0$——视界随机但期望总代价有限。

| 概念 | 定义 |
|------|------|
| **Proper 策略** | 从任意态以概率 1 到达 $t$ |
| **Improper 策略** | 非 proper |

Assumption 4.2.1 ⇔ 所有平稳策略 proper。更弱条件：存在 proper 策略，且 improper 策略在至少一态期望代价 $+\infty$（[BeT89], [Ber12] Ch.3）。

**反例**（无假设）：单态 $1$——停留代价 $a$ 或到 $t$ 代价 $b$；$a=0$ 或 $a<0$ 时 Bellman 无解或多解（[Ber18a] §3.1.1）。

---

### 主要命题（§4.13.1）

| 结论 | SSP | 折扣 |
|------|-----|------|
| VI 收敛：$J_k\to J^{*}$ | Prop 4.2.1 | Prop 4.3.1 |
| Bellman 方程解唯一 | Prop 4.2.2 | Prop 4.3.1 |
| 固定 $\mu$：$J_\mu=T_\mu J_\mu$ 唯一；策略 VI 收敛 | Prop 4.2.3 | Prop 4.3.1 |
| $\mu$ 最优 ⇔ 每态在 Bellman 右端取 min | Prop 4.2.4 | Prop 4.3.1 |
| $T,T_\mu$ 为压缩映射 | Prop 4.2.5（加权范数，$\rho<1$） | Prop 4.3.5（sup-范数，模 $\alpha$） |

Prop 4.2.5 保证 $J_{k+1}=TJ_k$ 收敛；收敛率 $\|J_k-J^{*}\|\le\rho^k\|J_0-J^{*}\|$。

**DP 算子**：

```math
(TJ)(i)=\min_{u\in U(i)}\Big[p_{it}(u)g(i,u,t)+\sum_{j=1}^n p_{ij}(u)\big(g(i,u,j)+J(j)\big)\Big], \qquad \text{(4.8)}
```

```math
(T_\mu J)(i)=p_{it}(\mu(i))g(i,\mu(i),t)+\sum_{j=1}^n p_{ij}(\mu(i))\big(g(i,\mu(i),j)+J(j)\big). \qquad \text{(4.9)}
```

压缩率与最大期望到达 $t$ 的步数 $-m^{*}(i)$ 相关：$v(i)=-m^{*}(i)$，$\|J_k-J^{*}\|\le\rho^k\|J_0-J^{*}\|$。

---

### 例 4.2.1（最大期望终止时间）

$g(i,u,j)\equiv -1$ → 目标为**尽量晚**终止；$-J^{*}(i)$ 为从 $i$ 到 $t$ 的**平均首达时间**。

单策略 $\mu$ 时：

```math
J_\mu(i)=-1+\sum_{j=1}^n p_{ij}(\mu(i))J_\mu(j).
```

---

### Q 因子（SSP）

```math
Q^{*}(i,u)=p_{it}(u)g(i,u,t)+\sum_{j=1}^n p_{ij}(u)\big(g(i,u,j)+J^{*}(j)\big).
```

Q-Bellman（Fig 4.2.2 增广 SSP）：

```math
Q^{*}(i,u)=p_{it}(u)g(i,u,t)+\sum_{j=1}^n p_{ij}(u)\Big(g(i,u,j)+\min_{v\in U(j)}Q^{*}(j,v)\Big). \qquad \text{(4.16 型)}
```

```math
J^{*}(j)=\min_{v\in U(j)}Q^{*}(j,v).
```

Q-VI：将 $J^{*}$ 换 $Q_k$，$\min_v Q_k(j,v)$ 换 $J_k(j)$。

---

### Cost shaping / 时序差分形式

任意 $V=(V(1),\ldots,V(n))$，$V(t)=0$，定义 $\hat J=J^{*}-V$ 与修正代价：

```math
\hat g(i,u,j)=\begin{cases}
g(i,u,j)+V(j)-V(i), & i,j\in\{1,\ldots,n\},\\
g(i,u,t)-V(i), & j=t.
\end{cases} \qquad \text{(4.10)/(4.11)}
```

**变分 Bellman**：$\hat J(i)=\min_u[p_{it}\hat g(i,u,t)+\sum_j p_{ij}(u)(\hat g(i,u,j)+\hat J(j))]$。

| 精确 DP | 近似 DP |
|---------|---------|
| 最优策略不变 | 次优策略对 $V$ **敏感**——cost shaping 可显著改善 FVI/PI |

---

### 与 Ch.1 归约

- §1.3.1：$p_{ij}(u)\in\{0,1\}$ 的确定性最短路；  
- §1.3.3：增广态 $(x_k,k)$ + 合并 $t$ 的 SSP。

---

## §4.3 Discounted Problems

> 折扣问题可归约为 SSP（Fig 4.3.1）；本节 Bellman/VI 形式见 (4.12)、(4.19)。压缩性由 Prop 4.3.5 直接给出；§4.4 误差界中的 $(1-\alpha)$ 因子来自此处。

### Bellman 与 VI

```math
J^{*}(i)=\min_{u\in U(i)}\sum_{j=1}^n p_{ij}(u)\big[g(i,u,j)+\alpha J^{*}(j)\big]. \qquad \text{(4.12)}
```

```math
J_{k+1}(i)=\min_{u\in U(i)}\sum_{j=1}^n p_{ij}(u)\big[g(i,u,j)+\alpha J_k(j)\big]. \qquad \text{(4.19)}
```

---

### 主要结果（Prop 4.3.1–4.3.5）

| 结果 | 内容 |
|------|------|
| VI 收敛 | $J_k\to J^{*}$ |
| Bellman 唯一解 | $J^{*}$ 满足 (4.12) 且唯一 |
| 策略评估 | $J_\mu(i)=\sum_j p_{ij}(\mu(i))[g(i,\mu(i),j)+\alpha J_\mu(j)]$ 唯一 |
| 最优性 | $\mu$ 最优 ⇔ 每态在 (4.12) 达 min |
| **压缩** | $\|TJ_1-TJ_2\|_\infty\le\alpha\|J_1-J_2\|_\infty$（Prop 4.3.5） |

结论与 §4.2 命题组对应；§4.4 将说明：$T$ 虽压缩，但「VI + 加权最小二乘投影」复合映射未必压缩（例 4.4.1）。

**算子**：

```math
(TJ)(i)=\min_{u}\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha J(j)\big], \qquad \text{(4.13)}
```

```math
(T_\mu J)(i)=\sum_j p_{ij}(\mu(i))\big[g(i,\mu(i),j)+\alpha J(j)\big]. \qquad \text{(4.14)}
```

**策略线性方程**：$(I-\alpha P_\mu)J_\mu=g_\mu$（矩阵形式）。

---

### Q 因子与 Q-VI (4.16)–(4.17)

```math
Q^{*}(i,u)=\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha J^{*}(j)\big],\quad J^{*}(i)=\min_u Q^{*}(i,u).
```

```math
Q^{*}(i,u)=\sum_j p_{ij}(u)\Big[g(i,u,j)+\alpha\min_{v\in U(j)}Q^{*}(j,v)\Big]. \qquad \text{(4.16)}
```

```math
Q_{k+1}(i,u)=\sum_j p_{ij}(u)\Big[g(i,u,j)+\alpha\min_{v}Q_k(j,v)\Big]. \qquad \text{(4.17)}
```

Q-VI 为 Q-learning 的模型基础。

---

### 折扣 cost shaping

```math
\hat g(i,u,j)=g(i,u,j)+\alpha V(j)-V(i).
```

变分 Bellman：$\hat J=J^{*}-V$ 满足 $\hat J(i)=\min_u\sum_j p_{ij}(u)[\hat g(i,u,j)+\alpha\hat J(j)]$。

---

### 例 4.3.1（资产出售）

每步报价 $i\in\{v_1,\ldots,v_n\}$，概率 $p(j)$；接受则获得 $(1+r)^{-k}i_k$。折扣因子 $\alpha=1/(1+r)$。

```math
J^{*}(i)=\max\Big\{i,\;\frac{1}{1+r}\sum_{j=1}^n p(j)J^{*}(j)\Big\}.
```

临界值 $c=\frac{1}{1+r}\sum_j p(j)J^{*}(j)$：**当且仅当** $i>c$ 时出售。相关报价/未知分布 → 信念态上的部分信息 DP（Ch.2–3 近似）。

---

### SSP vs 折扣对照

| 维度 | SSP | 折扣 |
|------|-----|------|
| 终止 | 物理/问题 $t$ | 可选人工 $1-\alpha$ 吸收 |
| 压缩 | 加权范数 $\rho<1$（依赖 $m,\rho$） | $\|\cdot\|_\infty$ 模 $\alpha$ |
| Bellman | 含 $p_{it}g(i,u,t)$ | 无直达 $t$ 项 |
| 归约 | 原生 | → SSP（Fig 4.3.1） |

---

## §4.4 Exact and Approximate Value Iteration

### 精确 VI (4.18)–(4.19)

每轮 $J_{k+1}=TJ_k$。Prop 4.2.1 / 4.3.1：对任意初值 $J_0$ 有 $J_k\to J^{*}$。$n$ 大时每轮 $O(n|U|)$，引出 fitted VI。

---

### 近似 / Fitted VI

沿用 §3.3：初值 $\tilde J_0$，迭代 $\tilde J_{k+1}\approx T\tilde J_k$。在样本态 $i^s$ 上回归 $(T\tilde J_k)(i^s)$：

```math
r_{k+1}\in\arg\min_r\sum_{s=1}^q\big(\tilde J(i^s,r)-\beta^s\big)^2,\quad \beta^s=(T\tilde J_k)(i^s). \qquad \text{(4.24)}
```

(4.24) 中 $\beta^s$ 由**第 $k$ 轮**的 $\tilde J_k$ 经 $T$ 生成，再拟合得 $\tilde J_{k+1}$；与精确 VI 的 $J_{k+1}=TJ_k$ 不同，中间含回归误差。

| | 精确 VI | Fitted VI |
|--|---------|-----------|
| 更新 | $J_{k+1}=TJ_k$ | $\tilde J_{k+1}=\mathcal{R}(T\tilde J_k)$（$\mathcal{R}$：加权 LS） |
| 压缩性 | $T$ 压缩（Prop 4.2.5 / 4.3.5） | 「VI + 投影」未必压缩 |
| 收敛 | Prop 4.2.1 / 4.3.1 | 无一般保证；例 4.4.1 中 $\tilde J_k\to\infty$ |

---

### 理想误差界 (4.22)

若对所有 $k,i$：

```math
\big|\tilde J_{k+1}(i)-\min_u\sum_j p_{ij}(u)[g(i,u,j)+\alpha\tilde J_k(j)]\big|\le\delta,
```

则渐近：

```math
\|\tilde J_k-J^{*}\|_\infty\le\frac{\delta}{1-\alpha},\qquad
\|J_{\tilde\mu_k}-J^{*}\|_\infty\le\frac{2\delta}{(1-\alpha)^2}.
```

**问题**：自然 LS-FVI **未必**满足 (4.22)。

---

### 例 4.4.1（FVI 误差放大）— 详述

**设定**（Fig 4.4.1）：两态 $1,2$，单策略；确定转移 $1\to 2$，$2\to 2$；零代价。

```math
J(1)=\alpha J(2),\quad J(2)=\alpha J(2)\ \Rightarrow\ J^{*}(1)=J^{*}(2)=0.
```

精确 VI：$J_{k+1}(1)=\alpha J_k(2)$，$J_{k+1}(2)=\alpha J_k(2)$。

**近似子空间** $S=\{(r,2r)\mid r\in\mathbb{R}\}$（含 $J^{*}$）。$\tilde J_k=(r_k,2r_k)$。

1. $T\tilde J_k=(\alpha\tilde J_k(2),\alpha\tilde J_k(2))=(2\alpha r_k,2\alpha r_k)$。  
2. 加权 LS：$\xi_1,\xi_2>0$，

```math
r_{k+1}\in\arg\min_r\big[\xi_1(r-2\alpha r_k)^2+\xi_2(2r-2\alpha r_k)^2\big]
\ \Rightarrow\ r_{k+1}=\alpha\zeta r_k,\quad
\zeta=\frac{2(\xi_1+2\xi_2)}{\xi_1+4\xi_2}>1. \qquad \text{(4.23)}
```

3. 若 $\alpha>1/\zeta$，则 $|r_k|\to\infty$，$\tilde J_k\to\infty$（虽 $J^{*}\in S$）。

| 权重 | 后果 |
|------|------|
| $\xi_1=\xi_2=1$ → $\zeta=6/5$ | $\alpha\in(5/6,1)$ 时发散 |
| $\xi_2\gg\xi_1$（稳态加权） | $\zeta\to 1$，$\alpha\zeta<1$ → 可收敛 |

**Fig 4.4.2**：加权投影将 $T\tilde J_k$ 投到 $(r,2r)$；$\alpha$ 接近 1 时投影点可远离 $J^{*}$。

---

### 稳态加权启发

按长期访问频率加权回归：在「好」启发策略下仿真至稳态，记录态 $i^s$ 做 (4.24)。**无一般理论保证**（[Ber12] §6.3）；例 4.4.1 中应令 $\xi_2\gg\xi_1$。

---

#### 要点

- 近似 VI 的病理来自 **非压缩复合映射**，非 Bellman 本身。  
- 样本态选择、权重与探索同等重要。

#### 注意点

- SSP 定性类似；界中常出现 $(1-\alpha)$ 型因子时来自折扣分析。

---

### VI 与 PI：无函数逼近误差下的两种算法模板

有限状态、tabular 表示下，Bellman 最优控制有两条经典精确迭代路线（详见 [`00-algorithm-taxonomy.md`](00-algorithm-taxonomy.md) §4–6）：

| | **值迭代（VI）** | **策略迭代（PI）** |
|--|------------------|-------------------|
| 每轮核心 | $J_{k+1}=TJ_k$ | 评估 $J_{\mu_k}=T_{\mu_k}J_{\mu_k}$ → 改进 $\mu_{k+1}$ |
| 策略何时出现 | 收敛后贪心，或每轮隐式 | 每轮显式更新 $\mu_k$ |
| 典型代价 | 每轮 $O(n\|U\|)$，轮数可多 | 评估 $O(n^3)$ 级，轮数常少 |
| 近似版 | Fitted VI（§4.4） | 近似 PI / Actor–Critic（§4.6–4.7） |

二者同属**值空间** paradigm；差别在 **VI 型 vs PI 型**算法模板。Fitted VI 在近似 regime 下等价于 $m_k=1$ 的乐观 PI + 回归（§4.5.2）。

---

## §4.5 Policy Iteration

### 算法框架（Fig 4.5.1）

每轮：**策略评估**（求 $J_{\mu_k}$，满足 $J_{\mu_k}=T_{\mu_k}J_{\mu_k}$）→ **策略改进**（用 $J_{\mu_k}$ 做一步前瞻取 min）→ 若 $J_{\mu_{k+1}}=J_{\mu_k}$ 则停止。

$J_{\mu_k}$ 为策略 $\mu_k$ 的代价，一般 $J_{\mu_k}\ge J^{*}$。Rollout = 单次评估+改进；PI = 重复至最优。

---

### §4.5.1 Exact Policy Iteration

**SSP 评估**（Prop 4.2.3）：

```math
J_{\mu_k}(i)=p_{it}(\mu_k(i))g(i,\mu_k(i),t)+\sum_j p_{ij}(\mu_k(i))\big[g(i,\mu_k(i),j)+J_{\mu_k}(j)\big].
```

**SSP 改进**：

```math
\mu_{k+1}(i)\in\arg\min_u\sum_j p_{ij}(u)\big[g(i,u,j)+J_{\mu_k}(j)\big].
```

**折扣评估 (4.25)**：

```math
J_{\mu_k}(i)=\sum_j p_{ij}(\mu_k(i))\big[g(i,\mu_k(i),j)+\alpha J_{\mu_k}(j)\big].
```

**折扣改进 (4.26)**：

```math
\mu_{k+1}(i)\in\arg\min_u\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha J_{\mu_k}(j)\big].
```

**终止**：$J_{\mu_{k+1}}=J_{\mu_k}$ ⇒ $\mu_k$ 最优。

**Prop 4.5.1**：SSP 与折扣下，$J_{\mu_{k+1}}(i)\le J_{\mu_k}(i)$（单调改进），有限策略集下**有限步**终止于最优策略。

**改进证明梗概（折扣）**：令 $J_N$ 为「前 $N$ 步用 $\tilde\mu$，之后用 $\mu$」的代价；由 (4.26) 得 $J_1\le J_\mu$，归纳 $J_{N+1}\le J_N\le J_\mu$，令 $N\to\infty$ 得 $J_{\tilde\mu}\le J_\mu$。

---

### 例 4.5.1（寻宝 / Treasure Hunting）

态 $i$ = 未找到宝藏数；$0$ 为终止。搜索代价 $c$/天；找到 $m$ 个的概率 $p(m|i)$，期望 $r(i)=\sum_m mp(m|i)$ 随 $i$ 单调增。

```math
J^{*}(i)=\max\Big\{0,\;r(i)-c+\sum_{m=0}^i p(m|i)J^{*}(i-m)\Big\}.
```

- $\mu_0$：永不搜索 → $J_{\mu_0}\equiv 0$。  
- $\mu_1$：$r(i)>c$ 时搜索。  
- $\mu_2$：与 $\mu_1$ 相同 → **两轮 PI 即终止**，$\mu_2$ 最优。

---

### §4.5.2 Optimistic / Multistep Lookahead PI

精确 PI 需解 $J_{\mu_k}=T_{\mu_k}J_{\mu_k}$。**乐观 PI**：用 $m_k$ 次 $T_{\mu_k}$ 迭代近似 $J_{\mu_k}$；$m_k$ 有限时称为 optimistic（§4.5.2）。

**改进 (4.33)**：$\mu_k(i)\in\arg\min_u\sum_j p_{ij}(u)[g(i,u,j)+\alpha J_k(j)]$。

**乐观评估 (4.34)**：$\hat J_{k,0}=J_k$；对 $m=0,\ldots,m_k-1$，

```math
\hat J_{k,m+1}(i)=\sum_j p_{ij}(\mu_k(i))\big[g(i,\mu_k(i),j)+\alpha\hat J_{k,m}(j)\big];
```

令 $J_{k+1}=\hat J_{k,m_k}$。

| $m_k$ | 含义 |
|-------|------|
| $m_k=1$ | 与 VI 本质相同（原文：essentially identical to VI） |
| $m_k\to\infty$ | 精确 PI |
| $1<m_k<\infty$ | VI 与精确 PI 之间的折中 |

**Prop 4.5.2**（折扣）：$J_k\to J^{*}$，$J_{\mu_k}\to J^{*}$。Fitted VI（§4.4）= $m_k=1$ 的乐观 PI + 回归；受例 4.4.1 型误差放大影响。

**Multistep 改进**：精确评估 $J_{\mu_k}$ 后，解 $\ell$ 阶段问题（终端 $J_{\mu_k}$），取首控为 $\mu_{k+1}$；评估近似时较长前瞻可部分补偿评估误差（§4.5.2）。

### §4.5.3 Policy Iteration for Q-Factors

```math
Q_{\mu_k}(i,u)=\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha Q_{\mu_k}(j,\mu_k(j))\big]. \qquad \text{(4.35)}
```

```math
\mu_{k+1}(i)\in\arg\min_{u\in U(i)}Q_{\mu_k}(i,u). \qquad \text{(4.36)}
```

与代价空间 PI 数学等价；Fig 4.2.2 增广问题视角。异步/乐观 Q-PI 有收敛复杂性（[WiB93]；[BeY12], [YuB13a] 解决折扣/SSP）。

---

### 近似 PI vs 近似 VI

| 方法 | 稳定性（例 4.4.1 型） | 界（同 $\delta$ 水平） |
|------|----------------------|----------------------|
| 近似 VI | $\tilde J_k$ 可 $\to\infty$ | $\delta/(1-\alpha)$ 型（若 (4.22) 成立） |
| 近似 PI | $\{J_{\mu_k}\}$ 有界 | 常更优（§4.6.3） |

---

## §4.6 Approximation in Value Space — Performance Bounds

### 框架 (4.37)

一步前瞻：

```math
\tilde\mu(i)\in\arg\min_{u\in U(i)}\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha\tilde J(j)\big]. \qquad \text{(4.37)}
```

$\tilde J$ 来源同 Ch.2（Fig 4.6.1）：参数化、问题近似、rollout、聚合、MCTS 等。

**$\ell$ 步前瞻**：$\tilde\mu=\hat\mu_0$，其中 $\hat\mu_0,\ldots,\hat\mu_{\ell-1}$ 最小化

```math
\min_{\mu_0,\ldots,\mu_{\ell-1}}\mathbb{E}\Big\{\sum_{k=0}^{\ell-1}\alpha^k g(i_k,\mu_k(i_k),i_{k+1})+\alpha^\ell\tilde J(i_\ell)\Big\}.
```

等价：$\tilde J\leftarrow T^{\ell-1}\hat J$，再一步 min —— 或 $\tilde\mu$ 满足 $T_{\tilde\mu}(T^{\ell-1}\tilde J)=T(T^{\ell-1}\tilde J)$。

**近似 PI 序列** $\mu_0,\ldots,\mu_m$：每 $\mu_k$ 得 $\tilde J_{\mu_k}$ → 改进 → 末 $\tilde J_{\mu_m}$ 作前瞻。

---

### §4.6.1 Limited Lookahead — Prop 4.6.1

**(a) $\ell$ 步前瞻**：

```math
\|J_{\tilde\mu}-J^{*}\|_\infty\le\frac{2\alpha^\ell}{1-\alpha}\|\tilde J-J^{*}\|_\infty. \qquad \text{(4.38)/(4.99)}
```

**常数平移加强**：$\|\tilde J-J^{*}\|$ 可换为 $\min_\beta\max_i|\tilde J(i)+\beta-J^{*}(i)|$（$J_{\tilde\mu}$ 对 $\tilde J$ 加常数不变）。

**(b) 控集限制** $ \underline{U}(i)\subset U(i)$，$\hat J\le\tilde J+c$：

```math
J_{\tilde\mu}(i)\le\hat J(i)+\frac{c}{1-\alpha}. \qquad \text{(4.41)}
```

**例 4.6.1（界紧性）**（Fig 4.6.2）：两态，$\tilde J(1)=-\epsilon$，$\tilde J(2)=\epsilon$；停留策略 $\mu$ 为一步前瞻最优；$J_\mu(1)=\frac{2\alpha\epsilon}{1-\alpha}=\|\tilde J-J^{*}\|\cdot\frac{2\alpha}{1-\alpha}$——(4.38) 取等（$\ell=1$）。

**注意**：界改善 ≠ 多步 lookahead **一定**优于一步（cf. Ch.2 例 2.2.1）；说的是**界**随 $\ell$ 变紧。

---

### §4.6.2 Rollout

**纯 rollout**：$\tilde J=J_\mu$（基策略/heuristic）。

```math
\tilde\mu(i)\in\arg\min_{u\in\underline{U}(i)}\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha J_\mu(j)\big].
```

**Prop 4.6.2**：若 $\mu(i)\in\underline{U}(i)$，则 $J_{\tilde\mu}\le J_\mu$（一步 PI）。

**Parallel rollout**（例 4.6.2）：基策略 $\mu_1,\ldots,\mu_M$，

```math
\tilde J(i)=\min_m J_{\mu_m}(i)\ \Rightarrow\ J_{\tilde\mu}(i)\le\min_m J_{\mu_m}(i).
```

**截断 rollout**（Fig 4.6.3）：$\ell$ 步前瞻 + $m$ 步 $\mu$ rollout + 终端 $\tilde J$。

**Prop 4.6.3(a)**：

```math
\|J_{\tilde\mu}-J^{*}\|_\infty\le\frac{2\alpha^\ell}{1-\alpha}\|T_\mu^m\tilde J-J^{*}\|_\infty.
```

**(b)** 条件 (4.42)–(4.43)：终端近似接近 $J_\mu$ 时，$\tilde\mu$ 几乎改进 $\mu$（within $2c/(1-\alpha^2)$）。

**实践**：TeG96 西洋双陆棋、AlphaGo（MCTS + NN 终端/策略）——终端 NN 近似可能**破坏**相对基策略的改进保证。

---

### §4.6.3 Approximate PI — Prop 4.6.4–4.6.5

**误差模型**：

```math
\max_i|\tilde J_{\mu_k}(i)-J_{\mu_k}(i)|\le\delta, \qquad \text{(4.44)}
```

```math
\max_i\Big[\sum_j p_{ij}(\mu_{k+1}(i))[\cdots+\alpha\tilde J_{\mu_k}(j)]-
\min_u\sum_j p_{ij}(u)[\cdots+\alpha\tilde J_{\mu_k}(j)]\Big]\le\epsilon. \qquad \text{(4.45)}
```

**Prop 4.6.4**（折扣，[BeT96] §6.2.2）：

```math
\limsup_{k\to\infty}\max_i|J_{\mu_k}(i)-J^{*}(i)|\le\frac{\epsilon+2\alpha\delta}{(1-\alpha)^2}.
```

**Fig 4.6.4**：早期单调进步 → 进入宽度 $\approx(\epsilon+2\alpha\delta)/(1-\alpha)^2$ 的**振荡带**（界常悲观）。

**Prop 4.6.5**：若策略**聚合收敛** $\mu_{k+1}=\mu_k=\tilde\mu$（Ch.5 聚合常见）：

```math
\max_i|J_{\tilde\mu}(i)-J^{*}(i)|\le\frac{\epsilon+2\alpha\delta}{1-\alpha}.
```

**对比 4.4.1**：$\{J_{\mu_k}\}$ 有界，无 $\tilde J\to\infty$ 类发散；策略有限 ⇒ 代价序列有界。

---

## §4.7 Simulation-Based PI with Parametric Approximation

> 本节主体为**仿真 + 参数化近似 PI**（§4.5 的实现）；§4.7.1 的 Actor–Critic 是对 PI「评估–改进」两步的 RL 命名，非独立 AC 算法族。

### §4.7.1 Self-Learning and Actor–Critic

**Self-learning**（§4.7.1）：仿真近似 PI——由策略轨迹经验改进策略；不构造转移模型（非系统辨识）。

| 角色 | 功能 |
|------|------|
| **Critic** | **策略评估**：对当前 $\mu_k$ 近似 $J_{\mu_k}$ 或 $Q_{\mu_k}$（$\tilde J_{\mu_k}$、$\tilde Q_{\mu_k}$）；仿真 + LS/TD |
| **Actor** | **策略改进**：用 Critic 的 $\tilde J_{\mu_k}$（或 $\tilde Q_{\mu_k}$）做一步前瞻 min 得 $\mu_{k+1}$；或策略空间回归（§2.1.3） |

每轮（原文 (a)(b)）：

(a) **Critic**：在 $\mu_k$ 下采样，回归得 $\tilde J_{\mu_k}$（policy evaluation；非对 $\mu_k$ 本身做优劣判定）。

(b) **Actor**：给定 $\tilde J_{\mu_k}$，

```math
\mu_{k+1}(i)\in\arg\min_u\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha\tilde J_{\mu_k}(j)\big].
```

或在样本态 $i^s$ 上求 $\arg\min_u[\cdots]$，再推广为完整策略（§2.1.3）。

Critic 输出值函数（$J$ 或 $Q$）；Actor 输出策略（$\mu$ 或 $\mu(i,\theta)$）。二者对应 PI 的评估–改进两步（§4.5、Fig 4.7.1–4.7.2）。

---

### §4.7.2 Model-Based 变体（Fig 4.7.1）

知 $p_{ij}(u)$：

**评估 (4.47)**：

```math
r\in\arg\min_r\sum_{s=1}^q\big(\tilde J_\mu(i^s,r)-\beta^s\big)^2.
```

$\beta^s$：从 $i^s$ 出发、用 $\mu$ 仿真 $N$ 步折扣代价 + 终端 $\alpha^N\hat J(i_N)$。

**改进 (4.48)**：$\tilde\mu(i)\in\arg\min_u\sum_j p_{ij}(u)[g+\alpha\tilde J(j,r)]$。

**增量梯度**：

```math
r_{k+1}=r_k-\gamma^k\nabla_{r_k}\big(\tilde J(i^{s_k},r_k)-\beta^{s_k}\big)^2.
```

**轨迹复用**：长轨迹 $(i_0,i_1,\ldots,i_N)$ 可为多个起始态提供 tail 样本。

**Bias–variance**：短轨迹 → 偏差大（终端 $\hat J$ 误差）、方差小；长轨迹相反。衔接 TD($\lambda$)、LSTD($\lambda$)、LSPE($\lambda$)（§4.9）。

---

### §4.7.3 Model-Free 变体（Fig 4.7.2）

**Model-free** 指改进 (4.50) 不需显式 $p_{ij}(u)$；$\beta^s$ 仍由 simulator 生成（与 §4.7.2 相同）。

**Q 评估 (4.49)**：

```math
r\in\arg\min_r\sum_{s=1}^q\big(\tilde Q_\mu(i^s,u^s,r)-\beta^s\big)^2.
```

$\beta^s$：$(i^s,u^s)$ 首步后按 $\mu$ 仿真 $N$ 步的样本 Q。

**改进 (4.50)**：$\tilde\mu(i)\in\arg\min_u\tilde Q_\mu(i,u,r)$。

**探索缺陷**：轨迹复用后多为 $(i,\mu(i))$，$(i,u)$、$u\neq\mu(i)$ 欠采样。

**两阶段**：先 $\tilde J_\mu$（4.47），再回归近似 $\sum_j p_{ij}(u)[g+\alpha\tilde J_\mu(j,r)]$ → 值+策略双近似，可轨迹复用。

---

### §4.7.4 Implementation Issues

| 议题 | 要点 |
|------|------|
| **架构** | 线性 → 闭式 LS；NN → 非线性优化 |
| **Cost shaping** | $\hat g=g+\alpha V(j)-V(i)$；$V\approx J^{*}$ 或 $J_{\mu_k}$，每轮可更新 |
| **探索** | 稳态加权、多初始态、短轨迹 + 终端近似；**行为策略** vs **目标策略** → off/on-policy |
| **$\epsilon$-greedy** | 小概率随机控以增强探索 |
| **振荡** | [Ber96]：策略循环重复（Fig 4.6.4）；聚合等特殊结构可收敛（Prop 4.6.5，Ch.5） |

**探索–利用**：行为策略混合、Russo–Van Roy [RuV16] 等；与 Ch.5 §5.3.4–5.3.5 衔接。

---

## §4.8 Q-Learning

> **定位**（§4.8 开篇）：Q-learning 是对 $Q^*$ 的**随机 VI**（*"stochastic version of VI"*），直接更新最优 $Q$，**避免 PI 对 successive policies 的多次策略评估**。与 §4.5.3 / §4.7.3 的 **Q-PI**（评估 $Q_\mu$ → 改进 $\mu$）算法结构不同。

| | 对象 | 结构 | 节 |
|--|------|------|-----|
| PI | $J_\mu$ | 外环 $\{\mu_k\}$：评估 → 改进 | §4.5、§4.7.2 |
| **Q-PI** | $Q_\mu$ | 同上，(4.35)–(4.36) / (4.49)–(4.50) | §4.5.3、§4.7.3 |
| **Q-learning** | $Q^*$ | $Q_{k+1}\approx FQ_k$（(4.51)–(4.53)），无显式 $\mu_k$ 外环 | §4.8 |

$Q(i,u)$ 含动作 → 改进 $\arg\min_u Q(i,u)$ 不需显式 $p_{ij}$（§4.7.3）；Q-learning 更新只需 $(i,u,g,j)$ 样本，不必手算 (4.51) 中的期望。

**非 PI 循环**：Q-learning **不**做「固定 $Q$ → 求 $\mu$ → 固定 $\mu$ → 拟合 $Q_\mu$」两阶段。每步用当前 $Q_k$ 构造 Bellman 目标 $g+\alpha\min_v Q_k(j,v)$ 更新 $Q$；$\mu(i)=\arg\min_u Q(i,u)$ 随 $Q$ 同步变化，无显式 $\mu_k$ 外环。收敛后在线控：$\mu^{*}(i)=\arg\min_u Q^{*}(i,u)$（同 (3.30)）。

### Q-Bellman 与算子 $F$

```math
Q^{*}(i,u)=\sum_j p_{ij}(u)\Big[g(i,u,j)+\alpha\min_{v\in U(j)}Q^{*}(j,v)\Big].
```

```math
(FQ)(i,u)=\sum_j p_{ij}(u)\Big[g(i,u,j)+\alpha\min_{v}Q(j,v)\Big]. \qquad \text{(4.51)}
```

$F$ 为模 $\alpha$ 压缩 → $Q_{k+1}=FQ_k\to Q^{*}$。

---

### Watkins Q-learning（随机 VI）

采样 $(i_k,u_k)$，$j_k\sim p_{i_k\cdot}(u_k)$：

```math
Q_{k+1}(i,u)=(1-\gamma^k)Q_k(i,u)+\gamma^k(F_k Q_k)(i,u), \qquad \text{(4.52)}
```

```math
(F_k Q_k)(i,u)=\begin{cases}
g(i_k,u_k,j_k)+\alpha\min_v Q_k(j_k,v), & (i,u)=(i_k,u_k),\\
Q_k(i,u), & \text{否则}.
\end{cases} \qquad \text{(4.53)}
```

**无模型**：仅需 $(i,u,g,j)$ 样本。

**更新方式**：Watkins 版为**增量随机近似**——每样本更新一个 $(i_k,u_k)$，非批量收集 $\beta^s$ 再 LS 回归。

| 方法 | 目标 | 更新 | 节 |
|------|------|------|-----|
| **Watkins Q-learning** | $Q^{*}$，$FQ_k$ | 单样本 SA (4.52)–(4.53) | §4.8 |
| **拟合 Q / FQI** | Q-Bellman 右端 | 批量 $\beta^s$，LS (3.28) | Ch.3 §3.4 |
| **Q-PI + LS** | $Q_\mu$ 的 MC 样本 | (4.49)–(4.50)，显式 $\mu_k$ 外环 | §4.7.3 |

三者共用 Bellman 型目标（$g+\alpha\min Q$），但 Watkins 追 $Q^{*}$ 且为 VI 结构；后两者含回归批处理，Q-PI 保留 PI 外环。

---

### 收敛条件（标准 SA）

| 条件 | 含义 |
|------|------|
| 各 $(i,u)$ 无限访问 | 探索充分 |
| $j_k$ 独立采样 | 给定 $(i_k,u_k)$ |
| $\sum_k\gamma^k=\infty$，$\sum_k(\gamma^k)^2<\infty$ | 如 $\gamma^k=c_1/(k+c_2)$ |

⇒ $Q_k\to Q^{*}$ w.p.1（[Tsi94], [BeT96], [Ber12] §6.1.4）。嵌入异步 SA + 异步 DP（[Ber82], [BeT89]）。

---

### Off-policy 与函数逼近

表格 Q-learning：**off-policy**（行为策略 ≠ greedy $\min_v Q$ 目标仍收敛，Prop 条件见上表）。

**函数逼近**：表格情形有收敛证；线性/NN $\tilde Q$ 无 Prop 4.6 型界，收敛一般不保证（§4.9 deadly triad）。Ch.3 (3.28)、§4.7.3 (4.49) 的批量 LS 与 Watkins 增量 SA 是不同实现路径。

**线性/NN + 极端乐观 Q-PI** → SARSA 等（§4.8 末）；§4.8 末段将参数化 Watkins 与 approximate PI 衔接。

---

### Optimistic Q-PI / SARSA 梗概

特征 $\tilde Q(i,u,r)=\phi(i,u)'r$。每步：

1. 转移 $(i_k\to i_{k+1})$；  
2. $u_{k+1}=\arg\min_u\tilde Q(i_{k+1},u,r_k)$（可加 $\epsilon$-探索）；  
3. $r_{k+1}=r_k-\gamma^k\phi(i_k,u_k)q_k$，

```math
q_k=\phi(i_k,u_k)'r_k-\alpha\phi(i_{k+1},u_{k+1})'r_k-g(i_k,u_k,i_{k+1}).
```

**SARSA**：on-policy（用 $u_{k+1}$）；**Q-learning**：off-policy（用 $\min_v Q(j,v)$）。函数逼近 + 极端乐观更新 → 行为复杂，无 Prop 4.6 型界。

SSP 版 Q-learning 见 [BeT96], [Ber12]；详 Ch.5 §5.4。

---

## §4.9 Additional Methods — Temporal Differences

本节摘要：**线性**架构下仿真策略评估；可跳过不影响主线。

---

### 策略评估目标

折扣：$J_\mu=T_\mu J_\mu$，

```math
(T_\mu J)(i)=\sum_j p_{ij}(\mu(i))\big[g(i,\mu(i),j)+\alpha J(j)\big]. \qquad \text{(4.54)/(4.55)}
```

近似流形 $\mathcal{M}=\{\tilde J(\cdot,r)\}$。

---

### 三种抽象途径

| 途径 | 做法 |
|------|------|
| (a) 直接 | $\Pi(J_\mu)$ 投影到 $\mathcal{M}$ |
| (b) $N$ 步 VI + 投影 | $\Pi(T_\mu^N\hat J)$ |
| (c) 投影方程 | $J_\mu=\Pi(T_\mu J_\mu)$ 的近似解 |

线性 $\mathcal{M}=\{\Phi r\}$，加权范数 $\|J\|_\xi^2=\sum_i\xi_i J(i)^2$。

**Monte Carlo 投影 (4.61)–(4.65)**：按 $\xi$ 采样 $i^s$，$\beta^s=J(i^s)+$ 噪声，

```math
r=\Big(\sum_s\phi(i^s)\phi(i^s)'\Big)^{-1}\sum_s\phi(i^s)\beta^s.
```

等价 LS (4.66)。长期样本频率 $\to\xi_i$（(4.67)）即可，不必预设 $\xi$。

---

### §4.7.2 的投影视角 (4.68)

```math
\tilde J_\mu\approx\Pi(T_\mu^N\hat J).
```

$N$ 大或 $\hat J\approx J_\mu$、$q$ 大 ⇒ $\tilde J_\mu\approx\Pi(J_\mu)$（相对 $\|\cdot\|_\xi$ 最优近似）。

---

### TD($\lambda$)、LSTD($\lambda$)、LSPE($\lambda$)

**投影方程 (4.69)**：

```math
\Phi r=\Pi\big(T_\mu^{(\lambda)}\Phi r\big),
```

```math
(T_\mu^{(\lambda)}J)(i)=(1-\lambda)\sum_{\ell=0}^\infty\lambda^\ell(T_\mu^{\ell+1}J)(i).
```

$\lambda=0$ → 一步 Bellman；$\lambda\to 1$ → 接近 Monte Carlo / $\Pi(J_\mu)$。

**TD(0) 更新**：

```math
J(i_k)\leftarrow J(i_k)+\gamma^k\big(g_k+\alpha J(i_{k+1})-J(i_k)\big). \qquad \text{(4.79 型)}
```

**时序差分 (4.77)**：

```math
d_s(r)=\phi(i^s)'r-g(i^s,i^{s+1})-\alpha\phi(i^{s+1})'r.
```

**LSTD(0)**：将 (4.74) 化为 $Cr=d$ 一次矩阵求逆；**LSPE(0)**：增量版 (4.73)，可 hot start 下一策略；**TD(0)**：单样本、无逆矩阵，简单但慢且 fragile。

---

### Bias–variance（Fig 4.9.1）

间接 法（TD）解 $\Phi r_\lambda^{*}\neq\Pi(J_\mu)$ 一般；界 (4.80)：

```math
\|J_\mu-\Phi r_\lambda^{*}\|_\xi\le p\|J_\mu-\Pi J_\mu\|_\xi,\quad
p=\frac{\sqrt{1-\alpha^{2\lambda}}}{1-\alpha^\lambda},\quad
\alpha_\lambda=\frac{\alpha(1-\lambda)}{1-\alpha\lambda}.
```

$\lambda\to 1$：$r_\lambda^{*}\to\Pi(J_\mu)$（**TD(1)** 视角）；$\lambda$ 小：模拟噪声小、**偏差**大。

**Direct 法**（§4.7.2，$N\to\infty$）→ $\Pi(J_\mu)$。

---

### SARSA vs Q-learning（函数逼近）

| | SARSA | Q-learning |
|---|--------|------------|
| 目标 | $g+\alpha Q(i',u')$，$u'$ 实际选取 | $g+\alpha\min_v Q(i',v)$ |
| On/off | On-policy | Off-policy |
| 与 PI | 极端乐观 Q-PI | 随机 VI |

---

### Deadly triad（[SuB18], [Ber12]）

```text
函数逼近 + bootstrapping + off-policy  ⇒  可能发散
```

**缓解**：投影 TD、梯度 TD、重要性采样修正、限制 off-policy 等。表格 Q-learning 无此三要素同时作用时收敛有保证。

---

### 与 DP 术语

| RL | DP |
|----|-----|
| Backup | 单态 $(T_\mu J)(i)$ 或 $(TJ)(i)$ |
| Sweep | 全 $i$ 更新 |
| Episode | 到终止或截断的轨迹 |

---

## §4.10 Exact and Approximate Linear Programming

### 精确 LP

$J^{*}$ 为满足 (4.81) 约束的**分量最大**向量：

```math
J(i)\le\sum_j p_{ij}(u)\big[g(i,u,j)+\alpha J(j)\big],\quad \forall i,u. \qquad \text{(4.81)}
```

```math
\max\sum_{i=1}^n J(i)\quad\text{s.t. (4.81)}. \qquad \text{(4.82)}
```

**验证梗概**：从 $J_0\le TJ_0$ 出发，VI 单调性 ⇒ 任意可行 $J\le J^{*}$；$J^{*}$ 可行且最大。

Fig 4.10.1：两态 LP 几何——$J^{*}$ 在约束多面体「右上角」。若目标 $\sum_i\beta_i J(i)$ 且 $\beta_i>0$，$J^{*}$ 唯一最优。

---

### 近似 LP

```math
\tilde J(i,r)=\sum_{\ell=1}^m r_\ell\phi_\ell(i).
```

在状态子集 $\tilde I\subseteq I$、控子集 $\tilde U(i)\subseteq U(i)$ 上约束：

```math
\max\sum_{i\in\tilde I}\tilde J(i,r)\ \text{s.t.}\ \tilde J(i,r)\le\sum_j p_{ij}(u)[g+\alpha\tilde J(j,r)],\ \forall i\in I,\ u\in\tilde U(i).
```

**困难**：约束数可达 $O(n|U|_{\max})$；需**随机约束采样**（[DFV03], [DFV04]）。变量维 $m$ 可 moderate。

**策略评估 LP**：固定 $\mu$ 时约束更少——可用于近似 PI 中的评估步。

---

## §4.11 Approximation in Policy Space

> **定位**（§4.11 开篇）：值空间近似（§4.4–4.8、§4.7）之外的**简要**替代——直接参数化平稳策略 $\tilde\mu(i,r)$，优化 $r$（Fig 4.11.2）。可与值空间近似**并存**（policy network + value network；§2.1.5、§4.7.3 已有类似方案）。

| 路径 | 优化对象 | 典型训练 | 节 |
|------|----------|----------|-----|
| 值 / Q 空间 | $\tilde J$、$\tilde Q$ → $\arg\min_u$ 得 $\mu$ | VI、PI、Q-learning、§4.7 AC | §4.4–4.8 |
| **策略空间** | $\tilde\mu(\cdot,r)$ 直接 | (4.83) 代价优化 或 (4.90) 监督 | §4.11 |

**与 §4.7 区别**：§4.7 近似 PI——Critic 回归 $J_\mu$/$Q_\mu$，Actor 做 Bellman 型改进；§4.11 **不**走 Bellman 评估–改进环，而是 $\min_r\mathbb{E}[J_{\tilde\mu(r)}]$（仿真代价）或拟合专家/前瞻标签 (4.90)。

### 总述

```math
\min_r\ \mathbb{E}_{i_0}\big[J_{\tilde\mu(r)}(i_0)\big]. \qquad \text{(4.83)}
```

$i_0$ 固定时即 $\min_r J_{\tilde\mu(r)}(i_0)$（确定性问题更简单，§4.11.1）。

**两种训练途径**（§4.11 末段归纳）：

| § | 思路 | 核心式 |
|---|------|--------|
| **4.11.1** | 代价优化（梯度 / 随机搜索） | (4.83)、(4.84)、(4.87) |
| **4.11.2** | 专家 / 标签监督 | (4.90)；标签可来自专家、(4.91) 或 (4.92) |

---

### 例 4.11.1（供应链）

生产–库存–零售链（Fig 4.11.1）：库存 $<r_1$ 则订货至目标 $r_2$；$r=(r_1,r_2)$ 可仿真训练。

---

### 例 4.11.2（通过代价参数化策略）

```math
\tilde\mu(i,r)\in\arg\min_u\sum_j p_{ij}(u)\big[g(i,u,j)+\tilde J(j,r)\big].
```

$\tilde J$ 线性或 NN → **一步前瞻策略类**；$r$ 由 (4.83) 或监督学习确定。

---

### §4.11.1 代价优化：Policy Gradient 与随机搜索

#### 梯度法 (4.84)

$i_0$ 已知：

```math
r_{k+1}=r_k-\gamma^k\nabla_r J_{\tilde\mu(r_k)}(i_0). \qquad \text{(4.84)}
```

随机问题：$\nabla J$ 常不可解析，需 MC/有限差分 → **噪声大、需大量样本**（§4.11.1）；确定性问题则较易。

#### 随机策略与 log-likelihood trick

随机策略：态 $i$ 上 $u\sim p(u|i;r)$。将 $\min_r J_{\tilde\mu(r)}$ 化为 $\min_r\mathbb{E}_{p(z;r)}[F(z)]$（(4.85)–(4.86)）。**Log-likelihood trick**：

```math
\nabla_r\mathbb{E}_{p(\cdot;r)}[F(z)]=\mathbb{E}_{p(\cdot;r)}\big[\nabla_r\log p(z;r)\,F(z)\big].
```

**样本梯度 (4.87)**（增量实现 (4.86)）：

```math
r_{k+1}=r_k-\gamma^k\nabla_r\log p(z^k;r_k)\,F(z^k).
```

**模型无关**：只需轨迹代价 $F(z)$ 与 $\nabla_r\log p(z;r)$，**不需** $\nabla F$（§4.11.1）。

**局限**（原文）：(1) 随机策略极限分布非 atomic 时需从 $p(\cdot;r)$ 抽解；(2) 架构与采样需平衡探索；(3) 梯度估计噪声 → **慢收敛、局部极小、可靠性差**（§4.11.1 末段）。

---

### 例 4.11.3（折扣 REINFORCE）

轨迹 $z=\{i_0,u_0,i_1,u_1,\ldots\}$，

```math
F(z)=\sum_{m=0}^\infty\alpha^m g(i_m,u_m,i_{m+1}).
```

```math
\nabla_r\log p(z;r)=\sum_{m=0}^\infty\nabla_r\log p(u_m|i_m;r).
```

（转移 $\log p_{i_m i_{m+1}}(u_m)$ 与 $r$ 无关时消失。）

**问题**：随机策略极限非原子 ⇒ 需从 $p(\cdot;r)$ 抽解；参数族含确定性策略时可避免。

AlphaGo/AlphaZero：随机策略 + MCTS 沿改进方向调整，**非**经典 policy gradient（§2.4.2、§4.11.1 脚注）。

---

#### Cross-Entropy 随机搜索（Fig 4.11.3）

Policy gradient 的**替代**（§4.11.1）：同样通过采样迭代改进 $r$，但**不需**随机策略、**不需**代价可微、对局部极小较不敏感；收敛率保证有限，依赖领域启发（原文）。

**流程梗概**：

1. 在当前 $r_k$ 附近（椭球 $E_k$）随机采样多组 $r$；
2. 用 $\tilde\mu(r)$ 仿真代价，**接受**代价较低的样本；
3. $r_{k+1}$ = 接受样本的均值（及协方差更新 $E_{k+1}$）；
4. 重复——$r_k$ 为逐批更优样本的「中心点」（Fig 4.11.3；与进化策略、Salimans et al. [SHC17] 等同族）。

**特点**：实现简单、可并行；Tetris 等域有成功案例（[SzL06], [ThS09]）。与 policy gradient 同属 (4.83) 的采样优化路径。

---

### §4.11.2 Expert Supervised Training

在样本 $(i^s,u^s)$ 上拟合策略（§2.1.3 有限时域版；监督学习框架）：

```math
\min_r\sum_{s=1}^q\big\|u^s-\tilde\mu(i^s,r)\big\|^2. \qquad \text{(4.90)}
```

$u^s$ 来源（§4.11.2）：

| 来源 | 式 |
|------|-----|
| 人/软件专家 | 近最优控 $u^s$ |
| 值空间一步前瞻 | $u^s=\arg\min_u\sum_j p_{ij}(u)[g(i^s,u,j)+\tilde J(j)]$ **(4.91)** |
| Q 近似 | $u^s=\arg\min_u\tilde Q(i^s,u,r)$ **(4.92)** |

(4.91)/(4.92) 生成标签 → (4.90) 策略回归：**先值空间、再策略空间**（§4.11.2）；性能一般不超过所用专家或前瞻策略，但**在线**只需 $\tilde\mu(i,r)$，无 (4.91) 型实时 min（§4.11.2 末段）。

**局限**：(4.90) 监督策略难超专家；Tesauro 等：纯监督 backgammon 不如 TD/rollout（§4.11.2 脚注）。可作 actor 初始化或与自博弈（AlphaZero）对照。

---

## §4.12 Notes and Sources

| 主题 | 入口文献 |
|------|----------|
| 无限时域 DP 理论 | [Ber12], [Ber17], [Ber18a], [Put94] |
| 近似 DP / RL | [BeT96], [Ber10], [Ber15b] |
| SSP 弱假设 | [BeT89], [BeT91] |
| 乐观/异步 PI | [Ber12] Ch.2–3, [Ber18a] §2.5–2.6 |
| Q-learning / TD | [Wat89], [Tsi94], [BeT96], [SuB18], [BBD10] |
| LP 近似 | [DFV03], [DFV04], [DeF04] |
| 探索 | [RuV16], [RVK18] |
| Deadly triad | [SuB18] |

本章参考文献**非 exhaustive**；与 Ch.1–3 交叉引用。

---

## §4.13 Appendix（证明导读）

| 小节 | 内容 | 证明要点 |
|------|------|----------|
| **4.13.1 SSP** | Prop 4.2.1–4.2.5 | Assumption 4.2.1 ⇒ $P\{x_{km}\neq t\}\le\rho^k$；$mK$ 阶段代价夹 $J^{*}$ ⇒ VI 极限；Bellman 唯一性用 VI 复制；$T,T_\mu$ 加权压缩 |
| **4.13.2 折扣** | Prop 4.3.1–4.3.5 | 经 SSP 等价或直证；$\|T J_1-TJ_2\|_\infty\le\alpha\|J_1-J_2\|_\infty$ |
| **4.13.3 PI** | Prop 4.5.1–4.5.2 | 单调 $J_{\mu_{k+1}}\le J_{\mu_k}$；有限策略终止；乐观 PI：平移 $J_0$ 使 $TJ_0\le J_0$，归纳 $J^{*}\le J_k\le T^k J_0$ |
| **4.13.4 界** | Prop 4.6.1–4.6.5 | (a) 三角不等式 + $T,T_{\tilde\mu}$ 压缩；(b) $T_{\tilde\mu}^k$ 与 $c$ 求和几何级数；(4.44)–(4.45) ⇒ 近似 PI 误差递推 |

---

## 本章小结

完整 **paradigm / template 分类** 与全书章节组织见 [`00-algorithm-taxonomy.md`](00-algorithm-taxonomy.md) §5、§8。

| 需求 / 场景 | Paradigm | Template | 推荐节 / 方法 |
|-------------|----------|----------|---------------|
| 理论底座 | — | — | §4.1–4.3 + §4.13 |
| 小 MDP，知模型 | 值空间 | VI 或 PI | 精确 VI / PI |
| 大状态，知模型 | 值空间 | VI 或 PI | §4.4 FVI、近似 PI、§4.10 LP + Ch.5 |
| 仅仿真器 | 值空间 | VI 或 PI | §4.8 Q-learning、§4.9 TD、§4.7 actor–critic |
| 有次优界需求 | 值空间 | 前瞻 / PI | §4.6（Prop 4.6.1–4.6.5；$\alpha\to 1$ 时界变松） |
| 有启发 $\mu$ | 值空间 | PI（单轮） | Rollout（Prop 4.6.2） |
| 直接优化策略 | 策略空间 | — | §4.11 policy gradient |

---

*个人学习笔记；原著 Copyright Bertsekas / Athena Scientific。*

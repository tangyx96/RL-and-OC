# 第 2 章 Approximation in Value Space — 分节笔记

> **文献**：Dimitri P. Bertsekas, *Reinforcement Learning and Optimal Control*（Athena Scientific, 2019 draft）第 2 章。  
> **文本依据**：书稿第 2 章。  
> **定位**：第 1 章给出精确 Bellman 结构；本章在 $J_k^{*}$ 不可算时，讨论**次优控制**如何在「方便实现」与「足够好的性能」之间折中。有限时域为主，思想可延拓至第 4–5 章无限时域。

---

## 章首：为何近似、两大路线与实现维度

### 动机

精确 DP 常不可行，主因是 Bellman 所称的**维度灾难**：状态/阶段一增，计算与存储急剧膨胀。此外，问题结构可能早知，但部分参数要到控前才明确，**在线可用时间**有限。次优方案因此成为全书 RL/近似 DP 的主线。

### 路线一：值空间近似（Approximation in Value Space）

用 $\tilde J_k$ 替代最优 cost-to-go $J_k^{*}$，在 Bellman **结构**不变的前提下选控。随机有限时域的标准形式为：

```math
\tilde\mu_k(x_k) \in \arg\min_{u_k \in U_k(x_k)} \mathbb{E}\big\{ g_k(x_k,u_k,w_k) + \tilde J_{k+1}\big(f_k(x_k,u_k,w_k)\big) \big\}. \qquad \text{(2.1)}
```

(2.1) 右端对固定 $(x_k,u_k)$ 的期望可视为**近似 Q 因子**：

```math
\tilde Q_k(x_k,u_k) = \mathbb{E}\big\{ g_k(x_k,u_k,w_k) + \tilde J_{k+1}(f_k(x_k,u_k,w_k)) \big\},
```

于是 $\tilde\mu_k(x_k)\in\arg\min_{u_k}\tilde Q_k(x_k,u_k)$（对照 §1.2）。基于此最小化的值空间近似通常称为**一步前瞻（one-step lookahead）**：单步之后由 $\tilde J_{k+1}$ 截断未来。

**值空间内的两种实现路径**（书中并列，本章以 $\tilde J$ 为主，但 Q 路径同等重要）：

| 路径 | 做法 | 特点 |
|------|------|------|
| **$\tilde J$ 近似** | 先构造 $\tilde J_{k+1}$，再算 (2.1) 或 $\tilde Q_k=g+\tilde J_{k+1}$ | 与第 1 章 cost-to-go 记号一脉；问题近似、rollout、NN 等多为此类 |
| **直接 $\tilde Q$ 近似** | 不经过显式 $\tilde J_k$，直接拟合 $\tilde Q_k(x_k,u_k)$（§2.1.4） | 执行控时不必再算期望；与 Q-learning、Actor-Critic 中 critic 衔接 |

二者可组合：例如用问题近似得 $\tilde J_{k+1}$，再用模拟器回归 $\tilde Q_k$ 以降低在线 min 代价。

### 多步前瞻（$\ell>1$）与两步例子

**多步前瞻**：在 $\ell>1$ 个阶段上联合最小化，$k+\ell$ 之后仍用 $\tilde J_{k+\ell}$ 近似尾部；**只执行**首控 $u_k$（详 §2.2）。

书中用**两步前瞻**说明内层 $\tilde J_{k+1}$ 本身如何由更短视界 DP 得到：在 $(k+1,x_{k+1})$ 上

```math
\tilde J_{k+1}(x_{k+1}) = \min_{u_{k+1}\in U_{k+1}(x_{k+1})} \mathbb{E}\big\{ g_{k+1}(x_{k+1},u_{k+1},w_{k+1}) + \tilde J_{k+2}\big(f_{k+1}(x_{k+1},u_{k+1},w_{k+1})\big) \big\},
```

其中 $\tilde J_{k+2}$ 近似 $J_{k+2}^{*}$。于是在时刻 $k$ 对 $(u_k,u_{k+1})$ 做二维 min，终端用 $\tilde J_{k+2}$。

**等价视角**：$\ell$ 步前瞻 = 一步前瞻，但 lookahead 函数取为「$(\ell-1)$ 阶段 DP 的最优余值 + 终端 $\tilde J_{k+\ell}$」。$\ell$ 步仍单独讨论，因其实现问题（树规模、截断、滚动时域）与纯一步前瞻不同。

§2.1 先聚焦一步前瞻；多步延拓见 §2.2。

### 路线二：策略空间近似（Approximation in Policy Space）

**策略空间**在受限策略族 $\mu_k(x_k,r_k)$（常参数化，如 NN）上**直接**优化，而非每步解 (2.1)。

**主要优点**：系统在线运行时，执行 $\mu_k(x_k,r_k)$ 通常比 (2.1) 的 min + 期望**便宜得多**——实时环里这是关键约束。

**与值近似的组合（两阶段，书中重点）**——不必二选一，常「值空间定策略、策略空间降在线成本」：

1. **(a) 值空间**：得 $\tilde J_k$，经 (2.1) 定义次优策略 $\tilde\mu_k$。  
**(b) 策略回归**：在 $q$ 个样本 $(x_k^s,u_k^s)$ 上拟合 $\mu_k(x_k,r_k)$，其中 $u_k^s=\tilde\mu_k(x_k^s)$：

```math
r_k \in \arg\min_r \sum_{s=1}^q \big\| u_k^s - \mu_k(x_k^s,r) \big\|^2. \qquad \text{(2.8)}
```

训练完成后，在线只前向计算 $\mu_k(x_k,r_k)$，**不再**重复 (2.9) 级优化。MPC 计算过重时，此法是常见出口（§2.5.3）。纯策略空间方法（专家监督、策略梯度等）共享「在线快」这一优点，但 (a)+(b) 把 Bellman 结构保留在训练阶段。

本章主体为**值空间**，策略空间作为 §2.1.5 及第 3–4 章接口贯穿全文。

> **跨章索引**：值/策略范式、VI/PI 算法模板、精确与参数化表示，及全书章节结构，见 [`00-algorithm-taxonomy.md`](00-algorithm-taxonomy.md)。本章定名两大 paradigm；精确 VI/PI 见 Ch.4 §4.4–4.5——建议在 §4.5 读毕后对照该文 §4–6。

### Model-based 与 Model-free（本书界定）

本书以**期望是否用 Monte Carlo 采样**区分，而非单纯「有没有 $f,g$ 的公式」：

| 类型 | 含义 |
|------|------|
| **Model-based** | 对任意 $(x_k,u_k,w_k)$，$p_k(w_k\mid x_k,u_k)$ 及 $f_k,g_k$ **闭式可用**；(2.1) 等式中的 $\mathbb{E}[\cdot]$ 用**代数/求和/积分**算，不用 MC。 |
| **Model-free** | (2.1) 及相关式中的期望用 **Monte Carlo 模拟**估计。两种情形：(1) **无** $p_k$ 的数学模型，只有模拟器——给定 $(x_k,u_k)$ 可采样下一态 $x_{k+1}$ 与代价；(2) **有** $p_k$，但为效率仍选采样——仍算 model-free。 |

**小结**：采样/MC 是本书划分 model-based / model-free 的**决定性属性**。Model-free 方法**有模型时也可用**，以免「有模型却故意采样」产生术语歧义。

**与确定性的关系**：确定性问题无期望，通常归 **model-based**（即使 $f_k,g_k$ 靠 heavy 仿真算）。但确定性问题仍可能大量用 MC——如国际象棋/围棋完全确定，AlphaGo/AlphaZero 仍靠 MCTS 与随机化策略（§2.4.2）；第 4 章部分策略梯度同理。

**与前述要点的衔接**：

- (2.1) 的 $\mathbb{E}[\cdot]$ → model-based 代数 vs model-free 采样（§2.1.3 vs §2.1.4）。  
- $\tilde J$ vs 直接 $\tilde Q$ → 值空间两条路；Q 回归天然偏 model-free。  
- 策略回归 (2.8) → 在 (a) 用 model-based 或 model-free 算 $\tilde\mu_k$ 后，把结果压进 $\mu_k(x,r)$。

Fig. 2.1.1 概括值空间一步前瞻中 $\tilde J_{k+1}$ 的构造方式，以及 $u_k$ 的 min、$\mathbb{E}_{w_k}$ 的近似选项。

---

## §2.1 General Issues of Approximation in Value Space

### 总述

值空间方案可**独立**设计两个子问题（Fig. 2.1.1）：

1. **如何得 $\tilde J_k$（及/或 $\tilde Q_k$）**——lookahead 函数从哪来。  
2. **如何执行 (2.1)**——对 $u_k$ 的 min、对 $w_k$ 的期望如何实现（精确、确定性等价、Q 回归、MCTS 等）。

以下各小节分别展开；多步前瞻见 §2.2。

---

### §2.1.1 Methods for Computing Approximations in Value Space

构造 $\tilde J_k$（或间接支撑 $\tilde Q_k$）的**四类主路径**：

**(a) 问题近似（§2.3）**  
解一个**更简单**的关联问题，以其最优（或近最优）余值作为 $\tilde J$。包括：利用可分解结构、忽略部分不确定性、缩小状态空间。**聚合（aggregation）**为特例——代表状态上精确 DP + 插值，或分区后的 aggregate MDP；本书后续与第 5 章及 [Ber12] 衔接。

**(b) 在线近似优化（§2.4–§2.5）**  
用**基策略/启发式**在线滚动估计尾部代价。Rollout、MPC 为代表；基策略可来自 (a) 或启发式。特点：$\tilde J_{k+1}(x_{k+1})$ 常在**需要时**才算，而非整张表。

**(c) 参数化余值（第 3 章）**  
$\tilde J_k(x_k,r_k)$，由特征 $\phi_k(x_k)$ 与训练算法定 $r_k$（线性、NN 等）。离线训练、在线查值或快速前向。

**(d) 聚合（第 5 章）**  
可与 (a)–(c) **叠加**：例如先 NN 得粗 $\tilde J$，再用聚合做局部修正。

**组合**：上述任一路径均可与 **$u_k$ 的近似 min**、**期望的确定性等价**（§2.3.2）、**自适应采样 / MCTS**（§2.4.2）组合。值空间若走 **Q 路径**，(c) 可换成直接参数化 $\tilde Q_k$，(a)(b) 仍可用于提供 $\tilde J_{k+1}$ 以生成训练目标 $\beta^s=g+\tilde J_{k+1}$。

---

### §2.1.2 Off-Line and On-Line Methods

**离线（off-line）**  
控制开始前，对**全部**阶段 $k$ 计算（并存储或可快速求值）整张 $\tilde J_{k+1}$。在线执行 (2.1) 时只查 $\tilde J_{k+1}(x_{k+1})$，不再重算 Bellman 备份。**典型**：NN/参数化近似、聚合。

**在线（on-line）**  
当前 $x_k$ 已知后，**仅对实际会到达的**下一态 $x_{k+1}$ 计算 $\tilde J_{k+1}(x_{k+1})$ 并完成 (2.1)。整条轨迹只需 $N$ 次控计算，适合**再规划**、数据时变。**典型**：rollout、MPC。

**设计选择**  
挑战性问题上 $\tilde\mu_k(x_k)$ 本身常在线算（状态空间大难以存整张策略表）；但 $\tilde J_{k+1}$ 离线还是在线算是**独立**的设计旋钮。问题近似可离线或在线，取决于子问题如何解。**混合**亦常见：离线训 $\tilde J$，在线 rollout 精修。

**与 J / Q 路径**：离线 Q 回归在样本 $(x^s,u^s)$ 上批量拟合 $\tilde Q_k$；在线则每态现采现拟（代价更高，见 §2.1.4）。

---

### §2.1.3 Model-Based Simplification of the Lookahead Minimization

设 $f_k,g_k$ 与 $p_k(w_k\mid x_k,u_k)$ 已知，且 (2.2) 中期望**不用** Monte Carlo（model-based）。

**一步前瞻式 (2.2)**（与 (2.1) 同形，强调期望在内层）：

```math
\tilde\mu_k(x_k) \in \arg\min_{u_k \in U_k(x_k)} \mathbb{E}_{w_k}\Big\{ g_k(x_k,u_k,w_k) + \tilde J_{k+1}\big(f_k(x_k,u_k,w_k)\big) \Big\}. \qquad \text{(2.2)}
```

**两大子问题**：（1）算期望；（2）对 $u_k$ 做 min。在线 repeat 时两者都可能很贵。

**确定性等价（assumed certainty equivalence）**  
用扰动典型值 $\tilde w_k$（如均值）把随机问题**降维**为确定性：

```math
\tilde\mu_k(x_k) \in \arg\min_{u_k \in U_k(x_k)} \Big[ g_k(x_k,u_k,\tilde w_k) + \tilde J_{k+1}\big(f_k(x_k,u_k,\tilde w_k)\big) \Big]. \qquad \text{(2.3)}
```

$\tilde J$ 本身也常由确定性子问题得到（§2.3）。这是 model-based 下**近似期望**的常见手段，与 model-free 采样正交。

**关于 $u_k$ 的 min**：

| 情形 | 做法 |
|------|------|
| $U_k(x_k)$ 有限 | 穷举比较各 $u$ 的 (2.2)/(2.3)；可并行；可用整数规划 |
| 确定性 + 多步前瞻 | 化为**最短路**（label correcting、A* 等，[Ber98], [Ber17]） |
| 连续 $U_k$ | 离散化，或**非线性规划**（MPC，§2.5） |
| 随机 + 连续 | 随机规划；或转 **Q 因子参数化**（§2.1.4）分离「期望估计」与「选控」 |

**J 路径 vs Q 路径（model-based）**：(2.2) 对每个候选 $u$ 先算期望再 argmin；若 $U_k$ 很大，可先回归 $\tilde Q_k$ 再 min（(2.7)），期望在拟合阶段用 MC 或代数完成。

---

### §2.1.4 Model-Free Q-Factor Approximation in Value Space

**定位**：随机 **model-free** 是本书核心场景之一——$f_k,p_k,g_k$ 难建或不方便用，但有**模拟器**：给定 $(x_k,u_k)$ 可采样 $(x_{k+1},g_k)$。

**前提**：

- (a) 有上述模拟器；  
- (b) 已有 $\tilde J_{k+1}(x_{k+1})$（来源可 model-based 或 model-free，与子问题近似无关）。

**目标**：估计

```math
Q_k(x_k,u_k) = \mathbb{E}\big\{ g_k(x_k,u_k,w_k) + \tilde J_{k+1}(f_k(x_k,u_k,w_k)) \big\}, \qquad \text{(2.4)}
```

再 $\tilde\mu_k=\arg\min_u Q_k(x_k,u)$。对**每个** $u\in U_k(x_k)$ 多次模拟往往不可行 → **参数化 $\tilde Q_k(x_k,u_k,r_k)$ + 回归**（Fig. 2.1.2）。

**Q 回归三步**（值空间上的 model-free 实现）：

1. **采样**：$(x_k^s,u_k^s,x_{k+1}^s,g_k^s)$，$s=1,\ldots,q$；  
   $\beta_k^s = g_k^s + \tilde J_{k+1}(x_{k+1}^s)$ (2.5)（模拟器不必输出 $w_k^s$）。  
2. **拟合**：$\bar r_k \in \arg\min_r \sum_s \big(\tilde Q_k(x_k^s,u_k^s,r)-\beta_k^s\big)^2$ (2.6)（可加正则）。  
3. **执行**：$\tilde\mu_k(x_k)\in\arg\min_u \tilde Q_k(x_k,u,\bar r_k)$ (2.7)。

**要点**：

- **Model-free 体现在 MC**：生成 (2.5) 与 (2.6) 时不必显式调用 $f_k,p_k$；执行 (2.7) 只需 $\tilde Q$ 架构。  
- **双重近似**：$\tilde J_{k+1}$ 与 $\tilde Q_k$ 误差独立；(2.7) 一般**不等于**对 (2.2) 直接 min——为免建模付出的代价。  
- **与 J 路径关系**：仍属**值空间**——先 $\tilde J$ 定 tail，再 $\tilde Q$ 做选控；也可跳过显式 $\tilde J$ 直接学 $\tilde Q$（RL 文献常见）。  
- 架构与训练细节见第 3 章。

---

### §2.1.5 Approximation in Policy Space on Top of Value Space

**纯策略空间**：参数化 $\tilde\mu_k(x_k,r_k)$，在样本 $(x_k^s,u_k^s)$ 上最小化 (2.8)（可加正则）。$u_k^s$ 可来自**专家**（监督学习，§4.11）或值近似。

**值空间 + 策略空间（书中标准管道）**：

用 $\tilde J_{k+1}$ 经 **model-based (2.2)** 或 **model-free Q (2.10)** 生成标签：

```math
u_k^s \in \arg\min_u \mathbb{E}\big\{ g_k(x_k^s,u,w_k) + \tilde J_{k+1}(f_k(x_k^s,u,w_k)) \big\}, \qquad \text{(2.9)}
```

或 $u_k^s\in\arg\min_u \tilde Q_k(x_k^s,u,\bar r_k)$ (2.10)。

2. 用 (2.8) 拟合 $\mu_k(x_k,r_k)\approx \tilde\mu_k$。

**因果链**：Bellman/前瞻（值空间）→ 高质量 $(x,u)$ 标签 → 策略回归（策略空间）→ **在线只查 $\mu(x,r)$**。

**优势**：与纯策略空间相同——在线无 (2.9) 级优化；比每步 (2.1) 更适合硬实时。MPC、大规模 min 的常见补救（§2.5.3）。

**参数化**：NN 或问题相关特征（第 3 章）；$r_k$ 可按阶段 $k$ 独立或共享。

---

### §2.1.6 When is Approximation in Value Space Effective?

**$\tilde J_k \approx J_k^{*}$ 非必要**：若 $\tilde J_k(x)-J_k^{*}(x)\equiv c$（与 $x$ 无关常数），(2.1) 仍得**最优**策略。

**相对余值**：$\tilde J_k(x)-\tilde J_k(x')\approx J_k^{*}(x)-J_k^{*}(x')$ 是更合理的启发，但**仍忽略**首段（或前 $\ell$ 段）代价在排序中的作用——绝对余值差小不代表控选对。

**Q 误差斜率**（Fig. 2.1.3）：设 $u_k^{*}$ 最优、$\tilde u_k$ 最小化 $\tilde Q_k$。若 $Q_k(x_k,u)-\tilde Q_k(x_k,u)$ 在 $u_k^{*},\tilde u_k$ 附近**变化平缓**（小「斜率」），则 $\tilde u_k$ 的 $Q_k$ 损失通常可控；若 $Q_k$ 与 $\tilde Q_k$ 仅差**与 $u$ 无关的常数**，两式 argmin 等价。**Advantage** 等 Q 差分在函数逼近误差下往往更稳健（第 3 章 §3.4）。

**J 路径 vs Q 路径的质量**：J 路径看 $\tilde J$ 是否保持**相对排序**；Q 路径直接看 $\tilde Q$ 的斜率——后者更贴「选控」任务，故直接 $\tilde Q$ 在 RL 中更常见。

**局限**：缺乏与问题无关的次优性证书；评估策略多靠结构与仿真——近似 DP/RL 的共性困难。

---

## §2.2 Multistep Lookahead

### 定义与实现

**$\ell$ 步前瞻（Fig. 2.2.1）**：在 $(k,x_k)$ 上联合优化 $u_k,\ldots,u_{k+\ell-1}$（或等价地 $\mu_k,\ldots,\mu_{k+\ell-1}$），终端用 $\tilde J_{k+\ell}$ 截断尾部；**只执行**首控 $u_k$，其余 $k+1,\ldots,k+\ell-1$ 上的最优控**丢弃**，下一时刻重新规划。

```math
\tilde\mu_k(x_k) \in \arg\min_{u_k,\ldots,u_{k+\ell-1}} \mathbb{E}\Big\{ g_k(x_k,u_k,w_k) + \sum_{m=k+1}^{k+\ell-1} g_m(x_m,u_m,w_m) + \tilde J_{k+\ell}(x_{k+\ell}) \Big\}.
```

**两步前瞻的等价表述**（与 §2.0 一致）：对由 $x_k$ 可达的每个 $x_{k+1}=f_k(x_k,u_k,w_k)$，内层

```math
\tilde J_{k+1}(x_{k+1}) = \min_{u_{k+1}} \mathbb{E}\big\{ g_{k+1}(x_{k+1},u_{k+1},w_{k+1}) + \tilde J_{k+2}(f_{k+1}(\cdot)) \big\},
```

再在 $x_k$ 上对 $u_k$ 做外层 min——等价于以 $x_k$ 为初态、$\tilde J_{k+2}$ 为终端的**两阶段 DP**；算出的 $\mu_{k+1}(x_{k+1})$ 仅用于求 $u_k$，不直接执行。

**$\ell>2$**：同理——在 $x_k$ 解 $\ell$ 阶段 DP，终端 $\tilde J_{k+\ell}$，取首控。**阶段截断**：$k>N-\ell$ 时前瞻长度改为 $N-k$。

§2.1 的确定性等价、自适应采样、model-free Q 实现均可延拓至多步（书中 §2.1.2–2.1.3 的简化）。

---

### §2.2.1 Multistep Lookahead and Rolling Horizon

#### $\tilde J_{k+\ell}$ 怎么选

与一步前瞻相同：问题近似、rollout、参数化、聚合等（§2.1.1）。**滚动时域（rolling horizon）** 是特殊情形：取 $\tilde J_{k+\ell}(x)\equiv 0$，或 $\tilde J_{k+\ell}(x)=g_N(x)$，用**足够长**的 $\ell$ 使真实尾部在常数意义下可被 $\ell$ 步精确优化「吸收」——本质上仍是多步前瞻 + **极简终端**。

无限时域：常取平稳 $\tilde J_k\equiv\tilde J$，得平稳策略。折扣问题可 (i) 长 $\ell$ + 终端 0；或 (ii) 短 $\ell$ + 好的终端 $\tilde J$ 补偿（第 4 章）。

#### 为何 $\ell$ 越大，对 $\tilde J_{k+\ell}$ 的要求越低

有效 cost-to-go 近似可拆成**两分量**（书中核心直觉）：

| 分量 | 内容 | 如何得到 |
|------|------|----------|
| **(a)** | $\ell$ 步前瞻中**最后 $(\ell-1)$ 个阶段**的代价 | $(\ell-1)$ 阶段子问题经**精确优化**（DP / 最短路 / NLP） |
| **(b)** | 终端近似 $\tilde J_{k+\ell}(x_{k+\ell})$ | 启发式、0、$g_N$、NN 等 |

总近似 ≈ (a) + (b)。$(\ell-1)$ 段是精确算的，只要 **(b) 相对 (a) 可忽略**，整体就较准——$\ell$ 足够大时，精确优化覆盖的步数多，终端权重自然下降，故**粗 $\tilde J_{k+\ell}$（甚至 $\equiv 0$）有时够用**。

#### 为何「$\ell$ 越大越好」**不总成立**

直觉上 $\ell\uparrow$ → 更多精确优化 → 策略应更好；但策略对 **$k+\ell$ 之后** 的状态是「盲」的——只看 $\tilde J_{k+\ell}$ 对远期**特别有利或不利**态的编码，**不**做精确前瞻。若 $\tilde J_{k+\ell}$ 很差或过于简化（如恒 0），更长 $\ell$ 反而把决策绑在**错误的终端信息**上。

**例 2.2.1**（4 阶段确定性最短路，Fig. 2.2.2；$\tilde J_k\equiv 0$）：

- 初态可选 $u$（上路）或 $u'$（下路）；其余态只有唯一控。  
- 上路弧权：$0,1,2,1$，总代价 **4**（最优）。  
- 下路弧权：$0,2,0,10$，总代价 **12**（次优）。  
- **2 步前瞻**（$\tilde J_2=0$）：比较 $0+1$ vs $0+2$ → 选 **$u$**（优）。  
- **3 步前瞻**（$\tilde J_3=0$）：比较 $0+1+2$ vs $0+2+0$ → 选 **$u'$**（劣）。

**机理**：代价在 lookahead **边缘**突变——2 步视界末尾看到代价 0（下路第 3 弧），3 步视界才暴露其后代价 10；更长前瞻被**误导性的零终端**骗向短路。说明：**$\ell$ 与 $\tilde J_{k+\ell}$ 必须联合设计**，不能单靠加大 $\ell$。

**何时滚动时域较可靠**（书中脚注）：$k+\ell$ 步后的状态分布大致**与当前** $(x_k,u_k)$ **无关**，或集中在**低成本**态附近时，$\tilde J_{k+\ell}\equiv 0$ 类近似较安全——否则终端盲区会伤策略。

---

### §2.2.2 Multistep Lookahead and Deterministic Problems

#### 随机 vs 确定性：计算瓶颈

**随机**多步前瞻：每步需求解**随机 DP**（每层 min + 期望），视界 = $\ell$ → 计算常 **prohibitive**（除非状态/控极少，或靠采样/MCTS 近似）。

**确定性**多步前瞻：子问题也是确定性的 → 有限状态可化为**最短路**（Fig. 2.2.3；label correcting、A* 等）；连续状态可用 **NLP**（§2.5 MPC）。因此**长 $\ell$ + 滚动时域**在确定性问题上特别常用。

#### 部分确定性形式（Partially Deterministic Multistep Lookahead）

随机问题上的**混合**做法（书中 §2.2.2 末；与 §2.3.2 确定性等价一脉）：

**思路**：在 $x_k$ 处  
- **第一阶段**保留 $w_k$ 的随机性；  
- **未来** $w_{k+1},\ldots,w_{k+\ell-1}$ **固定**为典型值（均值、情景等），直到 lookahead 末端。

这样 $k+1$ 之后的计算可全部用**确定性**方法。

**三步流程**：

1. **算 $\tilde J_{k+1}(x_{k+1})$**：对每个候选下一态，解 $(\ell-1)$ 步**确定性最短路**（扰动取 $\tilde w_{k+1},\ldots,\tilde w_{k+\ell-1}$），得从 $x_{k+1}$ 起 $(\ell-1)$ 段最优 tail + 终端 $\tilde J_{k+\ell}$ 的代价。  
**算近似 Q**（**只**在第一阶段对 $w_k$ 取期望）：

```math
\tilde Q_k(x_k,u_k) = \mathbb{E}_{w_k}\Big\{ g_k(x_k,u_k,w_k) + \tilde J_{k+1}\big(f_k(x_k,u_k,w_k)\big) \Big\}.
```

3. **选控**：$\tilde\mu_k(x_k)\in\arg\min_{u_k\in U_k(x_k)} \tilde Q_k(x_k,u_k)$（同 (2.7) 形）。

**解读**：

- 与一步前瞻 (2.1) 同形，但 $\tilde J_{k+1}$ 不是外给启发式，而是 **$(\ell-1)$ 步确定性 DP 的最优余值**（在固定未来扰动下）。  
- 第一阶段仍 model-based（代数期望）或 model-free（对 $w_k$ 采样）；内层最短路为 model-based 确定性。  
- 将不确定量固定为典型值 = **（假设）确定性等价**；详 §2.3.2。确定性问题上的多步前瞻思想在 **§2.4.1 rollout** 中再次出现。

---

### §2.2 小结

| 主题 | 要点 |
|------|------|
| **定义** | $\ell$ 阶段 DP + 终端 $\tilde J_{k+\ell}$；只执行首控；$k>N-\ell$ 时缩短视界 |
| **$\ell$ 与 $\tilde J$** | 有效近似 = 精确 $(\ell-1)$ 段 + 终端 (b)；$\ell\uparrow$ 降低对 (b) 的要求，但不保证策略改进 |
| **陷阱** | 对 $k+\ell$ 之后「失明」；粗终端 + 长 $\ell$ 可在边缘误导（例 2.2.1） |
| **滚动时域** | $\tilde J_{k+\ell}\equiv 0$ 或 $g_N$ + 长 $\ell$；适用无限时域/折扣；依赖 $k+\ell$ 态分布条件 |
| **确定性** | 多步 → 最短路 / NLP；长 lookahead 可行 |
| **部分确定性** | $w_k$ 随机 + 未来 $w$ 固定 → 内层确定性最短路得 $\tilde J_{k+1}$，再对 $w_k$ 算 $\tilde Q_k$ |
| **与后文** | 确定性等价 §2.3.2；确定性 rollout §2.4.1；MPC §2.5 |

**实践取舍**：$\ell$ 增大改善精确优化深度，但计算随分支/控空间约 $\ell$ 次方级膨胀；随机问题常退化为 $\ell=1$ + 好 $\tilde J$，或部分确定性 / MCTS。MPC 可视为确定性长 $\ell$ + 终端约束（常驱至 0）的工程标准件。

---

## §2.3 Problem Approximation

核心：用**关联但更简单**的问题的最优余值（或近优余值）作为 $\tilde J$。除聚合外，本节强调：

1. **强制分解（enforced decomposition）**  
2. **概率结构简化（确定性等价类）**

---

### §2.3.1 Enforced Decomposition

**适用场景**：多子系统问题，耦合出现在动力学、代价或**控制约束**中，但耦合「相对弱」——无精确定义，靠问题结构识别。思路：**人为解耦**子系统，得到更简问题或更便宜的 tail 代价计算，各子系统可**隔离**处理。

**实现方式**（确定性可离线/在线；随机问题常离线算 $\tilde J_k$、在线用 (2.1)）：

---

#### 方法一：逐子系统优化（Optimization of One Subsystem at a Time）

控制 $u_k=(u_k^1,\ldots,u_k^n)$ 时，算 $\tilde J_k(x_k)$ 或一步前瞻中的 tail 项可：

1. 固定其余子系统控制为**名义值**（上一轮最优或零）；  
2. **轮流**只对某一子系统的控制序列 $\{u_k^i,u_{k+1}^i,\ldots\}$ 做优化；  
3. 可选：优化子系统顺序、**多轮循环**（每轮用最新结果作名义值）——类似**坐标下降**。

可与 (2.1) 的近似 min、确定性等价、自适应采样组合。

**例 2.3.1（车辆路径，Fig. 2.3.1）**

- $n$ 辆车在图上移动；节点有价值，**首辆**经过者收值，后续无值。  
- 状态 = 各车位置 + 已访问节点集 → 维数指数级；**单车**子问题可 DP 或启发式。  
- **一步前瞻 + enforced decomposition**：  
  - 枚举 $x_k$ 上所有车辆联合移动 → 后继 $x_{k+1}$；  
  - 对每个 $x_{k+1}$：固定车辆顺序（如先 1 后 2），**逐车**求路径——先算车 1 最优路径（假设车 2 不动），再算车 2（计入车 1 已收值）；  
  - 路径总价值 = $\tilde J_{k+1}(x_{k+1})$；  
  - 在 $x_k$ 选使 $\tilde J_{k+1}$ 最大的联合移动。  
- 变体：多种车辆顺序、旅行代价、每车任务上限等。

---

#### 方法二：约束松弛（Constraint Relaxation）

耦合**仅**在控制约束（资源分配），子系统动力学**可分解**时：把耦合约束 $U$ 换为**更大**的 decoupled 集 $\bar U\supset U$，解 relaxed 问题得 $\tilde J$；原问题可行解需另行验证。

**例 2.3.2（Restless Multi-Armed Bandit, RMAB）**

- $n$ 个项目，每步**至多做一个**；做 $i$ 得 $R_i(x_i)$，$x_i$ 按 $f^i(x_i,w_i)$ 演化；不做则被动演化且得 $R^i(x_i)$。  
- 完全 DP 状态 $(x_1,\ldots,x_n)$ 维数爆炸。

**可分离近似**：

```math
\tilde J_k(x_1,\ldots,x_n)=\sum_{i=1}^n \tilde J_k^i(x_i).
```

一步前瞻选项目 $i$ 最大化（与 $i$ 无关项可略）：

```math
\tilde m_k^i(x_i)=R_i(x_i)-R^i(x_i)+\mathbb{E}\Big[\tilde J_{k+1}^i\big(f^i(x_i,w_i)\big)-\tilde J_{k+1}^i\big(\bar f^i(x_i,w_i)\big)\Big],
```

其中 $\bar f^i$ 表示「不做 $i$」时的被动转移；策略为 $\text{work on } i \text{ if } \tilde m_k^i(x_i)=\max_j \tilde m_k^j(x_j)$——**指数型 index 规则**（Whittle index 等在最优情形下结构类似）。

**$\tilde J_k^i$ 从哪来**：

- **单项目全视界 DP**：假设 $k+1$ 起只做项目 $i$，其余永不激活——常可算；  
- **参数化** $\tilde J_{k+1}^i(x_i,r_{k+1}^i)$ + 第 3 章训练。

---

#### 方法三：Lagrange / 价格分解（Constraint Decoupling by Lagrangian Relaxation）

硬耦合约束不直接去掉，而是加入 **Lagrangian 惩罚**，解**可分离**子问题。例 2.3.3 推广 RMAB：每子系统有 $u_k^i$，约束 $u_k\in U$（如 $\sum_i c_i u_k^i\le b$）。

- **平均松弛**：$\sum_{k,i} c_i u_k^i \le Nb$ 替代每步 $\sum_i c_i u_k^i\le b$（(2.15)→(2.16)）；  
- 对 (2.16) 引入乘子 $\lambda\ge 0$，阶段代价加 $\lambda\big(\sum_i c_i u_k^i - Nb/N\big)$ 类项 (2.17)；  
- 子系统 $i$ 可**独立**优化 → $\tilde J_k=\sum_i \tilde J_k^i$ 形式的**下界近似**（对偶/Lagrangian 松弛，[BeT97], [Ber16a]）；  
- $\lambda$ 可手调或最大化下界；亦可用时变 $\lambda_k$。

**价格分解**：资源约束下引入「价格」信号，交替更新子系统决策与价格——同一思想族。

---

#### 与 §2.3.2 的交叉

子系统仅通过**扰动分布**耦合时，CEC 可「解耦」未来扰动（例 2.3.6）——见下节。

---

### §2.3.2 Probabilistic Approximation — Certainty Equivalent Control

通过修改**概率结构**做问题近似：把随机扰动换成**典型值**（均值、情景等），再当**确定性**问题解。最常用的是 **CEC（Certainty Equivalent Controller）**。

与 §1.3.7 **Certainty Equivalence** 的关系：LQ 问题中策略与确定性相同；CEC 是更一般的**启发式**——**假定**等价成立，把 $w_k$ 固定为 $\tilde w_k$，用确定性 OC 工具处理随机/部分信息问题。

**优点**：每步（或离线一次）解**确定性**最优控制，比随机 DP 便宜得多；得控序列后**只用首控**，其余丢弃（与 MPC/滚动时域同型）。

---

#### CEC 基本形式

对每 $(x_k,u_k)$ 选典型扰动 $\tilde w_k(x_k,u_k)$（常为 $\mathbb{E}[w_k\mid x_k,u_k]$，扰动空间凸时合理）。

**变体 A：在线再规划（(2.18)）**  
在 $(k,x_k)$ 解尾部确定性问题：

```math
\min_{u_i\in U_i(x_i),\, i=k,\ldots,N-1} \Big\{ g_N(x_N)+\sum_{i=k}^{N-1} g_i\big(x_i,u_i,\tilde w_i(x_i,u_i)\big) \Big\},
```

约束 $x_{i+1}=f_i(x_i,u_i,\tilde w_i(x_i,u_i))$。最优序列 $\{\tilde u_k,\ldots,\tilde u_{N-1}\}$ 中取 $\mu_k(x_k)=\tilde u_k$。

**变体 B：离线策略（(2.19)）**  
对**整问题**用 DP 解确定性策略 $\mu_k^d(x_k)$，在线 $\tilde\mu_k=\mu_k^d(x_k)$。

两变体**性能等价**；A 适再规划/参数时变，B 适离线表查。

**部分状态信息**：$x_k$ 未知时用估计 $\hat x_k$ **当作**真状态代入 (2.18)/(2.19)（与部分 CEC 衔接）。

---

#### 带启发式的确定性等价控制（Certainty Equivalent Control with Heuristics）

(2.18) 的尾部确定性 DP 仍可能难解 → 用**启发式**得次优 $\{\tilde u_k,\ldots,\tilde u_{N-1}\}$，仍只执行 $\tilde u_k$。

**重要增强（(2.20)–(2.21)）**：对**首控** $u_k$ 精确 min，启发式只负责 $k+1,\ldots,N-1$：

```math
\tilde u_k \in \arg\min_{u_k\in U_k(x_k)} \Big\{ g_k(x_k,u_k,\tilde w_k(x_k,u_k)) + H_{k+1}\big(f_k(x_k,u_k,\tilde w_k(x_k,u_k))\big) \Big\},
```

$H_{k+1}(x_{k+1})$ = 从 $x_{k+1}$ 起按启发式滚动的 tail 代价。这是 **一步前瞻 + $H_{k+1}$** 与 **CEC（$w$ 固定典型值）** 的混合；$H_{k+1}$ 可闭式或**仿真**（对每个候选 $u_k$ 算 $x_{k+1}$ 再跑启发式）。

**例 2.3.4（停车 + 概率估计）**  
例 1.3.3 中 $p(k)$ 原为常数；现 $p(k)$ 为基于观测的**估计**，状态含 belief → 无穷维/难精确 DP。CEC：到达 $k$ 时把**前方**各空位概率**冻结**为当前 belief，当固定概率问题用例 1.3.3 的快速 DP **在线**求控。例：用已遇空闲比例 $R(k)$ 调整 $ \hat p(m,R(k))=\gamma p(m)+(1-\gamma)R(k)$；精确 DP 在 $R(k)$ 上仍指数，CEC 子优策略易在线实现。

---

#### 部分确定性等价控制（Partial Certainty Equivalent Control）

CEC 不必把所有 $w$ **全部**固定：只对**部分**量取典型值，其余保留随机性。

**典型模式**：部分信息 → 用状态估计 $\tilde x_k$ **当作**完美信息，扰动 $w_k$ 仍随机，对**完整随机完美信息问题**离线求最优策略 $\{\mu_k^p\}$，在线 $\tilde\mu_k=\mu_k^p(\tilde x_k)$。

**例 2.3.5（不诚实的旅店老板）**  
$m$ 档房价 $r_i$，报价 $r_i$ 被接受概率 $p_i$。若知剩余顾客数 $y$ 与空房 $x$，最优值 $\tilde J(x,y)$ 满足 (2.22) 的 DP。若 $y$ 未知、只有分布 → 精确为 POMDP。部分 CEC：用 $y$ 的估计 $\tilde y$（如期望取整），用 (2.22) 的 $\tilde J$ 做一步前瞻，选 $r_i$ 最大化 $p_i r_i + \tilde J(x-1,\tilde y-1)-\tilde J(x,\tilde y-1)$——**当作** $\tilde y$ 确定。

与 §2.2.2 **部分确定性多步前瞻**（只固定未来 $w$、保留当前 $w_k$ 随机）同属「部分等价」族。

---

#### 确定性等价控制的其他变体（Other Variations）

**例 2.3.6（解耦扰动分布）**  
$n$ 个子系统 $x_{k+1}^i=f^i(x_k^i,u_k^i,w_k^i)$，但 $w_k^i$ 的分布依赖**全状态** $x_k$。Enforced decomposition + CEC：对子系统 $i$，用其他子系统未来状态的**名义值** $\tilde x_{k+1}^j,\ldots$，使 $w_{k+1}^i,\ldots$ 的分布只依赖**局部** $x_{k+1}^i,\ldots$；每子系统解局部随机/确定性问题，取首控 $u_k^i$。

**例 2.3.7（情景 / Scenarios，(2.23)）**  
单条名义扰动轨迹 $\tilde w_{k+1},\ldots$ 的 CEC 可能偏乐观/悲观 → 用 **$q$ 条情景** $w^s(x_{k+1})=(w_{k+1}^s,\ldots,w_{N-1}^s)$：

```math
\tilde J_{k+1}(x_{k+1},r)=\sum_{s=1}^q r_s C_s(x_{k+1}),
```

$C_s$ = 情景 $s$ 下从 $x_{k+1}$ 起最优或启发式 tail 代价；$r=(r_1,\ldots,r_q)$ 为权重（概率向量，可时变、「聚合概率」）。情景可随机/仿真生成；与 **rollout**（§2.4）、Monte Carlo 思想相通——多轨迹比单典型值更稳。

**一般技巧**（问题依赖）：简化转移律、分阶段固定不同子集的不确定性、model-free 采样生成情景等。

**小结**：CEC 族 = 固定典型值 + 确定性（或启发式）尾部；完整 CEC → 启发式 CEC → 部分 CEC → 情景/解耦变体，计算与 fidelity 逐级可调。一般**非最优**；常与 rollout 组合（基策略来自近似问题最优策略时，rollout = 以该问题 $J^{*}$ 为终端的多步前瞻）。

---

## §2.4 Rollout

**目标**：**策略改进**——给定基策略/启发式 $\bar\pi$，经有限前瞻 + 基策略尾部代价得 rollout 策略 $\tilde\pi$，在序贯一致或序贯改进条件下**不劣于**基策略（第 4 章将 PI 视为反复 rollout）。

**一步形式**：$\tilde J_{k+1}(x_{k+1})=$ 从 $x_{k+1}$ 起运行基策略的 tail 代价。  
**$\ell$ 步推广**：基策略从 $x_{k+\ell}$ 起算 $\tilde J_{k+\ell}$（Fig. 2.4.1）；长视界可**截断** + 终端 $\tilde J$。

基策略为**近似问题的最优策略**时，rollout 等价于以该问题最优余值为终端的多步前瞻。

**组合**：有限 horizon rollout + 终端 $J_\mu$ 或 $\tilde J$ 估计（第 4 章 §4.5.3）。

---

### §2.4.1 On-Line Rollout for Deterministic Finite-State Problems

在 $x_k$，对每个 $u_k\in U_k(x_k)$，令 $x_{k+1}=f_k(x_k,u_k)$，基**启发式**（确定性下称 base heuristic）生成 $\{x_{k+1},\ldots,x_N\}$ 及控制，得 tail 代价。选最小者：

```math
\tilde\mu_k(x_k) \in \arg\min_{u_k} \tilde Q_k(x_k,u_k), \quad \tilde Q_k(x_k,u_k) = g_k(x_k,u_k) + H_{k+1}(f_k(x_k,u_k)), \qquad \text{(2.24-2.25)}
```

$H_{k+1}(x_{k+1})$ 为从 $x_{k+1}$ 起的启发式总代价（不含首项 $g_k$）。

**例 2.4.1（TSP + 最近邻）**：状态为部分回路；对每个候选下一城市，用最近邻补全回路得 $H_{k+1}$，选最优扩展（Fig. 2.4.3）。读者可验证：**最近邻启发式是序贯一致的**（见下）。

---

#### 序贯一致（Sequentially Consistent）

**定义**：基启发式从 $x_k$ 生成轨迹 $\{x_k,x_{k+1},\ldots,x_N\}$ 时，若从**下一态** $x_{k+1}$ 再启动，仍生成**同一后缀** $\{x_{k+1},\ldots,x_N\}$，则称基启发式**序贯一致**——「保持航向」：起始态沿轨迹前进一步，启发式**不偏离**剩余路径。

**等价表述（书中强调）**：序贯一致 ⟺ 基启发式是**合法的 DP 策略** $\bar\pi=\{\mu_0,\ldots,\mu_{N-1}\}$ 的轨迹：

- 任一策略显然序贯一致（按 $\mu_k(x_k)$ 滚动，后缀不变）；  
- 反之，序贯一致的启发式**定义**策略：在 $x_k$ 选控使下一态落在启发式路径 $\{x_k,x_{k+1},\ldots\}$ 上的 $x_{k+1}$，即 $\mu_k(x_k)=u_k^{\text{heur}}$。

因此 $H_k(x_k)$ 不仅是「启发式总代价」，更是策略 $\bar\pi$ 从 $x_k$ 起的 **cost-to-go**——归纳证明 (2.27) 时，最后一步需 $H_k=g_k(x_k,u_k^{\text{heur}})+H_{k+1}(f_k(x_k,u_k^{\text{heur}}))$，即策略的 DP 方程。

**实践**：多数贪心类启发式（[Ber17] §6.4）在**大多数** $x_k$ 上序贯一致；**部分**态可能违反（例如 TSP 最近邻在特殊构型下改选下一城市）。MPC 的基启发式常**序贯改进**但**未必**序贯一致（§2.5.1）。

**命题（代价改进，(2.27)）**：若基启发式序贯一致，则 rollout 策略满足 $J_{k,\tilde\pi}(x_k)\le \hat J_k(x_k)=H_k(x_k)$（从每个 $(k,x_k)$ 起不劣于基启发式）。

---

#### 序贯改进（Sequentially Improving，弱于序贯一致）

不要求后缀一致，只要求**一步前瞻**不比启发式整段差：

```math
\min_{u_k\in U_k(x_k)} \big[ g_k(x_k,u_k) + H_{k+1}(f_k(x_k,u_k)) \big] \le H_k(x_k). \qquad \text{(2.29)}
```

即「最优启发式 Q 因子 $\le$ 启发式总代价」(2.30)。序贯一致 ⇒ 上式取**等号**（因 $H_k$ 恰为启发式在 $x_k$ 的 Q 因子）。仅序贯改进时也得 $J_{k,\tilde\pi}\le H_k$，证明用 (2.29) 替代归纳中依赖序贯一致的最后一步。

---

#### 归纳证明概要（(2.27)）

设 $J_{k,\tilde\pi}$ 为 rollout 策略从 $(k,x_k)$ 起的代价，$\hat J_k=H_k$ 为启发式总代价。基步 $k=N$：$J_{N,\tilde\pi}=g_N=H_N$。归纳步：对任意 $x_k$，记 $\tilde u_k=\tilde\mu_k(x_k)$，

```math
J_{k,\tilde\pi}(x_k)=g_k(x_k,\tilde u_k)+J_{k+1,\tilde\pi}(f_k(x_k,\tilde u_k)) \le g_k(x_k,\tilde u_k)+H_{k+1}(f_k(x_k,\tilde u_k)) =\min_u[g_k+H_{k+1}(f_k)] \le g_k(x_k,u_k^{\text{heur}})+H_{k+1}(f_k(x_k,u_k^{\text{heur}}))=H_k(x_k),
```

其中 $u_k^{\text{heur}}$ 为启发式在 $x_k$ 的首控；最后一步等式需**序贯一致**（$H_k$ 满足策略 $\bar\pi$ 的 DP 方程）。

---

#### 加固 rollout（Fortified Rollout，Fig. 2.4.4）

当基启发式**不满足**序贯改进时，普通 rollout 的 (2.27) 可能不保证。加固版**隐式**构造序贯改进的基启发式，使 (2.29) 成立。

**机制**（从 $x_0$ 逐步构造）：

- **永久轨迹** $P^k=\{x_0,u_0,\ldots,u_{k-1},x_k\}$：已确定、不可回退的前缀。  
- ** tentative 轨迹** $T^k=\{x_k,u_k,\ldots,x_N\}$，代价 $C(T^k)$：当前「从 $x_k$ 到终点」的最佳后缀估计；初始 $T^0$ = 基启发式从 $x_0$ 的整条轨迹。不变量：$P^k\cup T^k$ 是到阶段 $k$ 为止**全程最佳**已知轨迹。

在 $x_k$：对每个 $u_k$ 做普通 rollout，得最优后缀 $ \tilde T^k$、代价 $C(\tilde T^k)$（首控 $\tilde u_k$）。**比较** $C(P^k\cup T^k)$ 与 $C(P^k\cup\tilde T^k)$：

- 若 $C(P^k\cup T^k)\le C(P^k\cup\tilde T^k)$：沿原 tentative 走，$x_{k+1}$ 取 $T^k$ 的下一态，$T^{k+1}$ 为 $T^k$ 的后缀；  
- 否则：采用 rollout 发现的更优后缀，$x_{k+1}=\tilde x_{k+1}$，$T^{k+1}=\{\tilde x_{k+1},\tilde u_{k+1},\ldots\}$。

**直观**：在 $x_k$ **跟随**当前 tentative，除非 rollout 从某 $x_{k+1}$ 重新跑启发式发现**更便宜**的全后缀。若启发式从 $x_{k+1}$ 生成的轨迹与 tentative 的尾部**不一致**，加固版与普通 rollout 结果不同。

**性质**：全程 $P^k\cup T^k$ 代价 ≤ 基启发式从 $x_0$ 的初始轨迹；若基启发式已序贯改进，加固版与普通 rollout **重合**。可视为对**修改后问题+修改后启发式**做普通 rollout（构造见 [BTW97], [Ber17] §6.4.2）。

---

#### 使用多个启发式（Superheuristic）

常有 $M$ 个候选基启发式。构造 **superheuristic**：给定 $x_{k+1}$，让每个启发式 $m$ 生成轨迹 $\tilde T_{k+1}^m$ 与代价 $C(\tilde T_{k+1}^m)$，取**代价最小**者作为 superheuristic 的输出——再以其为基做 rollout。

**性质**：若各基启发式均序贯改进，superheuristic 也序贯改进；存在加固版 rollout，使最终轨迹代价 **≤** 每个基启发式从 $x_0$ 单独出发的轨迹。

**用途**：组合最近邻 / 2-opt / 贪心等不同 TSP 启发式，rollout 在「多专家」中选优 tail。

---

#### 具有多步前瞻的 rollout（Multistep Lookahead Rollout，Fig. 2.4.5）

一步 rollout 只在 $u_k$ 上 min，tail 由基启发式从 $x_{k+1}$ 算起。**$\ell$ 步推广**：

- 在 $x_k$ 枚举（或搜索）$\ell$ 步内的控制/状态树；  
- 对每个 **$\ell$ 步可达叶态** $x_{k+\ell}$，运行基启发式得 $H_{k+\ell}(x_{k+\ell})$；  
- 叶代价 = 从 $x_k$ 到 $x_{k+\ell}$ 的 $\ell$ 段精确代价 + $H_{k+\ell}$；  
- 选全局最小叶，**只执行**首控 $\tilde u_k$，下一态 $x_{k+1}=f_k(x_k,\tilde u_k)$。

**两步例**：对所有 $x_{k+2}$ 跑启发式，比较 $g_k+g_{k+1}+H_{k+2}$，取最优 $(\tilde u_k,\tilde u_{k+1})$ 中的 $\tilde u_k$。

**截断 rollout + 终端 $\tilde J$**：长视界时，从叶态只跑基启发式 **$H$ 步**，再加终端 $\tilde J$ 补偿剩余——与 §2.2 滚动时域、第 4 章 rollout+终端估计一致。

**选择性深度（Selective Depth，Fig. 2.4.6）**：$\ell$ 步树爆炸时，先对 $x_{k+1}$ 候选用启发式**粗评**，只对 top 分支展开 $x_{k+2},\ldots$——限制启发式调用次数；与 MCTS 的 selective expansion 同族。

**多步加固版**：可与 tentative 轨迹机制结合（沿 §2.4.4 思路）。

**计算代价**：约为（基启发式单次代价）$\times$（一步：$|U_k|$；$\ell$ 步：lookahead 树规模；选择性深度：剪枝后子树规模）；可并行。

---

### §2.4.2 Stochastic Rollout and Monte Carlo Tree Search

将 rollout 推广到**有限状态随机问题**；基启发式取**策略** $\pi=\{\mu_0,\ldots,\mu_{N-1}\}$（确定性情形下即序贯一致）。更一般的「序贯改进基启发式」在随机情形理论可行，但书中指出**尚无**重要应用，故不展开。

---

#### 随机情形下的代价改进（(2.27) 随机版）

记 $J_{k,\pi}(x_k)$ = 从 $(k,x_k)$ 起执行基策略 $\pi$ 的期望代价；$J_{k,\tilde\pi}(x_k)$ = 从同一起点执行 rollout 策略 $\tilde\pi$ 的期望代价。在基策略序贯一致（随机 DP 意义下）时仍有：

```math
J_{k,\tilde\pi}(x_k) \le J_{k,\pi}(x_k), \quad \forall x_k,\, k.
```

**归纳证明**（与确定性同构）：

- 基步 $k=N$：$J_{N,\tilde\pi}=J_{N,\pi}=g_N$。  
- 归纳步：rollout 的 DP 方程 → 归纳假设 → rollout 定义（对 $u_k$ 取 min 的期望 Q）→ 基策略的 DP 方程。

**经验观察**：随机问题中 rollout 不仅**不劣于**基策略，且常带来**显著**改进（见章末案例）。

---

#### 基于仿真的 rollout 实现

在 $(k,x_k)$，对每个 $u_k\in U_k(x_k)$：

1. 采样大量扰动序列 $\{w_k,\ldots,w_{N-1}\}$（独立）；  
2. 首态 $x_{k+1}=f_k(x_k,u_k,w_k)$，之后按基策略 tail 滚动：$x_{i+1}=f_i(x_i,\mu_i(x_i),w_i)$；  
3. 轨迹代价样本 = 阶段代价之和 + 可选终端 $\tilde J$；  
4. Monte Carlo 平均得 Q 因子近似 $\tilde Q_k(x_k,u_k)$；  
5. 选控（(2.31)）：

```math
\tilde\mu_k(x_k) \in \arg\min_{u_k\in U_k(x_k)} \tilde Q_k(x_k,u_k).
```

$\tilde Q_k$ 估计的是

```math
Q_k(x_k,u_k)=\mathbb{E}\big[g_k(x_k,u_k,w_k)+J_{k+1,\pi}(f_k(x_k,u_k,w_k))\big],
```

其中 $J_{k+1,\pi}$ 为基策略 cost-to-go。**长视界**：截断 rollout 轨迹 + 终端代价近似（与 §2.4.1 截断 rollout 一致；极端可零步 rollout，仅用终端 $\tilde J$）。

---

#### 例 2.4.2（Backgammon [TeG96]，Fig. 2.4.7）

**背景**：双陆棋为**双人**零和博弈，非单决策者最优控制；本书未展开博弈 DP，但 rollout 仍可工程化。

**基策略**：TD-Gammon（NN + TD($\lambda$) 训练，Ch.4 §4.9）——一步/两步前瞻 + NN 终端余值。

**Rollout 流程**（给定局面与骰子）：

- 枚举所有合法走法；  
- 对每步走法，用 TD-Gammon 作基策略**模拟至终局**多条对局（骰子随机）；  
- Monte Carlo 平均得分；选平均得分最优走法。

**截断**：长对局截断 + 基于 TD-Gammon 的终端代价——基策略同时提供 tail 与终端估计。

**双人解释（书中关键）**：**不对等**对待两玩家——一方**固定**用 TD-Gammon，另一方用 rollout **改进**。故「策略改进」= 对 TD-Gammon 对手，rollout 玩家平均得分更高。**不保证**：双方都用 TD-Gammon 训练的 rollout 玩家，对**非** TD-Gammon 对手仍优于原版 TD-Gammon——合理假设，需**经验**验证。

**实践**：多数现代双陆棋程序源自 TD-Gammon；rollout 版最强但**实时**太慢（高分支因子 × 每步大量仿真），多用于**诊断** NN 程序质量。

---

#### Monte Carlo Tree Search（MCTS）

朴素 rollout 对**每个** $(x_k,u_k)$ 分配**相同**大量仿真 → 三问题：

| 问题 | 含义 |
|------|------|
| **(a) 轨迹过长** | 视界 $N$ 大或无限时域，仿真到终点代价高 |
| **(b) 弱控浪费** | 明显劣于其他的 $u_k$ 仍获同等采样 |
| **(c) 强控不足** |  promising 的 $u_k$ 值得更多多步前瞻/更深探索 |

**MCTS** 族方法在**计算经济**与**性能风险**间折中，核心手段：

- **(a) 补救**：有限长 rollout + 终端 $\tilde J$（可为 0 或辅助计算；基策略常即终端估计来源，如 Backgammon）；  
- **(b) 补救**：启发式/统计检验**早丢弃**劣控；  
- **(c) 补救**：对选定 $u_k$ **选择性加深**前瞻（类比确定性 selective depth，Fig. 2.4.6）。

**一般框架**：维护**前瞻搜索树**，随仿真逐步扩展；用**中间结果 + 统计检验**将仿真预算集中到最有希望的方向——在 **exploitation**（评估看似最优的控）与 **exploration**（探索采样不足的控）间平衡。思想源自 **multi-armed bandit**（章末文献 [Cou06], [BPW12], [Fu17] 等）。

**典型四步循环**（一步前瞻 MCTS）：

1. **Selection**：从根沿树选子节点（UCB/UCT 等）；  
2. **Expansion**：加新叶节点；  
3. **Simulation**：从叶态 rollout（可截断）得代价样本；  
4. **Backpropagation**：沿路径更新访问次数与 Q 估计。

模拟预算**自适应**倾斜到 promising 分支，而非均匀分配。

---

#### 例 2.4.3（一步前瞻 MCTS + 自适应采样，Fig. 2.4.8）

在 $x_k$，$m=|U_k(x_k)|$，控标为 $1,\ldots,m$。第 $\ell$ 次采样：

- 选控 $i_\ell$ 使 $T_{i,n}+R_{i,n}$ 最小（exploitation + exploration）；  
- 对 $i_\ell$ 抽一条 $\tilde Q_k(x_k,i_\ell)$ 样本 $S_{i_\ell}$；  
- 更新经验均值（控 $i$ 至少被采一次时）：

```math
Q_{i,n}=\frac{\sum_{\ell=1}^n \delta(i_\ell=i)\, S_{i_\ell}}{\sum_{\ell=1}^n \delta(i_\ell=i)}.
```

**停止准则**：自适应采样设计「选哪条控继续采」与「何时停」，使误选最小控的概率小、总样本数有限（ inferior 控少浪费）。

**常用指数**：$T_{i,n}=\bar Q_{i,n}$（样本均值）；**UCB 探索项**（最小化 cost 时符号与 reward 文献相反）：

```math
R_{i,n}=-c\sqrt{\frac{\log n}{s_i}},
```

$s_i$ = 控 $i$ 已被采样次数；$c>0$ 经验选取（$Q$ 归一化到 $[-1,0]$ 时分析建议 $c\approx\sqrt{2}$，[ACF02]）。

**多步 MCTS**：从 lookahead 树节点 $i$ 出发的轨迹，仍按 $T_{i,n}+R_{i,n}$ 选下一扩展点——**UCT**（UCB applied to trees，[KoS16]）；细节见章末文献。

---

#### 与 AlphaGo / AlphaZero

**AlphaGo [SHM16]**：整合本章多项技术 + Ch.3–4 的 NN 训练——离线训练 NN 作基策略/价值，**MCTS + rollout** 作在线多步前瞻，超人类围棋水平。

**AlphaZero [SHS17]**：与 AlphaGo 相似，**关键依赖 MCTS**，但**不用 rollout**（纯 NN prior/value + MCTS）。

棋类虽**确定性**，程序仍广泛使用**随机化策略**与 MCTS 采样（§2.0）——探索与训练数据生成需要。

---

#### 双人博弈中的 rollout

Backgammon 模式：固定对手用基策略，优化方用 rollout。零和博弈 DP 理论本书未覆盖；**解释需经验**（例 2.4.2）。围棋/象棋等通过 MCTS + NN 取得类似「在线改进」效果。

---

#### MCTS 的随机策略改进（Randomized Policy Improvement）

前述 rollout/MCTS 默认**确定性**策略：$\tilde\mu_k(x_k)$ 为唯一最优控 (2.31)。

**随机化基策略**：$x_k$ 上映射到 $U_k(x_k)$ 上的概率分布——采样轨迹方式与确定性基策略相同。

**得到随机 rollout 策略**：

- 将基策略概率向确定性 rollout 控 $\tilde\mu_k(x_k)$ **倾斜**（提高「最佳」控概率，降低其余）；或  
- **MCTS 频率计数**：自适应仿真中统计各 $u_k$ 被选/访问比例，按频率方向调整基策略概率——类似**梯度型**改进（§4.11；AlphaGo/AlphaZero 训练数据生成）。

AlphaGo/AlphaZero 用 MCTS 生成**局面–走法概率**供 NN 训练——值空间 + **策略空间**近似的组合，但非显式 policy gradient。

---

#### 方差缩减：比较 Q 差与 Advantage（(2.32)–(2.33)）

Rollout 选控取决于 **Q 因子差** $\tilde Q_k(x_k,u_k)-\tilde Q_k(x_k,\hat u_k)$，而非单个 Q 的绝对值。分别估计两个 Q 再相减，仿真误差可能被**放大**。

**共同随机数（common random numbers）**：若 $w_k$ 分布**不依赖** $(x_k,u_k)$，可直接采样**代价差**

```math
C_k(x_k,u_k,\mathbf w_k)-C_k(x_k,\hat u_k,\mathbf w_k),
```

其中 $\mathbf w_k=(w_k,\ldots,w_{N-1})$，

```math
C_k(x_k,u_k,\mathbf w_k)=g_k(x_k,u_k,w_k)+\sum_{i=k+1}^{N-1} g_i(x_i,\mu_i(x_i),w_i)+g_N(x_N),
```

$x_i$ 由 $(x_k,u_k)$ 与 $\mathbf w_k$ 及基策略 tail 确定。**同一** $\mathbf w_k$ 下比较两控 → 误差 $D_k=C_k-\tilde Q_k$ 在两控间**正相关**时，差分估计方差小于独立采样再相减。

**条件 (2.32)**：$\mathbb{E}[D_k(x_k,u_k,\mathbf w_k)\,D_k(x_k,\hat u_k,\mathbf w_k)]>0$（两控误差正相关）。

**充分条件 (2.33)**：首控 $u_k$ 对误差 $D_k$ 的影响相对扰动 $\mathbf w_k$ 较小（存在 $\gamma<1$ 使控间误差差期望 ≤ $\gamma$ 倍单控误差期望）→ (2.32) 成立 → **平均代价差样本**优于**独立 Q 平均再相减**。

**Advantage 函数**（rollout 外亦有用，§3.4）：

```math
A_k(x_k,u_k)=Q_k(x_k,u_k)-\min_{u\in U_k(x_k)} Q_k(x_k,u).
```

与 $u_k$ 无关的常数（如 $\min_u Q_k$）不影响控比较；参数化 $\tilde Q$ 时直接学 **advantage** 往往更稳。

---

## §2.5 On-Line Rollout for Deterministic Infinite-Spaces Problems

§2.4 的 rollout 在**离散控空间**上对 $U_k(x_k)$ **穷举**比较 Q 因子；连续/无限控空间下先离散化往往**不便且低效**。本节讨论**确定性**无限控空间的替代：**基启发式本身含连续优化**，用 NLP/最优控制解 lookahead min。

---

#### 连续控空间的一步 rollout（(2.34)–(2.35)）

```math
\tilde\mu_k(x_k)\in\arg\min_{u_k\in U_k(x_k)} \tilde Q_k(x_k,u_k), \qquad \tilde Q_k(x_k,u_k)=g_k(x_k,u_k)+H_{k+1}(f_k(x_k,u_k)).
```

若 $H_{k+1}$ 有**闭式可微**表达，且 $g_k,f_k$ 对 $u_k$ 可微，则 $\tilde Q_k$ 可微，(2.34) 用**梯度法**（无约束/约束 NLP 均可）——无需离散化控空间。

**限制**：闭式 $H_{k+1}$ 往往过强。**出路**：令 $H_{k+1}(x_{k+1})$ = 某 **$(\ell-1)$ 阶段**确定性最优控制问题的最优代价（与原问题相关）。则 (2.34) 的首步 min 与基启发式的 $(\ell-1)$ 步优化**无缝拼接**为 **$\ell$ 阶段**连续空间 NLP（Fig. 2.5.1），在线用标准非线性规划/最优控制求解。

**MPC** 即该框架在调节问题上的最重要实例。

---

### §2.5.1 Model Predictive Control

#### 调节问题与 LQ 的不足

**目标**：保持确定性系统状态接近**原点**（或给定 nominal 轨迹）——经典调节/跟踪问题。

- 动力学：$x_{k+1}=f_k(x_k,u_k)$；  
- 代价：$g_k(x_k,u_k)\ge 0$（常为二次型）；  
- 约束：$x_k\in X_k$，$u_k\in U_k(x_k)$；  
- **原点可零代价维持**（(2.36)）：$\exists\,\bar u_k\in U_k(0)$ 使 $f_k(0,\bar u_k)=0$，$g_k(0,\bar u_k)=0$。

**LQ 的两处局限**（Fig. 2.5.2 机器人避障为例）：

| 局限 | 说明 |
|------|------|
| **(a) 非线性** | 线性化模型用于控制可能不准 |
| **(b) 硬约束** | 二次惩罚是「软约束」，可能产生**违反**位置/速度/加速度/障碍约束的轨迹 |

**MPC** 融合：**多步前瞻** + **无限控空间 rollout（NLP）** + **确定性等价**；并天然支持**在线再规划**（障碍移动时重算）。

随机系统常见做法：用典型扰动值替不确定量 → 确定性版本（确定性等价，§2.3.2）。

---

#### MPC 算法（Fig. 2.5.3）

在 $x_k\in X_k$：

1. **(a)** 解 **$\ell$ 阶段**问题 (2.37)：最小化 $\sum_{i=k}^{k+\ell-1} g_i(x_i,u_i)$，满足 $x_{i+1}=f_i(x_i,u_i)$、$x_i\in X_i$、$u_i\in U_i(x_i)$，**终端** $x_{k+\ell}=0$；  
2. **(b)** 得最优序列 $\{\tilde u_k,\ldots,\tilde u_{k+\ell-1}\}$，**只执行** $\tilde u_k$，丢弃其余；  
3. **(c)** 观测 $x_{k+1}$ 后重复——下一步优化目标变为 $x_{k+\ell+1}=0$（比上一步多「瞄准」一步）。

**实际轨迹未必到 0**：每步只执行首控，$k+1$ 时生成的 $\tilde u_{k+1}$ 可能与上一序列中的 $\tilde u_{k+1}$ **不同**（因终端条件平移）。

**跟踪变体**：终端改为 nominal 轨迹上某点；代价改为偏离 nominal 的惩罚；可对关键段用**时变** $\ell_k$ 加强控制。

---

#### 约束能控（Constrained Controllability）

保证 (2.37) **可行**的充分条件：存在整数 $\ell>1$，使对**任意** $x_k\in X_k$，存在 $u_k,\ldots,u_{k+\ell-1}$ 在 $\ell$ 步内将 $x_{k+\ell}$ 驱至 **0**，且中间 $x_m\in X_m$、$u_m\in U_m(x_m)$ 均满足。

**直觉**：控约束不太紧、状态约束不允许离原点太远 → 通常可满足，且闭环倾向稳定。求合适的 $\ell$ 是重要设计问题。

**视界 $\ell$ 的选择**：若某 $\ell$ 满足约束能控，则更大 $\ell$ 也满足——大 $\ell$ 有利于控制但**每步 NLP 更大更难**；实践：先取满足能控的最小 $\ell$，再实验调性能。

---

#### 与 rollout 的关系

MPC 隐含的 $\tilde J$ = **基启发式** cost-to-go：在 $(\ell-1)$ 步内将状态驱至 0 并**此后保持** 0，同时满足约束、最小化 $(\ell-1)$ 段代价（Fig. 2.5.1 的 $(\ell-1)$ 阶段部分）。

记 $\hat J_k(x_k)$ = MPC 在 $x_k$ 所解 $\ell$ 阶段问题的最优值；$H_k(x_k)$、$H_{k+1}(x_{k+1})$ = 从 $x_k$、$x_{k+1}$ 起 $(\ell-1)$ 步驱至 0 的最优启发式代价。由最优性原理：

```math
\hat J_k(x_k)=\min_{u_k\in U_k(x_k)}\big[g_k(x_k,u_k)+H_{k+1}(f_k(x_k,u_k))\big].
```

少一阶段驱至 0 不会更便宜 → $\hat J_k(x_k)\le H_k(x_k)$。合并得**序贯改进** (2.38)：

```math
\min_{u_k\in U_k(x_k)}\big[g_k(x_k,u_k)+H_{k+1}(f_k(x_k,u_k))\big]\le H_k(x_k).
```

基启发式**序贯改进**但**非序贯一致**（§2.4.1；见例 2.5.1）——故 MPC 有 rollout 型代价改进，但 (2.27) 的严格等式归纳不适用。

---

#### 稳定性分析（(2.39)–(2.41)）

除约束外，常要求闭环**稳定**：无限时域总代价有限

```math
\sum_{k=0}^\infty g_k(x_k,u_k)<\infty.
```

由 (2.38) 逐步：

```math
g_k(x_k,u_k)+H_{k+1}(x_{k+1})\le H_k(x_k).
```

对 $k=0,\ldots,K$ telescoping：

```math
\sum_{k=0}^K g_k(x_k,u_k)\le H_0(x_0)-H_{K+1}(x_{K+1})\le H_0(x_0)<\infty,
```

（$H_0(x_0)$ 有限因约束能控保证 $x_0\to x_\ell=0$ 可行）。令 $K\to\infty$ 得 (2.39)——**总代价有上界**；**不保证**状态渐近到 0（例 2.5.1）。

---

#### 例 2.5.1（标量 LTI + 盒约束，$\ell=2$）

$x_{k+1}=x_k+u_k$，$g_k=x_k^2+u_k^2$，$|x_k|\le 1.5$，$|u_k|\le 1$。

**约束能控**：$u_0=-\mathrm{sgn}(x_0)$，$u_1=-x_1$ 可在 2 步内驱 $x_2=0$（$|x_0|\le 1.5$）。

**MPC 解析解**：$\tilde u_k=-\frac{3}{2}x_k$，$\tilde u_{k+1}=-(x_k+\tilde u_k)$；闭环 $x_{k+1}=x_k$——**Lyapunov 稳定**但 $x_0\neq 0$ 时**永不到 0**。

**非序贯一致**：从 $x_k=1$，基启发式给 $x_k=1,u_k=-1,x_{k+1}=\frac12,\ldots$；从 $x_{k+1}=\frac12$ 重启，后缀**不同**（$u_{k+1}$ 等改变）——违反 §2.4.1 后缀一致。

---

### §2.5.2 Target Tubes and Constrained Controllability

约束能控假设**掩盖**：控集可能不足以抵消系统**不稳定**倾向，无法在足够长的时间内保持 $x_k\in X_k$——一种**约束意义下的不稳定性**。

#### 例 2.5.2（不稳定标量系统，Fig. 2.5.4）

$x_{k+1}=2x_k+u_k$，$|u_k|\le 1$。

- **$0\le x_0<1$**：反复 $u_k=-1$ 使 $x_k$ 单调趋 0；到 $[0,\frac12]$ 后可用 $u=-2x$ 一步到 0。  
- **$-1<x_0\le 0$**：对称用 $u_k=+1$ 压向 0。  
- **$|x_0|>1$**：任意可行控下 $|x_k|\to\infty$——**不可控/不稳定**。

**状态约束** $X_k=[-\beta,\beta]$：

| $\beta$ | 结论 |
|---------|------|
| $0<\beta<1$ | 约束能控成立；$\ell$ 依赖 $\beta$（$0<\beta<1/2$ 时 $\ell=1$ 即可） |
| $\beta\ge 1$ | 从 $x_0\in[1,\beta]$ 无法在 $|u_k|\le 1$ 下驱至 0 → **约束能控失败**；需更大控集或更靠近原点的初值 |

**最大可达管道（largest reachable tube）**：$\bar X=\{x:|x|\le 1\}$ 反复应用。

---

#### 目标管道（Target Tube）

**定义**：管道 $\{\bar X^0,\bar X^1,\ldots,\bar X^N\}$，$\bar X^k\subset X_k$。**可达（reachable）**：对每个 $k$、$x_k\in\bar X^k$，$\exists u_k\in U_k(x_k)$ 使 $f_k(x_k,u_k)\in\bar X^{k+1}$。亦称 **effective target tube** [Ber71]；有界管道内可保持 ⟺ 一种**闭环稳定性**保证。

**原约束管道不可达**时：约束能控**不可能**——存在态一旦出管再也无法返回。须先算**可达子管道**作 MPC 状态约束。

**后向递推构造**（$N$ 阶段确定性问题）：

```math
\bar X^N=X_N, \qquad \bar X^k=\big\{x_k\in X_k \;\big|\; \exists u_k\in U_k(x_k): f_k(x_k,u_k)\in\bar X^{k+1}\big\}.
```

高维一般难精确算 $\bar X^k$；**内逼近**算法：椭球 [Ber71], [BeR71], [Ber72]；多面体 [BBM17]。

**例 2.5.2 续**：$X_k=\{x:|x|\le 1\}$ 时管道可达；$|x|\le 2$ 时 $x_0=2$ 一步即到 $|x_1|>2$ → 须缩至 $\bar X^k=\{x:|x|\le 1\}$。

**二次代价 + $\ell=2$ MPC**：需 $|x_0|\le 1$ 才可行；得 $\tilde u_k=-\frac{5}{3}x_k$，$x_{k+1}=\frac12 x_k\to 0$ **渐近**（与例 2.5.1 常值闭环对比）。

---

#### 可达 vs 约束能控

| 关系 | 说明 |
|------|------|
| 约束能控 ⇒ 管道可达 | 能有限步到 0 则必能在管道内滚动 |
| 管道可达 ⇏ 约束能控 | **边界效应**：$|x_k|\le 1$ 管道可达，但从边界 $\pm 1$ 无法在有限步内到 0（$|u_k|\le 1$）；去掉边界点后二者可同时成立 |

除边界情形外，管道可达通常蕴含约束能控。

---

### §2.5.3 Variants of Model Predictive Control

MPC 仅是方法论起点；许多变体与本章其他次优控制思想相连。

#### 终端处理变体

- **大终端惩罚**：不要求 $x_{k+\ell}=0$，而对 $x_{k+\ell}\neq 0$ 施加大惩罚——分析仍成立，若惩罚使 (2.38) 类序贯改进满足；  
- **小邻域 + 局部稳定器**：$\ell$ 步内进入原点附近小邻域，再切换其他方法设计的**稳定控制器**；  
- **Rollout + 终端余值**：与 §2.4、Ch.4 §4.6.1 无限时域变体结合，可处理不确定性/扰动。

---

#### 例 2.5.3（随机系统 + 确定性等价 MPC）

随机系统 $x_{k+1}=f_k(x_k,u_k,w_k)$，约束 $x_k\in X_k$，$u_k\in U_k(x_k)$，$w_k\in W_k$。

**鲁棒可达**：策略须在**最坏**扰动下仍保持管道 $\{X_0,X_1,\ldots\}$。可行控集缩小为

```math
\tilde U_k(x_k)=\big\{u_k\in U_k(x_k)\;\big|\; f_k(x_k,u_k,w_k)\in X_{k+1},\;\forall w_k\in W_k\big\},
```

需非空——同样依赖目标管道构造与足够「富」的 $U_k$。

**算法**（$\ell>1$，确定性等价基策略）：

1. 将 $w_{k+1},\ldots,w_{k+\ell-1}$ **固定**为典型值；  
2. 在 $\tilde U_k(x_k)$ 上 min Q 因子 (2.43)：

```math
\tilde Q_k(x_k,u_k)=\mathbb{E}\big[g_k(x_k,u_k,w_k)+H_{k+1}(f_k(x_k,u_k,w_k))\big],
```

$H_{k+1}$ = 从 $x_{k+1}$ 在固定扰动、控来自 $\tilde U_m$ 下 **$\ell-1$ 步确定性**驱至 0 的最优代价；  
3. 实现 = 首步 min 与 $(\ell-1)$ 步基启发式拼接的 **$\ell$ 阶段确定性 NLP**（可梯度法）。

**局限**：每态计算量大，**在线**可能过重。

**策略回归（§2.1.5）**：对大量样本态 $x_k^s$ 离线/批量算 $u_k^s=\arg\min \tilde Q_k(x_k^s,\cdot)$，再回归 $\tilde\mu_k(x)\approx u_k^s$——在值空间近似之上叠加**策略空间**近似，加速在线执行。

**其他**：最坏扰动下的管道 MPC 等见章末文献。

---

## 本章小结

| 机制 | 核心对象 | 典型保证/风险 |
|------|----------|----------------|
| 一步/多步前瞻 | $\tilde J_{k+\ell}$ | Q 误差斜率小则控排序稳健；$\ell$ 过大 + 差终端可退化 |
| 问题近似 | 简化问题的 $J^{*}$ | 启发式，需结构支持 |
| Rollout | $H_k$ 来自基策略/启发式 | 序贯改进 $\Rightarrow$ 不劣于基；加固可弱前提 |
| MPC | $\ell$ 步到 0 + 首控执行 | 约束能控 $\Rightarrow$ 有限总代价；未必到平衡点 |

**与后续章节**：参数化 $\tilde J$（Ch.3）、无限时域 VI/PI 与 rollout 界（Ch.4–5）、Actor-Critic 与 MCTS 工程化（Ch.4–5）。

---

*个人学习笔记；原著 Copyright Dimitri P. Bertsekas / Athena Scientific。*
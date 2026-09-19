# PPO（Proximal Policy Optimization）算法笔记

下文公式与流程对应的实现，见带注释的连续动作脚本[`ppo_continuous_action.py`](https://github.com/tangyx96/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)（自 [CleanRL](https://github.com/vwxyzjn/cleanrl) 分叉）。
网络结构见 [`ppo_mlp_notes.md`](ppo_mlp_notes.md)。

---

## 一、算法定位

PPO 是 John Schulman 等人于 2017 年提出的策略梯度方法，是当前最主流的深度强化学习算法之一，也是 OpenAI 的默认 RL 算法，被用于 ChatGPT 的 RLHF 训练中。

| 特性 | DQN | PPO |
|------|-----|-----|
| 类型 | 值函数方法（Value-based） | 策略梯度方法（Policy Gradient） |
| 输出 | 每个动作的 Q 值 | 动作的概率分布 |
| 学习方式 | Off-policy（经验池随机采样） | On-policy（当前策略收集，用完即弃） |
| 核心技巧 | 经验回放 + 目标网络 | Clipping + GAE |
| 网络结构 | 一个 Q 网络 | Actor（策略）+ Critic（价值）双网络 |

---

## 二、前置基础：策略梯度定理

强化学习的目标是最大化期望累积奖励：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]
$$

策略梯度定理给出了梯度：

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t \right]
$$

**直观理解**：

- $\hat{A}_t > 0$（动作比预期好）→ 增大选这个动作的概率
- $\hat{A}_t < 0$（动作比预期差）→ 减小选这个动作的概率
- $\hat{A}_t$ 绝对值越大，调整力度越大

### 策略梯度定理推导

下面从零开始，逐步推导策略梯度的形式。

**Step 1：定义目标函数**

强化学习的目标是最大化轨迹的期望累积奖励：

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]
$$

其中轨迹 $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \dots)$ 的概率为：

$$
p_\theta(\tau) = p(s_0) \prod_{t=0}^{T} \pi_\theta(a_t | s_t) \cdot p(s_{t+1} | s_t, a_t)
$$

**Step 2：将期望写成积分形式**

$$
J(\theta) = \int p_\theta(\tau) \cdot R(\tau) \, d\tau
$$

对 $\theta$ 求梯度：

$$
\nabla_\theta J(\theta) = \int \nabla_\theta p_\theta(\tau) \cdot R(\tau) \, d\tau
$$

**Step 3：使用对数求导技巧（Log-Derivative Trick）**

关键恒等式：

$$
\nabla_\theta \log p_\theta(\tau) = \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}
$$

移项得：

$$
\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \cdot \nabla_\theta \log p_\theta(\tau)
$$

代入梯度公式：

$$
\nabla_\theta J(\theta) = \int p_\theta(\tau) \cdot \nabla_\theta \log p_\theta(\tau) \cdot R(\tau) \, d\tau
$$

恢复为期望形式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \nabla_\theta \log p_\theta(\tau) \cdot R(\tau) \right]
$$

**Step 4：展开 $\log p_\theta(\tau)$**

$$
\log p_\theta(\tau) = \log p(s_0) + \sum_{t=0}^{T} \log \pi_\theta(a_t | s_t) + \sum_{t=0}^{T} \log p(s_{t+1} | s_t, a_t)
$$

对 $\theta$ 求梯度，**只有 $\log \pi_\theta(a_t|s_t)$ 这一项依赖 $\theta$**，其余项梯度为 0：

$$
\nabla_\theta \log p_\theta(\tau) = \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t)
$$

**Step 5：代入得原始策略梯度（REINFORCE）**

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot R(\tau) \right]
$$

这就是最原始的 REINFORCE 算法梯度：用整条轨迹的总回报 $R(\tau)$ 作为每个动作的"得分"。

**Step 6：引入 Baseline（减小方差）**

原始形式的问题：$R(\tau)$ 的方差很大。引入一个与 $\theta$ 无关的 baseline $b(s_t)$：

$$
\mathbb{E}\left[\nabla_\theta \log \pi_\theta(a_t|s_t) \cdot b(s_t)\right] = 0
$$

原因：$\int \pi_\theta(a|s) \cdot \nabla_\theta \log \pi_\theta(a|s) \cdot b(s) \, da = b(s) \cdot \nabla_\theta \int \pi_\theta(a|s) \, da = b(s) \cdot \nabla_\theta 1 = 0$

于是可以减去 baseline 而不改变期望：

$$
\nabla_\theta J(\theta) = \mathbb{E}\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \big(R(\tau) - b(s_t)\big) \right]
$$

**Step 7：用时序因果性化简**

把 $R(\tau)=\sum_{k=0}^{T}\gamma^k r_k$ 代入，得到双求和：

$$
\nabla_\theta J(\theta)
=\mathbb{E}\left[\sum_{t=0}^{T}\nabla_\theta\log\pi_\theta(a_t|s_t)\sum_{k=0}^{T}\gamma^k r_k\right]
=\mathbb{E}\left[\sum_{t=0}^{T}\sum_{k=0}^{T}\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot\gamma^k r_k\right]
$$

固定 $s_t$，score 对动作求平均必为 0（概率和为 1）：

$$
\mathbb{E}\big[\nabla_\theta\log\pi_\theta(a_t|s_t)\mid s_t\big]
=\int\pi_\theta(a|s_t)\,\nabla_\theta\log\pi_\theta(a|s_t)\,da
=\nabla_\theta\int\pi_\theta(a|s_t)\,da
=\nabla_\theta 1=0
$$

对每个 $(t,k)$ 分两种：

- **$k<t$**：到达 $s_t$ 时 $r_k$ 已确定，故 $\mathbb{E}[\nabla\log\pi\cdot r_k\mid s_t]=r_k\cdot\mathbb{E}[\nabla\log\pi\mid s_t]=r_k\cdot 0=0$，再取外层期望仍为 0。
- **$k\ge t$**：$r_k$ 还取决于 $a_t$，不能提出去，这才是真正的信号。

因此内层只留下 $k\ge t$。令 $G_t=\sum_{t'=t}^{T}\gamma^{t'-t}r_{t'}$（严格说前面还有 $\gamma^t$，实现里常省掉）：

$$
\nabla_\theta J(\theta)=\mathbb{E}\left[\sum_{t=0}^{T}\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot G_t\right]
$$

过去奖励对所有 $a_t$ 都一样，只加噪声；砍掉后期望不变、方差变小。这和 Step 6 的 baseline 是同一条恒等式。

**Step 8：用 Advantage 替换回报**

$Q^\pi(s_t,a_t)=\mathbb{E}[G_t\mid s_t,a_t]$。单条轨迹上 $G_t\neq Q$，但 $\nabla_\theta\log\pi_\theta(a_t|s_t)$ 只依赖 $(s_t,a_t)$，给定后可提出去：

$$
\begin{aligned}
\mathbb{E}\big[\nabla_\theta\log\pi\cdot G_t\big]
&=\mathbb{E}\Big[\mathbb{E}\big[\nabla_\theta\log\pi\cdot G_t\mid s_t,a_t\big]\Big]
\quad\text{（全期望公式）}\\
&=\mathbb{E}\Big[\nabla_\theta\log\pi\cdot\mathbb{E}[G_t\mid s_t,a_t]\Big]\\
&=\mathbb{E}\big[\nabla_\theta\log\pi\cdot Q^\pi(s_t,a_t)\big]
\end{aligned}
$$

Step 6 里任意只依赖 $s_t$ 的 $b(s_t)$ 都不改变期望。取 $b=V^\pi$，因为 $V^\pi(s)=\mathbb{E}[Q^\pi(s,A)\mid s]$（$A\sim\pi(\cdot\mid s)$），于是

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
\quad\Longrightarrow\quad
\mathbb{E}[A^\pi\mid s]=\mathbb{E}[Q^\pi\mid s]-V^\pi(s)=0.
$$

残差相对该状态平均水平，方差比 $G_t$ 或 $Q$ 小。

取 $f(s,a)=\nabla_\theta\log\pi_\theta(a|s)\,A^\pi(s,a)$，真正的梯度是「一条轨迹上对 $t$ **求和**，再对轨迹平均」：

$$
\nabla_\theta J(\theta)
=\mathbb{E}_\tau\left[\sum_{t=0}^{T}f(s_t,a_t)\right]
=\sum_{t=0}^{T}\mathbb{E}\big[f(s_t,a_t)\big]
$$

占用测度 $d^\pi$ 是各时刻分布的等权平均（等价于先均匀抽 $t\in\{0,\ldots,T\}$，再抽 $(s_t,a_t)$）：

$$
d^\pi(s,a)=\frac{1}{T+1}\sum_{t=0}^{T}P(s_t=s,\,a_t=a)
$$

代入期望定义：

$$
\begin{aligned}
\mathbb{E}_{(s,a)\sim d^\pi}[f]
&=\sum_{s,a}d^\pi(s,a)\,f(s,a)\\
&=\sum_{s,a}\left(\frac{1}{T+1}\sum_{t=0}^{T}P(s_t=s,a_t=a)\right)f(s,a)\\
&=\frac{1}{T+1}\sum_{t=0}^{T}\underbrace{\sum_{s,a}P(s_t=s,a_t=a)\,f(s,a)}_{=\mathbb{E}[f(s_t,a_t)]}\\
&=\frac{1}{T+1}\sum_{t=0}^{T}\mathbb{E}[f(s_t,a_t)]
\end{aligned}
$$

故 $\sum_{t=0}^{T}\mathbb{E}[f(s_t,a_t)]=(T+1)\,\mathbb{E}_{d^\pi}[f]$。$\theta\leftarrow\theta+\alpha g$ 里 $T+1$ 可并入 $\alpha$，故

$$
\nabla_\theta J(\theta)\propto\mathbb{E}_{(s,a)\sim d^\pi}\big[\nabla_\theta\log\pi_\theta(a|s)\,A^\pi(s,a)\big]
$$

代码把所有时间步放进同一 batch 做 `mean`，就是对 $d^\pi$ 的蒙特卡洛（$N=$ 轨迹数 $\times$ 步数）：

$$
\frac{1}{N}\sum_{i=1}^{N}\nabla_\theta\log\pi_\theta(a^{(i)}|s^{(i)})\,\hat A^{(i)}
$$

框里的 $\mathbb{E}$ 指这个平均，不是再对 $t$ 求和；下标 $t$ 只表示 buffer 里的某一对 $(s,a)$：

$$
\boxed{\nabla_\theta J(\theta)=\mathbb{E}\left[\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot\hat A_t\right]}
$$

**推导链总结**：

```text
J(θ) = E[R(τ)]
   ↓ 对数求导技巧
∇J = E[∇log p(τ) · R(τ)]
   ↓ 展开 log p(τ)，只有 π 项依赖 θ
∇J = E[Σ ∇log π(a|s) · R(τ)]
   ↓ 引入 baseline（减方差，不改变期望）
∇J = E[Σ ∇log π(a|s) · (R(τ) - b(s))]
   ↓ 时序因果性：过去奖励与 ∇logπ(a_t|s_t) 期望为 0
∇J = E[Σ ∇log π(a_t|s_t) · G_t]
   ↓ E[G_t|s,a]=Q；b=V（状态均值）得 A=Q-V
∇J = E_τ[Σ ∇log π · A] = E_{s,a}[∇log π(a|s) · A(s,a)]
```

---

## 三、痛点：为什么普通策略梯度不够好

1. **步长敏感**：学习率稍大，策略一步跨太远直接崩溃，且无法恢复（on-policy 数据已失效）
2. **数据利用率低**：每次更新后，之前收集的数据"过期"，必须重新收集
3. **更新不稳定**：策略空间是弯曲的，沿直线更新容易偏离最优方向

---

## 四、TRPO 的思路：信任区域约束

TRPO 提出：每次更新时限制新旧策略的 KL 散度，确保在"可信赖区域"内更新。

**问题**：约束优化涉及二阶导数（Hessian 矩阵），计算复杂，实现困难。

---

## 五、PPO 的核心创新：Clipping

PPO 用一个极其简单的裁剪操作替代了 TRPO 的复杂约束优化。

### 5.1 重要性采样比率

$$
r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}
$$

- $r_t = 1$：新旧策略在这个动作上概率相同
- $r_t > 1$：新策略更倾向于选这个动作
- $r_t < 1$：新策略更不倾向于选这个动作

### 5.2 裁剪目标函数

$$
L^{CLIP}(\theta) = \mathbb{E}\left[ \min\left( r_t(\theta) \cdot \hat{A}_t, \quad \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \cdot \hat{A}_t \right) \right]
$$

其中 $\epsilon$ 通常取 0.2。

### 5.3 分情况分析

**情况 A：$\hat{A}_t > 0$（好动作，想增大概率）**

- $r_t \leq 1+\epsilon$：正常更新，梯度正常
- $r_t > 1+\epsilon$：被裁剪，梯度为 0，停止更新

> 效果：好的动作概率最多增大到原来的 $(1+\epsilon)$ 倍，之后不再奖励。

**情况 B：$\hat{A}_t < 0$（坏动作，想减小概率）**

- $r_t \geq 1-\epsilon$：正常更新，梯度正常
- $r_t < 1-\epsilon$：被裁剪，梯度为 0，停止更新

> 效果：坏的动作概率最多减小到原来的 $(1-\epsilon)$ 倍，之后不再惩罚。

### 5.4 核心思想一句话

> 如果策略更新在可接受范围内（$r_t \in [1-\epsilon, 1+\epsilon]$），正常更新；如果太激进，直接裁剪掉梯度。这等价于给策略更新加了一个"软约束"。

---

## 六、$L^{CLIP}$ 与策略梯度的关系

$$
\nabla_\theta L^{CLIP} = \begin{cases}
r_t(\theta) \cdot \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \hat{A}_t, & |r_t - 1| \leq \epsilon \\[8pt]
0, & |r_t - 1| > \epsilon
\end{cases}
$$

**推导关键**：$\nabla_\theta r_t(\theta) = r_t(\theta) \cdot \nabla_\theta \log \pi_\theta(a_t|s_t)$

- 在裁剪范围内，梯度就是带重要性权重 $r_t$ 的策略梯度
- 超出裁剪范围，梯度被硬截断为 0

---

## 七、GAE：优势函数估计

策略梯度需要 $\hat{A}_t$，即「当前动作相对平均水平好多少」。精确的优势

$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)
$$

无法直接观测。GAE（Generalized Advantage Estimation，Schulman et al., 2016）把优势写成 **TD 残差的加权和**，用 $\lambda$ 在偏差与方差之间插值。

### 7.1 TD 误差：Bellman 残差与一步优势

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

这是价值函数 Bellman 方程的**一步残差**（TD(0) 误差）。若 $V = V^\pi$ 准确，则

$$
V^\pi(s_t) = \mathbb{E}\big[r_t + \gamma V^\pi(s_{t+1})\big]
$$

因此 $\delta_t$ 的含义是：实际发生的「即时奖励 + 折扣后的下一状态价值」，相对当前 $V(s_t)$ 多出或少掉多少。

- $\delta_t > 0$：这一转移比 Critic 预期更好（正 surprise）
- $\delta_t < 0$：比预期更差

又因 $Q^\pi(s_t,a_t) = \mathbb{E}[r_t + \gamma V^\pi(s_{t+1}) \mid s_t, a_t]$，当 $V = V^\pi$ 时

$$
\mathbb{E}[\delta_t \mid s_t, a_t] = A^\pi(s_t, a_t)
$$

即 **一步 TD 误差是优势的无偏估计**（条件于 $s_t,a_t$）。仅条件于 $s_t$ 时 $\mathbb{E}[\delta_t \mid s_t] = 0$，好坏动作相互抵消。Actor 需要的正是「这个动作相对平均好多少」，故 $\delta_t$ 是构造 $\hat{A}_t$ 的基本积木。

一步 bootstrap 方差小，但 $V$ 不准时偏差会直接进入 $\hat{A}_t$。为此引入更长视野。

### 7.2 $n$ 步优势：TD 误差的折扣和

$n$ 步回报与对应优势：

$$
G_t^{(n)} = r_t + \gamma r_{t+1} + \cdots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})
$$

$$
\hat{A}_t^{(n)} = G_t^{(n)} - V(s_t)
$$

它可改写成 TD 误差的折扣和。$n=1$ 即上一小节：

$$
\hat{A}_t^{(1)} = r_t + \gamma V(s_{t+1}) - V(s_t) = \delta_t
$$

$n=2$ 时加减一项 $V(s_{t+1})$，把两步真实奖励拆成两段一步残差：

$$
\begin{aligned}
\hat{A}_t^{(2)}
&= r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t) \\
&= \big(r_t + \gamma V(s_{t+1}) - V(s_t)\big)
  + \gamma\big(r_{t+1} + \gamma V(s_{t+2}) - V(s_{t+1})\big) \\
&= \delta_t + \gamma\,\delta_{t+1}
\end{aligned}
$$

归纳得：

$$
\hat{A}_t^{(n)} = \sum_{\ell=0}^{n-1} \gamma^\ell \delta_{t+\ell}
$$

> $n$ 步优势 = 从现在起连续 $n$ 个「一步 surprise」的折扣累加。后面的 $\delta$ 会修正前面用 $V$ bootstrap 带来的误差。

### 7.3 GAE：对所有 $n$ 步估计做指数加权

GAE 不是另造一种优势，而是把全部 $n$ 步估计做指数加权平均：

$$
A_t^{\mathrm{GAE}(\gamma,\lambda)}
= (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} \hat{A}_t^{(n)}
$$

- $\lambda = 0$：只保留 $\hat{A}_t^{(1)} = \delta_t$（低方差、高偏差）
- $\lambda \to 1$：趋向 Monte Carlo / 全轨迹回报（低偏差、高方差）

将 $\hat{A}_t^{(n)} = \sum_{\ell=0}^{n-1} \gamma^\ell \delta_{t+\ell}$ 代入并交换求和顺序：固定 $\delta_{t+\ell}$，它出现在所有 $n > \ell$ 的项中，系数为

$$
(1-\lambda)\sum_{n=\ell+1}^{\infty}\lambda^{n-1} = \lambda^\ell
$$

于是得到展开式：

$$
\boxed{A_t^{\mathrm{GAE}} = \sum_{\ell=0}^{\infty} (\gamma\lambda)^\ell \delta_{t+\ell}
= \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots}
$$

有效折扣是 $\gamma\lambda$：$\gamma$ 来自回报折扣，$\lambda$ 来自「是否采信更远的真实奖励」。

### 7.4 递推公式与倒序计算

对展开式做一步移位：

$$
A_{t+1}^{\mathrm{GAE}} = \delta_{t+1} + (\gamma\lambda)\delta_{t+2} + (\gamma\lambda)^2\delta_{t+3} + \cdots
$$

立刻得到递推：

$$
\boxed{A_t^{\mathrm{GAE}} = \delta_t + (\gamma\lambda) \cdot A_{t+1}^{\mathrm{GAE}}}
$$

轨迹末端取 $A_T = 0$（或最后一步只保留 $\delta_{T-1}$），因此实现上必须**从后往前**扫描。因 $A_t \approx G_t - V(s_t)$，加回价值得到 GAE 配套的回报目标：

$$
R_t = A_t + V(s_t)
$$

- $\hat{A}_t$：更新 Actor（$\theta$），进 $L^{\mathrm{CLIP}}$ / 加权策略梯度
- $R_t$：更新 Critic（$\phi$），回归拟合 $V_\phi(s_t)\approx R_t$

### 7.5 $\lambda$ 的作用：偏差–方差权衡

| $\lambda$ | 含义 | 特点 |
|-----------|------|------|
| $\lambda = 0$ | 只看一步 TD 误差 | 低方差，高偏差 |
| $\lambda = 1$ | 看全部未来 TD 误差 | 高方差，低偏差 |
| $\lambda = 0.95$（默认） | 折中 | 实践常用 |

$\lambda$ 不是再乘一个折扣，而是在「信模型 $V$」与「信轨迹上的真实奖励」之间插值：

| | $V$ 准确 | $V$ 不准 |
|--|--|--|
| **任意 $\lambda$** | $\mathbb{E}[\delta_{t+\ell} \mid s_t, a_t] = 0\ (\ell \ge 1)$，GAE 对 $A^\pi$ **无偏** | 后面的 $\delta$ 带着真实奖励，能纠正错误的 $V$ |
| **$\lambda$ 小** | 少用未来噪声，方差小 | 更依赖当前 $V$，偏差大 |
| **$\lambda$ 大** | 许多随机 $\delta$ 相加，方差大 | 更接近真实回报，偏差小 |

与 PPO 的衔接：正的 $\hat{A}_t$ 提高 $\pi(a_t \mid s_t)$，负的则压低。$\delta_t$ 给出每一步的局部评价；GAE 把未来残差按 $\gamma\lambda$ 折进来，得到比纯 TD(0) 更稳、比纯 Monte Carlo 噪声更小的优势估计。

---

## 八、Actor-Critic 架构

```text
     ┌──────────────────┐
     │   共享特征提取    │
     └──────┬───────────┘
            │
    ┌───────┴───────┐
    ▼               ▼
┌─────────┐ ┌─────────┐
│  Actor  │ │ Critic  │
│  π(a|s) │ │  V(s)   │
└─────────┘ └─────────┘
 输出动作概率  输出状态价值
```

- **Actor**：决定"做什么"，输出动作概率分布
- **Critic**：评价"做得好不好"，输出状态价值 $V(s)$，用于计算 Advantage

上图是概念框图。CleanRL `ppo_continuous_action.py` 里 Actor / Critic 是**两套独立 MLP**，不共享特征；均值由 MLP 给出，标准差是与状态无关的 `logstd`。展开说明见 [`ppo_mlp_notes.md`](ppo_mlp_notes.md)。

---

## 九、完整训练流程

```text
1. 用当前策略 π_θ 与环境交互，收集 num_steps 步数据
   每步记录：(s, a, r, log π_old(a|s), V(s), done)

2. 用 GAE 倒序计算每个时间步的 advantage 和 return
   A_t = δ_t + γλ·A_{t+1}
   R_t = A_t + V(s_t)

3. 将收集的数据打乱，分成 minibatches

4. 重复 update_epochs 次（默认 4 次）：
   对每个 minibatch：
     a. 计算 r_t = π_new(a|s) / π_old(a|s)
     b. 计算裁剪后的策略损失 L_CLIP
     c. 计算价值函数损失（也有裁剪）
     d. 计算熵损失（鼓励探索）
     e. 总损失 = L_CLIP - ent_coef * entropy + vf_coef * v_loss
     f. 反向传播，梯度裁剪，更新参数
     g. 估计新旧策略的 KL，并统计落入裁剪区的样本比例（诊断；可选按 KL 提前结束本批更新）

5. 丢弃旧数据，回到步骤 1
```

---

## 十、总损失函数

$$
L_{total} = L^{CLIP} - c_{ent} \cdot H(\pi_\theta) + c_{vf} \cdot L^{VF}
$$

- **$L^{CLIP}$**：裁剪后的策略损失（见第五节）
- **$-c_{ent} \cdot H(\pi_\theta)$**：熵正则化（负号 = 最大化熵），鼓励探索，防止过早收敛
- **$c_{vf} \cdot L^{VF}$**：价值函数损失，让 Critic 准确估计 $V(s)$

实现里对应 `L_total = L_CLIP - ent_coef * entropy + vf_coef * v_loss`。

### 10.1 熵 $H(\pi_\theta)$

状态 $s$ 上策略分布的 Shannon 熵。离散动作：

$$
H(\pi_\theta(\cdot|s)) = -\sum_a \pi_\theta(a|s)\log\pi_\theta(a|s)
$$

连续动作常用对角高斯的解析熵：

$$
H(\pi_\theta(\cdot|s)) = \frac{1}{2}\sum_i \log\bigl(2\pi e\,\sigma_i(s)^2\bigr)
$$

损失里对旧策略采集的状态取期望：

$$
H(\pi_\theta) = \mathbb{E}_{s\sim\pi_{\theta_{old}}}\bigl[H(\pi_\theta(\cdot|s))\bigr]
$$

熵越大，动作越不确定。减去 $c_{ent} H$ 等价于最大化熵；$c_{ent}$（`ent_coef`）通常取 $0.01$。

### 10.2 价值损失 $L^{VF}$

回归目标是 GAE 回报（第九节）：$\hat{R}_t = \hat{A}_t + V_{\theta_{old}}(s_t)$。未裁剪时为 MSE：

$$
L^{VF} = \mathbb{E}\Bigl[\bigl(V_\theta(s_t) - \hat{R}_t\bigr)^2\Bigr]
$$

价值裁剪（与 $L^{CLIP}$ 同构）：先把 $V$ 的变化限制在 $\pm\epsilon$，再取较坏的一边，避免 Critic 一步跳太远：

$$
V^{clip}_t = V_{\theta_{old}}(s_t) + \mathrm{clip}\bigl(V_\theta(s_t) - V_{\theta_{old}}(s_t),\; -\epsilon,\; \epsilon\bigr)
$$

$$
L^{VF} = \mathbb{E}\Bigl[\max\Bigl(
  \bigl(V_\theta(s_t)-\hat{R}_t\bigr)^2,\;
  \bigl(V^{clip}_t-\hat{R}_t\bigr)^2
\Bigr)\Bigr]
$$

$c_{vf}$（`vf_coef`）通常取 $0.5$。

### 10.3 策略偏移的度量

$L^{\mathrm{CLIP}}$ 在每个状态–动作对上限制 $r_t$，并不直接约束分布距离。同一批数据上多轮更新后，$\pi_\theta$ 仍可能整体远离 $\pi_{\theta_{old}}$。TRPO 用

$$
\mathbb{E}_{s}\bigl[\mathrm{KL}\bigl(\pi_{\theta_{old}}(\cdot\mid s)\,\|\,\pi_\theta(\cdot\mid s)\bigr)\bigr]
$$

界定可信步长；PPO 的 clip 是其替代，该期望本身仍是「策略走了多远」的本征量。精确 KL 须对动作积分。在旧策略采样下 $r_t=\pi_\theta(a_t\mid s_t)/\pi_{\theta_{old}}(a_t\mid s_t)$，有

$$
\mathbb{E}[-\log r_t],
\qquad
\mathbb{E}[(r_t-1)-\log r_t]
$$

（Schulman 的一阶与 k3 估计）。二者在 $r_t\equiv 1$ 时为零；后者对 $r_t>0$ 非负，偏差较小。它们是对 KL 的估计，不进入 $L_{total}$。若估计值超过预定阈值，可停止本批后续轮次，相当于在 clip 之外再加一条显式的信赖域。

落入 $|r_t-1|>\varepsilon$ 的样本比例刻画裁剪是否已经「顶死」：比例高则多数轨迹已离开线性可信区；接近零则更新几乎未触边界。与 KL 估计一样，这是对步长几何的观察，不是优化目标。

---

## 十一、PPO 成功的关键设计

| 设计 | 解决的问题 | 实现方式 |
|------|-----------|---------|
| **Clipping** | 策略更新太激进导致崩溃 | `clip(r_t, 1-ε, 1+ε)` |
| **GAE** | Advantage 估计的偏差-方差权衡 | `λ` 参数控制 |
| **多轮更新** | 数据利用率低 | `update_epochs`，同一批数据反复用 |
| **熵正则化** | 策略过早收敛，探索不足 | `-ent_coef * entropy` |
| **梯度裁剪** | 梯度爆炸 | `max_grad_norm` |
| **价值裁剪** | Critic 更新太激进 | 对 value loss 也做裁剪 |
| **学习率衰减** | 后期收敛不稳定 | 线性衰减到 0 |
| **KL 估计 / 裁剪占比** | clip 不保证分布距离；多轮后仍可能偏离 | 用 $r_t$ 估计 KL；可选超阈即停本批更新 |

---

## 十二、直观类比

把 PPO 训练想象成**学骑自行车**：

- **Actor**：你的身体，决定怎么动
- **Critic**：你的大脑，判断"这个姿势稳不稳"
- **Advantage**：偏离平衡的程度（正 = 稳了，负 = 要倒了）
- **Clipping**：你不会一次做太大动作调整（否则会摔），而是小步调整
- **多轮更新**：同一段经验反复琢磨"刚才哪里做得不够好"
- **熵正则化**：偶尔尝试新姿势，防止固化成坏习惯

---

## 十三、关键超参数

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `num_envs` | 4 | 并行环境数量 |
| `num_steps` | 128 | 每次收集步数 |
| `update_epochs` | 4 | 每批数据重复训练轮数 |
| `clip_coef` | 0.2 | 裁剪系数 ε |
| `gamma` | 0.99 | 折扣因子 |
| `gae_lambda` | 0.95 | GAE 的 λ |
| `ent_coef` | 0.01 | 熵正则化系数 |
| `vf_coef` | 0.5 | 价值损失权重 |
| `max_grad_norm` | 0.5 | 梯度裁剪阈值 |
| `target_kl` | 不启用 | 估计 KL 超过该值则结束本批更新 |
# PPO（Proximal Policy Optimization）算法笔记

下文公式与流程对应的实现，见带注释的连续动作脚本[`ppo_continuous_action.py`](https://github.com/tangyx96/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)（自 [CleanRL](https://github.com/vwxyzjn/cleanrl) 分叉）。
网络结构见 [`ppo_mlp_notes.md`](ppo_mlp_notes.md)。

---

## 一、算法定位

PPO 是 John Schulman 等人于 2017 年提出的策略梯度方法，长期是深度强化学习中的代表性基线，也曾用于 InstructGPT 的 RLHF 训练。具体系统是否采用 PPO 取决于训练阶段与实现版本，不能由这一公开实例推及所有后续模型。

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

以归一化折扣占用测度 $d_\gamma^{\pi_\theta}$ 为采样分布时，策略梯度可写为（忽略与 $\theta$ 无关的比例常数）：

$$
\nabla_\theta J(\theta)
\propto
\mathbb E_{(s,a)\sim d_\gamma^{\pi_\theta}}
\left[
\nabla_\theta\log\pi_\theta(a\mid s)\,A^{\pi_\theta}(s,a)
\right].
$$

若直接对一条轨迹求和，则每个时刻还带有 $\gamma^t$；常见 PPO 实现使用 rollout 的等权经验分布，得到的是相应的样本梯度代理。两种约定在后文分别说明。

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

给定选择 $a_t$ 之前的完整历史 $\mathcal F_t$，score 对动作求平均必为 0（概率和为 1）：

$$
\mathbb{E}\big[\nabla_\theta\log\pi_\theta(a_t|s_t)\mid\mathcal F_t\big]
=\int\pi_\theta(a|s_t)\,\nabla_\theta\log\pi_\theta(a|s_t)\,da
=\nabla_\theta\int\pi_\theta(a|s_t)\,da
=\nabla_\theta 1=0
$$

对每个 $(t,k)$ 分两种：

- **$k<t$**：在 $\mathcal F_t$ 下，$r_k$ 已确定，故 $\mathbb{E}[\nabla\log\pi\cdot r_k\mid\mathcal F_t]=r_k\cdot\mathbb{E}[\nabla\log\pi\mid\mathcal F_t]=r_k\cdot0=0$，再取外层期望仍为 $0$。
- **$k\ge t$**：$r_k$ 还取决于 $a_t$，不能提出去，这才是真正的信号。

因此内层只留下 $k\ge t$。令 $G_t=\sum_{t'=t}^{T}\gamma^{t'-t}r_{t'}$，则对上述折扣回报目标有

$$
\nabla_\theta J(\theta)=\mathbb{E}\left[\sum_{t=0}^{T}\gamma^t\nabla_\theta\log\pi_\theta(a_t|s_t)\cdot G_t\right].
$$

$\gamma^t$ 随 $t$ 变化，不能作为统一常数吸收到学习率。许多 PPO 实现对采样到的时间步等权平均，不显式乘 $\gamma^t$；这对应实践中的代理目标或未折扣状态访问分布，而不是上式所定义折扣目标的逐项无偏估计。下文推导若采用严格折扣目标，均应保留 $\gamma^t$；描述常见实现时则单独说明等权采样约定。

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

取 $f(s,a)=\nabla_\theta\log\pi_\theta(a|s)\,A^\pi(s,a)$。对折扣目标，梯度是「一条轨迹上按 $\gamma^t$ 加权求和，再对轨迹平均」：

$$
\nabla_\theta J(\theta)
=\mathbb{E}_\tau\left[\sum_{t=0}^{T}\gamma^t f(s_t,a_t)\right]
=\sum_{t=0}^{T}\gamma^t\mathbb{E}\big[f(s_t,a_t)\big].
$$

有限时域的归一化折扣占用测度为

$$
d_{\gamma,T}^\pi(s,a)
=\frac{1}{C_T}\sum_{t=0}^{T}\gamma^tP(s_t=s,\,a_t=a),
\qquad
C_T=\sum_{t=0}^{T}\gamma^t.
$$

代入期望定义：

$$
\begin{aligned}
\mathbb{E}_{(s,a)\sim d_{\gamma,T}^\pi}[f]
&=\frac{1}{C_T}\sum_{t=0}^{T}\gamma^t\mathbb{E}[f(s_t,a_t)].
\end{aligned}
$$

故 $\sum_t\gamma^t\mathbb{E}[f(s_t,a_t)]=C_T\mathbb{E}_{d_{\gamma,T}^\pi}[f]$。$C_T$ 与 $\theta$ 无关，可并入学习率，故

$$
\nabla_\theta J(\theta)\propto\mathbb{E}_{(s,a)\sim d_{\gamma,T}^\pi}\big[\nabla_\theta\log\pi_\theta(a|s)\,A^\pi(s,a)\big].
$$

令 $T\to\infty$ 可得常用定义 $d_\gamma^\pi(s,a)=(1-\gamma)\sum_{t\ge0}\gamma^tP(s_t=s,a_t=a)$。若改用各时刻等权的经验分布，则应将所得更新称为 PPO 的采样代理，而不再与上述折扣占用测度混写。

常见代码把 rollout 中所有时间步放进同一 batch 做 `mean`。这相当于对该批次诱导的等权经验分布 $\hat d_{\mathrm{rollout}}$ 做蒙特卡洛平均，而不是显式按 $d_{\gamma,T}^\pi$ 的 $\gamma^t$ 加权：

$$
\frac{1}{N}\sum_{i=1}^{N}\nabla_\theta\log\pi_\theta(a^{(i)}|s^{(i)})\,\hat A^{(i)}
$$

实现中的期望指这个经验平均，下标 $t$ 只表示 buffer 里的一对 $(s_t,a_t)$。因此更准确的写法是 PPO 所用的样本梯度代理

$$
\boxed{
g_{\mathrm{PPO}}
=\mathbb{E}_{(s_t,a_t)\sim\hat d_{\mathrm{rollout}}}
\left[\nabla_\theta\log\pi_\theta(a_t|s_t)\,\hat A_t\right]
}
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

## 三、普通策略梯度的数值局限

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
- $r_t > 1+\epsilon$：对该样本，裁剪分支成为常数，该分支的梯度为 0

> 效果：当该样本的比率越过 $1+\epsilon$ 后，继续增大比率不再提高它对裁剪代理目标的贡献。

**情况 B：$\hat{A}_t < 0$（坏动作，想减小概率）**

- $r_t \geq 1-\epsilon$：正常更新，梯度正常
- $r_t < 1-\epsilon$：对该样本，裁剪分支成为常数，该分支的梯度为 0

> 效果：当该样本的比率越过 $1-\epsilon$ 后，继续减小比率不再提高它对裁剪代理目标的贡献。

### 5.4 小结

裁剪只截平朝有利方向走得过远时的样本目标：$\hat A_t>0$ 时截平 $r_t>1+\epsilon$ 的收益，$\hat A_t<0$ 时截平 $r_t<1-\epsilon$ 的收益；反向越界不截平。它不是对 $r_t$ 或 KL 的硬约束：共享参数受到其他样本和后续 minibatch 的更新后，某个样本的比率仍可落在区间之外。PPO 以这一单侧代理替代 TRPO 的显式 KL 约束。

---

## 六、$L^{CLIP}$ 与策略梯度的关系

$\nabla_\theta r_t(\theta)=r_t(\theta)\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)$。$\min$ 取未裁剪项时，

$$
\nabla_\theta\big(r_t(\theta)\,\hat A_t\big)
=r_t(\theta)\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat A_t;
$$

取裁剪项时该支路对 $\theta$ 为常数，梯度为零。由第五节，未裁剪项被选中当且仅当优势尚未把比率推过有利一侧的边界：

$$
\nabla_\theta L^{\mathrm{CLIP}}
=
\mathbb{E}\left[
\begin{cases}
r_t(\theta)\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat A_t,
& \hat A_t\ge 0,\; r_t\le 1+\epsilon
\quad\text{或}\quad
\hat A_t<0,\; r_t\ge 1-\epsilon,\\[6pt]
0,
& \text{其余}.
\end{cases}
\right].
$$

$\hat A_t>0$ 且 $r_t<1-\epsilon$，或 $\hat A_t<0$ 且 $r_t>1+\epsilon$ 时，比率虽已离开 $[1-\epsilon,1+\epsilon]$，仍走未裁剪支路。

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
| **任意 $\lambda$** | 精确 $V^\pi$、on-policy 转移和正确终止处理下，未来残差对当前历史的条件均值为零，故 GAE 对 $A^\pi$ 无偏 | 后面的 $\delta$ 带着真实奖励，可减弱错误 bootstrap 的影响，但不能一般地消除函数逼近偏差 |
| **$\lambda$ 小** | 少用未来噪声，方差小 | 更依赖当前 $V$，偏差大 |
| **$\lambda$ 大** | 许多随机 $\delta$ 相加，方差大 | 更接近真实回报，偏差小 |

与 PPO 的衔接：正的 $\hat{A}_t$ 提高 $\pi(a_t \mid s_t)$，负的则压低。$\delta_t$ 给出每一步的局部评价；GAE 把未来残差按 $\gamma\lambda$ 折进来，得到比纯 TD(0) 更稳、比纯 Monte Carlo 噪声更小的优势估计。

上表第一格可由过滤信息严格说明。令 $\mathcal F_{t+\ell}$ 表示在时刻 $t+\ell$ 选择动作之前的历史。若 $V=V^\pi$ 且后续动作按 $\pi$ 采样，则

$$
\mathbb E[\delta_{t+\ell}\mid\mathcal F_{t+\ell}]=0,
\qquad \ell\ge1.
$$

再对给定 $(s_t,a_t)$ 使用迭代期望，即得 $\mathbb E[\delta_{t+\ell}\mid s_t,a_t]=0$。实践中 $V_\phi\ne V^\pi$，且 rollout 有限，GAE 因而仍是带偏差的优势估计；$\lambda$ 调节的是这种近似误差与采样方差之间的权衡。

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

## 九、训练过程

### 9.1 完整训练流程

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
     e. 总损失 = -L^CLIP - ent_coef * entropy + vf_coef * v_loss
     f. 反向传播，梯度裁剪，更新参数
     g. 估计新旧策略的 KL，并统计落入裁剪区的样本比例（诊断；可选按 KL 提前结束本批更新）

5. 丢弃旧数据，回到步骤 1
```

### 9.2 单次更新过程

（1）**优势加权。** 策略梯度在当前策略采得的样本上，对 $\nabla_\theta\log\pi_\theta(a_t\mid s_t)$ 作加权平均，权重是优势 $\hat A_t\approx Q^\pi(s_t,a_t)-V^\pi(s_t)$。在参数空间中，该梯度指向 $\log\pi_\theta(a_t\mid s_t)$ 的上升方向；沿它走一小步，$a_t$ 的概率局部增大。更新沿它与 $\hat A_t$ 的乘积前进：$\hat A_t>0$ 时 $a_t$ 的概率上升，$\hat A_t<0$ 时下降。

（2）**重要性比。** 同一批数据上 $\theta$ 离开采样时的 $\theta_{\mathrm{old}}$。$\min$ 取 $r_t\hat A_t$ 时，梯度为

$$
r_t(\theta)\,\nabla_\theta\log\pi_\theta(a_t\mid s_t)\,\hat A_t,
\qquad
r_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
$$

第一步 $r_t=1$，其期望即策略梯度。其后 $r_t$ 把旧动作上的目标改写成对新策略的估计。$\log\pi_\theta(a\mid s)$ 与采集时的 $\log\pi_{\theta_{\mathrm{old}}}(a\mid s)$ 都是标量；连续动作由对角高斯密度算出，$r_t$ 是二者之差的指数。$\nabla_\theta\log\pi_\theta$ 的维数等于 $\theta$，由反向传播得到。

（3）**裁剪。** $\hat A_t>0$ 且 $r_t>1+\epsilon$ 时，该动作的概率已超过旧策略的 $1+\epsilon$ 倍，$\min$ 取常数 $(1+\epsilon)\hat A_t$，梯度为零。$\hat A_t<0$ 且 $r_t<1-\epsilon$ 时，概率已低于 $1-\epsilon$ 倍，同样停住。反向越界仍沿 $r_t\hat A_t$ 把比率拉回。

（4）**弃批。** `update_epochs` 结束后丢弃该批。下一批动作由更新后的 $\pi_\theta$ 重新采样。

---

## 十、总损失函数

第五节的 $L^{\mathrm{CLIP}}$ 对 $\theta$ 最大化。实现中最小化

$$
L_{\mathrm{total}} = -L^{\mathrm{CLIP}} - c_{ent} \cdot H(\pi_\theta) + c_{vf} \cdot L^{VF}.
$$

前两项分别提高裁剪目标与策略熵，第三项拟合价值。$c_{ent}$（`ent_coef`）通常取 $0.01$，$c_{vf}$（`vf_coef`）通常取 $0.5$。

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

熵越大，动作越不确定。$L_{\mathrm{total}}$ 中的 $-c_{ent} H$ 使最小化该式时熵上升。

### 10.2 价值损失 $L^{VF}$

回归目标是 GAE 回报（第 9.1 节）：$\hat{R}_t = \hat{A}_t + V_{\theta_{old}}(s_t)$。未裁剪时为 MSE：

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

## 十一、PPO 的主要组成

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

## 十二、各组成部分的作用

Actor 参数化动作分布，Critic 估计状态价值，优势函数衡量所采动作相对当前基线的收益。裁剪目标限制单个样本的概率比贡献，多轮更新提高同一批 on-policy 数据的利用率，熵正则则抑制策略过早退化为近确定性分布。这些机制分别作用于更新幅度、估计方差、样本复用与探索，不能由裁剪项单独替代。

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

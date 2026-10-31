# 安全策略学习：累计约束、屏障函数与安全预测控制

机器人策略的“安全”至少有三种不可混用的含义。累计成本约束限制长期平均风险；机会约束限制随机违规的概率；状态约束要求每一时刻均留在安全集。它们对应不同的问题、算法和保证，不能仅靠在奖励中加入一个大惩罚项来替代。

本文将安全策略学习放在最优控制和强化学习的交界处：CMDP 方法在策略或占用测度层面处理累计约束；control barrier function（CBF）在动作层面保持安全集前向不变；安全 MPC、鲁棒 CBF 与 backup controller 处理模型误差、有限采样和不可行性。最优控制中的约束转录、SQP/QP 与 MPC 求解见 [`control-optimization-synthesis.md`](../../optimal-control-solvers/study-notes/control-optimization-synthesis.md)。

---

## 一、安全约束的三种语义

设策略为 $\pi$，安全相关成本为 $c(s,a)$，安全状态集为

$$
\mathcal C=\{x\mid h(x)\ge0\}.
$$

### 1.1 累计约束：CMDP

约束 Markov 决策过程（CMDP）要求

$$
\max_\pi J_r(\pi)
\qquad
\text{s.t.}\qquad
J_c(\pi)
=
\mathbb E_\pi\left[\sum_{t\ge0}\gamma^t c(s_t,a_t)\right]
\le d.
\tag{1}
$$

$c$ 可以表示能耗、碰撞次数、关节力矩超限次数或人类介入频率。式 (1) 是期望约束：单条轨迹仍可能违规，只要总体期望不超过预算。它适合资源预算和可容忍的风险，不等同于“绝不碰撞”。

### 1.2 概率约束

机会约束直接限制事件概率，例如

$$
\Pr_\pi\left(\exists t\le H:\;x_t\notin\mathcal C\right)
\le\delta.
\tag{2}
$$

它比式 (1) 更接近任务级安全，但估计罕见事件概率需要大量数据；用期望成本代替式 (2) 时必须说明这种替代带来的松弛。

### 1.3 逐步状态约束

硬安全要求为

$$
x_t\in\mathcal C,\qquad\forall t.
\tag{3}
$$

在连续时间中，若控制律使所有从 $\mathcal C$ 出发的轨迹始终留在 $\mathcal C$，则称 $\mathcal C$ 前向不变。CBF 的目的正是构造或近似这种控制律。保证依赖动力学、扰动界、控制输入限制和数值实现；离开这些假设后，不能把 CBF-QP 的输出笼统称为“硬安全”。

---

## 二、CMDP 的策略优化

### 2.1 拉格朗日松弛

对式 (1) 引入乘子 $\lambda\ge0$：

$$
\mathcal L(\pi,\lambda)
=J_r(\pi)-\lambda\bigl(J_c(\pi)-d\bigr).
\tag{4}
$$

固定 $\lambda$ 时，actor 最大化“修正奖励”

$$
r_\lambda(s,a)=r(s,a)-\lambda c(s,a).
\tag{5}
$$

乘子则按约束违背方向上升，例如

$$
\lambda\leftarrow\bigl[\lambda+\eta_\lambda(\widehat J_c-d)\bigr]_+.
\tag{6}
$$

该方法实现简单，可与 PPO、SAC 等 actor--critic 结构结合；但深度函数逼近、有限批量和双时间尺度学习会导致乘子滞后、振荡或在训练期持续违规。式 (6) 的收敛直觉不构成逐步安全保证。

### 2.2 CPO 与信赖域近似

Constrained Policy Optimization（CPO）从旧策略 $\pi_k$ 出发，在局部近似下同时约束性能、成本和策略 KL：

$$
\begin{aligned}
\max_\theta\quad& g_r^\top(\theta-\theta_k)\\
\text{s.t.}\quad&
J_c(\pi_k)+g_c^\top(\theta-\theta_k)\le d,\\
&\frac12(\theta-\theta_k)^\top F_k(\theta-\theta_k)\le\delta.
\end{aligned}
\tag{7}
$$

其中 $g_r,g_c$ 为回报和成本代理的梯度，$F_k$ 为 Fisher 信息矩阵。它与 TRPO 共享局部 KL 几何，但增加了成本可行性半空间。CPO 的结论是基于策略差异界、精确或受控估计以及局部子问题可行性；有限样本和神经网络近似下，训练中仍可能违规。

CVPO 则更接近仓库中的 REPS/MPO 主线：先在非参数分布层面进行带约束的变分改进，再投影回参数化策略。该视角把安全约束与 KL 保守更新写在同一个优化问题中，但仍是式 (1) 意义上的累计约束。

---

## 三、控制屏障函数与安全过滤

### 3.1 控制仿射系统上的 CBF 条件

考虑连续动力学

$$
\dot x=f(x)+g(x)u,
\qquad u\in\mathcal U.
\tag{8}
$$

若 $h$ 对输入具有相对阶一，沿轨迹的导数为

$$
\dot h(x,u)=\nabla h(x)^\top f(x)+\nabla h(x)^\top g(x)u.
\tag{9}
$$

扩展类 $\mathcal K$ 函数 $\alpha$ 满足 $\alpha(0)=0$ 且严格递增。若对每个 $x\in\mathcal C$ 都能选到 $u$ 使

$$
\dot h(x,u)+\alpha(h(x))\ge0,
\tag{10}
$$

则该不等式限制状态不穿过边界 $h=0$ 向不安全区域移动。常取 $\alpha(h)=\kappa h$，$\kappa>0$。

这不是“让 $h$ 增大”的一般目标：当系统位于安全集内部，$h$ 可以下降，只要下降速度受式 (10) 控制。多个约束、输入饱和、相对阶大于一和离散采样均会改变可行条件。

### 3.2 将任意 actor 接入 CBF-QP

策略给出名义动作 $u_{\rm nom}=\pi_\theta(o)$ 后，安全滤波器解

$$
\begin{aligned}
u_{\rm safe}=\arg\min_{u\in\mathcal U}\quad
&\frac12\|u-u_{\rm nom}\|_W^2\\
\text{s.t.}\quad
&\nabla h(x)^\top f(x)+\nabla h(x)^\top g(x)u
+\alpha(h(x))\ge0.
\end{aligned}
\tag{11}
$$

这一步把 actor 的性能偏好投影到局部安全可行集。它可放在 PPO、SAC、扩散策略或手工控制器之后，因此 CBF-RL 不是一种单独的 RL 更新公式，而是一类“策略提议 + 安全动作选择”的系统结构。

式 (11) 有三项常被忽略的前提：

1. 状态 $x$、动力学 $f,g$ 或其可信误差界可用；
2. 在当前状态和输入边界下，约束 QP 可行；
3. 离散执行、时延和跟踪误差已留出足够安全裕度。

若 QP 不可行，松弛变量只能在“尽量少违规”的意义下恢复问题，不能维持原来的前向不变性结论。实际系统应定义 fallback controller、紧急停止或可验证的恢复集。

### 3.3 CBF-RL 的三种耦合方式

**后置过滤。** 训练和部署均执行 $u_{\rm safe}$，但 actor 更新仍把 $u_{\rm nom}$ 视为环境动作。这种做法最易部署，却造成策略动作与实际执行动作不一致；若 filter 经常激活，critic 学到的是被滤波闭环的价值，而 actor 未必学会主动避开边界。

**可微安全层。** 将式 (11) 的解映射嵌入计算图，使梯度通过 QP/KKT 系统传到 actor。它要求活跃集在局部稳定且数值求解可微；活跃约束切换处通常只有分段可微。可微不等于模型误差下仍安全。

**联合学习。** 可同时学习策略、动力学和 $h_\psi(x)$，或用数据估计不确定性并收紧式 (10)。这提高了表达能力，也把证书本身变成近似对象；若不额外验证，学习到的 barrier 不能直接继承解析 CBF 的保证。

---

## 四、不确定性、安全 MPC 与安全探索

### 4.1 鲁棒化

若真实动力学为

$$
\dot x=f(x)+g(x)u+w,
\qquad w\in\mathcal W,
\tag{12}
$$

则式 (10) 应对最坏扰动留出余量，例如

$$
\nabla h^\top(f+gu)+\alpha(h)
\ge
\sup_{w\in\mathcal W}[-\nabla h^\top w].
\tag{13}
$$

右端需要可靠的扰动集合；任意从数据估计的 ensemble 方差并不能自动充当鲁棒界。随机系统中也可采用 chance-CBF 或风险度量，但得到的是概率保证而非确定性不变性。

### 4.2 安全 MPC

安全 MPC 在有限预测时域中显式优化状态、输入和终端约束：

$$
\begin{aligned}
\min_{u_{0:H-1}}\quad&
\sum_{k=0}^{H-1}\ell(x_k,u_k)+V_f(x_H)\\
\text{s.t.}\quad&x_{k+1}=f(x_k,u_k),\\
&x_k\in\mathcal X_{\rm safe},\quad u_k\in\mathcal U,\\
&x_H\in\mathcal X_f.
\end{aligned}
\tag{14}
$$

若终端集和局部控制律满足递归可行性条件，MPC 能提供比单步过滤更长视野的约束处理。它的代价是每个控制周期均需解优化问题，并依赖模型可信度。策略网络可作为 warm start、终端策略或候选轨迹分布；反过来，MPC 也可生成安全示范或 shield 训练数据。

### 4.3 训练期安全与部署期安全

部署前仅验证最终策略，不能保证训练采样安全。训练期安全需要过滤器、保守初始集、人工接管或仿真到真机的逐级迁移；部署期安全还需处理观测噪声、延迟、执行器饱和和故障恢复。二者都不能由单独的 CMDP 目标或单独的 CBF 定理覆盖。

---

## 五、选择方法时应先问什么

| 问题 | 更合适的起点 | 不足之处 |
|------|--------------|----------|
| 能耗、磨损、干预次数预算 | CMDP / 拉格朗日 / CPO | 不保证每一步不违规 |
| 已知几何边界、低层状态和动力学 | CBF-QP filter | 需要可行性与模型误差裕度 |
| 预测地平线内多约束耦合 | 安全 MPC | 在线求解开销和模型依赖 |
| 未知模型、随机扰动、视觉观测 | 鲁棒/概率安全层 + 在线验证 | 保证取决于误差界与感知可靠性 |
| 真机探索 | 分级约束、shield、人工接管 | 仍需定义不可恢复时的停机策略 |

安全方法的选择应由约束语义和可访问信息决定，而不是由 actor 是 PPO、SAC 还是扩散策略决定。将所有违规都折入标量 reward，通常只得到一个权重敏感的软偏好，无法替代式 (2)、(3) 或 (10) 中的明确约束。

---

## 六、文献与代码

1. Achiam et al., *Constrained Policy Optimization*, ICML 2017. [PMLR](https://proceedings.mlr.press/v70/achiam17a.html)
2. Liu et al., *Constrained Variational Policy Optimization for Safe Reinforcement Learning*, ICML 2022. [PMLR](https://proceedings.mlr.press/v162/liu22b.html)
3. Ames et al., *Control Barrier Functions: Theory and Applications*, ECC 2019. [arXiv](https://arxiv.org/abs/1903.11199)
4. Cheng and Zhao, *Safe Reinforcement Learning Using Robust Control Barrier Functions*, 2021. [arXiv](https://arxiv.org/abs/2110.05415)
5. Suttle et al., *Sampling-based Safe Reinforcement Learning for Nonlinear Dynamical Systems*, AISTATS 2024. [PMLR](https://proceedings.mlr.press/v238/suttle24a.html)
6. Wachi et al., [Safe Control Gym](https://github.com/utiasDSL/safe-control-gym) — 安全控制与 RL 基准环境。

建议阅读顺序：先读式 (1)--(7) 区分累计约束与策略信赖域，再读式 (8)--(13) 理解 CBF 的安全语义，最后回到最优控制笔记中的约束 MPC。这样可避免把“期望约束收敛”“QP 可行”与“真实系统全程安全”混为同一层次的结论。

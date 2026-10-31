# 离线强化学习：数据支持、保守价值与策略提取

离线强化学习（offline reinforcement learning, offline RL）只使用预先收集的数据集

$$
\mathcal D=\{(s,a,r,s',d)\},
$$

学习最大化回报的策略，训练期间不再向环境查询新转移。它与带回放池的 off-policy RL 共享 Bellman 备份和 actor--critic 结构，但后者仍可通过新的交互数据纠正策略分布；离线 RL 不能离开数据覆盖的区域后再由环境反馈修正。

与 SAC 的最大熵策略改进、AWR/AWAC 的优势加权回归，以及分布空间中的 KL 正则化的关系见 [`sac_notes.md`](../foundations/sac_notes.md) 与 [`distribution-space-policy-optimization.md`](../foundations/distribution-space-policy-optimization.md)。本文集中讨论离线场景中“为何常规 off-policy RL 会失效，以及不同方法如何限制策略离开数据支持集”。

---

## 一、问题与数据分布

考虑折扣 MDP $\mathcal M=(\mathcal S,\mathcal A,P,r,\rho_0,\gamma)$。行为策略 $\mu$ 或若干行为策略的混合产生离线数据，其状态--动作分布记为 $d_{\mathcal D}(s,a)$。目标仍是

$$
J(\pi)=\mathbb E_\pi\left[\sum_{t\ge0}\gamma^t r(s_t,a_t)\right].
$$

离线设定的困难不在于目标改变，而在于 Bellman 备份和 actor 改进访问的分布不同。给定 critic $Q_\phi$，普通 actor 更新倾向于

$$
\pi_{\rm new}(\cdot\mid s)
\approx
\arg\max_\pi
\mathbb E_{a\sim\pi(\cdot\mid s)}[Q_\phi(s,a)].
\tag{1}
$$

若 $a$ 很少或从未在 $\mathcal D$ 中与状态 $s$ 一起出现，则 $Q_\phi(s,a)$ 只受函数逼近的外推决定。actor 会主动寻找这些未经数据约束、但被误估为高价值的动作；被选中的动作又进入 TD target，使误差沿 Bellman 递推放大。这是离线 RL 中常说的 extrapolation error 或 OOD action error。

问题也可用支持集表述。若存在 $s$ 使

$$
\pi(a\mid s)>0,
\qquad
\mu(a\mid s)=0,
$$

则基于行为数据的动作重要性比 $\pi(a\mid s)/\mu(a\mid s)$ 无定义。连续动作中严格的零密度较少见，但当比值极大时，有限样本估计同样不稳定。因此，离线 RL 不是“把 SAC 的环境交互关掉”，而是需要显式处理分布支持、价值悲观性或两者兼有。

---

## 二、三种基本策略

### 2.1 行为约束：不离数据太远

最直接的办法是在 actor 更新中保留行为策略 $\mu$。一般形式为

$$
\max_\pi\;
\mathbb E_{s\sim d_{\mathcal D},a\sim\pi}[Q_\phi(s,a)]
-\alpha\,\mathbb E_{s\sim d_{\mathcal D}}
D(\pi(\cdot\mid s)\|\mu(\cdot\mid s)).
\tag{2}
$$

散度可以是 KL、MMD、动作距离或参数化的行为克隆损失。它将策略改进限制在行为分布的局部邻域；代价是当数据中缺少高价值动作时，策略也难以超越数据。

TD3+BC 使用更简单的确定性 actor 目标：

$$
\max_\theta\;
\mathbb E_{(s,a)\sim\mathcal D}
\left[
\lambda Q_\phi(s,\pi_\theta(s))
-\|\pi_\theta(s)-a\|^2
\right].
\tag{3}
$$

第二项不是对真实 $\mu$ 的显式 KL，却在数据动作附近约束 actor。其优点是实现简单；其局限是单一平方距离难以表示多模态行为分布。

### 2.2 保守价值：让未知动作不显得过好

另一条路线不先约束 actor，而是令 critic 对数据外动作保持悲观。Conservative Q-Learning（CQL）的代表性正则为

$$
\mathcal L_{\rm CQL}(Q)
=
\alpha\,mathbb E_{s\sim\mathcal D}
\left[
\log\int_{\mathcal A}e^{Q(s,a)}\,\mathrm da
-\mathbb E_{a\sim\mathcal D(\cdot\mid s)}Q(s,a)
\right]
+\mathcal L_{\rm TD}(Q).
\tag{4}
$$

离散动作时积分变为 log-sum-exp。对 $Q(s,a)$ 求导可见，第一项压低所有动作的值，第二项相对抬高数据动作的值；二者共同扩大“数据动作与其他动作”的保守间隔。式 (4) 并不声称所有 $Q$ 值都是下界；其目标是在目标策略可能访问的区域抑制系统性高估。

悲观性与行为约束可以并用：前者改善 actor 看到的价值地形，后者减少 actor 前往不可靠区域的机会。MOPO、COMBO 等模型式离线方法则将模型不确定性罚入虚拟奖励，见 [`model-based-reinforcement-learning.md`](model-based-reinforcement-learning.md)。

### 2.3 不显式最大化 OOD 动作：IQL 与优势加权回归

Implicit Q-Learning（IQL）避免式 (1) 中显式的 $\max_a Q(s,a)$。先在数据动作上拟合 $Q$，再用 expectile 回归定义状态值：

$$
V_\psi
=
\arg\min_V
\mathbb E_{(s,a)\sim\mathcal D}
\left[
L_\tau\bigl(Q_\phi(s,a)-V(s)\bigr)
\right],
\tag{5}
$$

其中

$$
L_\tau(u)=|\tau-\mathbf 1\{u<0\}|u^2.
\tag{6}
$$

$\tau>1/2$ 时，$V$ 更接近数据动作价值分布的上尾。critic 的 target 只使用数据动作：

$$
y=r+\gamma(1-d)V_\psi(s').
\tag{7}
$$

最后以优势权重回归 actor：

$$
\max_\theta\;
\mathbb E_{(s,a)\sim\mathcal D}
\left[
\exp\!\left(\frac{Q_\phi(s,a)-V_\psi(s)}{\beta}\right)
\log\pi_\theta(a\mid s)
\right].
\tag{8}
$$

实践中权重通常截断。式 (8) 只提高数据中已有动作的概率，并不要求从连续动作空间求 $\arg\max Q$；这正是 IQL 在复杂离线数据上较稳健的原因。它与 AWR/AWAC 的共同变分来源见分布优化笔记第 5.3 节，但 IQL 的 critic 与值函数估计方式不同。

---

## 三、Bellman 误差为何不足以选策略

离线 RL 中，小 TD loss 不等于高真实回报。TD target 本身包含网络输出，且训练误差只在 $d_{\mathcal D}$ 上受约束；不同的策略可以在数据外产生完全不同的访问分布，却有近似相同的经验 Bellman 误差。

策略选择至少应报告以下量：

1. 数据归一化回报或真实环境评估；
2. 目标策略与行为数据的距离，例如动作 MSE、估计 KL 或 advantage 权重的有效样本量；
3. critic ensemble 的分歧、不确定性或 CQL gap；
4. 若不能在线评估，则采用保守的 off-policy evaluation，并明确其覆盖假设。

Fitted Q Evaluation、重要性采样、doubly robust 估计各自依赖模型正确性、密度比估计或两者之一。没有充分覆盖时，离线策略评估同样无法从有限数据中可靠地证明一个远离行为策略的 actor 更好。

---

## 四、方法之间的关系

离线算法可按“约束施加在哪里”组织：

| 路线 | 约束对象 | 代表方法 | 主要风险 |
|------|----------|----------|----------|
| 行为正则 | actor | BCQ、BEAR、BRAC、TD3+BC | 过于保守，不能利用高质量子集 |
| 保守估值 | critic | CQL | 价值整体偏低，超参数敏感 |
| 隐式策略提取 | value / actor | IQL、AWR、AWAC | 优势权重退化，依赖 $Q-V$ 质量 |
| 模型悲观 | 虚拟模型 rollout | MOPO、COMBO | 模型误差与不确定性校准 |

这些方法不是互斥类别。比如 actor 可以同时作行为回归，critic 可以采用 ensemble 悲观值；关键是明确每个正则在控制哪一种分布外误差。

---

## 五、与机器人策略学习的接口

机器人离线数据常是演示、失败轨迹、历史策略和人类介入片段的混合，而非单一固定行为策略。此时“接近 $\mu$”不是接近一个可精确写出的密度，而是留在数据的局部支持中。对多峰动作，显式高斯行为模型可能失真；扩散或 flow actor 则可提供更富表达力的行为先验。

[`rl100_notes.md`](../generative-rl/rl100_notes.md) 的离线阶段正属于“数据上的 $Q-V$ 优势 + 行为近端更新”的路线，但其 actor 是扩散去噪链，而不是式 (8) 中的单步密度。HIL-SERL 的 $50/50$ 演示--在线混合回放则是 offline-to-online 过渡：在线交互重新提供了纠错信号，但演示缓冲仍承担支持集锚点的作用。

离线到在线微调不应被理解为“离线误差自然消失”。初始策略若离数据支持很远，第一轮在线探索仍可能不安全；真实机器人应结合安全约束或过滤机制，见 [`safe-policy-learning.md`](safe-policy-learning.md)。

---

## 六、文献与代码

1. Fujimoto and Gu, *A Minimalist Approach to Offline Reinforcement Learning*, NeurIPS 2021. [arXiv](https://arxiv.org/abs/2106.06860)
2. Kumar et al., *Conservative Q-Learning for Offline Reinforcement Learning*, NeurIPS 2020. [project](https://sites.google.com/view/cql-offline-rl)
3. Kostrikov, Nair, and Levine, *Offline Reinforcement Learning with Implicit Q-Learning*, ICLR 2022. [OpenReview](https://openreview.net/forum?id=68n2s9ZJWF8)
4. Nair et al., *AWAC: Accelerating Online Reinforcement Learning with Offline Datasets*, 2020. [arXiv](https://arxiv.org/abs/2006.09359)
5. Tarasov et al., *CORL: Research-oriented Deep Offline Reinforcement Learning Library*, 2022. [GitHub](https://github.com/tinkoff-ai/CORL)

建议先读分布优化笔记第 5.3 节与 SAC 笔记，再读 IQL 的式 (5)--(8)，最后对照 CQL 的 critic 正则。这样可以把“指数加权的策略投影”和“悲观 Bellman 估值”区分为两条互补而非等价的路线。

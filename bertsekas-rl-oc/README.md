# Bertsekas — Reinforcement Learning and Optimal Control

Dimitri P. Bertsekas, *Reinforcement Learning and Optimal Control*（Athena Scientific 2019 draft）。

| 路径 | 用途 |
|------|------|
| [`Reinforcement learning and optimal control.pdf`](<Reinforcement learning and optimal control.pdf>) | 书稿 |
| [`study-notes/`](study-notes/) | 分节笔记 |

行内公式 `$...$`，块级 ` ```math `；最优星号写成 `J_k^{*}`。

| 笔记 | 状态 |
|------|------|
| [`00-algorithm-taxonomy.md`](study-notes/00-algorithm-taxonomy.md) | 跨章索引与术语约定 |
| [`ch01-exact-dp-study-notes.md`](study-notes/ch01-exact-dp-study-notes.md) | 精确有限时域 DP |
| [`ch02-approximation-in-value-space-study-notes.md`](study-notes/ch02-approximation-in-value-space-study-notes.md) | 值空间近似、rollout 与 MPC |
| [`ch03-parametric-approximation-study-notes.md`](study-notes/ch03-parametric-approximation-study-notes.md) | 参数化近似与增量算法 |
| [`ch04-infinite-horizon-rl-study-notes.md`](study-notes/ch04-infinite-horizon-rl-study-notes.md) | 无限时域 RL 与精确/近似 PI、VI |
| [`ch05-infinite-horizon-approximate-study-notes.md`](study-notes/ch05-infinite-horizon-approximate-study-notes.md) | 近似 DP、Q-learning 与策略梯度 |

笔记统一采用 $α$ 表示折扣因子、$η_k$ 表示迭代步长，并区分“显式调用模型”的 model-based 方法与只使用交互样本的 model-free 方法；Monte Carlo 只是估计手段，不单独构成两者的定义。

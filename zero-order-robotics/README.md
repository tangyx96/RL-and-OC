# Zero-Order Optimization Techniques for Robotics

Jordana, Zhang, Amigo, Righetti, *An Introduction to Zero-Order Optimization Techniques for Robotics*（[arXiv:2506.22087](https://arxiv.org/abs/2506.22087) v2, 2025）。实验代码：[zoo-rob](https://github.com/ajordana/zoo-rob)。

| 笔记 | 内容 |
|------|------|
| [`study-notes/zero-order-optimization-study-notes.md`](study-notes/zero-order-optimization-study-notes.md) | 按论文 §I–§VI |
| [`study-notes/zero-order-optimal-control-extension.md`](study-notes/zero-order-optimal-control-extension.md) | 玻尔兹曼目标；CEM、MPPI、朗之万、SVGD、DIAL-MPC 按权重、协方差与扩散项区分 |

[`optimal-control-solvers`](../optimal-control-solvers/) 讨论依赖局部导数模型的 SQP、DDP 与可微 MPC；本目录研究仅能批量查询代价或回报时，如何用随机平滑、控制序列分布和种群统计构造更新。优化变量取控制序列时得到采样式轨迹优化与 MPC，取策略参数时得到黑箱策略搜索；动作分布上的策略更新见 [`policy-learning`](../policy-learning/)。

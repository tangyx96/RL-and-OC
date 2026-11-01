# 最优控制与数值优化

| 文献 | |
|------|--|
| Kelly, *An Introduction to Trajectory Optimization* (*SIAM Review*, 2017) | [DOI](https://doi.org/10.1137/16M1062569)；直接配点、转录误差与网格细化 |
| Giftthaler 等, *A Family of Iterative Gauss--Newton Shooting Methods* (IROS 2018) | [arXiv:1711.11006](https://arxiv.org/abs/1711.11006) |
| Mastalli 等, *A Feasibility-Driven Approach to Control-Limited DDP* | [arXiv:2010.00411](https://arxiv.org/abs/2010.00411) |
| Jordana 等, *Stagewise Implementations of SQP for MPC* (2023) | HAL / 会议稿 |
| Jordana 等, *Structure-Exploiting SQP for MPC* (*IEEE TRO*, 2025) | 期刊扩展 |
| Abhijeet & Chakravorty, *A Sequential Quadratic Programming Perspective on Optimal Control* | [arXiv:2510.03475](https://arxiv.org/abs/2510.03475) |
| Tassa, Erez, Todorov, *Synthesis and Stabilization of Complex Behaviors through Online Trajectory Optimization* (IROS 2012) | DDP 值函数与在线轨迹优化 |
| Sousa-Pinto & Orban, *Primal-Dual iLQR* | [arXiv:2403.00748](https://arxiv.org/abs/2403.00748) |
| Jordana 等, Supplementary Material | KKT → Thomas → Riccati |

## 笔记导航

- [数值优化基础](study-notes/numerical-optimization.md)：以数值优化为主线，系统讨论局部模型、线搜索与信赖域、梯度法、Newton 法、拟 Newton 法、共轭梯度、非线性最小二乘、KKT 条件、二次规划、罚函数、障碍函数、增广拉格朗日、SQP、原始—对偶内点法以及近端与 ADMM 方法。重点关注收敛条件、病态性、KKT 线性系统和结构化数值线性代数。
- [最优控制求解器综合](study-notes/control-optimization-synthesis.md)：按“连续 OCP—直接转录—局部二次模型—结构化 OCP-QP—全局化—实时闭环”的主线，讨论 iLQR、DDP、GNMS、FDDP、SQP、配点法与 SQP-RTI，并以近端点和信息 oracle 为接口连接结构化非光滑优化、零阶优化与策略学习。

两份笔记相互衔接：数值优化笔记解释一般优化算法及其线性代数基础；综合笔记进一步说明这些工具如何用于最优控制、轨迹优化和实时模型预测控制。

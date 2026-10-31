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

笔记：[`study-notes/control-optimization-synthesis.md`](study-notes/control-optimization-synthesis.md)。文章按“连续 OCP—直接转录—局部二次模型—结构化 OCP-QP—全局化—实时闭环”的主线，讨论 iLQR、DDP、GNMS、FDDP、SQP、配点法与 SQP-RTI；后部以近端点和信息 oracle 为接口，区分结构化非光滑优化、零阶轨迹优化与策略学习。

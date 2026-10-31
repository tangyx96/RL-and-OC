# RL and Optimal Control — 读书笔记

按文献分目录。`bertsekas-rl-oc/` 保留书稿 PDF，`optimal-control-solvers/` 保留最优控制求解器相关论文与笔记，其余目录是专题笔记。

| 目录 | 文献 | 笔记 |
|------|------|------|
| [`bertsekas-rl-oc/`](bertsekas-rl-oc/) | Bertsekas, *Reinforcement Learning and Optimal Control*（2019 draft） | [`study-notes/`](bertsekas-rl-oc/study-notes/) |
| [`zero-order-robotics/`](zero-order-robotics/) | Jordana 等, *Zero-Order Optimization*（[arXiv:2506.22087](https://arxiv.org/abs/2506.22087)） | [`study-notes/`](zero-order-robotics/study-notes/) |
| [`optimal-control-solvers/`](optimal-control-solvers/) | Jordana SQP/MPC 系列；Chakravorty（[arXiv:2510.03475](https://arxiv.org/abs/2510.03475)）；Tassa 等（IROS 2012）；Sousa-Pinto 与 Orban, Primal-Dual iLQR（[arXiv:2403.00748](https://arxiv.org/abs/2403.00748)） | [`control-optimization-synthesis.md`](optimal-control-solvers/study-notes/control-optimization-synthesis.md) |
| [`policy-learning/`](policy-learning/) | 策略优化、离线与模型式 RL、安全约束、生成式动作策略 | [`README.md`](policy-learning/README.md)；统一视角：[`分布空间中的策略优化`](policy-learning/foundations/distribution-space-policy-optimization.md)；基础：[`PPO`](policy-learning/foundations/ppo_notes.md)、[`SAC`](policy-learning/foundations/sac_notes.md)；扩展：[`离线 RL`](policy-learning/data-model-safety/offline-reinforcement-learning.md)、[`模型式 RL`](policy-learning/data-model-safety/model-based-reinforcement-learning.md)、[`安全策略学习`](policy-learning/data-model-safety/safe-policy-learning.md)；生成式：[`Diffusion Policy`](policy-learning/generative-policies/diffusion_policy_notes.md)、[`Flow Matching`](policy-learning/generative-policies/flow_matching_notes.md)、[`VLA`](policy-learning/generative-policies/vla_notes.md)、[`WAM`](policy-learning/generative-policies/wam_notes.md) |
| [`simulators/mjlab/`](simulators/mjlab/) | mjlab（[arXiv:2601.22074](https://arxiv.org/abs/2601.22074)）；应用层对照 [unitree_rl_mjlab](https://github.com/tangyx96/unitree_rl_mjlab) | [`mjlab-introduction.md`](simulators/mjlab/mjlab-introduction.md) |
| [`simulators/rsl_rl/`](simulators/rsl_rl/) | RSL-RL（[arXiv:2509.10771](https://arxiv.org/abs/2509.10771)）；Isaac Lab / mjlab 的默认 on-policy 后端 | [`rsl-rl-introduction.md`](simulators/rsl_rl/rsl-rl-introduction.md) |

这些目录按优化对象和可用信息逐步展开。Bertsekas 给出 Bellman 方程、MPC 与策略梯度的序贯决策基础；`optimal-control-solvers/` 从连续 OCP 出发，经直接转录、局部二次模型和结构化 KKT 求解到实时 MPC，并讨论具有显式结构的非光滑项；`zero-order-robotics/` 研究仅能查询代价或 rollout 时的随机平滑、MPPI、CEM、CMA 类方法和参数空间策略搜索；[`分布空间中的策略优化`](policy-learning/foundations/distribution-space-policy-optimization.md) 进一步以动作分布、轨迹分布和占用测度为优化对象，讨论 IGO、REPS/MPO/AWR、TRPO/PPO、SAC、CEM/CMA-ES、MPPI 与 diffusion-RL 中的 KL 正则化和分布投影。

建议阅读路线为：[`最优控制求解器`](optimal-control-solvers/study-notes/control-optimization-synthesis.md) → [`零阶优化`](zero-order-robotics/study-notes/zero-order-optimization-study-notes.md)与[`零阶最优控制扩展`](zero-order-robotics/study-notes/zero-order-optimal-control-extension.md) → [`分布空间中的策略优化`](policy-learning/foundations/distribution-space-policy-optimization.md)。策略学习随后沿四条支线展开：actor--critic（PPO、SAC）、数据与模型（离线 RL、模型式 RL、安全约束）、生成式动作策略（Diffusion Policy、Flow Matching、VLA、WAM），以及生成式策略的奖励后训练（DPPO、RL-100）；`simulators/` 按仿真器与训练库归档。

版权在原书 / 原论文，仅供个人学习。

# RL and Optimal Control — 读书笔记

按文献分目录。`bertsekas-rl-oc/` 保留书稿 PDF，`optimal-control-solvers/` 保留最优控制求解器相关论文与笔记，其余目录是专题笔记。

| 目录 | 文献 | 笔记 |
|------|------|------|
| [`bertsekas-rl-oc/`](bertsekas-rl-oc/) | Bertsekas, *Reinforcement Learning and Optimal Control*（2019 draft） | [`study-notes/`](bertsekas-rl-oc/study-notes/) |
| [`zero-order-robotics/`](zero-order-robotics/) | Jordana 等, *Zero-Order Optimization*（[arXiv:2506.22087](https://arxiv.org/abs/2506.22087)） | [`study-notes/`](zero-order-robotics/study-notes/) |
| [`optimal-control-solvers/`](optimal-control-solvers/) | Jordana SQP/MPC 系列；Chakravorty（[arXiv:2510.03475](https://arxiv.org/abs/2510.03475)）；Tassa 等（IROS 2012）；Sousa-Pinto 与 Orban, Primal-Dual iLQR（[arXiv:2403.00748](https://arxiv.org/abs/2403.00748)） | [`control-optimization-synthesis.md`](optimal-control-solvers/study-notes/control-optimization-synthesis.md) |
| [`policy-learning/`](policy-learning/) | 在线强化学习；生成式动作策略；扩散策略的后训练 | 专题综述：[`distribution-space-policy-optimization.md`](policy-learning/distribution-space-policy-optimization.md)。在线：[`ppo_notes.md`](policy-learning/ppo_notes.md)、[`ppo_mlp_notes.md`](policy-learning/ppo_mlp_notes.md)、[`sac_notes.md`](policy-learning/sac_notes.md)、[`flashsac_notes.md`](policy-learning/flashsac_notes.md)、[`hil_serl_notes.md`](policy-learning/hil_serl_notes.md)。生成式：[`diffusion_policy_notes.md`](policy-learning/diffusion_policy_notes.md)、[`flow_matching_notes.md`](policy-learning/flow_matching_notes.md)、[`vla_notes.md`](policy-learning/vla_notes.md)、[`wam_notes.md`](policy-learning/wam_notes.md)。衔接：[`dppo_notes.md`](policy-learning/dppo_notes.md)、[`rl100_notes.md`](policy-learning/rl100_notes.md) |
| [`simulators/mjlab/`](simulators/mjlab/) | mjlab（[arXiv:2601.22074](https://arxiv.org/abs/2601.22074)）；应用层对照 [unitree_rl_mjlab](https://github.com/tangyx96/unitree_rl_mjlab) | [`mjlab-introduction.md`](simulators/mjlab/mjlab-introduction.md) |
| [`simulators/rsl_rl/`](simulators/rsl_rl/) | RSL-RL（[arXiv:2509.10771](https://arxiv.org/abs/2509.10771)）；Isaac Lab / mjlab 的默认 on-policy 后端 | [`rsl-rl-introduction.md`](simulators/rsl_rl/rsl-rl-introduction.md) |

这些目录按优化对象和可用信息逐步展开。Bertsekas 给出 Bellman 方程、MPC 与策略梯度的序贯决策基础；`optimal-control-solvers/` 从连续 OCP 出发，经直接转录、局部二次模型和结构化 KKT 求解到实时 MPC，并讨论具有显式结构的非光滑项；`zero-order-robotics/` 研究仅能查询代价或 rollout 时的随机平滑、MPPI、CEM、CMA 类方法和参数空间策略搜索；[`分布空间中的策略优化`](policy-learning/distribution-space-policy-optimization.md) 进一步以动作分布、轨迹分布和占用测度为优化对象，讨论 IGO、REPS/MPO/AWR、TRPO/PPO、SAC、CEM/CMA-ES、MPPI 与 diffusion-RL 中的 KL 正则化和分布投影。

建议阅读路线为：[`最优控制求解器`](optimal-control-solvers/study-notes/control-optimization-synthesis.md) → [`零阶优化`](zero-order-robotics/study-notes/zero-order-optimization-study-notes.md)与[`零阶最优控制扩展`](zero-order-robotics/study-notes/zero-order-optimal-control-extension.md) → [`分布空间中的策略优化`](policy-learning/distribution-space-policy-optimization.md)。`policy-learning/` 的其余笔记随后分三组展开：在线强化学习（PPO、SAC、HIL-SERL 等）、生成式动作策略（Diffusion Policy、Flow Matching、VLA、WAM），以及接在两组之间的 DPPO 与 RL-100；`simulators/` 按仿真器与训练库归档。

版权在原书 / 原论文，仅供个人学习。

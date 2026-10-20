# RL and Optimal Control — 读书笔记

按文献分目录。`bertsekas-rl-oc/` 保留书稿 PDF，`sqp-oc/` 保留论文 PDF，其余目录是笔记。

| 目录 | 文献 | 笔记 |
|------|------|------|
| [`bertsekas-rl-oc/`](bertsekas-rl-oc/) | Bertsekas, *Reinforcement Learning and Optimal Control*（2019 draft） | [`study-notes/`](bertsekas-rl-oc/study-notes/) |
| [`zero-order-robotics/`](zero-order-robotics/) | Jordana 等, *Zero-Order Optimization*（[arXiv:2506.22087](https://arxiv.org/abs/2506.22087)） | [`study-notes/`](zero-order-robotics/study-notes/) |
| [`sqp-oc/`](sqp-oc/) | Jordana SQP/MPC 系列；Chakravorty（[arXiv:2510.03475](https://arxiv.org/abs/2510.03475)）；Tassa 等（IROS 2012）；Sousa-Pinto 与 Orban, Primal-Dual iLQR（[arXiv:2403.00748](https://arxiv.org/abs/2403.00748)） | [`control-optimization-sqp-synthesis.md`](sqp-oc/study-notes/control-optimization-sqp-synthesis.md) |
| [`policy-learning/`](policy-learning/) | 在线强化学习；生成式动作策略；扩散策略的后训练 | 在线：[`ppo_notes.md`](policy-learning/ppo_notes.md)、[`ppo_mlp_notes.md`](policy-learning/ppo_mlp_notes.md)、[`sac_notes.md`](policy-learning/sac_notes.md)、[`flashsac_notes.md`](policy-learning/flashsac_notes.md)、[`hil_serl_notes.md`](policy-learning/hil_serl_notes.md)。生成式：[`diffusion_policy_notes.md`](policy-learning/diffusion_policy_notes.md)、[`flow_matching_notes.md`](policy-learning/flow_matching_notes.md)、[`vla_notes.md`](policy-learning/vla_notes.md)、[`wam_notes.md`](policy-learning/wam_notes.md)。衔接：[`dppo_notes.md`](policy-learning/dppo_notes.md)、[`rl100_notes.md`](policy-learning/rl100_notes.md) |
| [`simulators/mjlab/`](simulators/mjlab/) | mjlab（[arXiv:2601.22074](https://arxiv.org/abs/2601.22074)）；应用层对照 [unitree_rl_mjlab](https://github.com/tangyx96/unitree_rl_mjlab) | [`mjlab-introduction.md`](simulators/mjlab/mjlab-introduction.md) |
| [`simulators/rsl_rl/`](simulators/rsl_rl/) | RSL-RL（[arXiv:2509.10771](https://arxiv.org/abs/2509.10771)）；Isaac Lab / mjlab 的默认 on-policy 后端 | [`rsl-rl-introduction.md`](simulators/rsl_rl/rsl-rl-introduction.md) |

Bertsekas 提供 Bellman / MPC / 策略梯度骨架；Zero-Order 用随机搜索统一 TO 与 RL；SQP 组用数值优化统一 iLQR、DDP 与结构利用型 QP；`policy-learning/` 分三组：在线强化学习（PPO、SAC、HIL-SERL）、生成式动作策略（Diffusion Policy、Flow Matching、VLA、WAM），以及接在两组之间的 DPPO 与 RL-100（DPPO 用 PPO 微调 Diffusion Policy；RL-100 把同一裁剪目标用于真机的离线与在线后训练）；`simulators/` 按仿真器与训练库归档；其中 `mjlab/` 对应 GPU 并行仿真上的 manager-based 机器人 RL 环境，`rsl_rl/` 对应接到这些环境上的 on-policy 训练循环。

版权在原书 / 原论文，仅供个人学习。

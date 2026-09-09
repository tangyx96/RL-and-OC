# RL and Optimal Control — 读书笔记

按文献分目录。论文目录只保留笔记；书（Bertsekas）另有抽取正文与脚本。

| 目录 | 文献 | 笔记 |
|------|------|------|
| [`bertsekas-rl-oc/`](bertsekas-rl-oc/) | Bertsekas, *Reinforcement Learning and Optimal Control*（2019 draft） | [`study-notes/`](bertsekas-rl-oc/study-notes/) |
| [`zero-order-robotics/`](zero-order-robotics/) | Jordana 等, *Zero-Order Optimization*（[arXiv:2506.22087](https://arxiv.org/abs/2506.22087)） | [`study-notes/`](zero-order-robotics/study-notes/) |
| [`sqp-oc/`](sqp-oc/) | Jordana SQP/MPC 系列；Chakravorty（[arXiv:2510.03475](https://arxiv.org/abs/2510.03475)） | [`control-optimization-sqp-synthesis.md`](sqp-oc/study-notes/control-optimization-sqp-synthesis.md) |
| [`clean-rl/`](clean-rl/) | PPO / SAC / FlashSAC | [`ppo_notes.md`](clean-rl/ppo_notes.md)、[`sac_notes.md`](clean-rl/sac_notes.md)、[`flashsac_notes.md`](clean-rl/flashsac_notes.md) |
| [`simulators/mjlab/`](simulators/mjlab/) | mjlab（[arXiv:2601.22074](https://arxiv.org/abs/2601.22074)）；应用层对照 [unitree_rl_mjlab](https://github.com/tangyx96/unitree_rl_mjlab) | [`notes/mjlab-introduction.md`](simulators/mjlab/notes/mjlab-introduction.md) |

Bertsekas 提供 Bellman / MPC / 策略梯度骨架；Zero-Order 用随机搜索统一 TO 与 RL；SQP 组用数值优化统一 iLQR、DDP 与结构利用型 QP；`clean-rl/` 把策略梯度与 off-policy Actor-Critic 落到推导上；`simulators/` 按仿真器归档；其中 `mjlab/` 对应 GPU 并行仿真上的 manager-based 机器人 RL 环境。

版权在原书 / 原论文，仅供个人学习。

Bertsekas 从 PDF 重新抽章节：

```bash
cd bertsekas-rl-oc
python3 scripts/extract_chapters.py
python3 scripts/split_chapter_parts.py
python3 scripts/fix_study_notes_math.py
```

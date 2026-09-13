# RSL-RL

[RSL-RL](https://github.com/leggedrobotics/rsl_rl)（Schwarke 等，[arXiv:2509.10771](https://arxiv.org/abs/2509.10771)）是面向机器人学习的轻量 GPU 强化学习库：默认 PPO，可选学生–教师蒸馏，环境须实现其 `VecEnv` 接口。物理与 MDP 不在库内，由 Isaac Lab、mjlab、Legged Gym、MuJoCo Playground 等提供。

| Notes | Content |
|-------|---------|
| [`rsl-rl-introduction.md`](rsl-rl-introduction.md) | 导读：定位与架构、VecEnv、learn 循环、PPO / 蒸馏、配置、与 mjlab 的衔接 |

官方文档：[leggedrobotics.github.io/rsl_rl](https://leggedrobotics.github.io/rsl_rl/)。mjlab 侧训练入口见 [`../mjlab/mjlab-introduction.md`](../mjlab/mjlab-introduction.md) 第六部分。

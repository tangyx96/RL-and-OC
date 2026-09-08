# mjlab

[mjlab](https://github.com/mujocolab/mjlab)（Zakka 等，[arXiv:2601.22074](https://arxiv.org/abs/2601.22074)）是面向刚体机器人学习的轻量框架：Isaac Lab 式的 manager-based 环境 API，物理后端为 GPU 加速的 MuJoCo Warp。

| 笔记 | 内容 |
|------|------|
| [`mjlab-introduction.md`](mjlab-introduction.md) | 导读：定位与两层架构、仿真层与管理器层、生命周期、训练入口、实现路径 |

应用层示例贯穿 [unitree_rl_mjlab](https://github.com/tangyx96/unitree_rl_mjlab)（将 mjlab 作为库，在本地维护机型、任务与 Train → Play → Sim2Real）。官方文档：[mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/main/index.html)。

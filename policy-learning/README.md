# 策略学习

本目录讨论机器人策略的表示、估计与改进。笔记不按单一算法流派排列，而是沿“优化对象—可访问信息—更新几何—部署接口”组织：同一策略可以由价值函数、行为数据、学习模型或安全过滤器共同塑造。

## 阅读地图

### 策略优化与 actor--critic

- [`foundations/distribution-space-policy-optimization.md`](foundations/distribution-space-policy-optimization.md)：KL 正则、指数重权、占用测度、REPS/MPO/AWR、TRPO/PPO、SAC、MPPI 与 diffusion-RL 的统一视角。
- [`foundations/ppo_notes.md`](foundations/ppo_notes.md)：on-policy 策略梯度、GAE 和 clipping。
- [`foundations/ppo_mlp_notes.md`](foundations/ppo_mlp_notes.md)：CleanRL 连续动作 PPO 的函数近似器。
- [`foundations/sac_notes.md`](foundations/sac_notes.md)：最大熵 actor--critic、soft Bellman 与重参数化。
- [`foundations/flashsac_notes.md`](foundations/flashsac_notes.md)：面向高维机器人控制的 SAC 实现设计。

### 数据、模型与约束

- [`data-model-safety/offline-reinforcement-learning.md`](data-model-safety/offline-reinforcement-learning.md)：离线数据支持、保守 critic、行为正则与 IQL。
- [`data-model-safety/model-based-reinforcement-learning.md`](data-model-safety/model-based-reinforcement-learning.md)：学习动力学、模型 rollout、规划与世界模型。
- [`data-model-safety/safe-policy-learning.md`](data-model-safety/safe-policy-learning.md)：CMDP、CBF-RL、安全过滤与安全 MPC。
- [`real-robot-systems/hil_serl_notes.md`](real-robot-systems/hil_serl_notes.md)：人类介入、混合回放和真机 SAC。

### 生成式动作策略与模仿学习

- [`generative-policies/diffusion_policy_notes.md`](generative-policies/diffusion_policy_notes.md)：DDPM 动作序列、ELBO 与 receding horizon。
- [`generative-policies/flow_matching_notes.md`](generative-policies/flow_matching_notes.md)：conditional flow matching 与流式动作策略。
- [`generative-policies/vla_notes.md`](generative-policies/vla_notes.md)：语言条件行为克隆与离散动作 token。
- [`generative-policies/wam_notes.md`](generative-policies/wam_notes.md)：未来视觉辅助的 world action model。

### 生成式策略的奖励后训练

- [`generative-rl/dppo_notes.md`](generative-rl/dppo_notes.md)：将扩散去噪链写成内层 MDP 后的 PPO 微调。
- [`generative-rl/rl100_notes.md`](generative-rl/rl100_notes.md)：扩散策略的离线、在线真机后训练与一步蒸馏。

## 与其他目录的关系

[`../optimal-control-solvers/`](../optimal-control-solvers/) 从已知动力学上的轨迹优化、约束转录和结构化 KKT 求解出发；模型式 RL 将其中的模型改为由数据学习，并在规划与策略更新之间分配计算。[`../zero-order-robotics/`](../zero-order-robotics/) 讨论只能查询代价或 rollout 时的随机搜索、MPPI 与参数搜索；策略学习进一步指定策略分布、critic、数据集或模型所提供的信息接口。

建议先读分布优化笔记建立统一坐标，再按问题选择分支：离线数据优先读离线 RL；已知或可学习动力学优先读模型式 RL；存在明确安全边界时读安全策略学习；生成式动作策略则从 Diffusion Policy 或 Flow Matching 进入，再读 DPPO 与 RL-100。

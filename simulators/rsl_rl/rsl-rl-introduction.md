# RSL-RL 导读

本文面向尚未使用 RSL-RL 的读者，说明其问题定位、组件分层、环境接口、PPO / 蒸馏循环，以及与 mjlab、Isaac Lab 一类仿真栈的衔接。技术细节以官方文档 [leggedrobotics.github.io/rsl_rl](https://leggedrobotics.github.io/rsl_rl/) 与仓库 [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl) 为准。仿真与 MDP 编排不在本库内；mjlab 一侧见 [`mjlab-introduction.md`](../mjlab/mjlab-introduction.md) 第六部分。PPO 公式推导见 [`clean-rl/ppo_notes.md`](../../clean-rl/ppo_notes.md)。

## 笔记如何组织

官方文档按安装、Overview、Configuration、各 API 分章。下文按「架构 → 组件 → 用法」组织，并把库本身与上层仿真框架分开。

| 部分 | 内容 |
|------|------|
| 一、定位与动机 | 定义、相对通用 RL 库的取舍、适用范围 |
| 二、组件架构 | Runner、Algorithm、Model、Module 如何衔接 |
| 三、环境接口 | `VecEnv`、TensorDict 观测、`obs_groups` |
| 四、Runner 与学习循环 | `learn()`、checkpoint、导出、多 GPU |
| 五、算法 | PPO 与 Student-Teacher Distillation |
| 六、模型、分布与扩展 | MLP / RNN / CNN、动作分布、RND 与对称性 |
| 七、配置 | 嵌套字典、最小 YAML、观测路由 |
| 八、与仿真栈的衔接 | mjlab / Isaac Lab / Legged Gym |
| 九、边界与源码索引 | 误用、阅读顺序、对照仓库 |

建议先读第一、二、四部分，再按需要查阅第三、五、七部分。在 mjlab 或 Isaac Lab 中改超参数时以第七、八部分为索引。

---

## 一、定位与动机

### 1.1 定义

RSL-RL 是面向**机器人学习**的轻量 GPU 强化学习库，由 ETH Zürich 机器人系统实验室（RSL）维护。PyPI 包名为 `rsl-rl-lib`，导入名为 `rsl_rl`。当前文档对应的发行版本约为 5.x（仓库 `pyproject.toml` 记为 5.5.1）；Python 3.9 及以上，依赖 PyTorch 2.6 与 TensorDict。许可证为 BSD-3-Clause。

它不提供物理仿真，也不定义奖励或观测项。职责限于：

1. 以 `VecEnv` 接口驱动已向量化的并行环境；
2. 用 on-policy 循环采集轨迹并更新策略（默认 PPO）；
3. 在需要时把特权教师策略蒸馏为学生策略，并导出 JIT / ONNX。

安装：

```bash
pip install rsl-rl-lib
```

开发安装：克隆仓库后 `pip install -e .`。通常不必单独装：Isaac Lab、mjlab、Legged Gym 会将其列为依赖。

### 1.2 设计策略

通用 RL 库（Stable-Baselines3、RLlib、CleanRL 等）算法面宽，便于基准对比，但模块层次深，改一行算法细节往往要穿过多层抽象。机器人侧更常见的需求是：在大规模 GPU 仿真上跑通 PPO，并针对形态对称、特权观测、实机部署做少量扩展。

论文（Schwarke 等，[arXiv:2509.10771](https://arxiv.org/abs/2509.10771)）给出的原则是：

1. **代码量小、可改。** 典型改动落在 Runner、Algorithm、Network 三个文件，不必 fork 整库；配置里可用 `class_name` 注入自定义类。
2. **算法集合刻意收窄。** 主路径是 PPO 与 DAgger 式行为克隆；辅以对称性增强与 RND 好奇奖励。
3. **GPU-only 吞吐。** 观测、动作、存储均在 PyTorch 张量上，面向数千并行环境；支持多 GPU / 多节点。
4. **为 sim-to-real 预留出口。** 学生–教师蒸馏、JIT / ONNX 导出、超时自举等实现细节按机器人部署习惯处理。

最初版本（v1.0.2）用于 Rudin 等 *Learning to Walk in Minutes*（CoRL 2022），在大规模并行仿真上把四足行走缩短到数分钟。此后成为 Isaac Lab、Legged Gym、mjlab、MuJoCo Playground 的默认 on-policy 后端之一。

### 1.3 与相近库的比较

| 库 | 主要特点 | 较适用的情形 |
|----|----------|----------------|
| RSL-RL | 代码短、PPO + 蒸馏、GPU 向量环境 | 腿式 / 操作机器人的大规模 on-policy 训练 |
| CleanRL | 单文件实现、公式与代码一一对应 | 读懂算法、改实现细节做对照 |
| Stable-Baselines3 | 算法多、Gymnasium 单环境为主 | 小规模基准、非机器人任务 |
| rl-games | 高吞吐、偏游戏与 Isaac 生态 | 需要其特定算法或配置习惯 |
| skrl / RLlib | 模块化、算法面宽 | 算法对比或分布式通用 RL |

对已有 mjlab / Isaac Lab 任务的用户：环境与 MDP 在仿真栈中定义，RSL-RL 只接收 `VecEnv` 与一份 runner 配置。算法公式与 CleanRL 的 PPO 一致；差异在向量化存储、超时处理、特权观测路由和导出路径。

### 1.4 范围

较适用：连续控制、数千并行环境的 on-policy 训练、把仿真特权策略蒸馏成实机可部署策略、在 PPO 上加对称性或好奇探索。

明确不作为目标：算法大而全的基准平台、纯模仿学习（没有在线学生 rollout 的离线 BC）、off-policy 方法（SAC / DQN）、以及物理、奖励、域随机化的定义。后者属于仿真栈；mjlab 中对应 Entity、Manager 与 `ManagerBasedRlEnv`。

---

## 二、组件架构

可将系统看成四个核心组件加若干附属件。Runner 是对外 API；Algorithm 管学习规则；Model 管网络；Module 是网络积木。环境由调用方实现。

```
用户 / 仿真栈
        │  实现 VecEnv（step / get_observations）
        ▼
   OnPolicyRunner 或 DistillationRunner
        │  learn()：采集 → 回报 → 更新 → 日志 / 保存
        ▼
   Algorithm（PPO 或 Distillation）
        │  act / process_env_step / compute_returns / update
        ├── Model（actor+critic，或 student+teacher）
        │        └── Module（MLP、归一化、Gaussian 分布、RNN、CNN）
        ├── RolloutStorage
        └── Extension（可选：RND、Symmetry）
```

**Runner。** 实现主循环 `learn()`，并提供 `save()` / `load()`、`export_policy_to_jit()`、`export_policy_to_onnx()`、`get_inference_policy()`。构造时根据配置解析算法类、初始化 Logger，并按环境变量决定是否进入分布式。

**Algorithm。** `act(obs)` 根据观测采样动作；`process_env_step(...)` 把一步写入 storage；`compute_returns(obs)` 用 GAE 算优势（蒸馏中为空操作）；`update()` 对已采集 batch 做多轮优化。`construct_algorithm()` 根据观测维与 `num_actions` 装配网络。

**Model。** 一次前向分三步：`get_latent()` 从 TensorDict 选出并拼接观测组（可归一化、可过 RNN / CNN）；MLP 映射；若配置了分布则采样或取确定性输出。PPO 使用 actor 与 critic；蒸馏使用 student 与 teacher；底层可以是同一类 `MLPModel`。

**Module。** MLP、`EmpiricalNormalization`、`GaussianDistribution` 等，由 Model 构造并持有。Algorithm 不直接操作这些积木。

附属组件：`VecEnv` 定义环境协议；`RolloutStorage` 按 `(num_steps_per_env, num_envs, …)` 存轨迹并产生 minibatch；Extension 挂在 PPO 上改奖励或损失；Utils 负责 `resolve_callable()`、`resolve_obs_groups()`、日志后端。

自定义算法或网络不必改库源码：在配置字典里把 `class_name` 写成可导入路径即可。文档称这是对 pip 安装版本做研究原型的主要入口。

---

## 三、环境接口

RSL-RL **不**构造仿真。调用方必须提供一个实现 `rsl_rl.env.VecEnv` 的对象。该接口假设环境已向量化且同步：每步对全部 `num_envs` 施加同一形状的动作，返回同一结构的观测。

### 3.1 必要属性与方法

| 成员 | 含义 |
|------|------|
| `num_envs` | 并行环境数 |
| `num_actions` | 动作维 |
| `max_episode_length` | 最大回合长度；可为标量或逐环境张量 |
| `episode_length_buf` | 当前回合已走步数 |
| `device` | 环境张量所在设备 |
| `cfg` | 任意配置对象，供日志写入 |
| `get_observations()` | 返回当前观测 `TensorDict` |
| `step(actions)` | 返回 `(obs, rewards, dones, extras)` |

`step` 的动作为 `(num_envs, num_actions)`；`rewards` 与 `dones` 为 `(num_envs,)`。环境须实现 **same-step reset**：某环境 `done` 时，同一次 `step` 返回的观测已是重置后的初始观测，而不是终止态。Runner 在 rollout 前**不**调用 `reset()`，只调用 `get_observations()`。因此包装器（如 mjlab 的 `RslRlVecEnvWrapper`）通常在构造时自行 `reset()` 一次。

### 3.2 观测组与 `obs_groups`

观测不是单一向量，而是 `TensorDict`：每个键是一个 **observation group**（如 `"policy"`、`"privileged"`）。Runner 配置中的 `obs_groups` 再把这些 group 绑到算法所需的 **observation set**：

| set（`obs_groups` 的键） | 用途 |
|--------------------------|------|
| `actor` | PPO actor 输入 |
| `critic` | PPO critic 输入 |
| `student` / `teacher` | 蒸馏双方输入 |
| `rnd_state` | RND 好奇网络输入 |

典型特权学习写法：

```python
obs_groups = {"actor": ["policy"], "critic": ["policy", "privileged"]}
```

actor 只看部署时仍存在的 `"policy"`；critic 额外看仿真里才有的 `"privileged"`。Model 的 `get_latent()` 按该列表拼接最后一维。缺项或配错由 `resolve_obs_groups()` 补全或报错。

### 3.3 `extras`

`step` 的第四返回值是字典。RSL-RL 读取其中两项：

- **`time_outs`**：因到时截断而非失败终止。PPO 用其做价值自举（见 5.2），避免把截断当成吸收态。
- **`log`**：键以 `/` 开头的标量或张量，写入 Logger；张量记均值。

Gymnasium 的 `terminated` 与 `truncated` 在进入 RSL-RL 前应合并为 `dones`，并把超时单独放入 `time_outs`。mjlab 包装器即做此转换。

---

## 四、Runner 与学习循环

对外入口是 `OnPolicyRunner`。`DistillationRunner` 继承它，仅在 `learn()` 开头检查教师权重已加载。

### 4.1 构造

```python
from rsl_rl.runners import OnPolicyRunner

runner = OnPolicyRunner(
    env=env,              # VecEnv
    train_cfg=train_cfg,  # 见第七部分
    log_dir="logs/exp",
    device="cuda:0",
)
runner.learn(num_learning_iterations=1500)
```

构造顺序：配置多 GPU → `env.get_observations()` → `algorithm.class_name.construct_algorithm(...)` → Logger。算法类通过 `resolve_callable` 解析，因此可换成自定义 PPO 子类。

### 4.2 `learn()` 一次 iteration

对 `it = 0 … num_learning_iterations-1`：

1. **Rollout**（`torch.inference_mode`）。重复 `num_steps_per_env` 次：`actions = alg.act(obs)` → `env.step(actions)` → 可选 NaN 检查 → `alg.process_env_step(...)`。每环境采集步数相同，总样本约为 `num_envs × num_steps_per_env`（多 GPU 时再乘 world size）。
2. **回报。** `alg.compute_returns(obs)`，用 rollout 结束后的 critic 值作 bootstrap。
3. **更新。** `alg.update()`：`num_learning_epochs` 轮、每轮 `num_mini_batches` 个 minibatch。
4. **日志与保存。** 每隔 `save_interval` 写 `model_{it}.pt`；结束时再存一份。

`init_at_random_ep_len=True` 时，把 `episode_length_buf` 随机到 `[0, max_episode_length)`。大规模并行下环境会在训练初期同步终止，造成轨迹相关；随机提前结束可打散相位。论文将此列为相对朴素 PPO 的实现细节之一。

Runner **不**在循环内 reset。reset 由环境在 `done` 的同一步完成。

### 4.3 保存、加载与导出

| 方法 | 作用 |
|------|------|
| `save(path)` | 算法 `state_dict`、iteration、可选 infos；并可上传到 W&B / Neptune |
| `load(path)` | 加载权重；`load_cfg` 可只恢复 actor 等子集 |
| `get_inference_policy()` | `eval` 模式的策略，默认确定性输出 |
| `export_policy_to_jit` / `export_policy_to_onnx` | 部署用图；ONNX 输入名为 `obs`，输出名为 `actions` |

回放不必再跑 `learn()`：

```python
runner.load("logs/exp/model_1499.pt")
policy = runner.get_inference_policy()
obs = env.get_observations()
for _ in range(1000):
    actions = policy(obs)          # 默认确定性
    obs, rewards, dones, extras = env.step(actions)
```

导出图接收**已拼接**的观测向量，而不是 TensorDict。部署侧须按训练时 actor 的 group 顺序拼接，并使用同一套归一化统计（统计量已写入 checkpoint / 导出模块）。

### 4.4 多 GPU 与日志

分布式由环境变量 `WORLD_SIZE`、`RANK`、`LOCAL_RANK` 触发（torchrun / 集群启动器）。`device` 必须等于 `cuda:{LOCAL_RANK}`。算法在 backward 后 `reduce_parameters()`，adaptive KL 的学习率在 rank 0 上调整再广播。

Logger 默认 TensorBoard。W&B / Neptune 在 runner 配置的 `logger` 字典中指定 `class_name` 与 `project_name`。顶层旧字段 `wandb_project` 已弃用。

---

## 五、算法

### 5.1 PPO 在循环中的位置

`PPO.act` 用当前 actor 随机采样动作，同时记下 log-prob、分布参数、critic 价值与（若有）RNN 隐状态。`process_env_step` 更新观测归一化，写入 reward / done，并处理超时与 RND。`compute_returns` 反序算 GAE。`update` 在新策略上重算 log-prob，做 PPO 裁剪更新。

与 [`ppo_notes.md`](../../clean-rl/ppo_notes.md) 中的目标一致。实现上需单独记住下列机器人场景中的处理。

### 5.2 超时自举

若 `extras` 含 `time_outs`，则

$$
r_t \leftarrow r_t + \gamma V(s_t) \mathbf{1}_{\mathrm{timeout}}.
$$

截断不是终止：后续价值仍应计入回报。失败（倾覆等）的 `done` 不走这项，GAE 中 `next_is_not_terminal = 1 - done` 会把后续截断。包装器必须把 Gymnasium 的 `truncated` 译成 `time_outs`，否则超时会被当成失败，价值被压低。

### 5.3 GAE

对 rollout 倒序：

$$
\delta_t = r_t + \gamma (1-d_t) V(s_{t+1}) - V(s_t)
$$

$$
\hat{A}_t = \delta_t + \gamma\lambda (1-d_t)\hat{A}_{t+1}
$$

$$
\hat{R}_t = \hat{A}_t + V(s_t)
$$

默认在整个 rollout 上标准化优势；`normalize_advantage_per_mini_batch=True` 则改在每个 minibatch 内标准化。

### 5.4 更新目标

每次 minibatch：

1. 若启用对称性，先把镜像样本拼进 batch。
2. 用当前 actor / critic 前向，得到新的 log-prob 与 $V_\theta(s)$。
3. `schedule="adaptive"` 时，若平均 KL 大于 `2 * desired_kl` 则学习率除以 1.5（下限 `1e-5`）；小于 `desired_kl / 2` 则乘 1.5（上限 `1e-2`）。
4. 概率比 $r_t(\theta)=\exp(\log\pi_\theta-\log\pi_{\mathrm{old}})$。实现里对 $-\hat{A} r$ 与裁剪项取逐元素最大（最小化该损失）；与标准 PPO 的 $\min$ 形式等价：

$$
L^{\mathrm{CLIP}} = \mathbb{E}\left[\min\left(r_t(\theta)\hat{A}_t,\quad \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
$$

5. 价值损失默认为裁剪版本（与 `clip_param` 共用 $\epsilon$）。
6. 总损失 $L^{\mathrm{CLIP}} + c_v L^V - c_e \bar{H}$。可选再加 mirror loss；RND 的预测误差用独立优化器。

梯度范数裁剪 `max_grad_norm`（默认 1.0）。`use_mixed_precision` 时前向与损失为 bfloat16 autocast，反向与 optimizer step 仍为 fp32。

常用超参与 mjlab G1 速度任务一致：`clip_param=0.2`，`entropy_coef=0.01`，`num_learning_epochs=5`，`num_mini_batches=4`，`gamma=0.99`，`lam=0.95`，`desired_kl=0.01`，`learning_rate=1e-3` 且 `schedule="adaptive"`。

### 5.5 Distillation

蒸馏是在线行为克隆，接近 DAgger：学生与环境交互，教师在相同观测上给出动作目标，学生用 MSE 或 Huber 拟合。`compute_returns` 为空。`gradient_length` 控制反传长度（RNN 时即 BPTT 窗口）。

`DistillationRunner.learn()` 要求事先 `load()` 教师。典型流程：先用 PPO 在特权观测上训练教师并保存；再构造 `DistillationRunner`，`obs_groups` 中 `teacher` 指向特权组、`student` 指向实机组，加载教师后蒸馏。教师在 `eval` 模式且不更新。

论文强调：库**不**提供纯离线模仿。没有在线学生 rollout 的数据集 BC 需要自写循环。

---

## 六、模型、分布与扩展

### 6.1 三种 Model

三者共享 MLP 头与可选输出分布；差异在 `get_latent()`。

| 类 | 输入 | 要点 |
|----|------|------|
| `MLPModel` | 仅 1D group | 拼接 → 可选 `EmpiricalNormalization` → MLP |
| `RNNModel` | 1D + 时序 | LSTM 或 GRU；storage 走 recurrent minibatch，隐状态跨步传递 |
| `CNNModel` | 1D + 若干 2D | 每个 2D group 独立 CNN；PPO 可 `share_cnn_encoders` 让 actor / critic 共用 |

`hidden_dims` 只描述 MLP 隐层，例如 `(512, 256, 128)`。输入维由观测决定，输出维由 `num_actions`（actor）或 1（critic）决定，配置中都不写。`activation` 常用 `"elu"`，加在隐层之间，不加在输出头。

`forward(..., stochastic_output=False)`：**默认确定性**。训练采样须显式 `stochastic_output=True`（`PPO.act` 已如此）。评估与导出走均值（Gaussian）或分布规定的确定性映射。

### 6.2 动作分布

仅 actor / student 通常带 `distribution_cfg`。critic / teacher 常为确定性回归头。

| 类 | 含义 |
|----|------|
| `GaussianDistribution` | 对角高斯；均值由 MLP 给出，标准差**与状态无关**，`std_type` 为 `"scalar"` 或 `"log"`，`learn_std` 控制是否学习 |
| `HeteroscedasticGaussianDistribution` | 标准差由网络逐样本预测 |
| `BetaDistribution` | 样本在 $[0,1]$，再线性缩放到 `action_range`（默认 $[-1,1]$） |

机器人关节位置控制几乎总是 `GaussianDistribution` + `std_type="scalar"`：各动作维共用一个 $\sigma$，初值 `init_std`（常为 1.0）。这与「观测正态化」不是一回事：`obs_normalization` 只标准化网络输入。

### 6.3 扩展

二者目前只挂在 PPO 上。

**RND**（Schwarke 等 CoRL 2023；原论文 Burda 等）。可训练 predictor 拟合固定 target 的嵌入，预测误差作内在奖励，加到外在奖励上。与原版 RND 的差别：好奇状态可以只是观测的一个子集（`rnd_state`），从而把探索压到门是否打开等任务相关坐标，而不是整个物理状态。稀疏奖励下可减少手写 shaping。权重可 constant / step / linear 退火。

**Symmetry**（Mittal 等 ICRA 2024）。用户提供镜像函数（观测与动作如何左右对换）。`use_data_augmentation` 把镜像轨迹拼进 minibatch；`use_mirror_loss` 另加一项，惩罚策略在镜像观测上与自身镜像动作不一致。可同时开启。不支持 recurrent 策略。四足左右对称时用来提高样本效率并抑制左右偏置。

---

## 七、配置

配置是传给 Runner 的嵌套字典，通常来自 YAML，或由 Isaac Lab / mjlab 的 dataclass 转出。顶层即 runner；其下为 `algorithm` 与各 Model（PPO 为 `actor` / `critic`，蒸馏为 `student` / `teacher`）。算法字典里可再挂 `rnd_cfg`、`symmetry_cfg`；模型字典里可挂 `distribution_cfg`。

### 7.1 最小 PPO 配置

```yaml
runner:
  num_steps_per_env: 24
  obs_groups: {"actor": ["policy"], "critic": ["policy", "privileged"]}
  save_interval: 100
  algorithm:
    class_name: PPO
  actor:
    class_name: MLPModel
    distribution_cfg:
      class_name: GaussianDistribution
  critic:
    class_name: MLPModel
```

未写出的字段用文档默认值（例如 `hidden_dims=[256,256,256]`，`activation=elu`，`clip_param=0.2`）。并行环境数**不属于**这份配置，而属于 `env.num_envs`。

### 7.2 Runner 字段

**调度。** `num_steps_per_env` 为每次 iteration 每环境采集的策略步（腿式速度任务常用 24）；`save_interval` 为 checkpoint 间隔。总更新次数由 `learn(num_learning_iterations=...)` 传入，不在 YAML 里；mjlab 则把它放进 `RslRlOnPolicyRunnerCfg.max_iterations`，由训练脚本转交给 `learn()`。

**观测路由。** `obs_groups` 必须覆盖当前算法用到的 set。配错时症状是 actor 维度与部署观测对不上，或 critic 没用上特权信息。

**算法块。** 见 5.4。`class_name` 必须能 `resolve_callable`（内置 `"PPO"` 即可）。

**网络块。** `obs_normalization=True` 时用训练中滑动的均值方差标准化输入。只改网络宽度或激活时，不必动算法块。

完整键表见官方 [Configuration](https://leggedrobotics.github.io/rsl_rl/guide/configuration.html)。上表只标出改任务时真正需要分清的层次：环境并行度、观测路由、网络、PPO 更新、调度。

---

## 八、与仿真栈的衔接

RSL-RL 出现在四条常见训练链中，位置相同：仿真栈负责物理与 MDP，本库负责 `learn()`。

| 仿真栈 | 物理 | 接到 RSL-RL 的方式 |
|--------|------|---------------------|
| Isaac Lab | Isaac Sim / PhysX | 官方 wrapper 实现 `VecEnv`；配置多为 `@configclass` |
| Legged Gym | Isaac Gym | 早期集成；任务脚本直接构造 `OnPolicyRunner` |
| mjlab | MuJoCo Warp | `RslRlVecEnvWrapper` + `MjlabOnPolicyRunner` |
| MuJoCo Playground | MJX / Warp | 环境侧适配 `VecEnv` |

### 8.1 mjlab

细节以 [`mjlab-introduction.md`](../mjlab/mjlab-introduction.md) 第六、九部分为准。此处只固定职责边界。

```
ManagerBasedRlEnv  （观测 / 奖励 / 终止 / 事件）
        │
        ▼
RslRlVecEnvWrapper  （dict 观测 → TensorDict；terminated+truncated → dones + extras["time_outs"]）
        │
        ▼
MjlabOnPolicyRunner(OnPolicyRunner)
        │  learn() 来自本库；子类只加 checkpoint 里的环境状态与 ONNX
        ▼
PPO
```

任务注册把 `ManagerBasedRlEnvCfg` 与 `RslRlOnPolicyRunnerCfg` 绑在同一 task id 上。后者转成上面的 `train_cfg` 字典。`num_envs` 在 `env.scene.num_envs`（CLI `--num-envs`），不在 agent 配置里。

包装器必须在构造时 `env.reset()`，因为 RSL-RL 假定此后只 `get_observations()` / `step()`。自备 `train.py` 时顺序仍是：构造 `ManagerBasedRlEnv` → 包装 → 构造 runner → `learn()`。

应用仓库（例如 [unitree_rl_mjlab](https://github.com/tangyx96/unitree_rl_mjlab)）把 `runner_cls` 换成在 `save` 时调用 `export_policy_to_onnx` 的子类；`deploy/` 只读 `policy.onnx`，不再构造 `VecEnv`。

### 8.2 从本库看过去时容易混淆的两点

1. **mjlab 的 `RslRlModelCfg` 比上游 YAML 更短。** 它把 `class_name: MLPModel` 等默认值藏进 dataclass；写出的往往只有 `hidden_dims`、`activation`、`obs_normalization`。对照本库文档时，把缺省项按第七部分补全即可。
2. **历史观测在哪一侧。** mjlab Observation Manager 可在 term 上做 frame stack，再把展平向量交给 RSL-RL。需要 LSTM 时改用 `RNNModel`，不要两套历史叠在一起却不改 `obs_groups`。

---

## 九、边界、常见误用与源码索引

### 9.1 适用范围

较适用：已有向量化 GPU 环境；连续控制 PPO；特权 critic + 部署 actor；教师蒸馏；对称机器人上的数据增强；稀疏奖励下用 RND。

不适合：在本库内实现 SAC 等 off-policy 并期望官方维护；无仿真、无 `VecEnv` 的单环境 Gym 脚本（勉强能包一层，但设计目标不是这个）；把奖励函数写进 Algorithm；用本库做算法动物园式的基准。

### 9.2 常见误用

- **忘记构造时 reset。** Runner 不会 reset；未 reset 的 `get_observations()` 可能是未初始化缓冲。
- **把 timeout 写进失败 `done` 且不设 `time_outs`。** 截断被当成终止，GAE 少一段 bootstrap。
- **actor 的 `obs_groups` 含特权项。** 训练能涨分，导出后实机没有这些通道。
- **导出后按错拼接顺序。** JIT / ONNX 吃的是拼接向量，顺序必须与 actor 的 group 列表一致。
- **评估时仍 `stochastic_output=True`。** `get_inference_policy()` 默认确定性；自己调 `forward` 时不要把训练采样旗标留下。
- **在 RNN 策略上开 Symmetry。** 库会直接报错。
- **把 `num_envs` 写进 runner YAML。** 并行度是环境属性；改它不会让已构造的 `VecEnv` 变大。

### 9.3 建议的阅读顺序

1. [Overview](https://leggedrobotics.github.io/rsl_rl/guide/overview.html) 与本文第一、二部分
2. [Environment](https://leggedrobotics.github.io/rsl_rl/api/env.html) 与第三部分
3. [Configuration](https://leggedrobotics.github.io/rsl_rl/guide/configuration.html) 与第七部分
4. 源码：`runners/on_policy_runner.py` 的 `learn()`，然后 `algorithms/ppo.py` 的 `act` / `process_env_step` / `compute_returns` / `update`
5. 仿真侧：[mjlab 训练一节](../mjlab/mjlab-introduction.md)；Isaac Lab 文档中的 RSL-RL 包装
6. 公式对照：[PPO 笔记](../../clean-rl/ppo_notes.md)

论文：Schwarke, Mittal, Rudin, Hoeller, Hutter, *RSL-RL: A Learning Library for Robotics Research*, [arXiv:2509.10771](https://arxiv.org/abs/2509.10771), 2025。在 mjlab / Isaac Lab 上发结果时，应同时引用该文与仿真栈各自的论文。

### 9.4 源码对照

包根目录为 `rsl_rl/`（发行包名 `rsl-rl-lib`）。

```
rsl_rl/
├── env/vec_env.py          # VecEnv 抽象
├── runners/
│   ├── on_policy_runner.py
│   └── distillation_runner.py
├── algorithms/
│   ├── ppo.py
│   └── distillation.py
├── models/                 # MLPModel, RNNModel, CNNModel
├── modules/                # MLP、CNN、RNN、分布、归一化
├── storage/                # RolloutStorage
├── extensions/             # RND, Symmetry
└── utils/                  # Logger, resolve_callable, resolve_obs_groups
```

| 问题 | 先看 |
|------|------|
| 一次 iteration 做什么 | `OnPolicyRunner.learn` |
| 超时如何进 GAE | `PPO.process_env_step` 中对 `time_outs` 的处理 |
| 优势如何计算 | `PPO.compute_returns` |
| clip 与 adaptive LR | `PPO.update` |
| 观测如何进网络 | `MLPModel.get_latent` |
| 部署图长什么样 | `MLPModel.as_onnx` / `as_jit` |
| 教师是否已加载 | `Distillation.teacher_loaded`；`DistillationRunner.learn` |
| mjlab 如何满足本接口 | `mjlab` 的 `RslRlVecEnvWrapper` |

Runner 对应「如何与环境交替」；Algorithm 对应「如何用这批数据改网络」；`obs_groups` 对应「哪一段观测允许在实机上出现」。仿真里的奖励与接触，仍在环境一侧，不在本库。

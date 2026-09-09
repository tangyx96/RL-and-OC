# mjlab 导读

本文面向尚未使用 mjlab 的读者，说明其问题定位、分层结构、MDP 的配置方式、任务注册与训练入口，以及适用范围。技术细节以官方文档 [mujocolab.github.io/mjlab](https://mujocolab.github.io/mjlab/main/index.html) 与仓库 [mujocolab/mjlab](https://github.com/mujocolab/mjlab) 为准。基于 mjlab 的多机型训练与实机部署见第九部分，对照仓库为 [unitree_rl_mjlab](https://github.com/tangyx96/unitree_rl_mjlab)。

## 笔记如何组织

官方文档按安装、Entity、各 Manager、RSL-RL 分章，缺少一条贯穿各层的阅读路径。下文按「架构 → 组件 → 用法」组织。

| 部分 | 内容 |
|------|------|
| 一、定位与动机 | mjlab 的定义、相对原生 MuJoCo 的优势、与相近框架的比较 |
| 二、两层架构 | 物理、MDP 编排与训练算法三者如何衔接 |
| 三、仿真层 | 实体、场景、传感器与 GPU 物理 |
| 四、管理器层 | 观测、动作、奖励等 MDP 项的组合 |
| 五、生命周期与时间尺度 | 环境步进顺序与时钟参数 |
| 六、训练与内置任务 | 注册表、CLI 与参考任务 |
| 七、边界与选型 | 适用范围与常见误用 |
| 八、源码结构与实现路径 | 对照仓库完成任务接入 |
| 九、应用示例 | unitree_rl_mjlab：Train → Play → Sim2Real |

建议先读第一、二、五部分，再按需要查阅第三、四、六部分。实现环境时以第八部分为索引。阅读应用仓库时先看第九部分，再回到第三、四、六、八部分对照实现。

---

## 一、定位与动机

### 1.1 定义

mjlab 是面向**刚体机器人学习**的轻量开源框架，其核心能力有二：

1. 以 **MuJoCo Warp** 在 GPU 上并行推进大量独立仿真世界；
2. 采用 Isaac Lab 提出的 **manager-based API**，将观测、奖励、终止、事件等 MDP 项作为可组合模块注册。

训练侧默认对接 **RSL-RL**（以 PPO 为代表的 on-policy 算法）。观测、奖励与动作均为 PyTorch 张量，并与 MuJoCo Warp 的 GPU 缓冲零拷贝共享。用户可直接访问原生 `MjModel` / `MjData`，中间不经过 USD 或跨仿真器翻译层。

依赖规模较小。无需安装即可运行演示：

```bash
uvx --from mjlab --refresh demo
```

训练要求 Linux 与 NVIDIA GPU（文档建议 CUDA 12.4 及以上）；评估可在 Linux、macOS 或 Windows（WSL）上进行。Python 版本为 3.10 及以上、3.14 以下。许可证为 Apache 2.0。

### 1.2 设计策略

仿真到真机的强化学习效果，在很大程度上取决于仿真是否可检查、可复现，以及是否支持大规模并行。现有工具大致处于两种极端：

- **Isaac Lab** 提供完整的 manager-based 环境 API，但依赖 Omniverse，安装与启动成本较高，适用于需要照片级渲染与 USD 管线的项目。
- **MuJoCo Playground** 抽象较少，单任务原型速度快，但环境定义偏于单体，多机器人、多任务时易于重复。
- **Newton** 面向多物理求解器与可微仿真，超出「刚体 MuJoCo + RL 环境」的范围。

mjlab 的取舍是：**上层沿用 Isaac Lab 的 manager-based API，底层将物理引擎换为 MuJoCo Warp**。前者给出可组合的 MDP 接口，后者把接触物理放到 GPU 上批量推进，官方给出的量级是约 10–100 倍加速，从而使大规模 on-policy 训练从按天计缩短到按小时计。工程原则如下（官方文档 *Why mjlab?*）：

1. **降低安装成本。** 不捆绑重量级运行时。
2. **物理过程可检查。** 仅绑定 MuJoCo Warp；跨仿真器可移植并非目标。
3. **与 MuJoCo 生态对齐。** MJCF、Menagerie 及既有 MuJoCo 工具无需翻译即可使用。

manager-based API 把环境拆成独立、可组合的管理器。

| 管理器 | 职责 |
|--------|------|
| 观测（Observation） | 定义策略与价值网络读取哪些量 |
| 动作（Action） | 把策略输出路由到执行器 |
| 奖励（Reward） | 加权组合多项标量，形成训练信号 |
| 终止（Termination） | 判定回合结束（失败或截断） |
| 事件（Event） | 在指定生命周期节点改仿真，域随机化由此进入 |

此外还有命令、课程与指标三类管理器，同样在第四部分说明。

### 1.3 相对原生 MuJoCo 的优势

原生 MuJoCo 以 CPU 上的 `mj_step` 推进单个（或少量线程上的）世界，适合检查模型与调试控制律。大规模强化学习需要的是数千个独立环境同时步进。MuJoCo Warp 把同一套 `MjModel` / `MjData` 范式搬到 GPU，mjlab 再在其上接好 MDP 与训练循环。

| 特性 | 原生 MuJoCo（CPU） | mjlab（MuJoCo Warp） |
|------|-------------------|----------------------|
| 计算载体 | CPU | GPU（NVIDIA） |
| 并行方式 | 多线程调用 `mj_step` | 一次步进更新数千个独立世界 |
| 性能 | 单环境 / 小批量基准 | 官方称约 10–100 倍；NVIDIA 报告相对旧版 MJX，locomotion 约 252 倍、manipulation 约 475 倍 |
| 典型用途 | 单环境仿真、算法与模型调试 | 大规模 RL 训练、参数扫描 |

吞吐的数量级（取决于机型、接触与 `num_envs`，以下为公开报告而非保证值）：物理步进可达约 $1.25\times 10^{4}$ 环境帧/秒；Warp（CUDA）训练可达 $10^{6}$ 步/秒量级，而 CPU 仿真常见约 $10^{5}$ 步/秒。

因此：需要检查单个 MJCF、调 PD 或看接触时，仍用原生 MuJoCo；需要在保留 MuJoCo 接触模型的前提下做大规模 on-policy 训练时，用 mjlab。它追求的是 Isaac Lab 量级的并行吞吐，而不是替换 MuJoCo 的建模方式。

### 1.4 与相近框架的比较

| 框架 | 主要特点 | 较适用的情形 |
|------|----------|----------------|
| mjlab | 依赖轻、原生 MuJoCo、PyTorch | 已有 MJCF，需要 GPU 并行 RL 与结构化环境 |
| Isaac Lab | 照片级渲染、USD、Omniverse | 必须使用 Isaac Sim 能力 |
| MuJoCo Playground | 抽象少 | 单任务快速原型 |
| Newton | 可变形体、可微仿真、多求解器 | 刚体 MuJoCo 不足以描述对象 |

对 Isaac Lab 用户而言，MDP 结构大体对应（reward、observation、action、command、termination、event、curriculum 等 manager）。主要差异在配置形态：Isaac Lab 使用嵌套 `@configclass`，mjlab 使用**名称到配置对象的字典**；场景不再使用 USD 与 `prim_path`，而使用 MJCF 与 `MjSpec`。

### 1.5 范围

mjlab 提供刚体机器人学习所需的仿真与 MDP 基础设施，包括深度与射线传感器。**高保真 RGB 渲染不在范围内。** 视觉策略仍可通过特权状态训练教师策略、再在外部渲染器上蒸馏得到。仓库提供速度跟踪、运动模仿、操作等参考任务。

实机通信、DDS 与电机 SDK 亦不在 mjlab 内；部署流程见第九部分。

---

## 二、两层架构

可将系统看成三个职责、由环境类连接的两层：MuJoCo Warp 只负责物理，`ManagerBasedRlEnv` 把原始状态翻译成 MDP，RSL-RL 只看到向量化环境接口。

```
MJCF / Python 配置
        │
        ▼
   Entity 组装为 MjSpec  ──► 编译 MjModel（CPU）
        │
        ▼
   MuJoCo Warp 上传 GPU  ──► 一份 MjData，N 个并行世界
        │                         ▲
        │                         │ 读写状态
        ▼                         │
   ManagerBasedRlEnv  （管理器层：观测、奖励、事件等）
        │
        ▼
   RSL-RL（PPO 等）
```

上图自上而下对应四个阶段。

**（1）物理世界的构建（CPU，仅初始化）。** MJCF 或 Python 给出机器人与场景；Entity 拼成 `MjSpec`，再编译为只读 `MjModel`（质量、惯性、几何、运动学树）。此阶段不进入训练循环。

**（2）并行仿真（GPU）。** `MjModel` 上传并编译为 Warp 后端。`MjData` 增加世界维，一份结构同时保存 N 个环境的状态。训练循环把形状为 `(N, …)` 的动作写入 GPU，再读回批量状态。域随机化、图捕获等实现细节见 3.1。

**（3）MDP 封装。** 物理输出的是关节角、速度等底层量。`ManagerBasedRlEnv` 用观测、奖励、终止、事件等管理器把它们变成算法所需的张量——例如用前进距离作奖励、用倾覆作终止。term 如何书写见第四部分。

**（4）策略更新。** RSL-RL 接收上述批量数据，运行 PPO 等算法，再把新动作经管理器写回第（2）阶段。与 runner、wrapper 的接口见第六部分。

训练主循环是（2）→（3）→（4）→（2）。配置、编译与上传只做一次。三个模块的数据都以 batch 形式留在 GPU 显存，避免逐步把状态拷回 CPU；这是吞吐的主要来源。

---

## 三、仿真层

### 3.1 MuJoCo Warp

MuJoCo Warp 是 MuJoCo 的 GPU 后端。它保留 `MjModel` / `MjData` 范式，并为数据增加**世界维**：一份 `MjData` 同时保存 N 个独立仿真实例的状态，从而在单次步进中更新大量环境。

默认情形下模型参数在各世界间共享。域随机化可将个别字段展开为 per-world 存储。仿真步进可捕获为 **CUDA graph**：内核序列录制一次后重放，以降低 CPU 调度开销。图捕获发生在环境启动阶段；回合 reset 与域随机化仍由 Python 执行，不撤销已捕获的图。

各环境为**独立世界**，不共享物理空间，互不发生碰撞。环境原点（`env_origins`）主要用于可视化排布，以及在程序化地形上将环境置于对应子地形块。

当前约束是：各世界共享同一份 `MjModel`（网格、几何与运动学树相同）。不同世界使用不同网格的异构仿真仍在上游推进；mjlab 以 `VariantEntityCfg` 提供过渡支持。

### 3.2 Entity

**Entity** 表示场景中的物理对象：机器人、被操作物、桌面、地形等。Isaac Lab 将其拆分为 `Articulation`、`RigidObject` 等多个子类；mjlab 使用单一 `Entity`，并以两个正交属性分类：

| 类型 | 例子 | 固定基座 | 内部关节 |
|------|------|----------|----------|
| 固定、非铰接 | 桌子、墙 | 是 | 否 |
| 固定、铰接 | 机械臂、门 | 是 | 是 |
| 浮动、非铰接 | 盒子、杯子 | 否 | 否 |
| 浮动、铰接 | 人形、四足 | 否 | 是 |

每个实体由 `EntityCfg` 描述。实践中通常必须提供 **`spec_fn`**（返回 `mujoco.MjSpec` 的可调用对象，最简情形为从 MJCF 加载）。其余字段在 spec 上叠加任务相关修改，而无需改写原始 XML：

- `init_state`：默认根位姿与关节角，写入 MuJoCo keyframe，供 reset 使用。关节名按正则匹配，后写条目覆盖先写条目。
- `articulation`：执行器配置。被动物体可省略。
- spec editor：碰撞策略、灯光、相机、材质、geom 属性补丁等。

固定基座实体会被包裹为 mocap body，否则所有并行环境将焊接于世界原点。包裹对用户透明，但**位姿写入发生在 reset 事件中**。若事件配置中缺少 `reset_root_state_uniform` 一类项，固定基座物体将停留在原点。

运行时状态读取由高到低分为三层：

1. **`entity.data`**：奖励与观测函数的主接口，张量形状为 `(num_envs, ...)`。
2. **传感器**：由 `env.scene["name"]` 索引。
3. **`env.sim.data` / `env.sim.model`**：完整 Warp 数组，使用全局 MuJoCo 索引，为零拷贝 PyTorch 视图。

在 term 中选择实体与关节时使用 `SceneEntityCfg("robot", joint_names=...)`。正则在 manager 初始化时解析为整数下标，步进过程中不再进行字符串匹配。

### 3.3 执行器

执行器将位置、速度或力矩指令转换为关节努力，配置于 `EntityCfg.articulation`。类型上应区分：

- **内建（builtin）**：在 `MjSpec` 中生成原生 MuJoCo actuator。速度相关阻尼由积分器隐式处理，在高增益或较大步长下更为稳定。文档默认积分器为 **`implicitfast`**。
- **显式（explicit）**：在 Python 中计算力矩，再经透传 actuator 写入。适用于自定义控制律、转矩–转速饱和或学习型执行器；积分器无法隐式处理其速度导数，数值稳定性通常弱于内建类型。

腿式机器人常用 `BuiltinPositionActuator`（PD 位置跟踪）；轮式关节可用速度执行器；电机饱和曲线可使用显式 `DcMotorActuator` 或内建 `BuiltinDcMotorActuator`（后者包含电气模型）。XML 中已有的 actuator 可用 `XmlActuator` 包装。

执行器配置可设置指令延迟，以模拟策略指令滞后、而板载 PD 仍读取最新关节状态的情形。这与观测延迟不同：后者使策略获得过时状态。二者分别对应「传感 → 策略」与「策略 → 电机」两段时延。

一份 `ActuatorCfg` 表示**同一硬件模型**作用于一组关节（刚度、阻尼、`armature` 等在组内均匀）。增益不同时应按关节组拆分配置，而不是在单份配置中使用 per-joint 字典。例如四足可将髋、大腿、小腿分为三个 `BuiltinPositionActuatorCfg`（小腿刚度与力矩上限较高），并用 `CollisionCfg` 指定参与接触的 geom、足端 `condim` 与摩擦。mjlab 提供这些配置类型；具体数值写在该机型的 `EntityCfg` 工厂函数中。

### 3.4 场景

`SceneCfg` 描述地形、实体字典、传感器元组与并行环境数。`Scene` 负责 MJCF 拼接、编译与运行时访问。

拼接时，每个实体的 `MjSpec` 以实体名为前缀挂到根 spec。名为 `"robot"` 的实体中，`base_link` 成为 `robot/base_link`。前缀用于避免重名，并为观测与奖励提供稳定命名空间。地形通常不带前缀。传感器在实体之后加入，可引用带前缀的元素。

`scene.compile()` 得到 `MjModel`；`Simulation` 经 Warp 将其上传至 GPU。随后 `scene.initialize()` 解析各实体在编译模型中的下标并分配状态缓冲。`scene.to_zip(path)` 可导出编译后的模型，供独立 MuJoCo viewer 检查。

跨实体约束（例如连接吊车与机器人的腱）不能写在任一实体的 XML 中，应使用 `SceneCfg.spec_fn`：在全部实体挂接完成、编译之前获得完整 `MjSpec` 再修改。

### 3.5 传感器

传感器配置于 `SceneCfg`，可附着于实体的 site 或关节，也可独立于实体。因此它不属于 `EntityCfg`。XML 中已有的 MuJoCo sensor 在拼接时自动发现，访问名带实体前缀，例如 `env.scene["robot/trunk_imu"]`。

四类内建传感器：

- **`BuiltinSensor`**：封装加速度计、陀螺、`framepos` 等原生类型，输出形状为 `[num_envs, dim]`。
- **`ContactSensor`**：从 MuJoCo 的扁平接触列表中筛选指定配对，并可做 reduction、腾空时间统计与子步历史。子步历史对 decimation 很重要：策略步内出现又消失的碰撞，若只读取最终 `found`，可能被漏检。速度跟踪中常见两类：足–地接触（`track_air_time=True`，供步态奖励）与自碰撞（`history_length` 常取与 `decimation` 相同，例如 4）。足–地接触若已启用腾空时间累计，通常不必再设 `history_length`。
- **`RayCastSensor`**：GPU 射线，用于地形扫描。粗糙地形任务常将网格射线挂在 pelvis 等机体坐标系；平地配置可从 scene 中移除该项。
- **`CameraSensor`**：RGB-D。高保真外观渲染仍非框架重点。

基类按仿真步缓存 `data`：同一步内多个奖励或观测项重复读取时只计算一次。IMU 等原生传感器常写在机器人 XML 中，观测通过 `builtin_sensor` 按前缀名读取，例如 `robot/imu_ang_vel`。

---

## 四、管理器层

### 4.1 Term 配置

管理器层的基本单元是**名称到配置对象的字典**。观测、奖励、终止、事件、课程、指标等项通常包含 `func`（可调用对象）与 `params`（关键字参数）。动作项与命令项使用专用配置类（如 `JointPositionActionCfg`、`UniformVelocityCommandCfg`），而非上述 `func` 模式。Manager 负责在仿真循环的相应位置调用各项、聚合输出并记录日志。

```python
rewards = {
    "alive": RewardTermCfg(func=mdp.is_alive, weight=1.0),
    "joint_torques": RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    ),
}
```

无状态计算使用函数 `func(env, **params)`。若需在初始化时缓存下标或在回合内保持状态，应实现为类：以 `(cfg, env)` 构造，调用签名与函数相同，并可实现 `reset(env_ids)`。

八个 manager 的职责如下。

| Manager | 职责 |
|---------|------|
| Observation | 组装观测组；噪声、裁剪、缩放、延迟、历史 |
| Action | 将策略输出切分并路由到执行器 |
| Reward | 加权求和；默认再乘 `step_dt` |
| Termination | 结束条件；区分失败与截断 |
| Event | 在 startup / reset / interval / step 触发；域随机化由此进入 |
| Command | 生成并重采样目标（速度、位姿、参考运动） |
| Curriculum | 按表现或训练步数调节难度 |
| Metrics | 记录不进入优化目标的诊断量 |

下文按「智能体接口 → 训练信号 → 对世界的干预」说明，不枚举全部内建函数。

### 4.2 观测

观测按 **group** 组织。组内各 term 按注册顺序在最后一维拼接，得到 `[num_envs, D]`。`enable_corruption` 控制整组是否施加噪声，从而使同一套 term 定义可用于带噪声的 actor 组与无噪声的 critic 组。

每步处理顺序为：

```
compute → noise → clip → scale → delay → history
```

延迟位于历史之前：历史堆叠的是已经延迟的读数。`history_length` 为 MLP 提供时间上下文（默认将时间维展平）；`delay_max_lag` 用环形缓冲模拟传感器时延。换算关系为 $\mathrm{lag}\approx\mathrm{latency}/\mathrm{step\_dt}$（观测延迟以**策略步**为单位）。缓冲区仅在启用时分配。

**非对称 actor-critic** 是速度跟踪等任务的常用写法：actor 组仅含真机可获得的量（带噪声 IMU、关节状态）；critic 组叠加特权信息（高度扫描、足端接触），并关闭 corruption。训练时价值网络读取 `obs["critic"]`，部署时策略仅读取 `obs["actor"]`。若真机没有基座线速度估计，模仿任务可从 actor 组删除 `base_lin_vel`、锚点位置等项，只需改写 `ObservationGroupCfg.terms`，不必另定义环境类。

内建观测包括基座线速度与角速度、投影重力、相对默认姿态的关节量、上一步动作、当前 command，以及具名传感器读数。自定义观测函数应返回 `[num_envs, D]`。

在 `flatten_history_dim=True` 且 `concatenate_terms=True` 时，mjlab 使用 **term-major** 顺序（先展平每个 term 的全部历史，再拼接 term）。从采用 time-major 顺序的框架迁移策略时，需要重排观测向量。

### 4.3 动作

Action manager 接收策略输出，按 term 切分，并映射到位置、速度或力矩等控制模式。共同参数包括：`entity_name`、以正则选择执行器的 `actuator_names`、`scale`、`offset`（关节类常用 `use_default_offset=True`，使动作 0 对应默认姿态），以及可选的 `clip`。

动作在**每一个物理子步**写入执行器目标，而不是每个策略步写入一次。观测延迟则以策略步计时。

除关节、腱与 site 努力外，**`DifferentialIKAction`** 将笛卡尔指令经阻尼最小二乘 IK 转换为关节位置目标，每个 decimation 子步求解一次，使末端在子步间连续跟踪。

Manager 保留最近三次动作，供 `last_action` 观测以及 `action_rate_l2` 等惩罚使用；reset 时清零，以避免回合边界处的信息泄漏。

### 4.4 奖励与终止

奖励项返回形状为 `[num_envs]` 的标量；负权重表示惩罚。默认 **`scale_rewards_by_dt=True`**：每项再乘以环境步长，使回合累计回报对仿真频率近似不变——同一任务在 50 Hz 与 200 Hz 下，每步贡献按比例缩小。日志中的 `Episode_Reward/<name>` 再除以回合时长，得到可跨实验比较的奖励率。

终止项返回布尔张量 `[num_envs]`。`time_out=True` 表示**截断**（Gym 的 `truncated`），否则为**失败**（`terminated`）。价值自举应越过截断、不越过失败。内建条件包括超时、姿态倾覆、根高度过低以及 NaN/Inf 检测。

`is_finite_horizon`（默认 `False`）进一步规定时间上限的语义：为 `False` 时，时限视为人为截断，智能体对时限之后的价值做自举；为 `True` 时，时限视为任务边界，对应终止信号且不再自举。

### 4.5 命令

命令项与其它 term 不同：**必须实现为类**，并继承 `CommandTerm`。它生成目标速度、参考轨迹或物体目标位姿，按 `resampling_time_range` 重采样，并在每次 reset 时无条件重采样。观测侧通过 `generated_commands` 按名称读入策略。

若环境没有 command，manager 空操作并返回空张量。内置任务提供：平面速度（`UniformVelocityCommand`）、操作举升目标（`LiftingCommand`），以及从 `.npz` 运动片段读入参考（`MotionCommand`）。

自定义 command 需实现 `_resample_command`、`_update_command`、`_update_metrics` 以及 `command` 属性。基类管理定时器。`_update_command` 在每步以 `env_ids=None`（表示全体环境）调用，并在 reset 之后仅对刚重置的环境再调用一次。若更新会推进内部状态（例如参考运动的帧下标），必须尊重 `env_ids`，否则重置部分环境会同时推进其余环境的状态。

### 4.6 事件与域随机化

Event 是在指定生命周期节点修改仿真的统一入口。`mode` 对应四个时间尺度：

| mode | 触发时机 | 典型用途 |
|------|----------|----------|
| `startup` | 环境初始化后一次，全体环境同时 | 训练全程固定、但环境间不同的参数（质量、转子惯量） |
| `reset` | 每个被重置的环境 | 恢复默认姿态、回合级随机化 |
| `interval` | 与回合边界无关的定时器 | 过程中的速度扰动、参数缓慢漂移 |
| `step` | 每步 | 自行管理持续时间的冲量等；计算应尽量轻量 |

默认配置包含 `reset_scene_to_default`。域随机化函数位于 `mjlab.envs.mdp.dr`，通过 event term 调用（例如 `dr.geom_friction`）。工厂函数先注册通用事件，机器人配置再填写 `geom_names`、`body_names`（足部碰撞 geom、质心所在连杆）以及 interval 扰动。需要展开为 per-world 的模型字段在初始化阶段完成，并重建 CUDA graph。

### 4.7 课程与指标

Curriculum 在每次 reset 时调用：根据行驶距离、训练步数等调节地形行、速度指令范围、奖励权重或终止阈值。程序化地形为 `num_rows × num_cols` 网格：列表示地形类型，行表示难度。`terrain_levels_vel` 根据本回合行驶距离升行或降行；到达最高行后随机重新分配，以保持各难度均有样本。

Metrics 的计算方式与奖励类似，但**没有权重、不乘 `dt`**，仅用于诊断（跟踪误差、接触力、能耗等）。聚合可选 mean、last、max、sum，写入 `Episode_Metrics/`。

---

## 五、生命周期与时间尺度

### 5.1 四个阶段

环境实例依次经过 Build、Initialize、Reset 与 Step。

**Build。** Scene 拼接 MJCF 并在 CPU 上编译 `MjModel`；Simulation 上传至 GPU，分配含 N 个世界的 `MjData`；捕获 `step`、`forward`、`reset`、`sense` 的 CUDA graph。

**Initialize。** 根据 term 字典构造各 manager；将正则解析为下标；分配观测历史与延迟缓冲；按域随机化需要展开模型字段并重建 CUDA graph；触发一次 startup 事件。

**Reset。** 在训练开始时，以及环境终止或超时时调用。reset 事件将场景恢复为初态（可含随机化）；command 重采样；观测历史清空。

**Step。** 官方顺序如下：

```
action_manager.process_action(action)
for _ in range(decimation):
    action_manager.apply_action()
    sim.step()
    scene.update()
termination_manager.compute()
reward_manager.compute()
metrics_manager.compute()
event_manager.apply(mode="step")
event_manager.apply(mode="interval")
[reset terminated envs]
sim.forward()
command_manager.compute()   # 刚 reset 的环境 dt=0
sim.sense()
observation_manager.compute()
```

step 与 interval 事件在 reset **之前**作用于终止前的状态；随后才重置、调用 `forward`、更新 command、感测并组装下一步观测。

### 5.2 三个时间参数

| 参数 | 含义 |
|------|------|
| `sim.mujoco.timestep` | 物理积分步长，默认 0.002 s（500 Hz） |
| `decimation` | 每个策略步内的物理步数 |
| `episode_length_s` | 回合时长（秒） |

策略频率为 $1/(\mathrm{timestep}\times\mathrm{decimation})$。速度任务常用 `timestep=0.005`、`decimation=4`，即 200 Hz 物理、50 Hz 策略；`episode_length_s=20` 时每回合为 1000 个策略步。运行时可读取 `env.physics_dt`、`env.step_dt`、`env.max_episode_length`。

在 `scale_rewards_by_dt=True` 时改变频率，奖励量级大致保持不变。物理稳定性仍取决于 `timestep` 与求解器设置，不能仅靠增大 `decimation` 维持。

---

## 六、训练与内置任务

### 6.1 与 RSL-RL 的衔接

训练侧默认使用 RSL-RL，衔接包括三部分。

1. **任务注册表。** 每个任务对应一对配置：`ManagerBasedRlEnvCfg` 与 `RslRlOnPolicyRunnerCfg`。`register_mjlab_task` 用字符串 id 绑定二者，并另存 `play_env_cfg`（关闭训练用随机化、延长回合）供评估。官方内置任务多用 `Mjlab-{Category}-{Terrain}-{Robot}`。可选参数 `runner_cls` 默认为 `MjlabOnPolicyRunner`；需要在保存 checkpoint 时导出 ONNX 时再换成自定义子类（见第九部分）。
2. **`RslRlVecEnvWrapper`。** 将观测字典转换为 RSL-RL 的 TensorDict；合并 `terminated` 与 `truncated` 为 `dones`，并将超时写入 `extras` 以便正确自举；可按配置裁剪动作。构造时调用 `env.reset()`，因为 RSL-RL 在收集 rollout 前不自行 reset。自备 `train.py` 时，仍是先构造 `ManagerBasedRlEnv`，再包装该层，然后交给 runner。
3. **配置 dataclass。** `RslRlOnPolicyRunnerCfg` 是 mjlab 对 RSL-RL `OnPolicyRunner` 的封装，由 `load_rl_cfg` 按 task id 取出，CLI 前缀为 `--agent`。它不描述物理或 MDP；并行环境数属于 `env.scene.num_envs`（`--num-envs`）。字段很多，下列只说明读配置时必须分清的三块；其余以 `train <task> --help` 与官方 *Training with RSL-RL* 为准。

**网络（`actor` / `critic`，类型为 `RslRlModelCfg`）。** 输入维由观测维决定，输出维由动作维或标量价值决定，均不在此填写。`hidden_dims` 给出全连接隐层的宽度序列：`(512, 256, 128)` 表示三层，宽分别为 512、256、128。`activation` 是这些隐层之间的逐元非线性（G1 速度任务常用 `"elu"`），一般不加在输出头上。actor 与 critic 可取不同宽度。

`obs_normalization=True` 作用于**网络输入**：对各维观测用训练中累计的均值与标准差做 $(o-\mu)/\sigma$（滑动统计），使量纲不同的通道尺度接近。它不改变仿真状态，也不把动作变成标准正态。

连续控制中，actor 的 `distribution_cfg` 指定**动作条件分布**，而非观测分布。`class_name="GaussianDistribution"` 表示 $\pi(a\mid o)=\mathcal{N}(\mu(o),\sigma)$：MLP 给出均值 $\mu$，标准差由该配置给出。训练时从该分布采样动作；评估时常取均值。`init_std` 为 $\sigma$ 的初值；`std_type="scalar"` 表示各动作维共用一个标准差。critic 只有价值头，不含此项。

**算法（`algorithm`，类型为 `RslRlPpoAlgorithmCfg`）。** 控制一次 iteration 内如何用已采集的 batch 更新网络。常用项如下。

| 字段 | 作用 |
|------|------|
| `learning_rate`、`schedule` | 优化步长；`adaptive` 按 KL 相对 `desired_kl` 调节 |
| `clip_param` | PPO 概率比裁剪（常取 0.2） |
| `entropy_coef` | 策略熵正则 |
| `value_loss_coef`、`use_clipped_value_loss` | 价值损失权重及是否对 $V$ 做裁剪 |
| `num_learning_epochs`、`num_mini_batches` | 对同一批数据扫描的轮数与 minibatch 划分 |
| `gamma`、`lam` | 折扣与 GAE-$\lambda$ |
| `max_grad_norm` | 梯度范数上限 |

**调度。** `num_steps_per_env` 为每次 iteration 每个环境采集的策略步数（G1 速度任务常用 24）；`max_iterations` 为更新次数；`save_interval` 为 checkpoint 间隔；`experiment_name` 决定日志目录 `logs/rsl_rl/{experiment_name}/`。续训、W&B 上传等字段多保持默认，需要时用 `--agent.resume` 等覆盖。

官方 Unitree G1 速度任务的典型写法如下（字段含义见上，不逐项重复）。

```python
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="g1_velocity",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=30_000,
    )
```

### 6.2 命令行

```bash
uv run train Mjlab-Velocity-Flat-Unitree-G1 --num-envs 4096

uv run train Mjlab-Velocity-Flat-Unitree-G1 \
    --num-envs 4096 \
    --agent.max-iterations 10000 \
    --agent.algorithm.learning-rate 3e-4 \
    --env.decimation 2
```

`--num-envs` 为顶层便捷参数，等价于覆盖 `env.scene.num_envs`。环境与算法的配置树可通过 tyro 用点分路径覆盖。CLI 使用连字符（`--num-envs`）而非下划线；布尔标志须显式给出 `True` 或 `False`（例如 `--agent.resume True`），以便与 W&B sweep 兼容。多 GPU 使用 `--gpu-ids`；官方 CLI 因 tyro 配置而采用 Python 字面量，例如 `--gpu-ids "[0, 1]"`。

回放：

```bash
uv run play Mjlab-Velocity-Flat-Unitree-G1 --wandb-run-path your-entity/mjlab/run-id
uv run play Mjlab-Velocity-Flat-Unitree-G1 --checkpoint-file path/to/model_1000.pt
```

`--agent` 可选 `trained`、`zero`、`random`。新任务宜先以 zero 与 random 检查奖励符号、终止条件与动作缩放。查看器可选原生 MuJoCo（`NativeMujocoViewer`）或浏览器端 Viser（`ViserPlayViewer`）。

日志目录形如 `logs/rsl_rl/{experiment_name}/{timestamp}/`，包含策略 checkpoint 以及 `params/env.yaml`、`params/agent.yaml`。带 ONNX 导出的 runner 还会在同目录写入 `policy.onnx`（及元数据）。

### 6.3 参考任务

mjlab 提供三类参考任务：

- **速度跟踪**：跟踪平面速度指令，例如 `Mjlab-Velocity-Flat-Unitree-G1`。
- **运动模仿**：`MotionCommand` 从 `.npz` 读入参考；官方使用运动资产 registry。
- **操作**：例如举升类 command。

Cartpole 教程给出最小可运行 MDP。文档 *Research* 页列出 PAL、Upkie、灵巧手等外部项目。建议将 `mjlab` 作为依赖引入，任务与 MJCF 放在应用仓库的 `src/`，而不是 fork mjlab 本体。若 MDP 与官方一致，只需 patch `make_velocity_env_cfg()`（ANYmal C 示例）；若需修改奖励、观测或导出逻辑，可将工厂与 `mdp/` 放入应用仓库，对 `mjlab.envs.mdp` 执行 `import *` 后再叠加本地 term。底层仍是同一套 `ManagerBasedRlEnvCfg`。完整的多机型示例见第九部分。

### 6.4 安装

一般用法：`uv add mjlab`，然后 `uv run demo`。开发 mjlab 本身：克隆仓库后执行 `uv sync`。此外支持 pip/conda、Docker（`ghcr.io/mujocolab/mjlab`）以及无需本地安装的 `uvx` 与 Colab。

---

## 七、边界、常见误用与延伸阅读

### 7.1 适用范围

较适用的情形包括：已有 MJCF 或 Menagerie 模型；需要数千并行环境的 on-policy RL；希望奖励、观测与随机化可分解、可复用；需要直接检查 MuJoCo 量。

明确不作为目标的包括：照片级仿真、USD 资产管线、跨 PhysX/Newton 的同一套环境代码、将可微物理作为一等能力、将高保真视觉作为默认观测，以及实机总线与 SDK。

### 7.2 常见误用

- **固定基座须由 reset 事件写入 `env_origins`。** 否则各并行环境重叠于原点。
- **截断与失败不宜混用。** 超时应设置 `time_out=True`；倾覆等失败条件不应标为 truncation。
- **奖励默认乘以 `dt`。** 从「每步原始标量」的实现迁移时，量级相差一个 `step_dt`。
- **动作按物理子步施加；观测延迟按策略步计时。** 建模时延时不应混用两种时钟。
- **各世界目前共享运动学树。** 物体形状随世界变化应使用 `VariantEntityCfg`，而不是仅修改一份 XML。
- **从 Isaac Lab 迁移主要是配置形态。** Manager 职责大体对应；场景由 USD 改为 `MjSpec`。

### 7.3 建议的阅读顺序

1. [Why mjlab?](https://mujocolab.github.io/mjlab/main/source/motivation.html) 与 [Architecture Overview](https://mujocolab.github.io/mjlab/main/source/architecture_overview.html)
2. [Environment Configuration](https://mujocolab.github.io/mjlab/main/source/environment_config.html)
3. 按任务需要阅读 Entity、Scene、Actuators、Sensors
4. 按 MDP 需要阅读 Observations、Actions、Rewards、Events、Commands
5. [Training with RSL-RL](https://mujocolab.github.io/mjlab/main/source/training/rsl_rl.html)
6. 实现：[Cartpole 教程](https://mujocolab.github.io/mjlab/main/source/tutorials/cartpole.html)；仅换机型见 [anymal_c_velocity](https://github.com/mujocolab/anymal_c_velocity)；多机型与部署见第九部分及 [unitree_rl_mjlab](https://github.com/tangyx96/unitree_rl_mjlab)
7. 若来自 Isaac Lab：[Migrating from Isaac Lab](https://mujocolab.github.io/mjlab/main/source/migration_isaac_lab.html)

论文：Zakka, Liao, Yi, Le Lay, Sreenath, Abbeel, *mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning*, [arXiv:2601.22074](https://arxiv.org/abs/2601.22074), 2026。使用 RSL-RL 时应同时引用其论文。

---

## 八、源码结构与实现路径

实现阶段的主要问题通常不是单个 Manager 的字段名，而是文件组织、CLI 如何发现任务，以及 XML、配置与注册三者如何衔接。逐步教程见 [Cartpole](https://mujocolab.github.io/mjlab/main/source/tutorials/cartpole.html)；仅更换机型见 [anymal_c_velocity](https://github.com/mujocolab/anymal_c_velocity)。依赖 mjlab 做多机型与实机部署的组织方式见第九部分。

### 8.1 三种扩展路径

| 路径 | 适用情形 | 参照 |
|------|----------|------|
| **A. 从零定义任务** | 动力学或 MDP 本身是新的 | `src/mjlab/tasks/cartpole/` |
| **B. 仅更换机器人** | 官方速度 / 跟踪 / 操作的 MDP 可直接使用 | `mjlab` 的 `tasks/velocity/config/g1/`，或 anymal 示例仓库 |
| **C. 应用仓库依赖 mjlab** | 需修改奖励与观测、增加机型、导出 ONNX、接入实机 | 第九部分 |

路径 A、B 将 mjlab 作为完整框架使用；路径 C 将其作为库：不 fork `mjlab` 源码，而以 `pip` 或 `setup.py` 固定版本，由本地 `train.py` 调用 `load_env_cfg`、`ManagerBasedRlEnv`、`RslRlVecEnvWrapper`。面向实机部署的课题更接近路径 C；路径 B 是其简化形式。

### 8.2 仓库结构

安装后的包根目录为 `src/mjlab/`。

```
src/mjlab/
├── entity/          # Entity、EntityCfg、EntityData
├── actuator/        # 内建 / 显式 / XML 执行器
├── sensor/          # Builtin / Contact / RayCast / Camera
├── scene/           # Scene、SceneCfg，MJCF 拼接与编译
├── sim/             # Simulation、MuJoCo Warp、timestep
├── terrains/        # 平面与程序化地形
├── envs/
│   ├── manager_based_rl_env.py   # 生命周期：reset / step
│   └── mdp/                      # 跨任务共用的 obs / reward / event / dr
├── managers/        # 八个 manager 与 *TermCfg
├── tasks/
│   ├── registry.py  # register_mjlab_task / list_tasks
│   ├── cartpole/    # 最小完整任务（XML + 单文件配置）
│   ├── velocity/    # 工厂配置 + 按机器人分的 config/
│   ├── tracking/
│   └── manipulation/
├── rl/              # RSL-RL runner、VecEnv wrapper、超参数 dataclass
├── asset_zoo/       # 内置机器人 EntityCfg（G1、Go1 等）
├── scripts/         # train / play / demo / list-envs / export-scene
└── viewer/          # native MuJoCo 与 Viser
```

修改时的对应关系：

- 物理与资产 → `entity` / `actuator` / `asset_zoo`，或应用仓库的机器人配置
- MDP 项 → `mjlab.envs.mdp`，或应用仓库本地 `mdp/`（对前者 `import *` 再叠加）
- 步进循环 → `envs/manager_based_rl_env.py`（应用仓库一般不应修改）
- 训练入口 → `tasks/registry.py`；应用仓库可自备 `scripts/train.py`，但仍调用 `load_env_cfg`

`pyproject.toml` 中的控制台命令：

| 命令 | 入口 |
|------|------|
| `train` | `mjlab.scripts.train:main` |
| `play` | `mjlab.scripts.play:main` |
| `demo` | `mjlab.scripts.demo:main` |
| `list-envs` | `mjlab.scripts.list_envs:main` |
| `export-scene` | `mjlab.scripts.export_scene:main` |

### 8.3 任务发现

`train <task_id>` 并不扫描目录推断任务名。登记一律写入 mjlab 的全局表；发现方式可以不同：

1. **entry point。** import `mjlab` 时，`_import_registered_packages()` 加载组 **`mjlab.tasks`**，从而导入应用包。ANYmal 示例采用该方式。
2. **显式 import。** 训练脚本执行 `import mjlab.tasks` 与 `import src.tasks`。后者通过 `mjlab.utils.lab_api.tasks.importer.import_packages` 递归导入含子包的目录；各机器人配置包的 `__init__.py` 调用 `register_mjlab_task`。中间目录必须包含 `__init__.py`，且应将纯工具包（如 `.mdp`）列入黑名单，以免将 term 模块当作任务包导入。
3. `train` / `play` 按 task id 调用 `load_env_cfg`、`load_rl_cfg`、`load_runner_cls`，构造 `ManagerBasedRlEnv`，再包装 `RslRlVecEnvWrapper`。

未调用 `register_mjlab_task` 的任务不会出现在 CLI 中。可用下列命令检查：

```bash
uv run list-envs
python scripts/list_envs.py
```

注册写在配置包的 `__init__.py` 中：

```python
from mjlab.tasks.registry import register_mjlab_task
from .env_cfgs import my_robot_flat_env_cfg
from .rl_cfg import my_robot_ppo_runner_cfg

register_mjlab_task(
    task_id="MyRobot-Flat",
    env_cfg=my_robot_flat_env_cfg(),
    play_env_cfg=my_robot_flat_env_cfg(play=True),
    rl_cfg=my_robot_ppo_runner_cfg(),
)
```

`play=True` 应关闭训练用随机化（扰动、观测 corruption）并延长回合。官方 id 多用 `Mjlab-` 前缀；应用仓库使用自有前缀即可，但须避免与已注册 id 冲突。需要自定义 `runner_cls` 时见第九部分。

### 8.4 路径 A：最小环境（Cartpole）

官方教程将任务置于两个文件：`cartpole.xml` 与 `cartpole_env_cfg.py`。组装顺序为仿真层 → 管理器层 → 注册，与第二至四部分对应。

**（1）XML。** 应能在独立 MuJoCo 中打开。cartpole 包含滑轨、铰链与一个 `<motor>`；在 `gear=10` 且 `ctrlrange="-1 1"` 时，策略输出经内部钳位后的最大力为 10 N。

**（2）Entity。** `spec_fn` 读取 XML；XML 中已有 actuator 时使用 `XmlActuatorCfg`；`init_state` 区分任务变体（摆起：铰链为 $\pi$；平衡：铰链为 0）。

```python
from pathlib import Path
import mujoco
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityCfg, EntityArticulationInfoCfg

_XML = Path(__file__).parent / "cartpole.xml"

def _get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(str(_XML))

def _get_cartpole_cfg(swing_up: bool = False) -> EntityCfg:
    return EntityCfg(
        spec_fn=_get_spec,
        articulation=EntityArticulationInfoCfg(
            actuators=(XmlActuatorCfg(target_names_expr=("slider",)),),
        ),
        init_state=EntityCfg.InitialStateCfg(
            joint_pos={"slider": 0.0, "hinge_1": math.pi if swing_up else 0.0},
            joint_vel={".*": 0.0},
        ),
    )
```

**（3）观测。** 用 `SceneEntityCfg` 限定关节。张量均带 batch 维 `[num_envs, ...]`。与 RSL-RL 对接时通常需要 `"actor"` 与 `"critic"` 两组；最小任务可共用同一套 term。自定义函数的签名与内建项相同：

```python
def pole_angle_cos_sin(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)
```

铰链角宜用余弦与正弦而非原始角：MuJoCo 的无限制铰链不回绕，旋转多圈后原始值会持续增大。

**（4）动作、奖励、终止、事件。** cartpole 使用 `JointEffortActionCfg` 写入 XML motor；终止仅含 `time_out=True` 的超时；reset 使用 `reset_joints_by_offset` 在 `init_state` 附近加入噪声。无碰撞时可在 `MujocoCfg(disableflags=("contact",))` 中关闭接触计算。

**（5）汇总为 `ManagerBasedRlEnvCfg`。** `scene.entities` 的键（如 `"cartpole"`）必须与所有 `SceneEntityCfg` / `entity_name` 一致。`num_envs=1` 仅为配置默认值，训练时用 `--num-envs` 或 `--env.scene.num-envs` 覆盖。

**（6）编写 `RslRlOnPolicyRunnerCfg` 并注册。** 字段含义见 6.1。Cartpole 等小规模任务可将 `hidden_dims` 取为 `(64, 64)`。然后：

```bash
uv run play Mjlab-Cartpole-Swingup --agent zero
uv run train Mjlab-Cartpole-Swingup --env.scene.num-envs 4096
```

奖励公式与噪声设置见官方 Cartpole 教程。

### 8.5 路径 B：工厂函数与机器人 patch

速度跟踪与运动模仿将任务逻辑与硬件配置分开。官方结构为：

```
tasks/velocity/
├── velocity_env_cfg.py      # make_velocity_env_cfg()
├── mdp/
├── rl/
└── config/g1/               # 调用工厂，再 patch
```

`env_cfgs.py` 的典型流程是 `cfg = make_velocity_env_cfg()`，随后只修改与硬件相关的字段。G1 速度任务中实际改写的对象包括：

- `cfg.scene.entities["robot"]` ← `EntityCfg`（`asset_zoo` 或本地工厂）
- `cfg.scene.sensors` ← `RayCastSensorCfg` 的 `frame`、`ContactSensorCfg`（足–地、自碰撞）
- `cfg.actions["joint_pos"]` ← `JointPositionActionCfg.scale`
- `cfg.commands["twist"]` ← `UniformVelocityCommandCfg`（模仿任务则为 `MotionCommandCfg` 的锚点与 body 列表）
- `cfg.events["foot_friction"]` / `base_com` 中的 `SceneEntityCfg` 正则
- `cfg.rewards["pose"]` 等按关节组给出的 `std`
- `play=True` 时的 `episode_length_s`，以及关闭 `push_robot` 与 actor corruption

路径 B 到此为止，工厂仍来自 `mjlab.tasks.velocity`。路径 C 在应用仓库保留同名工厂以便改 MDP，约定见第九部分。

关节名、PD、`armature` 与 `CollisionCfg` 应集中在机型常量模块中。`spec_fn` 用 `MjSpec.from_file` 加载 XML，网格经 `update_assets` 写入 `spec.assets`。传感器名称与观测中的 `sensor_name` 必须带实体前缀。

路径 B 可通过 uv 的 entry point 注册任务。

### 8.6 自定义 term 的约定

实现自定义项前，宜对照 `mjlab.envs.mdp` 中同类函数的签名。

| 种类 | 调用约定 | 返回 |
|------|----------|------|
| 观测 | `func(env, **params)` | `[num_envs, D]` |
| 奖励 | 同上 | `[num_envs]` |
| 终止 | 同上 | `bool[num_envs]` |
| 事件 | `func(env, env_ids, **params)` | 无返回值，就地修改仿真 |
| 课程 | `func(env, env_ids, **params)` | 标量、dict 或 `None`（供日志） |
| 命令 | 必须是 `CommandTerm` 子类 | `command` 属性为当前目标张量 |

`params` 中的 `SceneEntityCfg` 在 manager 初始化时解析正则；步进路径上使用 `joint_ids` 等整数切片。需要缓存或 `reset(env_ids)` 时应实现为类。读状态优先使用 `env.scene["robot"].data` 与 `env.scene["sensor_name"].data`，而不是直接索引全局 `qpos`。

### 8.7 验收顺序

建议按下列顺序检查，而不必一开始就使用数千并行环境。

1. XML 能在 MuJoCo viewer 中按预期运动；网格路径、关节限位与 actuator 名称正确。
2. `list-envs` 或应用仓库的列举脚本列出目标 `task_id`。
3. `play <id> --agent zero`：在 `use_default_offset=True` 时，零动作应对应默认姿态；固定基座物体应出现在各 `env_origins`。
4. `--agent random`：应产生明显运动，且仿真不立即出现 NaN（必要时启用 `--enable-nan-guard`）。
5. `export-scene` 检查拼接后的前缀名（例如 `robot/joint0`）。
6. 以较小的 `--env.scene.num-envs`（例如 16）运行若干 iteration，确认 `Episode_Reward/` 各项符号符合设计。
7. 再提高并行规模。上真机的步骤见第九部分。

若观测或奖励与关节对应关系不符，可打印 `SceneEntityCfg` 解析后的 `joint_names` / `joint_ids`，或在 viewer 中核对接接触传感器的 primary 正则。

### 8.8 源码对照

| 问题 | 文件 |
|------|------|
| manager 在一步中的调用顺序 | `mjlab/envs/manager_based_rl_env.py` |
| 内建观测 / 奖励 / 事件 / dr | `mjlab/envs/mdp/` |
| 速度跟踪的观测与 command | 官方或应用仓库的 `velocity_env_cfg.py` |
| 机器人如何 patch 进任务 | `config/<robot>/env_cfgs.py` |
| Entity、PD、碰撞 | `asset_zoo` 或应用仓库的机型常量 |
| 训练 CLI 与配置覆盖 | `mjlab/scripts/train.py` 或应用仓库 `scripts/train.py` |
| 任务如何进入注册表 | `register_mjlab_task`；entry point 或 `import_packages` |
| 策略如何离开仿真 | `mjlab.rl.exporter_utils`；应用仓库的 runner `save` |

场景对应「世界中有什么」，manager 字典对应「智能体与世界如何交互」，`register_mjlab_task` 对应「CLI 如何索引该任务」。

---

## 九、应用示例：unitree_rl_mjlab

[unitree_rl_mjlab](https://github.com/tangyx96/unitree_rl_mjlab) 将 mjlab 作为库（例如 `mjlab==1.2.0`），在本地维护机型资产、任务配置与训练脚本，并把导出的策略接入独立的实机进程。职责划分是：mjlab 提供 GPU 仿真与 MDP 编排；应用仓库负责机型特化、任务注册，以及 Train → Play → Sim2Real 中的部署环节。

### 9.1 流程与边界

完整流程为 **Train → Play → Sim2Real**。

- **Train / Play** 构造 `ManagerBasedRlEnv`、调用 RSL-RL，并用 `play` 回放。应用代码中凡 `from mjlab...` 的部分属于这两步。
- **Sim2Real** 将 runner 导出的 `policy.onnx` 交给 C++ 部署与 `unitree_sdk2`。`deploy/` 已脱离 mjlab：不再构造环境类，也不走 manager 步进。

因此：改奖励、观测、域随机化，仍在仿真侧的 manager 字典中完成；改电机通信、DDS 或板端控制，只动 `deploy/`。

### 9.2 目录与注册

```
src/
├── assets/robots/unitree_go2/   # XML、get_go2_robot_cfg()
└── tasks/
    ├── __init__.py              # import_packages → register_mjlab_task
    ├── velocity/
    │   ├── velocity_env_cfg.py
    │   ├── mdp/
    │   ├── rl/runner.py         # 保存时 export_policy_to_onnx
    │   └── config/go2/
    └── tracking/
scripts/train.py
scripts/play.py
deploy/
```

`src/assets` 对应第三部分的仿真层；`src/tasks` 对应第四部分的 manager 字典，不修改 `ManagerBasedRlEnv` 的步进顺序。`mdp/__init__.py` 先执行 `from mjlab.envs.mdp import *`，再导出本地 rewards、observations、commands。动作仍使用 `JointPositionActionCfg`，随机化仍使用 `mjlab.envs.mdp.dr`，PPO 仍使用 `mjlab.rl` 中的 dataclass。

任务发现采用第八部分的**显式 import**（可以没有 `mjlab.tasks` entry point）。注册示例：

```python
from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner
from .env_cfgs import unitree_g1_flat_env_cfg
from .rl_cfg import unitree_g1_ppo_runner_cfg

register_mjlab_task(
    task_id="Unitree-G1-Flat",
    env_cfg=unitree_g1_flat_env_cfg(),
    play_env_cfg=unitree_g1_flat_env_cfg(play=True),
    rl_cfg=unitree_g1_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
```

id 使用 `Unitree-` 等前缀，避免与官方 `Mjlab-` 冲突。该仓库在同一套速度跟踪 manager 上注册 `Unitree-Go2-Flat`、`Unitree-G1-Flat`、`Unitree-H1_2-Flat` 等；模仿任务常用 `--motion_file` 指向本地 npz（CSV 需先按控制频率重采样）。

`runner_cls` 换成在 `save` 时导出 ONNX 的子类（速度跟踪与模仿中的 `*OnPolicyRunner`）。checkpoint 目录中的 `policy.onnx` 及元数据供部署读取关节顺序与 PD 增益。

### 9.3 训练入口

`scripts/train.py` 先 `import src.tasks`，再 `load_env_cfg`、构造 `ManagerBasedRlEnv`、包装 `RslRlVecEnvWrapper`、交给 runner。并行规模常写成 `--env.scene.num-envs=4096`（与官方 `--num-envs` 等价）。`play.py` 直接构造 `NativeMujocoViewer` 或 `ViserPlayViewer`，不另实现可视化后端。

确认 `policy.onnx` 已写出后，再进入 `deploy/`；此后进程不再调用 mjlab。

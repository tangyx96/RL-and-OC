# HIL-SERL 算法笔记

下文对应 Luo, Xu, Wu, Levine, *Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning*（[arXiv:2410.21845](https://arxiv.org/pdf/2410.21845) v3，2025-03-20）。正文与补充材料的数字均按该版本。实现见 [rail-berkeley/hil-serl](https://github.com/rail-berkeley/hil-serl)。

训练入口为 `examples/train_rlpd.py`，分为 `--actor` 与 `--learner` 两个进程。数据经 agentlace 的队列 `actor_env` 与 `actor_env_intvn` 进入 learner。连续动作的损失位于 `serl_launcher/agents/continuous/sac.py` 的 `critic_loss_fn`、`policy_loss_fn` 与 `temperature_loss_fn`。双臂夹爪的离散 critic 位于 `sac_hybrid_dual.py` 的 `grasp_critic_loss_fn` 与 `sample_actions`。集成规模、LayerNorm 与熵备份开关由 `serl_launcher/utils/launcher.py` 的 `make_sac_pixel_agent` 设定。相对坐标在 `serl_robot_infra/franka_env/envs/relative_env.py` 的 `RelativeFrame`，介入在 `envs/wrappers.py` 的 `SpacemouseIntervention`。

soft Bellman、双 $Q$ 与温度对偶见 [`sac_notes.md`](sac_notes.md) 第五至七节。高维 off-policy 中的范数约束见 [`flashsac_notes.md`](flashsac_notes.md)。扩散策略上的真机后训练见 [`rl100_notes.md`](rl100_notes.md)。SAC 的推导沿用上述笔记。本文只记录 HIL-SERL 的改动：介入数据写入哪个缓冲、批次如何对半抽样、夹爪为何单独构成 MDP、观测与底层控制器如何把策略接到机器人，以及表中 $100\%$ 所对应的评估协议。

---

## 一、算法定位

HIL-SERL 是一套真机视觉操作的训练系统。策略是以图像和本体感觉为条件的高斯策略，优化目标为稀疏的任务成功。算法核为 RLPD（Ball et al., 2023）上的 SAC：每个批次从人提供的数据与在线数据中各抽一半，再按 SAC 更新 $Q$ 与 $\pi$。人参与两个阶段。训练前以遥操作采集约 $20$–$30$ 条演示；训练中用 SpaceMouse 在策略偏离时介入若干步。介入步写入演示缓冲与在线缓冲，策略自身产生的转移只写入在线缓冲。

样本效率来自若干系统选择。视觉骨干为冻结的 ImageNet ResNet-10。奖励由离线训练的二分类器给出，成功为 $1$，其余为 $0$。接触任务中，策略输出末端坐标系下的 $6$ 维速度，交由 $1000\,\mathrm{Hz}$ 的阻抗控制器执行，并限制参考位姿误差，从而约束接触力。动态任务输出末端坐标系下的前馈力旋量。夹爪由单独的 DQN 控制。

与 SERL（Luo et al., ICRA 2024）相比，演示缓冲之外增加了在线修正，任务范围扩展到双臂、接触装配和甩动等动态动作。与 HG-DAgger 相比，人机接口相同，修正数据用于带奖励的 off-policy 更新，监督克隆仅作对照。与本仓库中的 RL-100 相比，动作头为低维高斯加离散夹爪，人在训练中直接改写当前动作；RL-100 的动作头为扩散链，人主要提供成功与否的标签。

| | 行为克隆 / HG-DAgger | SERL | HIL-SERL | RL-100 |
|--|----------------------|------|----------|--------|
| 策略类 | 高斯或扩散，拟合人的动作 | SAC | SAC，另加夹爪 DQN | 扩散策略上的 PPO |
| 人的角色 | 演示；HG-DAgger 中途介入并做监督学习 | 仅演示 | 演示加中途介入；介入动作进入两个缓冲 | 稀疏成功标签，并遵守动作限制 |
| 批次 | 固定数据集 | 演示与在线数据 | 每个批次各一半 | 演示保留；展开轨迹并入后重新做行为克隆 |
| 奖励 | 无环境奖励 | 分类器或任务奖励 | 稀疏二值；抓取任务另有夹爪惩罚 | 稀疏成功；离线用 $Q-V$ |
| 真机时间 | 采集后离线训练 | 较短时域的插入类任务 | 多数任务 $1$–$2.5$ 小时，同步带 $6$ 小时 | 离线与在线采集同量级，约 $1.5$ 小时至十余小时 |

默认配置中的智能体名称为 `"drq"`。`train_rlpd.py` 实际构造的是上述 SAC，或带夹爪 critic 的混合 SAC。

---

## 二、MDP 与两份损失

环境为 $\mathcal{M}=\{\mathcal{S},\mathcal{A},\rho,\mathcal{P},r,\gamma\}$。$s$ 由图像与本体感觉组成，$a$ 为末端速度或力旋量，$\rho$ 为复位后的初始分布，$\mathcal{P}$ 未知。策略是神经网络参数化的高斯 $\pi(a\mid s)$。目标为

$$
\mathbb{E}\Big[\sum_{t=0}^{H}\gamma^t r(s_t,a_t)\Big].
$$

奖励的取值见第六节。单阶段任务在成功步取 $r=1$ 并结束回合，其余步为 $0$。记成功步为 $t_\star$，该轨迹的回报为 $\gamma^{t_\star}$，未成功则为 $0$，目标即

$$
\mathbb{E}\big[\gamma^{t_\star}\mathbf{1}_{\mathrm{成功}}\big].
$$

$\gamma<1$ 时，成功越早，回报越大，周期时间进入这个期望。多阶段任务的回报是各阶段奖励的折扣和。Jenga 把奖励放在每个回合固定的最后一步，成功的早晚不改变这个时刻。采用奖励分类器的单阶段任务中，上式的指示由该分类器给出。

论文将更新写成 RLPD 的两步。记演示缓冲与在线缓冲的转移分布为 $\mu_{\mathrm{demo}}$、$\mu_{\mathrm{online}}$，批次的经验分布为

$$
\mu=\frac12\mu_{\mathrm{demo}}+\frac12\mu_{\mathrm{online}}.
$$

$$
\begin{aligned}
\mathcal{L}_Q(\phi)
&=
\mathbb{E}_{(s,a,s')\sim\mu}
\Big[
\big(
Q_\phi(s,a)
-
\big(
r(s,a)+\gamma\,\mathbb{E}_{a'\sim\pi_\theta}[\bar Q(s',a')]
\big)
\big)^2
\Big],
\\
\mathcal{L}_\pi(\theta)
&=
-\mathbb{E}_{s\sim\mu}
\Big[
\mathbb{E}_{a\sim\pi_\theta}[Q_\phi(s,a)]
+
\alpha\,\mathcal{H}\big(\pi_\theta(\cdot\mid s)\big)
\Big].
\end{aligned}
$$

$\bar Q$ 为目标网络。$\mathcal{L}_Q$ 里的转移来自缓冲，$a'\sim\pi_\theta(\cdot\mid s')$ 由当前策略重新采样。$\mathcal{L}_\pi$ 的状态来自 $\mu$，内层动作是 $\pi_\theta$ 的新样本。人执行过的动作进入 $Q_\phi(s,a)$ 的回归，actor 再在这些状态上沿着 $Q$ 调整 $\pi$。损失中没有行为克隆项，也没有指向演示动作的正则。$\alpha$ 按 Haarnoja et al. (2018) 自适应调节。

介入转移写入两个缓冲，在 $\mu$ 下的权重高于只存在于在线缓冲的策略转移。$\mu_{\mathrm{demo}}$ 随介入追加纳入这些新转移，$\mu_{\mathrm{online}}$ 中亦有一份。缓冲的写入见第三节。

发布代码不把熵写入 $Q$ 的备份。`make_sac_pixel_agent` 设 `backup_entropy=False`。`critic_loss_fn` 的目标为

$$
y = r + \gamma\, m\, \min_i Q_{\bar\theta_i}(s', a'),
\qquad a'\sim\pi(\cdot\mid s'),
$$

其中 $m=1-\mathrm{done}$。成功步终止时该步 $m=0$，数值为 $1$ 的奖励停在该步。$y$ 是当前 $\pi$ 的熵外动作价值的一步自举，目标侧取两个 $Q$ 的最小。actor 取二者的平均，并另加熵：

$$
\mathcal{L}_\pi = -\mathbb{E}\big[Q_{\mathrm{mean}}(s,a) - \alpha \log\pi(a\mid s)\big].
$$

$a$ 经重参数化采样，形式见 SAC 笔记第六节。tanh 压缩高斯满足 $\mathbb{E}_{a\sim\pi}[-\log\pi(a\mid s)]=\mathcal{H}(\pi(\cdot\mid s))$，论文式 (2) 的分布熵与 `policy_loss_fn` 的样本对数概率在期望下一致。温度由 `GeqLagrangeMultiplier` 实现，最小化 $\alpha(\mathcal{H}-\bar{\mathcal{H}})$，约束 $\mathcal{H}\ge\bar{\mathcal{H}}$。未给定目标熵时 $\bar{\mathcal{H}}=-d/2$，$d$ 为 `create` 收到的动作维数；纯连续任务中 $d$ 即高斯维数，混合动作见第五节。初始温度为 $10^{-2}$。SAC 笔记第五节的 soft Bellman 在目标中还含 $-\alpha\log\pi(a'\mid s')$。本入口把熵写在 actor 损失与 $\alpha(\mathcal{H}-\bar{\mathcal{H}})$ 中。

---

## 三、两个缓冲与介入

Learner 维护两个回放。`demo_buffer` 在训练开始时载入离线演示，容量与在线缓冲相同，默认 $200000$。`replay_buffer` 初始为空，只接收 actor 进程送来的转移。Learner 从每个缓冲取 `batch_size // 2`（默认各 $128$），经 `concat_batches(..., axis=0)` 沿批次维拼成 $256$。代码注释标明这是 RLPD 的 $50/50$ 抽样。

Actor 以 $10\,\mathrm{Hz}$ 循环。`random_steps` 默认为 $0$，故没有均匀随机的预探索；探索来自高斯标准差与人的介入。`SpacemouseIntervention` 读取 SpaceMouse：平动范数大于 $0.001$ 记为介入。启用夹爪时，左键在 $[-1,-0.9]$ 上采样表示闭合，右键在 $[0.9,1]$ 上采样表示打开，否则该维为 $0$。环境执行替换后的动作。若 `info` 含 `intervene_action`，写入缓冲的动作改为人的动作，策略原采样被丢弃。

每条转移都进入在线队列。发生介入时，同一条再进入介入队列，由 learner 写入 `demo_buffer`。介入前后由策略执行的步骤只留在在线缓冲。这与论文第三节一致：人的片段在两个缓冲中各存一份，策略转移只在在线缓冲。

演示缓冲起初载入 $20$–$30$ 条遥操作，此后每次介入都追加进去，缓冲内介入转移所占的比例随训练提高。批次从两侧各取一半。在线缓冲同时保存策略转移与介入转移。介入动作用作 $\mathcal{L}_Q$ 的动作自变量，自举动作从当前 $\pi$ 抽取。

介入率定义为一个回合内被介入的时间步占比，再对 $20$ 个回合取滑动平均。训练初期介入更频繁；策略改善后，介入变短、变稀，收敛后该曲线降至 $0$。论文同时指出，训练早期若持续给出较长、且直接导致成功的介入，价值函数会被高估，训练不稳定。报告中收敛较快的操作方式，是给出短而具体的修正，其余时间由机器人自行探索。

参数同步是异步的。在线缓冲达到 `training_starts`（默认 $100$）条后，learner 开始更新，并立即将初始参数发给 actor。此后每 `steps_per_update`（默认 $50$）个 learner 迭代发布一次参数。Actor 通过 `recv_network_callback` 替换本地参数，发布不绑定在回合结束。环境步与梯度步异步，更新–数据比由 learner 的吞吐决定。每个 learner 迭代内部还有 `cta_ratio`（默认 $2$）：先做 `cta_ratio-1` 次只更新 critic（混合夹爪时包括 grasp critic），最后一次同时更新 critic、actor 与 temperature。注释说明，这样可以减少 CPU 到 GPU 的大批次传输。

---

## 四、代码中的 SAC 与论文所引的 RLPD

论文将算法核称为 RLPD，并给出上一节的单目标网络损失。RLPD 原文的常用配方包括：一半离线、一半在线，critic 使用 LayerNorm，约 $10$ 个 $Q$ 中随机抽取 $2$ 个取最小，以及较高的更新–数据比。HIL-SERL 正文未再给出集成规模。

`make_sac_pixel_agent` 及单臂、双臂两个混合工厂的实际设置为：

- `critic_ensemble_size=2`，`critic_subsample_size=None`：目标取两个 $Q$ 的最小，未使用 REDQ 的 $10$ 选 $2$；
- actor 与 critic 的 MLP 均为 $\tanh$、LayerNorm、$[256,256]$；
- `backup_entropy=False`；
- 策略为 tanh 压缩高斯，标准差以 $\exp$ 参数化，范围为 $[10^{-5},5]$；
- 折扣由任务配置传入，默认 $0.97$，补充材料中各任务为 $0.96$–$0.985$；
- 目标网络软更新系数沿用 `SACAgent.create` 的默认值 $0.005$。

实验设置以该训练入口为准。论文式 (1) 的 $\bar Q$ 在代码中是两个目标 $Q$ 的最小；式 (2) 的熵正则在代码中是样本对数概率。$Q$ 的备份用上一节的 $y$，固定点是熵外动作价值。LayerNorm 与 $50/50$ 抽样与 RLPD 配方一致。$10$ 成员集成未出现在该入口中。

Actor 与 critic 各有一套编码器，混合智能体再为 grasp critic 复制一套。各图像键的卷积在 Python 中共用同一个 `pretrained_encoder` 定义，Flax 按调用路径分别建立参数。`resnetv1-10-frozen` 设 `pre_pooling=True`，前向在池化前执行 `stop_gradient`，卷积核不接收任务梯度。其后的空间学习嵌入含 $8$ 个空间块，瓶颈维数为 $256$，dropout 为 $0.1$，再接 LayerNorm 与 $\tanh$，这一部分可训练。`load_resnet10_params` 把 SERL 发布的 ImageNet-1K ResNet-10 权重写入 `params["modules_actor"]` 下各图像键的 `pretrained_encoder`。critic 的卷积核保持初始化数值，混合智能体中 grasp critic 的卷积核同样如此。`create` 在载入之前已把 `target_params` 设为当时的 `params`，载入函数只替换 `params`。在线 actor 的卷积核由 `stop_gradient` 固定在 ImageNet 权重上，其目标编码器从载入前的随机参数出发，经系数 $0.005$ 的 Polyak 更新与之对齐。论文的写法是 actor 与 critic 各自编码，并共享冻结的预训练骨干。

图像先裁到任务区域，再缩放到 $128\times 128$。每个批次在计算损失前做随机裁剪。补充材料中，本体感觉编码器宽度为 $64$，策略 MLP 为 $256\times 256$，优化器为 Adam，学习率为 $3\times 10^{-4}$。Jenga 在成功率达到 $70\%$ 后将学习率降至 $3\times 10^{-5}$。其余任务表未列此衰减。

---

## 五、夹爪的离散 MDP

抓取任务将控制拆成两个 MDP。二者观测相同。连续 MDP 的动作是末端速度，离散 MDP 的动作是夹爪：单夹爪为开、合、保持三类，双夹爪为 $3^2=9$ 种组合。论文式 (3) 是 grasp critic 的平方时序误差，备份为 Double DQN：$\arg\max$ 使用在线网络，求值使用目标网络，目标网络以 Polyak 平均更新。`forward_grasp_critic` 选取动作，`forward_target_grasp_critic` 计算该动作的值。发布代码的目标为

$$
y_g = r + b + r_{\mathrm{grasp}} + \gamma\, m\, Q_{\theta'}(s', a'_g),
\qquad
a'_g = \arg\max_{a'} Q_\theta(s', a').
$$

$m=1-\mathrm{done}$。$b$ 为 `reward_bias`。`update` 先把它加进 `batch["rewards"]`，连续 critic 与 grasp critic 读到同一份奖励；工厂默认 $b=0$。$r_{\mathrm{grasp}}$ 为 `grasp_penalty`，加在 grasp critic 上。论文用这项小的负惩罚减少无谓开合。连续 $Q$ 仍用第二节的 $y$，奖励取加过 $b$ 的环境奖励。

夹爪在环境动作里编码为 $[-1,1]$。双臂将第 $6$ 维与第 $13$ 维四舍五入后加 $1$，得到 $\{0,1,2\}$，再合成为 $a_1\cdot 3+a_2\in\{0,\ldots,8\}$。单臂对最后一维做同一平移，得到三类。SpaceMouse 的开、合采样位于区间两端，四舍五入后分别对应开与合；保持对应 $0$。

高斯头只输出连续速度。双臂的 `action_dim` 为 $12$，单臂为环境动作维数减 $1$。`sample_actions` 从该高斯抽样或取其众数，再对 grasp $Q$ 取 $\arg\max$，映回 $\{-1,0,1\}$ 后拼入夹爪坐标。双臂因此得到 $14$ 维环境动作，夹爪位于第 $6$ 维与末维。`policy_loss_fn` 的 $\log\pi$ 和送入连续 $Q$ 的动作都是这条连续向量。`critic_loss_fn` 从缓冲里的环境动作切出同一段连续分量，单臂去掉最后一维，双臂取 $[0{:}6]$ 与 $[7{:}13]$，再与策略新抽样的连续动作做自举。连续 critic 的动作自变量是这段速度，grasp critic 的动作自变量是离散档。`train_rlpd.py` 的评估分支对连续部分调用 `argmax=False`，末端速度仍在采样，夹爪为贪心。

`create` 在未指定目标熵时取 $\bar{\mathcal{H}}=-d/2$，$d$ 为环境动作维数。双臂 $d=14$，$\bar{\mathcal{H}}=-7$，温度损失中的熵来自 $12$ 维高斯。单臂 $d=7$，$\bar{\mathcal{H}}=-3.5$，熵来自 $6$ 维高斯。

`setup_mode` 的四个取值决定智能体类型。`single-arm-fixed-gripper` 与 `dual-arm-fixed-gripper` 使用纯 SAC，`GripperCloseEnv` 将介入动作裁到前 $6$ 维。`single-arm-learned-gripper` 与 `dual-arm-learned-gripper` 才建立 grasp critic。补充材料将仪表板记为 $12$ 维速度加 $1$ 维离散夹爪。仓库中的双臂混合智能体实现两个夹爪、$9$ 类联合动作，且未附带仪表板配置。

---

## 六、奖励

除 Jenga 外，奖励为离线训练的二分类器。骨干同为预训练 ResNet-10，后接两层 MLP，损失为交叉熵，优化器为 Adam，学习率 $3\times 10^{-4}$，共 $100$ 次迭代。数据由遥操作采集：成功时按 SpaceMouse 键标为 $1$，其余标为 $0$。常见规模约为 $200$ 张正样本、$1000$ 张负样本，约 $10$ 条轨迹、$5$ 分钟，并按需补充假阳性与假阴性。评估集准确率通常高于 $95\%$，各任务表为 $95\%$–$99\%$。RAM 插入的补充图将「插入错误插槽」「未插到底」「仅插入一侧」标为负类。

分类器判定成功则奖励为 $1$，并可结束回合，否则奖励为 $0$。这个二值输出就是第二节单阶段回报里的 $r$。多阶段包装器按阶段累加，全部阶段满足后结束。RAM 示例配置在分类器之外另设门槛：sigmoid 大于 $0.85$，且末端 $z$ 高于 $0.04$，才给予奖励。该配置的 $r$ 由分类器输出与这一几何条件共同决定。论文的一般描述为二值成功。

Jenga 不使用分类器，奖励为回合结束时的人工标注。补充材料给出折扣 $0.96$，且每个回合都运行至最大长度，即 $20$ 个环境步，在 $10\,\mathrm{Hz}$ 下约 $2$ 秒。成功不提前终止。奖励记在最后一步，折扣只乘在这个固定下标上，信用分配沿这一短时域回传。

---

## 七、相对坐标与底层控制

每个回合开始时，末端相对基座的位姿 $T_s^{b_0}$ 在任务给定的随机化范围内抽取。策略观测的位置与姿态为

$$
T_{b_0}^{b_t} = \big(T_s^{b_0}\big)^{-1} T_s^{b_t},
$$

即当前末端在该回合初始末端坐标系中的位姿。速度表达在当前末端坐标系中。目标相对末端的几何不变时，基座系中的绝对位置可以改变。论文以此解释插入过程中主板被移动后策略仍能对准。

接触任务的动作是当前末端坐标系下的 $6$ 维速度 $v_{b_t}$。机器人接口要求基座系速度，二者由伴随映射联系：

$$
v_s = \mathrm{Ad}_{T_s^{b_t}}\, v_{b_t},
\qquad
\mathrm{Ad}_T
=
\begin{bmatrix}
R & 0 \\
[p]_\times R & R
\end{bmatrix}.
$$

`RelativeFrame.transform_action` 以 $6\times 6$ 矩阵左乘动作的前 $6$ 维，再交给内层环境。若该步为人的动作，`transform_action_inv` 将 `intervene_action` 变回策略坐标系，缓冲中保存的是策略应当输出的速度。观测中的 `tcp_pose` 改为相对复位位姿，$tcp\_vel$ 乘伴随矩阵的逆，变换到当前末端系。

阻抗控制器运行于 $1000\,\mathrm{Hz}$，接收 $10\,\mathrm{Hz}$ 的设定点。补充材料将末端力写为

$$
F = k_p e + k_d \dot e + F_{ff} + F_{\mathrm{cor}},
\qquad
e = p - p_{\mathrm{ref}},
$$

再经 $J^\top$ 转为关节力矩，并叠加零空间项。$p_{\mathrm{ref}}$ 远离当前位姿时，弹簧–阻尼项产生很大的接触力。论文限制参考误差 $|e|\le\Delta$，并给出这两项的上界

$$
k_p\,|\Delta| + 2 k_d\,|\Delta|\, f,
$$

其中 $f$ 为控制频率。参考位姿留在当前位姿的 $\Delta$ 邻域内，比例项至多为 $k_p|\Delta|$。误差落在 $[-\Delta,\Delta]$，一个控制周期 $1/f$ 内最多从 $-\Delta$ 变到 $\Delta$，差分不超过 $2|\Delta|f$，阻尼项至多为 $2k_d|\Delta|f$。$F_{ff}$ 与 $F_{\mathrm{cor}}$ 加在这个上界之外，仍按力的定义式进入控制器。RAM 示例把该限制实现为各轴位移裁剪，例如 $x$ 向 $0.0075\,\mathrm{m}$，平动刚度 $2000$、阻尼 $89$，动作缩放 $(0.01, 0.06, 1)$。这些数值属于该机器人配置。论文的一般描述只给出误差界的形式。

两项动态任务不使用上述闭环阻抗。策略输出当前末端系下的 $3$ 维前馈力旋量：Jenga 为 $F_x,F_z,\tau_z$，物体翻转为 $F_x,F_z,\tau_y$。控制器取 $\tau = J^\top w$ 并加上零空间项，在 $1000\,\mathrm{Hz}$ 执行、$10\,\mathrm{Hz}$ 更新设定点。操作空间中，力旋量与加速度之间还隔着惯性矩阵和科氏项；这条控制律把 $w$ 直接映成力矩。论文将 $w$ 视为期望加速度的近似，并报告该近似对这两项任务是足够的。

---

## 八、训练流程

图 3 将一次实验分为三步。

1. 选择相机。腕部相机提供随末端运动的视角；腕部视野不足时增加侧视。图像裁剪后缩放到 $128\times 128$。
2. 采集并训练奖励分类器。正负样本的采集见第六节。Jenga 跳过此步，改为回合结束时人工标注。
3. 采集 $20$–$30$ 条成功遥操作，只写入演示缓冲。复位为脚本运动或人工复位。随后 actor 以当前策略交互，learner 按第三节更新。需要夹爪的任务同时更新 grasp critic。人按第三节所述密度介入，直至策略收敛。

Jenga 不用在线介入，演示缓冲放入 $30$ 条专家轨迹。鞭梢速度高，在线介入难以实施，论文因此不采用在线修正。物体翻转仍使用分类器奖励和 $20$ 条演示，动作为力旋量；补充材料未记载在线介入，表 1a 的对照也改为离线行为克隆。

公开示例包括 RAM 插入、物体翻转、物体交接与 USB 拾取插入。主板的其余子任务、宜家装配、仪表板、同步带与 Jenga 在补充材料中有训练表，仓库未附带对应的全部任务配置。

---

## 九、实验

### 9.1 协议

训练时间包括脚本运动、策略展开、主动暂停，以及单张 RTX 4090 上的计算。除宜家整体装配为 $10$ 次试验外，成功率与周期时间均基于 $100$ 次评估。评估时按该任务的脚本或人工复位做初始随机化，幅度见补充材料，最大约数厘米与十余度。物体交接与两项动态任务不做位置随机化。周期时间只在成功回合上统计。

表 1a 的 BC 列，除 Jenga 与物体翻转外，均为 HG-DAgger：先用与 RL 相同数量的离线演示做行为克隆，再运行与 RL 相同的回合数，并尽量匹配每回合介入次数，以监督学习更新。Jenga 的对照是 $50$ 条演示上的行为克隆，物体翻转是 $200$ 条，均多于 RL 使用的 $20$–$30$ 条。

HIL-SERL 的介入率随成功率上升而下降，并降至 $0$。HG-DAgger 学习曲线中的成功率包含介入本身带来的成功，论文指出策略自身的成功率低于曲线；其介入率不持续下降，周期时间也未缩短。

摘要报告 $1$–$2.5$ 小时内接近完全成功，成功率约提高至 $2$ 倍，执行约快 $1.8$ 倍。正文将 $1$–$2.5$ 小时限定为几乎全部任务。同步带为 $6$ 小时。$100/49.7\approx 2.01$，与「约 $2$ 倍」一致。

### 9.2 表 1a

下表成功率为 $100$ 次评估的百分比。宜家整体为 $10$ 次试验中的成功次数。周期时间单位为秒。相对提升为论文所印数字。

| 任务 | 训练 (h) | BC / HG-DAgger | HIL-SERL | 周期，对照 | 周期，HIL-SERL |
|------|----------|----------------|----------|------------|----------------|
| RAM 插入 | $1.5$ | $29$ | $100$（$+245\%$） | $8.3$ | $4.8$（$1.7\times$） |
| SSD 装配 | $1$ | $79$ | $100$（$+27\%$） | $6.7$ | $3.3$（$2\times$） |
| USB 抓取插入 | $2.5$ | $26$ | $100$（$+285\%$） | $13.4$ | $6.7$（$2\times$） |
| 线缆卡扣 | $1.25$ | $95$ | $100$（$+5\%$） | $7.2$ | $4.2$（$1.7\times$） |
| 宜家侧板 1 | $2$ | $77$ | $100$（$+30\%$） | $6.5$ | $2.7$（$2.4\times$） |
| 宜家侧板 2 | $1.75$ | $79$ | $100$（$+27\%$） | $5.0$ | $2.4$（$2.1\times$） |
| 宜家顶板 | $1$ | $35$ | $100$（$+186\%$） | $8.9$ | $2.4$（$3.7\times$） |
| 宜家整体 | — | $1/10$ | $10/10$（$+900\%$） | — | — |
| 汽车仪表板 | $2$ | $41$ | $100$（$+144\%$） | $20.3$ | $8.8$（$2.3\times$） |
| 物体交接 | $2.5$ | $79$ | $100$（$+27\%$） | $16.1$ | $13.6$（$1.2\times$） |
| 同步带 | $6$ | $2$ | $100$（$+4900\%$） | $9.1$ | $7.2$（$1.3\times$） |
| Jenga 甩出 | $1.25$ | $8$ | $100$（$+1150\%$） | — | — |
| 物体翻转 | $1$ | $46$ | $100$（$+117\%$） | $3.9$ | $3.8$（$1.03\times$） |

论文所印平均为：成功率 $49.7$ 对 $100$（$+101\%$），周期 $9.6\,\mathrm{s}$ 对 $5.4\,\mathrm{s}$（$1.8\times$）。$49.7$ 等于上表十二个 $100$ 次评估任务的 BC 列均值，不含宜家整体的 $1/10$。$9.6$ 与 $5.4$ 等于有周期时间的十一个任务的均值，Jenga 与宜家整体未报告周期时间。

宜家整体将已训练的子策略串联，子任务之间以脚本运动过渡。抓取位姿在每个平动方向随机 $1\,\mathrm{cm}$，每个子策略最多尝试两次。$10$ 次试验中，对照成功 $1$ 次，HIL-SERL 成功 $10$ 次。论文以整机能够启动作为装配有效的定性证据。

物体交接的过程为从右篮抓起、交给左臂、放入左篮。紧随其后的成功判据写为「物体平放在右篮」。过程句与判据句所指篮子不同。

摘要中的 $1$–$2.5$ 小时、约 $2$ 倍成功率与 $1.8$ 倍速度是汇总描述。线缆卡扣的对照已达 $95\%$，成功率只提高 $5$ 个百分点，周期从 $7.2\,\mathrm{s}$ 降至 $4.2\,\mathrm{s}$。物体翻转的周期几乎不变（$3.9\,\mathrm{s}$ 至 $3.8\,\mathrm{s}$）。同步带的成功率增益最大，训练时间为 $6$ 小时，位于摘要区间之外。

### 9.3 各任务的训练设置

控制频率均为 $10\,\mathrm{Hz}$，优化器均为 Adam，学习率 $3\times 10^{-4}$，图像增强均为随机裁剪，本体编码器宽度均为 $64$。下表只列影响复现的项。$N$ 为在线转移数，$\gamma$ 为折扣，$H$ 为最大回合步数。

| 任务 | 动作 | 演示 | $N$ | $\gamma$ | $H$ | 复位 | 随机化 | 分类器 |
|------|------|------|-----|----------|-----|------|--------|--------|
| RAM | $6$ 维速度 | $20$ | $32000$ | $0.97$ | $100$ | 脚本 | $4\,\mathrm{cm}$ 的 $xy$，$6^\circ$ 的 $r_z$ | 腕部两相机，$97\%$ |
| SSD | $6$ 维速度 | $20$ | $21000$ | $0.97$ | $100$ | 脚本 | $2\,\mathrm{cm}$，$1^\circ$ | 两腕加侧视，$95\%$ |
| USB | $6$ 维速度 $+1$ 维夹爪 | $20$ | $50000$ | $0.98$ | $120$ | 脚本 | $2\,\mathrm{cm}$，$10^\circ$ | 仅侧视，$96\%$ |
| 线缆卡扣 | $6$ 维速度 $+1$ 维夹爪 | $20$ | $28000$ | $0.98$ | $120$ | 人工 | $4\,\mathrm{cm}$，$10^\circ$ | 两腕，$97\%$ |
| 宜家侧板 | $12$ 维速度 | $20$ | $31000$ / $36000$ | $0.98$ | $100$ | 脚本 | $8\,\mathrm{cm}$，$1^\circ$ | 侧视，$97\%$ |
| 宜家顶板 | $12$ 维速度 | $20$ | $18000$ | $0.97$ | $150$ | 脚本 | $3\,\mathrm{cm}$ | 三侧视，$95\%$ |
| 仪表板 | $12$ 维速度 $+1$ 维夹爪 | $20$ | $36000$ | $0.97$ | $200$ | 人工 | $2\,\mathrm{cm}$ | 两腕加侧视，$98\%$ |
| 交接 | $12$ 维速度 $+1$ 维夹爪 | $20$ | $43000$ | $0.97$ | $200$ | 人工 | 无 | 侧视，$99\%$；观测不含力 |
| 同步带 | $12$ 维速度 | $20$ | $108000$ | $0.97$ | $200$ | 人工 | $2\,\mathrm{cm}$ | 两侧视，$96\%$ |
| Jenga | $F_x,F_z,\tau_z$ | $30$ | $10000$ | $0.96$，不提前终止 | $20$ | 人工 | 无 | 回合末人工标注 |
| 翻转 | $F_x,F_z,\tau_y$ | $20$ | $25000$ | $0.985$ | $100$ | 脚本 | 无 | 腕部，$97\%$ |

接触任务的观测一般包括腕部或侧视图像、`tcp_pose`、`tcp_vel` 与力/力矩，抓取任务另加夹爪位置。Jenga 与物体翻转使用腕部、侧视、末端位姿与速度，以及关节角 $q$ 和角速度 $\dot q$，不使用力传感器。交接任务的训练表不含力/力矩。

$32000$ 条 RAM 转移在 $10\,\mathrm{Hz}$ 下约对应 $53$ 分钟的策略步。表 1a 的 $1.5$ 小时还包含复位、暂停与计算。同步带的 $108000$ 条是表中最长的在线数据，对应 $6$ 小时训练。

### 9.4 表 1b：三项任务上的对照与消融

三项任务为 RAM 插入、仪表板与物体翻转。列顺序与表头一致，三行均值等于表底的 $34,39,31,57,32,33,0,49,100$。

| 方法 | 数据设定 | RAM | 仪表板 | 翻转 |
|------|----------|-----|--------|------|
| Diffusion Policy | $200$ 条演示 | $27$ | $18$ | $56$ |
| HG-DAgger | 与 RL 相同的回合数与介入数；翻转列同表 1a 的 BC | $29$ | $41$ | $46$ |
| BC | $200$ 条演示 | $12$ | $35$ | $46$ |
| IBRL | 以 $200$ 条演示初始化 | $75$ | $0$ | $95$ |
| Residual RL | 同上 | $0$ | $0$ | $97$ |
| DAPG | 以 $200$ 条演示做动作正则 | $8$ | $18$ | $72$ |
| 从零训练 | 无演示、无介入 | $0$ | $0$ | $0$ |
| 仅演示 | $200$ 条，无在线介入 | $48$ | $0$ | $100$ |
| HIL-SERL | 约 $20$ 条演示加在线介入 | $100$ | $100$ | $100$ |

从零训练在三项任务上均为 $0$。将离线演示从约 $20$ 条增至 $200$ 条、但不做在线介入时，仪表板仍为 $0$，RAM 为 $48$，物体翻转达到 $100$。论文以仪表板的 $0$ 说明，复杂任务上在线介入不可省略；物体翻转一列则表明，偏开环的动态任务可以仅凭大量离线演示上的 RL 达到完全成功。

正文称 Diffusion Policy 在 RAM 与仪表板上分别为 $27\%$ 与 $28\%$，在物体翻转上为 $56\%$。表中仪表板为 $18$。$(27+18+56)/3=33.7$，四舍五入后等于所印平均 $34$；若仪表板取 $28$，均值约为 $37$，与平均行不符。表与平均行支持 $18$，正文该句写作 $28$。

Diffusion Policy 的补充设定为：观测块长度均为 $1$；RAM 的预测时域为 $8$、执行 $2$；仪表板的预测时域为 $8$、执行 $4$；物体翻转的预测时域为 $1$、执行 $1$。演示均为 $200$ 条。论文认为，表达能力更强的动作分布主要改善对人的运动的拟合，而这三项任务需要闭环视觉伺服，因此扩散策略低于 HIL-SERL，也低于 HG-DAgger。

其余各列的解释来自正文。Residual RL 依赖已经可用的 BC 基策略。RAM 与仪表板上 BC 本身较差，残差策略为 $0$；物体翻转上 BC 已有 $46\%$，残差达到 $97$。IBRL 的 actor 是 BC 与 RL 的混合，论文认为其行为更接近 BC，仪表板为 $0$；RAM 上 IBRL 为 $75$，高于同表 BC 的 $12$。DAPG 将动作正则到演示，三项结果都处于 BC 附近或更低。论文将 HIL-SERL 的差别表述为：off-policy 更新按当前目标对人的数据加权，早期借助人的数据起步，此后可以离开演示流形。物体翻转上，$20$ 条与 $200$ 条演示的 BC 分别为 $47\%$ 与 $46\%$，演示增至十倍后，模仿成功率基本不变。

### 9.5 可靠性、两类策略与论文所述局限

第五节用 RAM 任务中末端 $y$、$z$ 的访问热图描述一个漏斗：初始状态区域连接到插槽，接近目标时变窄，访问质量集中于能够成功的区域。$Q$ 方差较大的状态被定义为关键状态。动作先归一化到 $[-1,1]$，再叠加 $\epsilon\sim\mathrm{Unif}[-0.2,0.2]$，以 $100$ 个蒙特卡洛样本估计

$$
\mathrm{Var}[Q(s,a)]
=
\mathbb{E}_{\epsilon}
\Big[
\big(
Q(s, a+\epsilon)
-
\mathbb{E}_{\epsilon'}[Q(s, a+\epsilon')]
\big)^2
\Big].
$$

漏斗内部的状态同时具有较高的 $Q$ 与较高的这一方差，名义动作附近的价值随 $a$ 变化明显。论文结合访问热图，把该区域读成通向插槽的窄通道：通道内 $Q$ 较高，偏离后 $Q$ 下降。同一任务上，HG-DAgger 的访问更分散，漏斗更弱。论文将演示与修正视为名义轨迹，RL 借助像素输入与自主探索在其周围形成稳定区域，并援引 Burridge et al. (1999) 与 LQR-trees（Tedrake et al., 2010）。后者的漏斗来自状态空间中的反馈设计；本文图中的漏斗是访问计数与 $Q$ 的经验结果。

同一节以训练后高斯的标准差区分两类策略。RAM 上，接近目标前标准差约为 $0.6$，靠近目标后迅速下降，对应于闭环视觉伺服。Jenga 上标准差全程接近 $0$，均值在 $[-1,1]$ 内跨轨迹保持一致，对应于一次完成的开环甩动。仪表板与同步带还会出现脱离接触、重新对准后再插入的行为。论文将其与混合整数规划中的接触模态对照：这里的重新接近来自折扣回报下的闭环采样，规划问题中不显式枚举接触模式。

扰动实验是定性的，未另给成功率表。实验包括：插入过程中移动主板；物体交接与仪表板在抓取后强制打开夹爪，策略重新抓取并继续；同步带在变形与外力下继续装配；USB 抓取姿态较差时松开并再次抓取。论文将其归因于自我中心坐标，以及训练阶段对失败结果的自主探索。

讨论列出三项局限。更长时域上的样本复杂度尚不确定，文中建议借助预训练，或用视觉语言模型将长任务切分为短子任务。实验的随机化范围有限，也未在非结构化场景中检验泛化；文中建议以更长训练换取随机化，并将泛化交给大规模预训练的视觉模型。每项任务均从头训练。文中提出的后续方向，是用收敛策略采集数据并蒸馏到基础模型，以及预训练跨任务的价值函数再做微调。

---

## 十、与仓库内笔记的对应

| | SAC 笔记 | 本文的实现 |
|--|----------|------------|
| $Q$ 目标 | 第五节，含 $-\alpha\log\pi(a'\mid s')$ | `backup_entropy=False`，目标为 $r+\gamma m\min_i Q_i$，固定点为熵外动作价值 |
| 双 $Q$ | 目标取最小；actor 可用较小值或平均值 | 目标取两成员的最小；`policy_loss_fn` 对集成取平均 |
| $\alpha$ | 第七节的对偶，启发式为 $-\dim(\mathcal{A})$ | `GeqLagrangeMultiplier` 最小化 $\alpha(\mathcal{H}-\bar{\mathcal{H}})$；纯连续时 $\bar{\mathcal{H}}=-d/2$，初值 $10^{-2}$ |
| 数据 | 单一回放 | $\mu=\frac12\mu_{\mathrm{demo}}+\frac12\mu_{\mathrm{online}}$；介入步写入两边 |
| 动作 | 一个连续分布 | 连续高斯；抓取任务的夹爪另有 $3$ 或 $9$ 类 DQN，熵只统计连续头 |

FlashSAC 讨论的是仿真中扩大网络、降低更新–数据比，并以归一化稳定 critic。HIL-SERL 的 MLP 仍为 $256\times 256$，LayerNorm 开启，更新–数据比由异步 learner 与 `cta_ratio=2` 决定。稳定主要来自系统侧：actor 卷积核上冻结的 ImageNet ResNet-10、参考限幅的阻抗，以及短时介入。$Q$ 的固定点是熵外动作价值。

RL-100 笔记讨论扩散策略上的裁剪目标。两者都在真机上用稀疏成功压缩周期时间。成功即终止时，$\gamma<1$ 使回报等于 $\gamma^{t_\star}$，成功越早数值越大。数据角色不同。HIL-SERL 中，人在 $10\,\mathrm{Hz}$ 上替换当前动作，该转移立即进入两个缓冲，并出现在下一次 $50/50$ 批次中。RL-100 中，人提供轨迹级标签，策略更新发生在新轨迹并入数据集、重新做行为克隆和离线 PPO 之后。动作类也不同：此处为 $6$ 或 $12$ 维速度，或 $3$ 维力旋量，外加离散夹爪；RL-100 为点云条件下的去噪链或一步学生。表 1b 的 Diffusion Policy 按补充材料 C 训练，属于模仿基线；RL-100 的后训练是另一项实验。

DPPO 笔记的第二段不再读取演示，样本在使用后丢弃。HIL-SERL 的演示缓冲只增不减，在线缓冲同时保留成功与失败。它能够在数万条转移的量级上更新，是因为批次的一半始终来自人执行过的动作，并且 $Q$ 可以在旧转移上反复自举。早期过长的成功介入会持续留在这一半数据中。

---

## 十一、主要超参数

| 参数 | 代码 | 论文或默认值 | 作用 |
|------|------|--------------|------|
| 批次 | `batch_size=256`，各缓冲取一半 | RLPD 的 $50/50$ | 第三节 |
| 折扣 | 任务配置，默认 $0.97$ | $0.96$–$0.985$ | 成功即终止时，越早成功回报越大 |
| 集成 | `critic_ensemble_size=2`，`subsample=None` | 式 (1) 写单个 $\bar Q$ | 目标取最小 |
| 熵备份 | `backup_entropy=False` | 式 (1) 的目标不含 $\alpha\log\pi$ | 与 SAC 笔记第五节不同 |
| LayerNorm | actor 与 critic 均为真 | 正文未单列 | MLP 为 $[256,256]$，$\tanh$ |
| 温度 | 初值 $10^{-2}$；纯连续 $\bar{\mathcal{H}}=-d/2$，混合动作按环境维数 | 式 (2) 的自适应 $\alpha$ | 熵来自连续头 |
| 软更新 | `soft_target_update_rate=0.005` | 正文未单列 | 连续 $Q$ 与 grasp $Q$ 的目标网络 |
| `cta_ratio` | 默认 $2$ | 正文未单列 | 每次迭代 $1$ 次纯 critic、$1$ 次完整更新 |
| 发布周期 | `steps_per_update=50` | 周期性发送给 actor | 每 $50$ 个 learner 迭代发布一次 |
| 开训条件 | `training_starts=100`，`random_steps=0` | — | 不做均匀随机预填充 |
| 回放容量 | $200000$ | — | 两个缓冲各自的容量 |
| 学习率 | Adam，$3\times 10^{-4}$ | Jenga 在成功率 $70\%$ 后降至 $3\times 10^{-5}$ | 补充材料各表 |
| 图像 | $128\times 128$，随机裁剪 | 同左 | actor 载入冻结的 ImageNet ResNet-10，空间嵌入 $8\times 256$ |
| 控制 | 策略 $10\,\mathrm{Hz}$，底层 $1000\,\mathrm{Hz}$ | 同左 | 阻抗参考限幅，或 $J^\top w$ |
| 夹爪 | $3$ 类或 $9$ 类 | 式 (3) | 执行时取 $\arg\max$ |
| 分类器 | ResNet-10 加两层 MLP，$100$ 步 | 准确率通常高于 $95\%$ | RAM 配置另有阈值 $0.85$ 与 $z>0.04$ |

默认配置中 `eval_n_trajs` 为 $5$，用于脚本自测。论文表 1a 来自 $100$ 次评估，宜家整体来自 $10$ 次。

---

## 十二、参考文献与阅读顺序

1. Luo, Xu, Wu, and Levine, *Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning*, arXiv:2410.21845v3, 2025. 代码：[rail-berkeley/hil-serl](https://github.com/rail-berkeley/hil-serl)
2. Luo et al., *SERL: A Software Suite for Sample-Efficient Robotic Reinforcement Learning*, ICRA 2024.
3. Ball, Smith, Kostrikov, and Levine, *Efficient Online Reinforcement Learning with Offline Data*, ICML 2023.
4. Haarnoja et al., *Soft Actor-Critic Algorithms and Applications*, arXiv:1812.05905, 2018. [`sac_notes.md`](sac_notes.md)
5. Song et al., *Hybrid RL: Using Both Offline and Online Data Can Make RL Efficient*, ICLR 2023.
6. van Hasselt, Guez, and Silver, *Deep Reinforcement Learning with Double Q-learning*, AAAI 2016. 式 (3) 将 $\arg\max$ 与求值分开。
7. Kelly, Sidrane, Driggs-Campbell, and Kochenderfer, *HG-DAgger*, ICRA 2019.
8. Chi et al., *Diffusion Policy*, RSS 2023. 表 1b 的模仿基线。[`diffusion_policy_notes.md`](diffusion_policy_notes.md)
9. Johannink et al., *Residual Reinforcement Learning*, ICRA 2019.
10. Rajeswaran et al., *Learning Complex Dexterous Manipulation with Deep Reinforcement Learning and Demonstrations*, RSS 2018. DAPG。
11. Hu, Hendrycks, et al., *IBRL: Imitation Bootstrapped Reinforcement Learning*, 2024.
12. Burridge, Rizzi, and Koditschek, *Sequential Composition of Dynamically Dexterous Robot Behaviors*, IJRR 1999.
13. Tedrake, Manchester, Tobenkin, and Roberts, *LQR-Trees*, IJRR 2010.
14. Lei et al., *RL-100*, arXiv:2510.14830. [`rl100_notes.md`](rl100_notes.md)

建议先读 SAC 笔记第五至七节，再读本文第二至五节。实现上可核对 `train_rlpd.py` 的缓冲写入、`concat_batches` 的拼接轴，以及 `launcher.py` 中的 `backup_entropy` 与 `critic_ensemble_size`。表 1a 的汇总口径见第九节。

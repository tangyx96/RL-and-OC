# 模型式强化学习：学习动力学、规划与策略改进

模型式强化学习（model-based reinforcement learning, MBRL）从数据估计环境转移或潜变量世界模型，再用该模型进行规划、生成训练样本或直接优化策略。它处在最优控制与模型无关强化学习之间：模型提供反事实预测和梯度/采样接口，策略学习吸收规划结果并摊销在线计算。

本仓库的最优控制笔记讨论已知或显式给定的动力学模型；本文讨论模型由数据获得时，误差如何改变规划和策略学习。采样规划及模型预测控制的数值结构见 [`control-optimization-synthesis.md`](../../optimal-control-solvers/study-notes/control-optimization-synthesis.md)，离线数据上的保守学习见 [`offline-reinforcement-learning.md`](offline-reinforcement-learning.md)。

---

## 一、基本问题

真实环境满足

$$
s_{t+1}\sim P(\cdot\mid s_t,a_t),
\qquad
r_t=r(s_t,a_t).
$$

MBRL 从数据拟合参数模型

$$
\hat P_\omega(s'\mid s,a),
\qquad
\hat r_\omega(s,a),
\tag{1}
$$

然后选择下列至少一种用途：

1. **在线规划**：在 $\hat P_\omega$ 中优化短时域控制序列，仅执行首控；
2. **合成数据**：从真实数据状态出发，在模型中产生短 rollout，供 actor--critic 更新；
3. **策略梯度**：若模型可微，穿过时间展开的动力学反传；
4. **潜变量想象**：先从图像历史推断潜状态，再在潜空间中学习转移、价值和 actor。

模型式不等同于“有模型就不用策略”。MPC 可在每个控制周期重新求解；actor 可把规划结果蒸馏为快速策略；许多方法同时保留 actor 作为模型 rollout 的行为先验或终端价值近似。

---

## 二、模型误差与 rollout 长度

设真实与模型转移的单步误差在某种分布下受控。即使每步误差很小，递推模型也会把状态分布从真实轨迹推开。对长度 $H$ 的开环预测，误差一般由两部分累积：

$$
\text{trajectory error}
\approx
\sum_{t=0}^{H-1}
\text{model error at }(s_t,a_t)
\times
\text{dynamics sensitivity}.
\tag{2}
$$

式 (2) 不是普适等式，而是一个结构性提醒：长模型 rollout 同时经历误差累积和策略诱导的分布偏移。模型在行为数据附近拟合良好，不代表在优化器主动寻找的高回报轨迹上仍然可靠。

因此，MBRL 的核心设计不是单纯提高预测损失，而是决定：模型从何处分布学习、模型 rollout 从何处开始、rollout 多长、如何对不确定区域悲观，以及真实数据与虚拟数据以何种比例混合。

---

## 三、显式状态模型与采样规划

### 3.1 概率集成模型

PETS 一类方法用概率神经网络或 bootstrap ensemble 表示动力学：

$$
\hat P_m(s'\mid s,a)
=
\mathcal N\bigl(mu_m(s,a),\Sigma_m(s,a)\bigr),
\qquad m=1,\ldots,M.
\tag{3}
$$

成员间分歧可作为 epistemic uncertainty 的近似；每个成员输出的协方差描述 aleatoric uncertainty。二者都不是无条件可信的置信区间，尤其在模型结构错设或数据严重偏移时，但它们能为规划提供风险信号。

### 3.2 MPC 与 CEM

给定当前状态 $s_t$，在模型中评价控制序列

$$
U=(a_t,\ldots,a_{t+H-1}),
\qquad
\hat J(U)=
\mathbb E_{\hat P}\left[
\sum_{j=0}^{H-1}\gamma^j\hat r(s_{t+j},a_{t+j})
+\gamma^H\hat V(s_{t+H})
\right].
\tag{4}
$$

CEM 从搜索分布采样控制序列，保留高分精英并拟合新的搜索分布；MPPI 则对控制噪声作指数重权。两者在有限时域控制序列分布上优化，不等于直接训练全局状态反馈策略。关于这类分布更新的统一形式，见分布优化笔记第六节。

滚动时域执行只施加 $a_t$，下一周期重新观测并重解式 (4)。它能纠正部分模型误差，但不消除预测时域内的错误；终端价值、约束和不确定性处理仍决定规划器是否可靠。

---

## 四、Dyna 与短模型 rollout

### 4.1 MBPO 的混合数据观点

Model-Based Policy Optimization（MBPO）不要求模型从初始状态生成长轨迹。它从真实 replay buffer 采样 $s_0$，只在模型中展开较短的 $h$ 步：

$$
s_{j+1}\sim\hat P_\omega(\cdot\mid s_j,a_j),
\qquad a_j\sim\pi_\theta(\cdot\mid s_j),
\qquad j=0,\ldots,h-1.
\tag{5}
$$

虚拟转移加入模型缓冲，actor 与 critic 仍用 SAC 一类更新。短 rollout 牺牲了远期合成数据量，却控制了式 (2) 的误差积累；随着真实数据增多，rollout horizon 可逐渐增长。这里的“短”不是固定经验常数，而是模型精度、任务敏感度和真实数据比例共同决定的。

### 4.2 模型生成数据并非天然有益

若模型在某一区域高估奖励、低估碰撞概率或漏掉接触动力学，虚拟数据会把 critic 系统性推向错误固定点。模型生成样本的数量越大，错误 target 的权重也越大。故应分别记录真实/虚拟采样比例、rollout 起点分布与 horizon；只报告总交互步数不足以说明模型带来的样本效率。

---

## 五、模型悲观与离线 MBRL

离线 MBRL 同时面对数据支持限制和模型误差。MOPO 的基本思想是在模型中使用惩罚后的奖励

$$
\tilde r(s,a)=\hat r(s,a)-\lambda u(s,a),
\tag{6}
$$

其中 $u$ 是 ensemble disagreement、模型方差或其他不确定性分数。actor 因而不仅追求预测回报，也避免模型无法可靠预测的区域。

式 (6) 的有效性取决于 $u$ 是否与实际模型误差相关。ensemble 分歧低并不保证模型正确；反之，具有多模态但可预测的真实转移也可能呈现高方差。因此“悲观奖励”是降低模型利用风险的设计，并非自动的安全证明。COMBO 等方法还将保守 Q 正则与模型 rollout 联合，试图同时约束模型外推与 critic 外推。

---

## 六、潜变量世界模型与 Dreamer

像素观测下，直接预测下一帧常把建模容量耗在与控制无关的纹理上。潜变量模型先学习状态空间模型，例如 recurrent state-space model：

$$
z_t\sim q_\omega(z_t\mid z_{t-1},a_{t-1},o_t),
\qquad
\hat z_{t+1}\sim p_\omega(z_{t+1}\mid z_t,a_t),
\tag{7}
$$

并从 $z_t$ 预测奖励、终止和观测。Dreamer 在学习到的 prior dynamics 中想象轨迹：

$$
\hat z_{t+1}\sim p_\omega(\cdot\mid\hat z_t,a_t),
\qquad
a_t\sim\pi_\theta(\cdot\mid\hat z_t).
\tag{8}
$$

价值目标常以 $\lambda$-return 构造，actor 则通过可微想象轨迹的路径导数或混合梯度更新。世界模型训练、actor 优化和真实环境评估是三个不同环节：潜变量重建得好不必然意味着控制相关预测准确；想象回报提高也不必然迁移到真实环境。

世界模型适合长时序部分可观测问题，但其状态表示、终止预测和不确定性校准都是性能瓶颈。将未来视频作为辅助监督的 WAM 与此相关，但若模型不以动作条件形式预测转移，也不以其进行回报规划，就不能直接称为 MBRL；详见 [`wam_notes.md`](../generative-policies/wam_notes.md)。

---

## 七、与最优控制和策略学习的统一视角

| 对象 | 已知模型的最优控制 | 模型式 RL | 模型无关 RL |
|------|------------------|----------|------------|
| 动力学 | 已知 $f$ 或可微模拟器 | 数据拟合 $\hat P_\omega$ | 不显式使用模型 |
| 改进方式 | SQP、iLQR、MPC、CEM | 模型规划、Dyna、imagination | TD / policy gradient |
| 主要误差 | 局部二次近似、离散化、求解容差 | 模型偏差与分布偏移 | value estimation 与采样误差 |
| 在线代价 | 通常较高 | 规划型高，actor 型低 | 通常低 |

三者不是互斥阵营。可微模型允许把最优控制的灵敏度分析用于 actor；CEM/MPPI 可在学习模型上规划；策略网络可蒸馏 MPC；模型无关 critic 可充当模型规划的终端价值。真正需要核对的是优化对象、模型可信范围和更新时使用的数据分布。

---

## 八、文献与代码

1. Chua et al., *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models*, NeurIPS 2018. [arXiv](https://arxiv.org/abs/1805.12114)
2. Janner et al., *When to Trust Your Model: Model-Based Policy Optimization*, NeurIPS 2019. [project](https://sites.google.com/view/mbpo)
3. Yu et al., *MOPO: Model-based Offline Policy Optimization*, NeurIPS 2020. [arXiv](https://arxiv.org/abs/2005.13239)
4. Yu et al., *COMBO: Conservative Offline Model-Based Policy Optimization*, NeurIPS 2021. [arXiv](https://arxiv.org/abs/2102.08363)
5. Hafner et al., *Mastering Diverse Domains through World Models*, 2023. [DreamerV3](https://github.com/danijar/dreamerv3)
6. Janner et al., *Diffuser: Planning with Diffusion for Flexible Behavior Synthesis*, ICML 2022. [project](https://diffusion-planning.github.io/)

建议阅读顺序：先对照最优控制笔记中的 MPC 与 CEM，再读 MBPO 的短 rollout，最后读 MOPO/COMBO 的悲观性与 Dreamer 的潜空间想象。这样可以避免把“学习模型”“在模型中规划”和“用模型训练策略”视为同一件事。

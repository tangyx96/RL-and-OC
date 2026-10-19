# DPPO（Diffusion Policy Policy Optimization）算法笔记

下文对应 Ren et al., *Diffusion Policy Policy Optimization*（[arXiv:2409.00588](https://arxiv.org/pdf/2409.00588)）。实现见 [irom-princeton/dppo](https://github.com/irom-princeton/dppo)：去噪步的高斯密度在 `model/diffusion/diffusion_vpg.py`，PPO 损失在 `model/diffusion/diffusion_ppo.py`，采样与 GAE 在 `agent/finetune/train_ppo_diffusion_agent.py`。  
去噪过程与 $\epsilon$-prediction 见 [`diffusion_policy_notes.md`](diffusion_policy_notes.md)；裁剪目标与 GAE 见 [`ppo_notes.md`](ppo_notes.md)。

---

## 一、算法定位

DPPO 分两段训练，前一段结束才开始后一段。损失不相加，也不在同一次更新里交替。

第一段是 Diffusion Policy 的行为克隆。数据只有专家演示，损失只有去噪均方误差 $\|\epsilon-\epsilon_\theta\|^2$。这一段没有环境奖励，也不学习价值函数。网络学会的是：给定观测，从纯噪声出发，逐步生成一段接近专家的动作。演示覆盖有限或本身次优时，策略就停在模仿所能达到的水平上。

第二段才是 PPO。演示退出训练集，数据改为当前策略与环境交互的轨迹，用完即弃。优化目标换成折扣回报。网络仍是第一段的去噪模型；通常冻结前面的去噪步，只更新最后若干个去噪步。代码里可以为第二段加上拉回预训练动作的行为克隆项，论文默认关闭。附录中的从零训练跳过第一段、只做 PPO，用作对照。

第二段使用 PPO，需要可计算的对数概率。高斯策略直接给出 $\log\pi(\mathbf{a}\mid\mathbf{s})$。扩散策略送入环境的是干净动作 $\mathbf{a}^{0}$，它的条件分布是把去噪链的中间变量积分掉之后的边缘

$$
\pi_\theta(\mathbf{a}^{0}\mid\mathbf{s})
=\int p_\theta(\mathbf{a}^{0:K}\mid\mathbf{s})\,\mathrm{d}\mathbf{a}^{1:K},
$$

这个积分没有闭式。单次去噪是高斯，对数概率有闭式，PPO 的似然比就用它来计算。同一控制周期的优势只估计一次，再分给生成 $\mathbf{a}^{0}$ 的每一次去噪。$\gamma_{\mathrm{env}}$ 只沿控制周期折扣。

| | 第一段：行为克隆 | 高斯策略上的 PPO | 第二段：DPPO 微调 |
|--|------------------|------------------|-------------------|
| 数据 | 固定演示，可反复用 | 当前策略采样，用完即弃 | 当前策略采样，用完即弃；不再读演示 |
| 损失 | 去噪均方误差 | 环境回报上的裁剪目标 | 环境回报上的裁剪目标 |
| 写入梯度的密度 | — | 最终动作的对角高斯 | 每一次去噪的高斯 |
| 探索发生的位置 | 反向过程的随机项，不按回报更新 | 最终动作 | 每一次去噪，同时被第一段学到的去噪方向拉回演示附近 |

---

## 二、控制周期与动作序列

扩散的对象是长度 $T_p$ 的动作序列（`horizon_steps`）。环境执行其中前 $T_a\le T_p$ 步（`act_steps`），然后重新观测、再规划。下面的环境步 $t$ 指这样一次规划与执行，单步动作维数是 $D_a$。观测写成 $\mathbf{s}_t$；图像观测时它就是 Diffusion Policy 笔记里的 $\mathbf{O}_t$。

去噪指标 $k$ 从 $K$ 到 $0$：$k=K$ 是纯噪声，$k=0$ 是送去执行的序列。$\mathbf{a}_t^{k}\in\mathbb{R}^{T_p\times D_a}$。从 $\mathbf{a}_t^{k+1}$ 抽到 $\mathbf{a}_t^{k}$ 称为一个去噪步。一次控制周期含 $K$ 个去噪步，然后环境前进一步：

```text
s_t 已知，抽出 a_t^K ~ N(0, I)
    a_t^K → a_t^{K-1} → … → a_t^1 → a_t^0
执行 a_t^0 的前 T_a 步
环境返回奖励 R_t 与下一观测 s_{t+1}
```

第 $k$ 个去噪步的条件分布是各向同性高斯。网络 $\epsilon_\theta$ 在较噪的一端 $\mathbf{a}_t^{k+1}$ 上预测噪声，均值 $\mu_\theta$ 由噪声日程和这个预测写出（DDPM 的系数见 Diffusion Policy 笔记第 2.4.4 节）：

$$
\pi_\theta(\mathbf{a}_t^{k}\mid\mathbf{a}_t^{k+1},\mathbf{s}_t)
=\mathcal{N}\!\big(
\mathbf{a}_t^{k};\,
\mu_\theta(\mathbf{a}_t^{k+1},\mathbf{s}_t,k),\,
\sigma_{k+1}^{2}I
\big).
$$

$\sigma$ 由日程给出。DDIM 用更少的去噪步完成同一条链，方差前多一个系数 $\eta$。$\eta=1$ 时核仍是非退化高斯，训练用它，对数密度有定义，同时留下探索。$\eta=0$ 时更新变成确定性映射，评估用它。

---

## 三、去噪链上的策略梯度

环境目标是折扣回报

$$
\mathcal{J}(\theta)
=\mathbb{E}\!\left[\sum_{t\ge 0}\gamma_{\mathrm{env}}^{t}\,R(\mathbf{s}_t,\mathbf{a}_t^{0})\right].
$$

策略梯度把 $\nabla_\theta\log\pi_\theta(\mathbf{a}\mid\mathbf{s})$ 乘上该动作的优势（PPO 笔记第二节）。环境动作 $\mathbf{a}_t^{0}$ 的条件分布就是第一节的边缘 $\pi_\theta(\mathbf{a}^{0}\mid\mathbf{s})$，对中间变量 $\mathbf{a}^{1:K}$ 的积分没有闭式。

采样得到的是整条链，而不只是送入环境的 $\mathbf{a}_t^{0}$。链的联合密度按去噪步分解，开头的纯噪声与参数无关：

$$
p_\theta(\mathbf{a}_t^{0:K}\mid\mathbf{s}_t)
=\mathcal{N}(\mathbf{a}_t^{K};0,I)
\prod_{k=0}^{K-1}
\pi_\theta(\mathbf{a}_t^{k}\mid\mathbf{a}_t^{k+1},\mathbf{s}_t).
$$

对数对 $\theta$ 求导，标准正态项消失，剩下每个去噪步的对数密度：

$$
\nabla_\theta\log p_\theta(\mathbf{a}_t^{0:K}\mid\mathbf{s}_t)
=\sum_{k=0}^{K-1}
\nabla_\theta\log\pi_\theta(\mathbf{a}_t^{k}\mid\mathbf{a}_t^{k+1},\mathbf{s}_t).
$$

环境转移 $P(\mathbf{s}_{t+1}\mid\mathbf{s}_t,\mathbf{a}_t^{0})$ 的密度不含 $\theta$，与初始噪声一样，在对 $\theta$ 求导时消失。轨迹对数概率的参数梯度因此只剩各次去噪。以 $K=3$ 为例，一次控制周期为

```text
a^3 ~ N(0, I)
a^2 ~ π_θ(· | a^3, s_t)
a^1 ~ π_θ(· | a^2, s_t)
a^0 ~ π_θ(· | a^1, s_t)     送入环境
s_{t+1} ~ P(· | s_t, a^0)
```

对数导数是这三次条件分布的对数梯度之和。优势 $A_t$ 只评价 $\mathbf{a}_t^{0}$。$\gamma_{\mathrm{denoise}}=1$ 时，三次去噪共用这一个 $A_t$，样本上的梯度为

$$
\begin{aligned}
&\nabla_\theta\log\pi_\theta(\mathbf{a}^{2}\mid\mathbf{a}^{3},\mathbf{s}_t)\,A_t
+\nabla_\theta\log\pi_\theta(\mathbf{a}^{1}\mid\mathbf{a}^{2},\mathbf{s}_t)\,A_t\\
&+\nabla_\theta\log\pi_\theta(\mathbf{a}^{0}\mid\mathbf{a}^{1},\mathbf{s}_t)\,A_t.
\end{aligned}
$$

$A_t>0$ 时，每一项提高对应条件分布在已采样点上的概率：给定 $\mathbf{a}^{3}$ 更倾向于这次的 $\mathbf{a}^{2}$，给定 $\mathbf{a}^{2}$ 更倾向于这次的 $\mathbf{a}^{1}$，给定 $\mathbf{a}^{1}$ 更倾向于这次的 $\mathbf{a}^{0}$。三个因子同时增大，以后的去噪链更常沿这次的路径走到得分较高的 $\mathbf{a}^{0}$。$A_t<0$ 时三个因子同时减小。损失只含这些高斯条件分布的对数概率，不含边缘 $\log\pi_\theta(\mathbf{a}^{0}\mid\mathbf{s})$。去噪步数为 $K$ 时，从 $\mathbf{a}^{K}$ 到 $\mathbf{a}^{0}$ 共 $K$ 项，每一项都乘以同一个 $A_t$。

单个去噪步上，高斯的对数密度是

$$
\log\pi_\theta
=-\frac{1}{2\sigma^{2}}\|\mathbf{a}-\mu_\theta\|^{2}
-\frac{d}{2}\log(2\pi\sigma^{2}).
$$

日程给定的 $\sigma$ 不含 $\theta$，第二项的梯度为零。剩下

$$
\nabla_\theta\log\pi_\theta
=\frac{1}{\sigma^{2}}(\mathbf{a}-\mu_\theta)^{\top}\nabla_\theta\mu_\theta.
$$

$\mu_\theta$ 由 $\epsilon_\theta$ 仿射得到，上式对 $\theta$ 的导数只通过 $\epsilon_\theta$。这个 $\epsilon_\theta$ 就是第一段行为克隆得到的噪声预测网络，行为克隆使它接近注入的噪声。微调改变的是单次去噪高斯 $\pi_\theta(\mathbf{a}^{k}\mid\mathbf{a}^{k+1},\mathbf{s}_t)=\mathcal{N}(\mu_\theta,\sigma^{2}I)$ 的中心 $\mu_\theta$，而不是边缘 $\pi_\theta(\mathbf{a}^{0}\mid\mathbf{s})$。$\sigma$ 由日程固定。优势为正时，$\mu_\theta$ 靠近该去噪步已抽到的 $\mathbf{a}^{k}$，该高斯在此点的密度上升；优势为负时，$\mu_\theta$ 离开 $\mathbf{a}^{k}$，密度下降。

用概率流 ODE 数值估计边缘 $\log\pi(\mathbf{a}^{0}\mid\mathbf{s})$ 再反传，是另一条路线（论文附录 C.6）。它在 Hopper 上能改进预训练策略，在 Can 上成功率掉到零，而且必须穿过全部去噪步。DPPO 使用上面按去噪步分解的梯度。

---

## 四、两层马尔可夫决策过程

第三节的和式已经是策略梯度。把它套进 PPO，需要明确每个去噪步的状态、动作和奖励。一次控制周期内部再套 $K$ 个去噪步：

| | 第 $k$ 个去噪步 |
|--|-----------|
| 状态 | $(\mathbf{s}_t,\mathbf{a}_t^{k+1})$：观测不变，带上一个去噪步留下的、仍较噪的序列 |
| 动作 | $\mathbf{a}_t^{k}$：这一去噪步抽到的更干净的序列 |
| 奖励 | $k>0$ 时为 $0$；$k=0$ 时为环境奖励 $R(\mathbf{s}_t,\mathbf{a}_t^{0})$ |
| 转移 | $k>0$ 时只把刚抽到的序列写进下一状态；$k=0$ 时环境推进到 $\mathbf{s}_{t+1}$，并重新抽取 $\mathbf{a}_{t+1}^{K}\sim\mathcal{N}(0,I)$ |

$k=K$ 的纯噪声由转移抽样，网络从 $\mathbf{a}^{K}$ 开始才行动，每个控制周期行动 $K$ 次。普通策略梯度对单一时间指标求和，求和里的一步是去噪步，由 $(t,k)$ 标记。论文把它排成

$$
\bar{t}(t,k)=tK+(K-k-1).
$$

$\bar{t}$ 随 $t$ 增大；同一 $t$ 内，$k$ 从 $K-1$ 降到 $0$ 时 $\bar{t}$ 也增大。以 $K=3$ 为例，先走完当前周期从噪到净的三次转移，再进入下一周期：

| $\bar{t}$ | $(t,k)$ | 转移 |
|-----------|---------|------|
| $0$ | $(0,2)$ | $\mathbf{a}^{3}\to\mathbf{a}^{2}$ |
| $1$ | $(0,1)$ | $\mathbf{a}^{2}\to\mathbf{a}^{1}$ |
| $2$ | $(0,0)$ | $\mathbf{a}^{1}\to\mathbf{a}^{0}$，送入环境 |
| $3$ | $(1,2)$ | 下一周期的 $\mathbf{a}^{3}\to\mathbf{a}^{2}$ |

求和沿着 $\bar{t}$ 进行，形式与单层 MDP 相同。$\bar{t}$ 只规定这个顺序，每个去噪步并不是环境时间上的一拍。

以 $K=3$ 为例，一个控制周期内三个去噪步的当步奖励依次是 $0,0,R_t$。$\gamma_{\mathrm{env}}$ 只乘在相邻控制周期之间，三个去噪步的回报同为

$$
G_t=R_t+\gamma_{\mathrm{env}} R_{t+1}+\gamma_{\mathrm{env}}^{2} R_{t+2}+\cdots.
$$

每个控制周期只估计一次 $A_t$，三个去噪步共用。下一节用 $\gamma_{\mathrm{denoise}}^{k}$ 把这份 $A_t$ 分给第 $k$ 个去噪步。

若去噪步也乘这同一个折扣，记为 $\gamma$，从周期 $t$ 的第一个去噪步起，回报变成

$$
\gamma^{2} R_t+\gamma^{5} R_{t+1}+\gamma^{8} R_{t+2}+\cdots.
$$

相邻两次环境奖励之间隔着 $\gamma^{3}$。一般的 $K$ 下，间隔是 $\gamma^{K}$，有效时域约为原来的 $K$ 倍，策略梯度的方差增大（Psenka et al., 2023）。

Black et al. (2023) 把一条去噪链写成 MDP，奖励放在链的末端。控制任务在每个控制周期生成一条这样的链。

---

## 五、优势估计

### 5.1 只依赖观测的价值函数

Critic 是 $\tilde V_\phi(\mathbf{s}_t)$，输入不含带噪动作。实现即 `critic(obs)`。控制周期上的 TD 残差与 GAE 和 PPO 笔记第七节相同：

$$
\delta_t=R_t+\gamma_{\mathrm{env}}V(\mathbf{s}_{t+1})-V(\mathbf{s}_t),
\qquad
A_t=\delta_t+(\gamma_{\mathrm{env}}\lambda)\,A_{t+1}.
$$

倒序只沿 $t$。终止步的自举项乘 $1-\mathrm{terminated}$。价值回归的目标是 $R_t=A_t+V(\mathbf{s}_t)$，与 CleanRL 一致。

论文比较过三种优势：只依赖 $\mathbf{s}_t$、依赖 $(\mathbf{s}_t,\mathbf{a}^{1})$、以及 $Q(\mathbf{s}_t,\mathbf{a}^{0})$。Hopper 上后两种在微调中奖励更高；Can、Square 和更长的操作任务上，只读观测更稳。动作序列维数高，扩散采样又随机，让价值函数记住某个动作的值更难。$Q$ 的误差会直接改噪声网络，稀疏奖励下成功率容易从预训练水平掉下去。

### 5.2 去噪步上的优势权重

$$
\hat A_{t,k}=\gamma_{\mathrm{denoise}}^{k}\,A_t.
$$

$k=0$ 的权重是 $1$。$k$ 越大，这个去噪步越噪，权重越小。

$\gamma_{\mathrm{denoise}}=1$ 时，各去噪步权重相同，第三节的和式原样成立。$\gamma_{\mathrm{denoise}}<1$ 时，更新偏向末端、即将执行的那几个去噪步；估计量随之成为对环境目标的重加权。中间奖励为零只说明各去噪步共享同一份回报；$\gamma_{\mathrm{denoise}}^{k}$ 决定这份回报里有多少用于第 $k$ 个去噪步，是单独的设计。

代码里下标 $i=0$ 是微调段中最噪的去噪步，$i=K'-1$ 是最干净的去噪步，权重为 $\gamma_{\mathrm{denoise}}^{K'-1-i}$（`gamma_denoising ** (ft_denoising_steps - i - 1)`）。消融里 Gym 取 $0.8$ 较有效，Can 几乎不敏感，Square 上 $0.5$ 略好。它和下一节「冻结前面的去噪步」是同一意图的两种做法：少改还很噪的去噪步，或直接不改。

---

## 六、裁剪目标

采样时固定旧网络，记下每个去噪步的对数概率。更新时用新网络对同一个去噪步重算。似然比

$$
r_{t,k}(\theta)
=\frac{\pi_\theta(\mathbf{a}_t^{k}\mid\mathbf{a}_t^{k+1},\mathbf{s}_t)}
{\pi_{\theta_{\mathrm{old}}}(\mathbf{a}_t^{k}\mid\mathbf{a}_t^{k+1},\mathbf{s}_t)}
$$

表示新网络有多想重复这个去噪步。裁剪目标和 PPO 笔记第五节相同，宽度可以随去噪步变化：

$$
\mathcal{L}^{\mathrm{CLIP}}
=\mathbb{E}\Big[
\min\big(
r_{t,k}\,\hat A_{t,k},\;
\mathrm{clip}(r_{t,k},\,1-\varepsilon_k,\,1+\varepsilon_k)\,\hat A_{t,k}
\big)
\Big].
$$

优势为正时，这个去噪步的概率最多升到旧概率的大约 $1+\varepsilon_k$ 倍；优势为负时，最多降到大约 $1-\varepsilon_k$ 倍。越过有利一侧的边界后，该样本对噪声网络的梯度为零。

$T_p\times D_a$ 上若使用联合密度，比值会非常陡。实现因此做了缩放：对数概率截断到 $[-5,2]$；只保留实际执行的前 $T_a$ 步（`reward_horizon`，默认等于 `act_steps`），预测了但未执行的后缀不进入比值；再对这些坐标的对数密度取均值后做指数。均值使比值的尺度不随维数和序列长度膨胀，梯度仍经 $\mu_\theta$ 进入 $\epsilon_\theta$。优势在 batch 内标准化。

裁剪宽度沿微调段由紧到松。$i$ 仍是上一节的下标，

$$
\varepsilon(i)
=\varepsilon_{\mathrm{base}}
+(\varepsilon-\varepsilon_{\mathrm{base}})
\frac{e^{\rho\, i/(K'-1)}-1}{e^{\rho}-1}.
$$

默认 $\varepsilon_{\mathrm{base}}=10^{-3}$，$\rho=3$。最噪的去噪步几乎不能离开旧核，最干净的去噪步使用任务选定的 $\varepsilon$。$\varepsilon$ 在 $\{0.1,0.01,0.001\}$ 中选取，使 batch 里落入裁剪区的样本大约占 $10\%$ 到 $20\%$，并取仍满足这个比例的最大值。稀疏奖励的操作任务对这个比例敏感。

总损失与 PPO 相同：

$$
L_{\mathrm{total}}
=-\mathcal{L}^{\mathrm{CLIP}}
-c_{\mathrm{ent}}\,H
+c_{\mathrm{vf}}\,L^{\mathrm{VF}}.
$$

$\sigma$ 固定时，高斯熵只依赖日程，不改变 $\epsilon_\theta$。DDIM 的 $\eta$ 若可学习，熵通过 $\eta$ 进入更新；默认不学。第二段的损失默认不含去噪误差。末端几步上的探索仍留在演示附近，动作形态由此保持，不必再加行为克隆项。新旧策略的距离用 Schulman 的 k3 估计 $\mathbb{E}[(r-1)-\log r]$。超过 `target_kl` 就结束本批后续轮次。论文使用的阈值是 $1$，实践中很少触发。

---

## 七、微调的结构约束

### 7.1 末端去噪步

预训练常用较多去噪步来拟合多峰演示。前面的去噪步决定落到哪一种做法、粗轨迹是什么形状；后面的去噪步决定即将执行的数值。微调时复制一份参数：原网络冻结，负责前面的去噪步；$\theta_{\mathrm{ft}}$ 只负责最后 $K'$ 个去噪步，并接受 PPO。显存和反传深度随 $K'$ 增长。

$K'$ 过小（例如 $3$）时，末端的可调范围不够，渐近成功率下降。操作任务上 $K'=10$ 综合较好。状态观测常用预训练 $20$ 步、微调最后 $10$ 步；像素与 Furniture-Bench 常用预训练 $100$ 步、以 DDIM 微调 $5$ 步。

### 7.2 标准差的下界

余弦日程把末端的 $\sigma$ 收到约 $10^{-4}$。这个宽度足够生成样本，对强化学习的探索则太窄；同时 $\log\pi$ 里的 $1/\sigma^{2}$ 会把梯度放大。两处下界因此分开：

| | 用在哪里 | 代码 | 量级 |
|--|----------|------|------|
| $\sigma_{\min}^{\mathrm{exp}}$ | 采样 $\mathbf{a}=\mu+\sigma z$ | `min_sampling_denoising_std` | $0.01$–$0.1$ |
| $\sigma_{\min}^{\mathrm{prob}}$ | 计算 $\log\pi$ | `min_logprob_denoising_std` | $0.1$ |

采样下界决定探索宽度。Can 上它几乎不影响最终成功率；Square 上大约需要 $0.1$，否则策略在覆盖很窄的样本上更新后坍缩。下界提高后，轨迹仍贴着预训练的几种做法，只是每种做法周围更宽。似然下界可以高于采样时实际使用的标准差，比值里的 $\sigma$ 服务于数值稳定。

### 7.3 网络与序列长度

MLP 微调更稳。全连接把整个 $T_p$ 绑在一起，微调时若 $T_a<T_p$ 容易不稳定，因而两者取齐。UNet 沿动作序列做卷积，可以预训练用较长的 $T_p$、微调用较短的 $T_a$，例如 $16$ 与 $8$。较长的预测利于模仿阶段的时序一致；较短的执行缩短一次决策里的开环段。像素实验用 ViT 编码器加 MLP 动作头。

一条 PPO 样本是「一个控制周期 $\times$ 微调段中的一个去噪步」。批量大约是同任务高斯 PPO 的 $K'$ 倍。Robomimic 上更接近高斯 PPO 的批量有时样本效率更好。一批轨迹用完即弃，微调阶段不再读取专家演示。

---

## 八、训练流程

两段串行。第一段收敛并保存权重之后，第二段才采样环境。

**第一段，模仿。** 只使用演示。每个样本取观测与专家动作序列，随机抽取噪声水平 $k$ 和噪声 $\epsilon$，最小化 $\|\epsilon-\epsilon_\theta\|^2$。噪声日程为余弦。这一段不接触环境，不计算优势，也不做 PPO。得到的网络记为预训练策略。

**第二段，微调。** 复制一份参数作为 $\theta_{\mathrm{ft}}$，原来的网络冻结。此后损失里只有 PPO 与价值回归，演示不再进入。循环为：

```text
1. 用当前策略采样
   每个控制周期：
     从 a^K ~ N(0, I) 开始
     前面的去噪步用冻结的预训练网络，最后 K' 个去噪步用 θ_ft
     a ← μ_θ + σ z，σ 不低于 σ_exp_min
     执行 a^0 的前 T_a 步
     记录观测、整条链、环境奖励、log π_old
   评估时 DDIM 取 η = 0

2. 沿控制周期倒序做 GAE，得到 A_t 与 R_t = A_t + V(s_t)

3. 将（控制周期，微调段中的一个去噪步）摊平，重复若干轮：
     重算该去噪步的 log π_θ，σ 不低于 σ_prob_min
     优势取 γ_denoise^k A_t，并在 batch 内标准化
     按第六节计算裁剪损失，ε 从噪到净由紧到松
     价值网络回归 R_t
     梯度裁剪后更新 θ_ft 与价值网络
     k3 超过 target_kl 则结束本批

4. 丢弃这批轨迹，回到步骤 1
```

第二段的前若干迭代可以只更新价值网络。采样噪声下界和 $K'$ 的退火在代码里有接口，论文的配置没有使用。

---

## 九、流形上的探索

D3IL 的 Avoid 把这一点显出来：机械臂绕过障碍到达对侧，演示里有多条路径，微调的稀疏奖励只奖励最上方的一条。高斯和 GMM 也用 PPO，动作序列长度同为 $T_a=4$。

每个去噪步都加入噪声，覆盖变宽；已经学到的去噪方向同时把样本拉回演示附近。经过多个去噪步之后，轨迹在原有路径周围展开，绕障的形状还在。高斯噪声加在最终动作上，轨迹更散；GMM 盖住的范围更窄。若中间的去噪步改为确定性、只在最后一个去噪步加噪声，Avoid 上的路径失去这个形状，Can 和 Square 的收敛也更慢。

这依赖于预训练已经盖住有用的做法。奖励要求的行为落在演示范围之外时，这种探索偏保守：Furniture-Bench 的 Lamp、低随机性设定里，高斯略好。从零训练、没有演示流形时，论文也没有看到扩散相对高斯的明显优势。

微调过程中，动作分布沿去噪逐步收拢：粗结构留在较早的去噪步，末端继续修数值。往执行动作上持续加噪声时，高斯和 GMM 的成功率掉下去；DPPO 在至少保留四步去噪时还在。$T_a$ 增到 $16$ 时，高斯和 GMM 可以完全不提升。仿真到真机时，也用这个性质在仿真动作上加噪声，模拟不精确的底层控制器。

微调之后再加动作噪声，DPPO 的成功率下降更慢，并能从更大的初始区域走到奖励指定的路径。One-leg 上将仿真策略直接放到真机，$10\,\mathrm{Hz}$、不用真实数据：DPPO 为 $16/20$。高斯在仿真中达到 $88\%$，真机为 $0/20$。给高斯加上拉回预训练策略的行为克隆项后，仿真 $53\%$，真机 $50\%$。

---

## 十、实验对照

细节与种子见论文第五节。和算法选择有关的对照是：

| 对照 | 现象 |
|------|------|
| DQL、IDQL、DIPO、QSM，以及奖励加权、优势加权回归 | Gym 稠密奖励上 IDQL、DIPO 接近 DPPO。Robomimic 稀疏奖励上 DPPO 更稳，Transport 上差距最大。依赖 $Q$ 的方法常在微调初期掉点 |
| RLPD、Cal-QL、IBRL（微调时仍用专家数据） | HalfCheetah 上 off-policy 的样本效率更高。Kitchen 与 Robomimic 上 DPPO 的最终成功率更高。DPPO 的微调不再使用专家数据，因而依赖预训练是否盖住多种做法 |
| 高斯、GMM，同为 PPO | 状态下 Lift、Can 都能接近 $100\%$，Square 与 Transport 上 DPPO 更高，Transport 超过 $90\%$。像素下高斯在 Transport 上停在预训练的 $0\%$；DPPO 超过 $50\%$ |
| Furniture-Bench 上的高斯 MLP | 50 条仿真演示、按阶段给稀疏奖励。六种设定里 DPPO 都有提升。高斯在中等随机性下几乎归零。真机见第九节 |

从零训练（Gym，$T_a=1$，$10$ 步去噪）的最终回报与高斯相当，墙钟大约慢六倍，时间主要耗在多步采样。DPPO 的使用方式是先模仿、再微调。样本效率低于把专家数据留在回放池中的 off-policy 方法；换到的是长时程、稀疏奖励操作上的稳定。

---

## 十一、关键超参数

| 参数 | 代码 | 常见取值 | 作用 |
|------|------|----------|------|
| 去噪步数 $K$ | `denoising_steps` | 状态 20，像素 / 家具 100 | 预训练与采样的总步数 |
| 微调步数 $K'$ | `ft_denoising_steps` | 10，或 DDIM 的 5 | 只更新链的末端 |
| $T_p$ / $T_a$ | `horizon_steps` / `act_steps` | MLP 二者相等；UNet 可用 16 / 8 | 预测长度与执行长度 |
| $\gamma_{\mathrm{denoise}}$ | `gamma_denoising` | 0.8–1 | 降低较噪去噪步的优势权重 |
| $\sigma_{\min}^{\mathrm{exp}}$ | `min_sampling_denoising_std` | 0.01–0.1 | 采样时的探索宽度 |
| $\sigma_{\min}^{\mathrm{prob}}$ | `min_logprob_denoising_std` | 0.1 | 计算 $\log\pi$ 时的方差下界 |
| $\varepsilon$ | `clip_ploss_coef` | 0.1、0.01 或 0.001 | 最干净去噪步的裁剪宽度 |
| $\varepsilon_{\mathrm{base}}$ | `clip_ploss_coef_base` | $10^{-3}$ | 最噪去噪步的裁剪宽度 |
| $\gamma_{\mathrm{env}},\lambda$ | `gamma`, `gae_lambda` | 与 PPO 相同 | 控制周期上的 GAE |
| DDIM 的 $\eta$ | `eta` | 训练 1，评估 0 | 训练时保留高斯噪声 |
| 批量 | `batch_size` | 约为高斯 PPO 的 $K'$ 倍 | 样本同时覆盖 $t$ 与 $k$ |

---

## 十二、参考文献与阅读顺序

1. Ren et al., *Diffusion Policy Policy Optimization*, arXiv:2409.00588, 2024. 代码：[irom-princeton/dppo](https://github.com/irom-princeton/dppo)
2. Chi et al., *Diffusion Policy*, RSS 2023. [`diffusion_policy_notes.md`](diffusion_policy_notes.md)
3. Schulman et al., *Proximal Policy Optimization Algorithms*, 2017. [`ppo_notes.md`](ppo_notes.md)
4. Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, 2016.
5. Black et al., *Training Diffusion Models with Reinforcement Learning*（DDPO）, arXiv:2305.13301, 2023.
6. Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
7. Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021.
8. Wang et al., *Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning*（Diffusion-QL）, 2022.
9. Hansen-Estruch et al., *IDQL*, 2023.
10. Psenka et al., *Q-Score Matching*, arXiv:2312.11752, 2023.
11. Yang et al., *DIPO*, 2023.

建议先读 Diffusion Policy 笔记第 2.4 与第 3 节，以及 PPO 笔记第五、七节，再读本文第三至七节。对照实现时看 `get_logprobs` 与 `diffusion_ppo.py` 的 `loss`。

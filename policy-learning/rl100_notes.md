# RL-100 算法笔记

下文对应 Lei et al., *RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning*（[arXiv:2510.14830](https://arxiv.org/pdf/2510.14830) v3；*Science Robotics* 11(116), aed6267, 2026）。期刊版摘要另记折盒任务与 $1000/1000$ 次成功；下文的任务划分与计数按 arXiv v3。实现见 [Lei-Kun/RL-100](https://github.com/Lei-Kun/RL-100)。

去噪一步的高斯对数概率写在 `rl_100/unidpg/diffusion_policy/diffusers_patch/ddim_with_logprob.py`。离线更新是 `BehaviorProximalPolicyOptimization.update_distribution`，在线更新是同文件的 `dp_align_update_no_share`，类定义在 `rl_100/unidpg/uni_ppo.py`。IQL 的期望分位与 $Q-V$ 在 `rl_100/unidpg/critic.py`。行为策略是否前进由 `RL-100/train.py` 的 `finetune_dp3` 根据动力学模型上的 $Q$ 均值决定，复制权重的是 `rl_100/unidpg/ppo.py` 的 `set_old_policy`。一致性蒸馏在 `rl_100/policy/rl100_3d.py`。

$\epsilon$-prediction 与 DDIM 见 [`diffusion_policy_notes.md`](diffusion_policy_notes.md) 第四、八节；裁剪目标与 GAE 见 [`ppo_notes.md`](ppo_notes.md) 第五、七节；把裁剪目标写到每一步去噪上见 [`dppo_notes.md`](dppo_notes.md) 第三至六节。本文沿用这些推导，只记录 RL-100 的改动：离线优势的来源、行为策略的替换条件、一步学生的训练，以及真机数据如何并回。

---

## 一、算法定位

RL-100 是扩散视觉运动策略上的真机后训练。网络仍以观测为条件，沿去噪链生成动作；优化目标由拟合演示改为折扣回报。模仿、离线强化学习与在线强化学习使用同一个裁剪目标。三者的差别在于优势的估计方式、似然比分母中行为策略的更新时机，以及该阶段是否采集新轨迹。

训练按成本分成三段，一段结束才进入下一段。

第一段是行为克隆。数据是遥操作，损失是去噪均方误差。这一段没有环境奖励，也不学习价值函数。它给出低方差的初始去噪方向，后面的策略梯度在这个方向附近改概率。

第二段是迭代离线强化学习。每一轮从当前的模仿策略出发，先做离线裁剪更新，再用更新后的策略在真机上展开。轨迹并入数据集后重新做行为克隆，其结果作为下一轮的起点。第一轮的数据集仍只有遥操作。这一段贡献成功率的大部分增量。人只标注成功与否，并遵守动作限制。

第三段是在线、on-policy 的微调。优势改用 GAE，处理离线阶段还留下的失败。各任务的在线采集时数不同，七个任务的平均与离线采集同量级，见第十二节。

部署时另有一个学生网络，由多步扩散教师蒸馏为一次前向。学生与教师可以并存。执行时可以使用多步 DDIM，也可以使用一步策略。

| | 行为克隆 | DPPO 的在线微调 | RL-100 |
|--|----------|-----------------|--------|
| 数据 | 固定演示 | 当前策略采样，使用后丢弃，不再读取演示 | 演示保留在数据集中；策略轨迹并入后可重新做行为克隆 |
| 损失 | $\|\epsilon-\epsilon_\theta\|^2$ | 环境回报上的裁剪目标 | 同一裁剪目标；离线与在线只换优势 |
| 优势 | — | 环境步上的 GAE，可再乘 $\gamma_{\mathrm{denoise}}^{k}$ | 离线为 $Q-V$；在线为 GAE；一条链上的去噪步共用 |
| 行为策略 | — | 每批轨迹采完就换成当前网络 | 当前网络持续更新；分母里的网络只在模型评估变好时被替换 |
| 执行 | $K$ 步 DDIM | $K$ 步 DDIM，评估时 $\eta=0$ | $K$ 步教师，或一次前向的学生 |

仓库在同一套入口上还接了 flow 策略、flow 的一步蒸馏，以及可选的 IDQL 式评估（`unio4.idql_eval`，重复采样 $128$ 次）。论文正文的训练目标是扩散策略上的裁剪目标，加上一致性蒸馏。

---

## 二、环境步、动作块与去噪链

### 2.1 决策间隔

环境是 $\langle\mathcal{S},\mathcal{A},P,R,\gamma\rangle$。策略在 $s_t$ 选 $a_t$，最大化 $\sum_t\gamma^t R(s_t,a_t)$。视觉运动策略的条件是观测 $o_t$，含最近 $n_o$ 帧（论文取 $n_o=2$）的点云或图像，以及本体感觉 $q_t$。条件向量

$$
c_t=\big[\phi(o_i,q_i)\big]_{i=t-n_o+1}^{t}.
$$

一次决策的动作有两种形状。单步控制时 $a_t\in\mathbb{R}^{d_a}$，执行后立即再观测。动作块控制时一次决策给出

$$
a_t=[u_t,\ldots,u_{t+n_c-1}]\in\mathbb{R}^{n_c d_a},
$$

$n_c$ 通常为 $8$–$16$，随后 $n_c$ 个环境步开环执行。块内奖励先折扣再相加，

$$
R^{\mathrm{chunk}}_t=\sum_{j=0}^{n_c-1}\gamma^j R_{t+j},
$$

下一次决策的折扣是 $\gamma^{n_c}$。单步是 $n_c=1$。价值与 GAE 都按这个决策间隔计算，见第七节。

### 2.2 任务与执行频率

需要快速闭环的任务使用单步控制：Dynamic Push-T、Agile Bowling、Pouring。需要协调或轨迹平滑的任务使用动作块：Dynamic Unscrewing、Soft-towel Folding，以及 Orange Juicing 的放置与取出。骨干相同，输出头的维数不同。

| 任务 | 决策 | 本体 | 点云 | 动作 |
|------|------|------|------|------|
| Dynamic Push-T | 单步 | UR5，$6$ | $512\times 3$ | $2$，末端 $\Delta(x,y)$ |
| Agile Bowling | 单步 | 同上 | $512\times 3$ | $2$ |
| Pouring | 单步 | Franka $7$ + LeapHand $16$ | $512\times 3$ | $22$，臂的目标位姿与手关节 |
| Dynamic Unscrewing | 动作块 | 同上 | $1024\times 3$ | $23$，另加一维是否允许臂运动 |
| Soft-towel Folding | 动作块 | 双臂各 $7{+}1$ | $1024\times 3$ | $16$，双臂目标位姿与夹爪 |
| Orange Juicing | 动作块 | xArm $6{+}1$ | $1024\times 3$ | $8$，目标位姿、夹爪、任务结束 |

点云由 RealSense L515 的深度反投影到机器人基座，工作空间裁剪后再做最远点采样。部署时点云更新高于 $25\,\mathrm{Hz}$。策略指令 $30\,\mathrm{Hz}$：UR5 经 RTDE 插值到 $125\,\mathrm{Hz}$，xArm 位置线性插值、姿态 slerp 后 $200\,\mathrm{Hz}$ 执行，Franka 底层 $1000\,\mathrm{Hz}$。

### 2.3 去噪链

去噪链套在每一次决策里面。环境步 $t$ 写下标，去噪链上的量写上标，如 $a^{\tau_k}$、$s^k$。子采样日程

$$
\tau_K>\tau_{K-1}>\cdots>\tau_1,
$$

$K\ll T$（训练日程 $T$ 约 $50$–$1000$，采样 $K$ 约 $5$–$10$；真机效率对照里的 DDIM 是 $10$ 步）。$\tau_k$ 取自训练日程，下标越大噪声越强，链从 $\tau_K$ 走到干净动作。DPPO 笔记将同一次决策的去噪步记为 $k=K,\ldots,0$。

一次决策内部的子 MDP：

| | 第 $k$ 个去噪步 |
|--|----------------|
| 状态 | $s^k=(a^{\tau_k},\tau_k,c_t)$，$a^{\tau_K}\sim\mathcal{N}(0,I)$ |
| 动作 | $a^{\tau_{k-1}}$，从这一步的高斯子策略抽出 |
| 转移 | 把抽出的序列写入 $s^{k-1}$，观测 $c_t$ 不变 |
| 奖励 | $k>1$ 时为 $0$；走到干净动作时才接到环境奖励 |

$K$ 次高斯转移产生一个送去执行的动作。环境转移只依赖这个干净动作。

---

## 三、随机 DDIM 的对数概率

### 3.1 对数概率

PPO 需要 $\log\pi_\theta(a^{\tau_{k-1}}\mid a^{\tau_k},c_t)$。确定性 DDIM 的转移是 Dirac 分布，对数密度没有定义，训练使用带方差的随机形式。$\bar\alpha$ 是累积乘积；DDIM 原文把这个量记成 $\alpha_t$。$t>m$ 为子采样日程上相邻的噪声下标。

网络由 $\epsilon_\theta(a^{t},t,c_t)$ 还原干净动作

$$
\hat a^{0}
=\frac{a^{t}-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(a^{t},t,c_t)}{\sqrt{\bar\alpha_t}}.
$$

一步转移是

$$
\begin{aligned}
\mu_\theta(a^{t},t\to m)
&=\sqrt{\bar\alpha_m}\,\hat a^{0}
+\sqrt{1-\bar\alpha_m-\sigma_{t\to m}^{2}}\;\epsilon_\theta(a^{t},t,c_t),\\
a^{m}
&=\mu_\theta+\sigma_{t\to m}\,\varepsilon,
\qquad\varepsilon\sim\mathcal{N}(0,I).
\end{aligned}
$$

$\sigma_{t\to m}>0$ 时这一步是各向同性高斯

$$
\pi_\theta(a^{m}\mid a^{t},t\to m)
=\mathcal{N}\big(\mu_\theta,\,\sigma_{t\to m}^{2} I\big),
$$

$$
\log\pi_\theta
=-\frac{1}{2\sigma_{t\to m}^{2}}\|a^{m}-\mu_\theta\|^2+C,
\qquad
C=-\frac{d}{2}\log(2\pi\sigma_{t\to m}^{2}).
$$

$\sigma_{t\to m}$ 由日程与 $\eta$ 给定、不依赖 $\theta$ 时，$C$ 的梯度为零，$\mu_\theta$ 对 $\epsilon_\theta$ 是仿射，策略梯度只经过噪声网络。$\eta$ 由网络输出时，$\sigma_{t\to m}$ 进入 $C$、$1/(2\sigma_{t\to m}^{2})$ 以及 $\mu_\theta$ 中的 $\sqrt{1-\bar\alpha_m-\sigma_{t\to m}^{2}}$。

### 3.2 由前向边际推导

Diffusion Policy 笔记第 2.4.4 节的前向边际是

$$
q(a^{s}\mid a^{0})
=\mathcal{N}\!\big(\sqrt{\bar\alpha_s}\,a^{0},\,(1-\bar\alpha_s)I\big),
$$

即 $a^{s}=\sqrt{\bar\alpha_s}\,a^{0}+\sqrt{1-\bar\alpha_s}\,\epsilon_s$，$\epsilon_s\sim\mathcal{N}(0,I)$。训练损失只依赖这些边际。采样的一步从噪声水平 $t$ 走到 $m$，$t>m$，中间的训练下标可以空着。下面四步给出这一步的核。

#### （1）证明 $q(a^{m}\mid a^{t},a^{0})$

给定 $a^{0}$，$a^{t}$ 中的噪声是

$$
\epsilon_t
=\frac{a^{t}-\sqrt{\bar\alpha_t}\,a^{0}}{\sqrt{1-\bar\alpha_t}}\sim\mathcal{N}(0,I).
$$

$\epsilon_m$ 与 $\epsilon_t$ 在给定 $a^{0}$ 时同为标准正态，联合由相关系数 $c=\mathrm{Corr}(\epsilon_m,\epsilon_t)$ 决定。取 $c\in[0,1]$，使 $a^{m}$ 沿用 $a^{t}$ 的噪声方向，并用与 $\epsilon_t$ 独立的 $\varepsilon\sim\mathcal{N}(0,I)$ 写成

$$
\epsilon_m
=c\,\epsilon_t+\sqrt{1-c^{2}}\,\varepsilon.
$$

第二项的系数使 $\mathrm{Cov}(\epsilon_m)=c^{2}I+(1-c^{2})I=I$，故 $\epsilon_m\sim\mathcal{N}(0,I)$。$c=1$ 时两档共用同一噪声，$c=0$ 时相互独立。于是

$$
a^{m}
=\sqrt{\bar\alpha_m}\,a^{0}+\sqrt{1-\bar\alpha_m}\,\epsilon_m
$$

它满足 $m$ 档的前向边际。将 $\epsilon_m$ 代入后，与 $\epsilon_t$ 独立的 $\varepsilon$ 前的系数记为

$$
\sigma_{t\to m}
=\sqrt{1-\bar\alpha_m}\,\sqrt{1-c^{2}}.
$$

平方并移项得 $(1-\bar\alpha_m)c^{2}=1-\bar\alpha_m-\sigma_{t\to m}^{2}$。$c\in[0,1]$，故 $\epsilon_t$ 的系数是

$$
\sqrt{1-\bar\alpha_m}\,c
=\sqrt{1-\bar\alpha_m-\sigma_{t\to m}^{2}}.
$$

代回得

$$
a^{m}
=\sqrt{\bar\alpha_m}\,a^{0}
+\sqrt{1-\bar\alpha_m-\sigma_{t\to m}^{2}}\;\epsilon_t
+\sigma_{t\to m}\,\varepsilon.
$$

给定 $(a^{t},a^{0})$ 时 $\epsilon_t$ 已由 $a^{t}$ 确定，随机性只剩 $\varepsilon$，条件分布为下面的高斯。它即 DDIM 论文中的式 (7)：

$$
q(a^{m}\mid a^{t},a^{0})
=\mathcal{N}\!\left(
\sqrt{\bar\alpha_m}\,a^{0}
+\sqrt{1-\bar\alpha_m-\sigma_{t\to m}^{2}}\,
\frac{a^{t}-\sqrt{\bar\alpha_t}\,a^{0}}{\sqrt{1-\bar\alpha_t}},\;
\sigma_{t\to m}^{2} I
\right).
$$

$c^{2}\ge 0$ 要求 $0\le\sigma_{t\to m}\le\sqrt{1-\bar\alpha_m}$。

#### （2）证明 $q(a^{m}\mid a^{0})$ 不依赖 $\sigma_{t\to m}$

边际由条件分布对 $a^{t}$ 积分得到，

$$
q(a^{m}\mid a^{0})
=\int q(a^{m}\mid a^{t},a^{0})\,q(a^{t}\mid a^{0})\,\mathrm{d}a^{t}.
$$

$q(a^{t}\mid a^{0})=\mathcal{N}(\sqrt{\bar\alpha_t}\,a^{0},\,(1-\bar\alpha_t)I)$ 即

$$
a^{t}
=\sqrt{\bar\alpha_t}\,a^{0}+\sqrt{1-\bar\alpha_t}\,\epsilon_t,
\qquad
\epsilon_t\sim\mathcal{N}(0,I).
$$

条件均值中的分数就是该噪声，

$$
\frac{a^{t}-\sqrt{\bar\alpha_t}\,a^{0}}{\sqrt{1-\bar\alpha_t}}
=\epsilon_t.
$$

该高斯的密度是

$$
q(a^{t}\mid a^{0})
=(2\pi(1-\bar\alpha_t))^{-d/2}
\exp\Bigl(
-\frac{\|a^{t}-\sqrt{\bar\alpha_t}\,a^{0}\|^2}{2(1-\bar\alpha_t)}
\Bigr).
$$

指数中的平方写成 $\epsilon_t$，

$$
\frac{\|a^{t}-\sqrt{\bar\alpha_t}\,a^{0}\|^2}{1-\bar\alpha_t}
=\|\epsilon_t\|^2,
$$

故

$$
q(a^{t}\mid a^{0})
=(2\pi(1-\bar\alpha_t))^{-d/2}
\exp\bigl(-\tfrac12\|\epsilon_t\|^2\bigr).
$$

映射 $\epsilon_t\mapsto a^{t}$ 的雅可比矩阵是 $\sqrt{1-\bar\alpha_t}\,I$，行列式为 $\bigl(\sqrt{1-\bar\alpha_t}\bigr)^{d}=(1-\bar\alpha_t)^{d/2}$，

$$
\mathrm{d}a^{t}
=(1-\bar\alpha_t)^{d/2}\,\mathrm{d}\epsilon_t.
$$

归一化常数中的 $(1-\bar\alpha_t)^{-d/2}$ 与这个因子相乘为 $1$：

$$
\begin{aligned}
q(a^{t}\mid a^{0})\,\mathrm{d}a^{t}
&=(2\pi(1-\bar\alpha_t))^{-d/2}
\exp\bigl(-\tfrac12\|\epsilon_t\|^2\bigr)
\cdot(1-\bar\alpha_t)^{d/2}\,\mathrm{d}\epsilon_t\\
&=(2\pi)^{-d/2}
\exp\bigl(-\tfrac12\|\epsilon_t\|^2\bigr)\,\mathrm{d}\epsilon_t.
\end{aligned}
$$

因此对 $a^{t}$ 积分等于对 $\epsilon_t$ 取期望。条件分布里余下的随机性是与 $\epsilon_t$ 独立的 $\varepsilon\sim\mathcal{N}(0,I)$，被积的样本为

$$
a^{m}
=\sqrt{\bar\alpha_m}\,a^{0}
+\sqrt{1-\bar\alpha_m-\sigma_{t\to m}^{2}}\,\epsilon_t
+\sigma_{t\to m}\,\varepsilon.
$$

$\mathbb{E}[\epsilon_t]=\mathbb{E}[\varepsilon]=0$，故

$$
\begin{aligned}
\mathbb{E}[a^{m}\mid a^{0}]
&=\sqrt{\bar\alpha_m}\,a^{0}
+\sqrt{1-\bar\alpha_m-\sigma_{t\to m}^{2}}\,\mathbb{E}[\epsilon_t]
+\sigma_{t\to m}\,\mathbb{E}[\varepsilon]\\
&=\sqrt{\bar\alpha_m}\,a^{0}.
\end{aligned}
$$

$\mathrm{Cov}(\epsilon_t,\varepsilon)=0$。$z\sim\mathcal{N}(0,I)$ 时 $\mathrm{Var}(cz)=c^{2}I$，交叉项不出现：

$$
\begin{aligned}
\mathrm{Var}(a^{m}\mid a^{0})
&=(1-\bar\alpha_m-\sigma_{t\to m}^{2})\,\mathrm{Var}(\epsilon_t)
+\sigma_{t\to m}^{2}\,\mathrm{Var}(\varepsilon)\\
&\quad
+2\sigma_{t\to m}\sqrt{1-\bar\alpha_m-\sigma_{t\to m}^{2}}\,
\mathrm{Cov}(\epsilon_t,\varepsilon)\\
&=(1-\bar\alpha_m-\sigma_{t\to m}^{2})I+\sigma_{t\to m}^{2}I
=(1-\bar\alpha_m)I.
\end{aligned}
$$

$(\epsilon_t,\varepsilon)$ 联合为 $\mathcal{N}(0,I_{2d})$，$a^{m}$ 是它的仿射像，边际由上面的均值与方差确定：

$$
q(a^{m}\mid a^{0})
=\mathcal{N}\!\big(\sqrt{\bar\alpha_m}\,a^{0},\,(1-\bar\alpha_m)I\big).
$$

$\sigma_{t\to m}^{2}$ 在方差中相消。这是 DDIM 附录 Lemma 1：在根号有定义的范围内，改变 $\sigma_{t\to m}$ 不改变训练所用的边际。中间的训练下标因此不进入转移。

#### （3）证明 $\sigma_{t\to m}$ 取后验标准差

（1）中的 $\sigma_{t\to m}$ 只满足 $0\le\sigma_{t\to m}\le\sqrt{1-\bar\alpha_m}$。下面把它取成由 $m$ 走到 $t$ 的后验标准差。

$a^{m}$ 已服从 $q(a^{m}\mid a^{0})$。再加一层独立噪声 $\varepsilon'\sim\mathcal{N}(0,I)$，

$$
a^{t}=\sqrt{\gamma}\,a^{m}+\sqrt{1-\gamma}\,\varepsilon'.
$$

要让 $a^{t}$ 仍服从 $q(a^{t}\mid a^{0})=\mathcal{N}(\sqrt{\bar\alpha_t}\,a^{0},\,(1-\bar\alpha_t)I)$，取 $\gamma=\bar\alpha_t/\bar\alpha_m$。将 $a^{m}=\sqrt{\bar\alpha_m}\,a^{0}+\sqrt{1-\bar\alpha_m}\,\epsilon_m$ 代入。$\epsilon_m$ 与 $\varepsilon'$ 独立且均值为 $0$，

$$
\begin{aligned}
\mathbb{E}[a^{t}\mid a^{0}]
&=\sqrt{\gamma\bar\alpha_m}\,a^{0}
=\sqrt{\bar\alpha_t}\,a^{0},\\
\mathrm{Var}(a^{t}\mid a^{0})
&=\gamma(1-\bar\alpha_m)I+(1-\gamma)I
=(1-\gamma\bar\alpha_m)I
=(1-\bar\alpha_t)I.
\end{aligned}
$$

这一步的系数不读取 $a^{0}$，故 $q(a^{t}\mid a^{m},a^{0})=q(a^{t}\mid a^{m})$，

$$
q(a^{t}\mid a^{m})
=\mathcal{N}\!\big(\sqrt{\gamma}\,a^{m},\,(1-\gamma)I\big).
$$

贝叶斯公式

$$
q(a^{m}\mid a^{t},a^{0})
=\frac{q(a^{t}\mid a^{m})\,q(a^{m}\mid a^{0})}{q(a^{t}\mid a^{0})}.
$$

分母不含 $a^{m}$，后验正比于分子。记 $\beta=1-\gamma$。似然对 $a^{m}$ 配平方，

$$
\|a^{t}-\sqrt{\gamma}\,a^{m}\|^2
=\gamma\Bigl\|a^{m}-\frac{a^{t}}{\sqrt{\gamma}}\Bigr\|^2,
$$

$$
q(a^{t}\mid a^{m})
\propto
\exp\Bigl(
-\frac{\gamma}{2\beta}
\Bigl\|a^{m}-\frac{a^{t}}{\sqrt{\gamma}}\Bigr\|^2
\Bigr).
$$

指数与 $\mathcal{N}(a^{t}/\sqrt{\gamma},\,(\beta/\gamma)I)$ 相同，归一化常数不同，精度为 $\tau_1=\gamma/\beta$。先验 $q(a^{m}\mid a^{0})=\mathcal{N}(\sqrt{\bar\alpha_m}\,a^{0},\,(1-\bar\alpha_m)I)$ 的精度为 $\tau_2=1/(1-\bar\alpha_m)$。两个高斯核相乘，精度相加、均值按精度加权（Diffusion Policy 笔记附录 A.3）：

$$
\frac{1}{\sigma^{2}}
=\tau_1+\tau_2
=\frac{1}{1-\bar\alpha_m}+\frac{\gamma}{\beta},
\qquad
\sigma^{2}
=\frac{(1-\bar\alpha_m)\,\beta}{\beta+\gamma(1-\bar\alpha_m)}.
$$

分母 $\beta+\gamma(1-\bar\alpha_m)=1-\gamma+\gamma-\gamma\bar\alpha_m=1-\bar\alpha_t$，而 $\beta=1-\bar\alpha_t/\bar\alpha_m$，故

$$
\sigma^{2}
=\frac{1-\bar\alpha_m}{1-\bar\alpha_t}\left(1-\frac{\bar\alpha_t}{\bar\alpha_m}\right).
$$

后验均值

$$
\mu
=\sigma^{2}\Bigl(
\tau_1\cdot\frac{a^{t}}{\sqrt{\gamma}}
+\tau_2\cdot\sqrt{\bar\alpha_m}\,a^{0}
\Bigr).
$$

两项的系数是

$$
\sigma^{2}\cdot\frac{\tau_1}{\sqrt{\gamma}}
=\sigma^{2}\cdot\frac{\sqrt{\gamma}}{\beta}
=\frac{(1-\bar\alpha_m)\sqrt{\gamma}}{1-\bar\alpha_t},
\qquad
\sigma^{2}\cdot\tau_2\cdot\sqrt{\bar\alpha_m}
=\frac{\beta\sqrt{\bar\alpha_m}}{1-\bar\alpha_t}.
$$

$$
\mu
=\frac{\beta\sqrt{\bar\alpha_m}}{1-\bar\alpha_t}\,a^{0}
+\frac{(1-\bar\alpha_m)\sqrt{\gamma}}{1-\bar\alpha_t}\,a^{t}.
$$

用 $\sqrt{\bar\alpha_t}=\sqrt{\gamma\bar\alpha_m}$ 把上式收成 $\sqrt{\bar\alpha_m}\,a^{0}$ 加上 $(a^{t}-\sqrt{\bar\alpha_t}\,a^{0})$ 的倍数。$a^{t}$ 那一项乘 $\sqrt{\bar\alpha_t}$ 得

$$
\frac{(1-\bar\alpha_m)\sqrt{\gamma}}{1-\bar\alpha_t}\cdot\sqrt{\bar\alpha_t}
=\frac{(1-\bar\alpha_m)\,\gamma\,\sqrt{\bar\alpha_m}}{1-\bar\alpha_t},
$$

$a^{0}$ 的系数减去它，

$$
\sqrt{\bar\alpha_m}
-\frac{(1-\bar\alpha_m)\,\gamma\,\sqrt{\bar\alpha_m}}{1-\bar\alpha_t}
=\sqrt{\bar\alpha_m}\cdot\frac{1-\bar\alpha_t-\gamma+\gamma\bar\alpha_m}{1-\bar\alpha_t}
=\sqrt{\bar\alpha_m}\cdot\frac{\beta}{1-\bar\alpha_t}.
$$

这里用了 $\gamma\bar\alpha_m=\bar\alpha_t$，分子化为 $1-\gamma=\beta$。因此

$$
\mu
=\sqrt{\bar\alpha_m}\,a^{0}
+\frac{(1-\bar\alpha_m)\sqrt{\gamma}}{1-\bar\alpha_t}
\big(a^{t}-\sqrt{\bar\alpha_t}\,a^{0}\big).
$$

（1）中条件分布的均值是

$$
\sqrt{\bar\alpha_m}\,a^{0}
+\sqrt{1-\bar\alpha_m-\sigma^{2}}\;
\frac{a^{t}-\sqrt{\bar\alpha_t}\,a^{0}}{\sqrt{1-\bar\alpha_t}}.
$$

二者一致，当

$$
\sqrt{1-\bar\alpha_m-\sigma^{2}}
=\frac{(1-\bar\alpha_m)\sqrt{\gamma}}{\sqrt{1-\bar\alpha_t}}.
$$

将 $\sigma^{2}=(1-\bar\alpha_m)\beta/(1-\bar\alpha_t)$ 代入左边的平方，

$$
\begin{aligned}
1-\bar\alpha_m-\sigma^{2}
&=(1-\bar\alpha_m)\Bigl(1-\frac{\beta}{1-\bar\alpha_t}\Bigr)
=(1-\bar\alpha_m)\frac{1-\bar\alpha_t-\beta}{1-\bar\alpha_t}\\
&=(1-\bar\alpha_m)\frac{\gamma-\bar\alpha_t}{1-\bar\alpha_t}
=\frac{\gamma\,(1-\bar\alpha_m)^{2}}{1-\bar\alpha_t},
\end{aligned}
$$

其中 $\gamma-\bar\alpha_t=\gamma(1-\bar\alpha_m)$。开方即得上一式。所以（1）中的条件分布在这个 $\sigma^{2}$ 下就是后验 $\mathcal{N}(\mu,\sigma^{2}I)$。

$m$ 为 $t$ 的前一档时，$\gamma=\alpha_t=1-\beta_t$，该方差即 Diffusion Policy 笔记第 4.2 节的 $\tilde\beta_t$。$m$ 与 $t$ 不相邻时，$\gamma$ 只含 $\bar\alpha_t$ 与 $\bar\alpha_m$，中间下标不参与。

DDIM 用 $\eta\ge 0$ 缩放这条标准差，

$$
\sigma_{t\to m}
=\eta\sqrt{\frac{1-\bar\alpha_m}{1-\bar\alpha_t}\left(1-\frac{\bar\alpha_t}{\bar\alpha_m}\right)}.
$$

$\eta$ 改变 $\sigma_{t\to m}$，（1）中 $\epsilon_t$ 的系数 $\sqrt{1-\bar\alpha_m-\sigma_{t\to m}^{2}}$ 随之改变。$\eta=1$ 时方差与均值都是这一后验。$\eta=0$ 时方差为零，该系数回到 $\sqrt{1-\bar\alpha_m}$。$\eta\in[0,1]$ 时 $\sigma_{t\to m}^{2}\le 1-\bar\alpha_m$。

#### （4）证明该采样式即 3.1 节的 $\mu_\theta$

采样时没有 $a^{0}$。以 $\epsilon_\theta(a^{t},t,c_t)$ 代替 $\epsilon_t$，并按 3.1 节还原 $\hat a^{0}$，代入（1）的条件分布，即得 3.1 节的 $\mu_\theta$。

$\sigma_{t\to m}=0$ 时新噪声 $\sigma_{t\to m}\varepsilon$ 消失，$\epsilon_\theta$ 的系数回到 $\sqrt{1-\bar\alpha_m}$，

$$
a^{m}
=\sqrt{\bar\alpha_m}\,\hat a^{0}
+\sqrt{1-\bar\alpha_m}\,\epsilon_\theta.
$$

消去 $\hat a^{0}$ 后即

$$
\frac{a^{m}}{\sqrt{\bar\alpha_m}}
=\frac{a^{t}}{\sqrt{\bar\alpha_t}}
+\left(
\sqrt{\frac{1-\bar\alpha_m}{\bar\alpha_m}}
-\sqrt{\frac{1-\bar\alpha_t}{\bar\alpha_t}}
\right)\epsilon_\theta.
$$

$a^{m}$、$a^{t}$ 与 $\hat a^{0}$ 共线，且 $\sqrt{\bar\alpha_m}>\sqrt{\bar\alpha_t}$，故 $a^{m}$ 比 $a^{t}$ 更靠近 $\hat a^{0}$。

$m$ 紧邻 $t$ 且 $\eta=1$ 时，$\sqrt{\bar\alpha_m}/\sqrt{\bar\alpha_t}=1/\sqrt{\alpha_t}$，而

$$
\sqrt{1-\bar\alpha_m-\tilde\beta_t}
=\frac{\sqrt{\alpha_t}\,(1-\bar\alpha_m)}{\sqrt{1-\bar\alpha_t}}.
$$

代入 $\mu_\theta$ 并合并 $\epsilon_\theta$ 的系数，分子为 $\alpha_t(1-\bar\alpha_m)-(1-\bar\alpha_t)=\alpha_t-1=-\beta_t$，得到 Diffusion Policy 笔记第 4.3 节的

$$
\mu_\theta
=\frac{1}{\sqrt{\alpha_t}}
\left(a^{t}-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta\right).
$$

这条 $\sigma_{t\to m}$ 使前向过程回到 Markov 链，生成过程回到 DDPM。

---

## 四、共享的环境优势

轨迹概率里，对 $\theta$ 的对数导数只留下各次决策的去噪核。环境初始分布、环境转移、以及 $a^{\tau_K}\sim\mathcal{N}(0,I)$ 都不含 $\theta$。按 PPO 笔记第二节，

$$
\nabla_\theta\log p_\theta(\tau)
=\sum_{t\ge 0}\sum_{k=1}^{K}
\nabla_\theta\log\pi_\theta(a_t^{\tau_{k-1}}\mid a_t^{\tau_k},c_t).
$$

回报只在干净动作进入环境之后才出现。去噪链内部的中间奖励为零，链内又不另打折扣，所以生成同一个 $a_t$ 的 $K$ 个去噪步共用该决策的优势 $A_t$。第 $t$ 次决策对梯度的贡献是

$$
\sum_{k=1}^{K}
\nabla_\theta\log\pi_\theta(a_t^{\tau_{k-1}}\mid a_t^{\tau_k},c_t)\,A_t.
$$

$A_t>0$ 时，每个去噪核都提高已抽样点上的密度，后续采样更常到达该干净动作；$A_t<0$ 时各核同时降低密度。DPPO 再乘 $\gamma_{\mathrm{denoise}}^{k}$，使更新集中于链的末端，并通常冻结前段去噪步。RL-100 不使用该权重。论文的和式对全部 $K$ 步使用同一个 $A_t$。

裁剪目标与 PPO 笔记第五节相同，比值按去噪步分开裁：

$$
\begin{aligned}
J_i(\pi)
&=\mathbb{E}\Bigg[\sum_{k=1}^{K}
\min\Big(
r_k(\pi)\,A_t,\;
\mathrm{clip}(r_k(\pi),1-\epsilon,1+\epsilon)\,A_t
\Big)
\Bigg],\\
r_k(\pi)
&=\frac{\pi(a^{\tau_{k-1}}\mid s^k)}{\pi_i(a^{\tau_{k-1}}\mid s^k)}.
\end{aligned}
$$

$\pi_i$ 是第 $i$ 次迭代的行为策略，也是采样这条链的网络。优势为正时，该去噪步的概率最多升到旧概率的约 $1+\epsilon$ 倍；优势为负时最多降到约 $1-\epsilon$ 倍。越过有利一侧的边界后，该样本对噪声网络的梯度为零。策略损失是 $\mathcal{L}_{\mathrm{RL}}=-J_i(\pi)$。

实现中，每个去噪步计算自身的裁剪损失后即执行一次 `optimizer.step()`。离线更新位于 `update_distribution` 的时间步循环内，在线更新位于 `dp_align_update_no_share` 内。下一步的似然比使用刚刚更新的网络。论文将 $K$ 步写入同一次期望中的求和。学习率较小时，二者接近；学习率较大时，逐步更新使链的后段使用前段已经改变的参数。

动作块由 `offline_chunk_ratio_mode` 选择比值的归约。`scalar` 将窗口内各坐标的对数概率相加后取指数，整块共用一个比值。和的长度是 $n_c d_a$，比值随这个维数离开 $1$，大量样本落在 $1\pm\epsilon$ 上。`per_step` 对块内每个环境步各算一个比值，只累加该步的 $d_a$ 个坐标。默认优势是整块一个数，乘到每一个去噪步上。DPPO 笔记第 6.1 节对执行坐标取几何平均，维数留在平均的分母里。

---

## 五、离线优势：数据集上的 $Q-V$

离线阶段没有当前策略的轨迹，无法沿该轨迹计算 GAE。优势改由数据集上的两个网络给出：

$$
A^{\mathrm{off}}(s,a)=Q_\psi(s,a)-V_\psi(s).
$$

两个网络按 Implicit Q-Learning 训练。备份时不从 $\pi$ 采样下一动作，因此 $Q$ 的目标只使用数据集支撑内的动作。

记数据分布为 $\mu$。$V_\psi(s)$ 拟合的是 $Q(s,a)$ 在 $a\sim\mu(\cdot\mid s)$ 下的 $\tau$-期望分位，损失

$$
\mathcal{L}_V
=\mathbb{E}_{(s,a)\sim\mathcal{D}}
\big[\,|\tau-\mathbf{1}(u<0)|\,u^2\,\big],
\qquad
u=Q_{\bar\psi}(s,a)-V_\psi(s).
$$

$u>0$ 时权重是 $\tau$，$u<0$ 时权重是 $1-\tau$。$\tau>1/2$ 时，价值低于数据里那些高 $Q$ 的惩罚更重，$V$ 被拉向数据动作上 $Q$ 的上尾。$\tau\to 1$ 时，$V(s)$ 接近数据支撑上的 $\max_a Q(s,a)$，这个最大只在 $\mu$ 的支撑里取。代码把 $\tau$ 存成 `critic.omega`：`loss = target_q - value`，`loss > 0` 时权重为 `omega`。论文没有给出真机的 $\tau$。

$Q$ 的目标用 $V(s')$ 自举，不采样 $a'\sim\pi$：

$$
\mathcal{L}_Q
=\mathbb{E}\Big[\big(Q_\psi(s,a)-(r+\gamma V_\psi(s'))\big)^2\Big].
$$

动作块把 $r$ 换成 2.1 节的块奖励，自举折扣换成 $\gamma^{n_c}$（`critic.py` 里 `gamma ** n_action_steps`）。双 $Q$ 时优势取较小的那个，$A=\min(Q_1,Q_2)-V$，即 `get_advantage`。目标 $Q$ 按 `tau` 做软更新，期望分位回归读的是这份目标 $Q$。

由此得到的 $Q-V$ 定义在数据集中的动作上。离线 PPO 实际评分的动作可以不同于数据集中的原动作。`update_distribution` 在数据集的观测上调用 `_old_policy.all_step_logprob`，由行为策略重新采样整条去噪链，再用链末端的干净动作计算 $Q-V$。状态来自数据，去噪链相对 $\pi_i$ 是新采样的，$Q$ 在这些新动作上属于外推。裁剪与第六节的替换条件将该外推限制在 $\pi_i$ 的邻域内。

优势在 batch 内标准化后再乘比值。代码在乘之前 `.detach()`，演员损失不更新 critic。可选地把优势送进 $\exp(\beta A)$ 并截断到 $100$（`temperature`）；论文的离线优势是未取指数的 $Q-V$。

离线阶段论文冻结模仿阶段的视觉编码器 $\phi^{\mathrm{IL}}$，只更新各模块的任务头。`finetune_dp3` 在 `unio4.fix_encoder` 为真时把策略编码器设成 `eval()`。

动作块的优势还有三条用动力学模型拆开的路径，配置名是 `offline_chunk_adv_mode`：`per_step_vdelta` 在块内逐步做价值差，`chunk_vdelta_scalar` 把整块收成一个价值差，`chunk_vdelta_gae` 再沿若干个块做 GAE。论文正文用的是整块一个 IQL 优势，对应默认的 `scalar_iql`。

---

## 六、行为策略的替换条件

似然比的分母与采样去噪链的网络都是 $\pi_i$。若每次梯度步之后都将 $\pi_i$ 替换为最新网络，离线数据上的更新就不再相对一个固定的行为分布。RL-100 用模型评估决定 $\pi_i$ 是否前进。

在数据集上学一个转移模型 $\hat T(s'\mid s,a)$。从数据中的状态出发，用候选策略在 $\hat T$ 里展开 $H$ 步，

$$
\widehat J^{\mathrm{AM\text{-}Q}}(\pi)
=\mathbb{E}_{(s,a)\sim(\hat T,\pi)}
\Big[\sum_{t=0}^{H-1} Q_\psi(s_t,a_t)\Big].
$$

论文在

$$
\widehat J(\pi)-\widehat J(\pi_i)\ge\delta,
\qquad
\delta=0.05\,\big|\widehat J(\pi_i)\big|
$$

时接受候选，令 $\pi_{i+1}\leftarrow\pi$；否则 $\pi_{i+1}\leftarrow\pi_i$。这是 Uni-O4 的 AM-Q 门控。比较的是 $Q$ 沿 $\hat T$ 的累积，环境回报不进入这个式子。$\hat T$ 若有偏差，门控可能接受一个在模型中 $Q$ 更高、在真机上更差的策略。

实现与这条规则有两处差别，都在 `finetune_dp3`。

1. 估计量是 `dynamics.rollout` 返回的轨迹 $Q$ 均值。`dynamics_eval_batch.py` 里的 `rollout` 同样把沿轨迹收集的 $Q$ 取平均后返回。固定步数时均值与求和只差常数 $H$；提前终止时两者不再成比例。
2. 比较对象是该均值是否严格大于迄今最优均值（`current_mean_qs > best_mean_qs`）。通过后调用 `set_old_policy`，将当前网络整体深拷贝为 `_old_policy`。代码未使用 $0.05|\widehat J|$ 这一相对阈值。门控只决定分母网络的替换时机。当前网络的参数每步都在更新；未超过历史最优时保留当前参数，不做回滚。

因此，离线循环中始终存在两个网络：接收梯度的 $\pi$，以及仅当想象轨迹的 $Q$ 均值严格超过历史最优时才更新的 $\pi_i$。比值 $r_k=\pi/\pi_i$ 在两次替换之间可以偏离 $1$，裁剪将其限制在 $1\pm\epsilon$ 内。

转移模型在 `rl_100/unidpg/transition_model/`。集成网络或扩散动力学二选一，由 `dynamics_type` 决定；输出可以只预测最后一帧观测，或预测整段观测窗口（`prediction_mode` 的 `last` / `full`）。

---

## 七、在线优势：决策间隔上的 GAE

在线阶段有当前策略自己的轨迹，优势改用 GAE。论文把在线损失写成

$$
\mathcal{L}_{\mathrm{RL}}^{\mathrm{on}}
=-J_i(\pi)
+\lambda_V\,\mathbb{E}\big[(V_\psi(s_t)-\hat V_t)^2\big],
$$

并写 $\hat V_t=\sum_{l\ge 0}\gamma^l r_{t+l}$。实现里的回归目标是 $A_t+V(s_t)$，其中 $A_t$ 是 GAE，也就是 $\lambda$-return（`v_target = adv + vs`）。$\lambda<1$ 时它短于完整折扣回报，与 PPO 笔记第七节、DPPO 笔记第 5.1 节一致。

决策之间的折扣按动作块长度取。`dp_align_update_no_share` 中

$$
\delta_t
=r_t+(\gamma^{n_c})(1-d_t^{\mathrm{w}})V(s_{t+1})-V(s_t),
$$

$$
A_t
=\delta_t+(\gamma^{n_c}\lambda)\,(1-d_t)\,A_{t+1}.
$$

$n_c=1$ 时就是普通 GAE。同一决策的 $K$ 个去噪步继续共用这个 $A_t$。`use_adv_norm` 为真时，$A_t$ 在批内标准化。

价值损失默认是 $(V-v_{\mathrm{target}})^2$。`is_clip_value` 为真时，先把 $V$ 相对采样时的价值裁到 $\pm\epsilon$，再与未裁剪的平方误差取较大者，这是 PPO 的价值裁剪。`value_recon` 为真时，critic 自己的编码器再加上第十节的重构与 VIB 项。演员与价值的梯度分开反传，梯度范数裁到 $0.5$。

在线比值的近似 KL 用 Schulman 的 k3 估计 $\mathbb{E}[(r-1)-\log r]$。它不进入损失。`use_lr_decay` 关闭时，`AdaptiveScheduler` 用它调节演员和价值网络的学习率；打开时学习率随训练步线性下降。DPPO 用同一个估计去触发 `target_kl` 早停，这里用它调步长。

熵项在 actor 损失中被注释。探索由去噪步的高斯噪声提供。`use_bc` 为真时，在线 PPO 之前先做一段对齐行为克隆（`train_align`）。论文的在线阶段不包含该项。

可选的 `clip_std_decay` 把调度器的 `clip_std_max` 随在线步数从 `clip_std_max` 往下收。这是第八节方差上界的在线日程。

---

## 八、去噪标准差的上下界

余弦或类似日程将链末端的 $\sigma$ 收得很小。过小会使探索消失，对数密度中的 $1/\sigma^2$ 还会放大梯度；过大则使动作偏离演示邻域，增加真机执行风险。论文将每一步的标准差裁入区间：

$$
\tilde\sigma_k=\mathrm{clip}(\sigma_k,\sigma_{\min},\sigma_{\max}),
$$

并取 $\sigma_{\min}=0.01$、$\sigma_{\max}=0.8$。采样分布变成 $\mathcal{N}(\mu_\theta,\tilde\sigma_k^2 I)$。上界避免早期去噪步的大幅度探索，下界避免末端变成确定性映射。

仿真消融（Adroit door）中，上界 $0.8$ 最稳定；收至 $0.1$ 则探索不足，渐近成功率更低；不裁剪则曲线振荡、置信区间变宽。论文将 $0.8$ 用于 Adroit、MuJoCo locomotion 与真机单步任务，将 $0.1$ 用于 Meta-World 与真机动作块任务。动作块已在 $n_c$ 步上开环执行，块内较大的噪声会使整段轨迹偏离预训练行为，因此动作块任务使用更小的上界。

---

## 九、一步学生

$K$ 步 DDIM 的延迟随 $K$ 增长。一致性模型学习从任意噪声水平直接映射到干净动作。论文的蒸馏目标是

$$
\mathcal{L}_{\mathrm{CD}}(w)
=\mathbb{E}\Big[
\big\|
C_w(a^{\tau},\tau,c)
-\mathrm{sg}\big[\Psi_\varphi(a^{\tau},\tau\to 0)\big]
\big\|_2^2
\Big].
$$

$\Psi_\varphi$ 是多步教师，$\mathrm{sg}$ 表示停梯度。教师继续吃 $\mathcal{L}_{\mathrm{RL}}$，学生的误差不回传到教师。总损失

$$
\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{RL}}+\lambda_{\mathrm{CD}}\,\mathcal{L}_{\mathrm{CD}}.
$$

推理时 $a^{0}=C_w(a^{\tau_K},\tau_K,c)$，$a^{\tau_K}\sim\mathcal{N}(0,I)$，一次前向。构造函数里 `cm_inference_steps` 默认为 $1$，`ddim_inference_steps` 默认为 $10$。

实现按 `distill_loss_type` 分三条，都在 `distill_update` 里选择。

| 配置 | 函数 | 学生在拟合什么 |
|------|------|----------------|
| `action` | `compute_ddim2cm_loss_action` | 教师 `predict_action` 的完整动作；教师停梯度。与上式同一角色 |
| `action_same_noise` | `compute_ddim2cm_loss_action_same_noise` | 同上，教师与学生共用初始噪声 |
| `back_up` | `compute_ddim2cm_loss` | Song et al. 的相邻步一致性蒸馏，见下 |

`back_up` 做相邻步的一致性蒸馏。它随机取一个扩散下标，学生用边界缩放写出

$$
C=c_{\mathrm{skip}}(\tau)\,a^{\tau}+c_{\mathrm{out}}(\tau)\,\hat a^{0}_\theta,
$$

教师只做一步 DDIM 得到 $a^{\tau_{\mathrm{prev}}}$，目标由学生的 EMA 网络在 $\tau_{\mathrm{prev}}$ 上再写一次同样的缩放。损失是这两个输出的均方误差。相邻步一致沿日程传递，推理时仍可从纯噪声一步跳到动作。EMA 的更新率是 `training.ema_decay`。在线蒸馏时教师取当前 `self.model`，离线蒸馏时教师是冻结副本。

flow 策略不走这三条。`is_flow` 时 `distill_update` 调用 `compute_flow_distill_loss`，蒸馏结束后 `promote_distilled_model` 把学生权重复制回默认策略，并把推理步数改成 `flow_distill_inference_steps`。扩散学生与 flow 学生因此是两条代码路径；论文正文的学生是一致性模型。

蒸馏阶段由 `distill_phase` 选择：`after_dp` 在行为克隆之后，`after_offline` 在离线 PPO 之后，`online` 与在线 PPO 交替。`online` 时 `update_phase` 决定学生在每个去噪步、每个 iteration 或一轮末尾更新。

仿真中，Skip-Net 的一步策略约 $3.9\mathrm{M}$ 参数、$378\,\mathrm{Hz}$，U-Net 约 $39.2\mathrm{M}$、$133\,\mathrm{Hz}$；同图中的 DPPO 约 $30\,\mathrm{Hz}$，DSRL 约 $35\,\mathrm{Hz}$、$52.3\mathrm{M}$。真机整段任务的墙钟加速小得多。Orange Juicing 的放置任务上，一步学生为 $9.2\,\mathrm{s}$，十步 DDIM 为 $10.2\,\mathrm{s}$，DP-2D 为 $10.6\,\mathrm{s}$，相对 DDIM 约 $1.11$ 倍。其余时间消耗在点云处理与底层插值上，控制频率受相机帧率限制，约为 $30\,\mathrm{Hz}$。论文中由 $100\,\mathrm{ms}$ 降至 $10\,\mathrm{ms}$ 的是策略网络的前向时间；整段任务的墙钟时间为上述 $9.2\,\mathrm{s}$ 与 $10.2\,\mathrm{s}$。

---

## 十、观测编码器

条件编码器 $\phi$ 与噪声网络一起用行为克隆训练。点云用 DP3 的三维编码器，代码里带重构与信息瓶颈的是 `DP3EncoderReconVIB`。两项正则是

$$
\begin{aligned}
\mathcal{L}_{\mathrm{recon}}
&=\beta_{\mathrm{recon}}\big(d_{\mathrm{Chamfer}}(\hat o,o)+\|\hat q-q\|_2^2\big),\\
\mathcal{L}_{\mathrm{KL}}
&=\beta_{\mathrm{KL}}\,\mathrm{KL}\big(\phi(z\mid o,q)\,\|\,\mathcal{N}(0,I)\big).
\end{aligned}
$$

模仿阶段的总损失为去噪误差加上这两项。构造函数的默认系数为 $\beta_{\mathrm{recon}}=0.5$、$\beta_{\mathrm{KL}}=10^{-3}$。强化学习微调时，论文将两个系数除以 $10$，使策略仍能更新，同时避免特征被探索轨迹过度改写。

离线阶段冻结 $\phi$（第五节）。仿真消融比较微调期间的三种做法：编码器继续更新并保留重构与 VIB 时，成功率最高且最稳定；去掉这两项后稳定性下降；完全冻结编码器则上限更低。冻结是离线更新的默认设置。消融表明，在线微调时可以让编码器以较小权重继续适应。

采集离线数据时，若 `data_collect=True`，去噪改为随机采样；`use_vib=True` 时，编码器的潜变量也保留噪声（`ppo.force_stochastic_online`）。正式在线微调前应关闭 `data_collect`，否则评估轮次仍按采集模式写入轨迹。

---

## 十一、训练流程

外层是数据循环，内层依次为 critic 更新、离线裁剪更新、可选蒸馏与在线裁剪更新。外层循环不由单次 `train.py` 自动完成：真机轨迹经 `tools/teleop_off2off_data/data_prepare.py` 并入 zarr 后，下一轮训练需要重新启动。

```text
D ← 遥操作
π_IL ← 行为克隆(D)          # ε-prediction，可加重构与 VIB

重复 M 轮:
    在 D 上训练 IQL 的 (Q, V) 与转移模型 T̂
    π ← π_IL
    重复离线步:
        在 D 的观测上用行为策略 π_i 抽去噪链
        A ← Q(s, a^0) − V(s)，批内标准化
        对每个去噪步做裁剪更新（第四节）
        每 eval_step 步在 T̂ 上估计轨迹 Q 均值
        若 is_update_old_policy 且超过历史最好值: π_i ← deepcopy(π)
    用 π 或一步学生在真机上展开，人给成功标签
    D ← D ∪ 新轨迹
    π_IL ← 行为克隆(D)      # 在新分布上重拟合去噪场

在线:
    用当前策略采样，按决策间隔做 GAE
    对每个去噪步做裁剪更新，并回归 λ-return
    若 distill_phase = online: 同时蒸馏一步学生
```

重新做行为克隆，是因为裁剪更新只在 $\pi_i$ 的邻域内移动去噪核。若并入的轨迹包含新的行为，单次局部似然比更新不足以重拟合整个噪声场。去噪监督损失在合并后的数据上重新拟合 $\epsilon_\theta$，多峰仍由扩散模型表示。下一轮开始时执行 $\pi\leftarrow\pi_{\mathrm{IL}}$，离线 PPO 从这次行为克隆的结果出发。本轮离线 PPO 的权重只用于真机展开；展开轨迹与成功标签留在 $D$ 中，这组参数不作为下一轮的初始策略。行为克隆的优化器是否从这组权重热启动，训练流程没有单独写明。

`data_prepare.py` 有三种模式。`raw_to_npy` 把遥操作回合收成数组；`build_zarr` 从遥操作和选中的展开数据新建训练集；`extend_zarr` 把新的展开目录接到已有 zarr 上，不改原文件。训练读到的字段是

```text
point_cloud, next_point_cloud, state, next_state,
action, next_action, reward, return, done, timeout
```

以及 `meta/episode_ends`。展开数据在 yaml 的 `rollout_sources` 里按轮次启用。离线训练脚本与采集脚本必须使用同一套网络选项（策略族、二维或三维、单步或动作块、`horizon`、`n_action_steps`、`n_obs_steps`、是否 VIB / 重构、扩散或 flow 的日程），否则检查点对不齐。

奖励分两种。除 Dynamic Push-T 外，操作者在成功时用键盘给予 $+1$，其余时间为 $0$，另加每步的时间惩罚，以及动作相对近期滑动平均的抖动惩罚。论文未给出后两项的系数。Dynamic Push-T 使用稠密奖励

$$
r_t=r_{\mathrm{pose}}+r_{\mathrm{static}}+r_{\mathrm{smooth}},
$$

$r_{\mathrm{pose}}=\exp(-3e)-1$，$e$ 是 T 形块与目标位姿的 $\mathrm{SE}(3)$ 误差；$r_{\mathrm{static}}$ 在块几乎不动时为 $-1$，否则为 $0$；$r_{\mathrm{smooth}}=-5\|a_t-a_{t-1}\|_2^2$。

---

## 十二、实验

### 12.1 评估协议

真机成功率在图 5 的初始区域内随机放置物体，并按各任务的成功定义统计。倒水任务中，不可避免的反弹飞溅不计为失败。榨汁的压杆由动觉示教轨迹回放，不由策略生成。一步学生未在榨汁取出上评估：演示中推与抓之间已有逆运动学造成的位姿跳变，一步策略对噪声更敏感。下文的 $100\%$ 是在这些约定下的计数。

在线训练曲线（Agile Bowling）记录的是带探索噪声的展开，低于确定性评估。论文用它观察策略是否低于离线阶段的水平，不用它代替正式成功率。

### 12.2 成功率

模仿基线是二维 Diffusion Policy 与三维 DP3。RL-100 先做迭代离线，再做在线；在线执行器分十步 DDIM 与一步一致性模型。

| 任务 | DP-2D | DP3 | 迭代离线 | 在线 DDIM | 在线一步 |
|------|-------|-----|----------|-----------|----------|
| Dynamic Push-T | 40（20/50） | 64（32/50） | 90（45/50） | 100（50/50） | 100（50/50） |
| Agile Bowling | 14（7/50） | 80（40/50） | 88（44/50） | 100（50/50） | 100（50/50） |
| Pouring | 42（21/50） | 48（24/50） | 92（46/50） | 100（50/50） | 100（50/50） |
| Soft-towel Folding | 46（23/50） | 68（34/50） | 94（47/50） | 100（50/50） | 100（250/250） |
| Dynamic Unscrewing | 82（41/50） | 70（35/50） | 94（47/50） | 100（50/50） | 100（50/50） |
| Juicing，放置 | 78（39/50） | 88（44/50） | 94（47/50） | 100（100/100） | 100（50/50） |
| Juicing，取出 | 48（24/50） | 76（38/50） | 86（43/50） | 100（50/50） | — |
| 未加权平均 | 50.0 | 70.6 | 91.1 | 100 | 100（六个任务） |

未加权平均由 $50.0$ 升至 $70.6$，再升至 $91.1$，增幅最大的阶段是离线训练。在线阶段消除剩余失败：DDIM 共 $400/400$，一步学生在已评估的六个任务上为 $500/500$，合计 $900/900$。相对 DP3，离线阶段使 Pouring 从 $48\%$ 升至 $92\%$，Unscrewing 从 $70\%$ 升至 $94\%$，Bowling 仅从 $80\%$ 升至 $88\%$。Bowling 的模仿基线已经较高，离线阶段可改进的失败较少。

### 12.3 分布变化、少样本与扰动

零样本不改权重。Pouring 把颗粒换成水 $90\%$；Push-T 改表面摩擦 $100\%$，加干扰物 $80\%$；Bowling 改表面 $100\%$；折叠换未见形状 $80\%$。四项平均 $90\%$。

少样本是每个变体再训 $1$–$3$ 小时。折叠换材料 $100\%$，Bowling 球瓶对调 $100\%$，倒水换容器 $60\%$，平均 $86.7\%$。容器几何变了之后，已有的倾倒原语只覆盖一部分。

执行中施加人工扰动。折叠的抓取阶段与预折叠阶段各为 $90\%$，拧螺母在反向拧长达 $4\,\mathrm{s}$ 时为 $100\%$，Push-T 全程拖拽为 $100\%$，平均 $95\%$。商场部署中，榨汁机器人连续约 $7$ 小时为随机顾客出汁，论文报告未出现失败。保龄对照每侧 $25$ 次，策略成功 $25$ 次，人成功 $14$ 次。Push-T 在相同墙钟时间内，策略成功 $20$ 次，采集演示的熟练操作者成功 $17$ 次，初学者成功 $13$ 次。

### 12.4 步数、墙钟与数据量

成功轨迹的环境步数：折叠从 DP-2D 的 $390$ 步降至一步学生的 $312$ 步（$1.25$ 倍）；拧螺母从 $361$ 步降至 DDIM 的 $280$ 步（$1.29$ 倍）。$\gamma<1$ 的回报偏好更短的成功轨迹，步数下降由此产生。蒸馏改变的是前向次数。

若将失败也计入 Push-T 的平均步数，则由 $822$（DP-2D）降至 $658$（DP3）、$382$（离线）和 $322$（在线 DDIM），相对 DP-2D 为 $2.55$ 倍。失败回合更早终止，因此平均步数的降幅大于成功轨迹上的加速。

七个任务的平均采集量为：人 $115$ 条、$1.8\,\mathrm{h}$；迭代离线 $566$ 条、$6.5\,\mathrm{h}$；在线 $434$ 条、$5.6\,\mathrm{h}$。人的采集约占全部采集时间的 $13\%$。各任务的在线时数差异很大。倒水的在线采集为 $1.5\,\mathrm{h}$，离线为 $6.8\,\mathrm{h}$；榨汁放置的在线采集为 $12.5\,\mathrm{h}$，离线为 $10.5\,\mathrm{h}$。引言将在线阶段描述为把约 $95\%$ 提升到 $99\%$ 以上的短预算。表 7 的平均在线时数与离线同量级，各任务的时数应分开看。

### 12.5 仿真对照与消融

仿真任务包括 MuJoCo locomotion（本体感觉）、Adroit（door、hammer、pen）和 Meta-World（box-close、disassemble、peg-insert-side、push-wall）。后两组使用三维视觉加本体感觉。对照方法为 DPPO、ReinFlow 与 DSRL。论文对图 14 的读数为：halfcheetah-medium-v2 上 RL-100 达到 $10000$，DPPO 约 $4500$，DSRL 约 $3000$；Adroit 的 door 与 hammer 接近成功率 $1$，DPPO 停在约 $0.9$，ReinFlow 在 hammer 上低于 $0.6$；peg-insert-side 上 RL-100 达到 $1$，ReinFlow 不超过 $0.2$。上述数字是论文对曲线图的读数，图中未给出逐点数值表。论文同时报告，收敛大约发生在 $1$–$2\mathrm{M}$ 步，对照方法大约为 $3$–$5\mathrm{M}$ 步，且置信区间更窄。

Adroit door 上的消融：

| 改动 | 现象 |
|------|------|
| 点云相对 RGB | 点云学得更快，最终成功率更高；场景干净时，几何裁剪把把手和接触面留在观测里 |
| $\sigma$ 上界 | $0.8$ 最好；$0.1$ 探索不足；不裁则振荡 |
| 一步学生相对十步 DDIM | 学习曲线接近，最终成功率都约 $1$ |
| 微调时的编码器 | 更新编码器并保留重构与 VIB 时最好；去掉两项后稳定性下降；冻结编码器则上限更低 |
| $\epsilon$-prediction 相对 $x_0$-prediction | 最终成功率约 $1.0$ 对约 $0.6$ |

$\epsilon$-prediction 与 $x_0$-prediction 还原干净动作的方式是

$$
\hat a^{0}
=\frac{a^{t}-\sqrt{1-\bar\alpha_t}\,f_\theta}{\sqrt{\bar\alpha_t}}
\quad(\epsilon),\qquad
\hat a^{0}=f_\theta
\quad(x_0).
$$

$t$ 较大时，$1/\sqrt{\bar\alpha_t}$ 放大 $\epsilon_\theta$ 的误差。同一网络在预训练后滚动时，$\hat a^{0}$ 的经验方差在 Hopper 上为 $0.1316$ 对 $0.0870$，在 Adroit door 上为 $0.0589$ 对 $0.0290$（各 $100$ 次前向）。模仿与开环执行中，方差更小的 $x_0$-prediction 更稳定。在线策略梯度将 $\epsilon$-prediction 放大的方差留在去噪链中作为探索，后训练因此采用噪声预测。DP3 一类模仿实现常用 $x_0$-prediction。

---

## 十三、和 DPPO 笔记的对应

两条笔记写的是同一个两层 MDP。读的时候按下面更换符号和开关。

| | DPPO 笔记 | 本文 |
|--|-----------|------|
| 去噪一步 | 采样链上 $k+1\to k$，$k$ 越大越噪 | 子采样日程上 $\tau_k\to\tau_{k-1}$，可跳过训练日程 $T$ 的下标 |
| 优势 | $A_t$，再乘 $\gamma_{\mathrm{denoise}}^{k}$ | 离线 $Q-V$，在线 GAE；链上不另乘折扣 |
| 更新哪些步 | 冻结前段，只训最后 $K'$ 步，$\epsilon$ 由紧到松 | $K$ 步同一个 $\epsilon$；实现逐步 `step()` |
| 数据 | 第二段不再读演示 | 演示保留，并入展开，重新行为克隆 |
| 保守更新 | 裁剪与 k3 早停 | 裁剪；$\pi_i$ 仅在想象轨迹的 $Q$ 均值超过历史最优时替换；k3 用于调节学习率 |
| 方差下界 | 采样与对数概率各一个下界 | 论文给 $[\sigma_{\min},\sigma_{\max}]=[0.01,0.8]$ |
| 部署 | 评估时 $\eta=0$ 的 DDIM | 十步教师或一步学生 |

DPPO 的第二段为纯 on-policy，不再使用专家数据，样本效率较低；在稀疏奖励下，它较少偏离预训练成功率。RL-100 将 on-policy 更新限于最后阶段，此前在不断增长的数据集上使用 IQL 优势与模型门控。真机实验中，从模仿到约 $90\%$ 的提升发生在 on-policy 阶段之前。

---

## 十四、关键超参数

| 参数 | 代码 | 论文或默认 | 作用 |
|------|------|------------|------|
| 采样步数 $K$ | `ddim_inference_steps` | 默认 $10$；论文子采样 $5$–$10$ | 教师的去噪步 |
| 学生步数 | `cm_inference_steps` | 默认 $1$ | 部署前向次数 |
| 观测帧 $n_o$ | `n_obs_steps` | $2$ | 条件里的历史 |
| 动作块 $n_c$ | `n_action_steps` | 块任务约 $8$–$16$ | 决策间隔与 $\gamma^{n_c}$ |
| $\eta$ | 构造函数 `eta` | 默认 $1$ | DDIM 方差系数 |
| $\sigma$ 区间 | `clip_std_max`，论文另有下界 $0.01$ | 上界 $0.8$ 或 $0.1$ | 第八节 |
| 裁剪宽度 | `unio4.clip_ratio`，在线 `ppo.epsilon` | 论文记为 $\epsilon$ | $1\pm\epsilon$ |
| 期望分位 $\tau$ | `critic.omega` | 论文未给真机值 | 第五节，$V$ 的上尾 |
| 接受阈值 | — | 论文 $\delta=0.05|\widehat J|$ | 实现改为严格超过历史最好 $Q$ 均值 |
| $\gamma,\lambda$ | `critic.gamma`，`ppo.lamda` | 与 PPO 相同 | 在线按 $\gamma^{n_c}$ 使用 |
| $\beta_{\mathrm{recon}}$，$\beta_{\mathrm{KL}}$ | 构造函数 | $0.5$，$10^{-3}$；微调时除以 $10$ | 第十节 |
| 离线冻结编码器 | `unio4.fix_encoder` | 离线冻结 | 在线消融建议继续小权重更新 |
| 蒸馏时机 | `distill_phase` | `after_dp` / `after_offline` / `online` | 第九节 |
| 蒸馏损失 | `distill_loss_type` | `action` / `action_same_noise` / `back_up` | 第九节 |
| 离线块优势 | `offline_chunk_adv_mode` | 默认 `scalar_iql` | 整块一个 $Q-V$ |
| 梯度范数 | 硬编码 $0.5$ | — | 演员与价值 |
| 想象长度 | `unio4.rollout_length` | 论文的 $H$ | AM-Q 的展开步数 |

`is_linear_decay` 为真时，离线阶段的学习率、价值学习率与裁剪宽度随 `bppo_steps` 线性下降。`finetune_dp3` 在第 $200$ 步之后关闭 `is_clip_decay` 与 `is_bppo_lr_decay`。这是代码中的日程，论文未单独报告。

---

## 十五、参考文献与阅读顺序

1. Lei et al., *RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning*, arXiv:2510.14830, 2025；*Science Robotics* 11(116), aed6267, 2026. 代码：[Lei-Kun/RL-100](https://github.com/Lei-Kun/RL-100)
2. Chi et al., *Diffusion Policy*, RSS 2023. [`diffusion_policy_notes.md`](diffusion_policy_notes.md)
3. Schulman et al., *Proximal Policy Optimization Algorithms*, 2017. [`ppo_notes.md`](ppo_notes.md)
4. Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016.
5. Ren et al., *Diffusion Policy Policy Optimization*, arXiv:2409.00588, 2024. [`dppo_notes.md`](dppo_notes.md)
6. Kostrikov, Nair, and Levine, *Offline Reinforcement Learning with Implicit Q-Learning*, ICLR 2022.
7. Lei et al., *Uni-O4: Unifying Online and Offline Deep Reinforcement Learning with Multi-Step On-Policy Optimization*, ICLR 2024.
8. Zhuang, Lei, et al., *Behavior Proximal Policy Optimization*, ICLR 2023.
9. Song, Dhariwal, Chen, and Sutskever, *Consistency Models*, ICML 2023.
10. Ho, Jain, and Abbeel, *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
11. Song, Meng, and Ermon, *Denoising Diffusion Implicit Models*, ICLR 2021.
12. Ze et al., *3D Diffusion Policy*, RSS 2024.
13. Luo et al., *SERL*, ICRA 2024；*HIL-SERL*, *Science Robotics*, 2025.

建议先读 Diffusion Policy 笔记第四节与第八节、PPO 笔记第五与第七节、DPPO 笔记第三至六节，再读本文第三至七节。对照实现时看 `update_distribution` 里优势与裁剪的顺序，以及 `finetune_dp3` 里 `dynamics.rollout` 与 `set_old_policy` 的条件。

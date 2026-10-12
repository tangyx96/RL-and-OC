# DPPO（Diffusion Policy Policy Optimization）算法笔记

下文公式对应 Ren et al., *Diffusion Policy Policy Optimization*（[arXiv:2409.00588](https://arxiv.org/pdf/2409.00588)）。实现见 [irom-princeton/dppo](https://github.com/irom-princeton/dppo)：去噪链上的似然在 `model/diffusion/diffusion_vpg.py`，PPO 损失在 `model/diffusion/diffusion_ppo.py`，采样与 GAE 在 `agent/finetune/train_ppo_diffusion_agent.py`。  
条件去噪与 $\epsilon$-prediction 见 [`diffusion_policy_notes.md`](diffusion_policy_notes.md)；裁剪目标与 GAE 见 [`ppo_notes.md`](ppo_notes.md)。

---

## 一、算法定位

DPPO 用策略梯度微调已经行为克隆好的 Diffusion Policy。网络仍是条件去噪模型，预训练目标仍是去噪均方误差；微调阶段改为最大化环境回报。做法是把每一步去噪写成内层马尔可夫决策过程（Markov decision process, MDP）的一个转移，把与环境的交互写成外层 MDP，得到一条两层链。链上每一步的策略都是高斯，对数密度有闭式，因而可以直接套 Proximal Policy Optimization（PPO）。

扩散策略的边缘 $\pi(\mathbf{a}^{0}\mid\mathbf{s})$ 没有闭式密度。以往工作因此多用 $Q$ 学习或加权回归去改扩散策略，并认为把策略梯度穿过去噪链会把有效时域乘上去噪步数、方差过大（Psenka et al., 2023）。DPPO 把梯度放在每一跳高斯核上，优势只在环境步上估计一次，再按去噪步分配。环境时域不被去噪步数乘长。

| | Diffusion Policy | 高斯 PPO | DPPO |
|--|------------------|----------|------|
| 训练信号 | 专家动作上的 $\epsilon$-MSE | 环境回报 | 预训练为 $\epsilon$-MSE，微调为环境回报 |
| 策略密度 | 边缘 $p(\mathbf{a}^{0}\mid\mathbf{s})$ 无闭式 | 对角高斯 | 每个去噪核为高斯 |
| 数据 | 固定演示，可反复用 | 当前策略采样，用完即弃 | 微调阶段同样 on-policy |
| 探索 | 反向过程的随机项 | 加在最终动作上 | 每一跳去噪都加噪声，同时被预训练的去噪方向拉回数据流形 |

动作仍是 chunk。记预测长度 $T_p$（`horizon_steps`）、执行长度 $T_a\le T_p$（`act_steps`）、单步动作维 $D_a$。$\mathbf{a}_t^{k}\in\mathbb{R}^{T_p\times D_a}$ 是环境步 $t$、去噪指标 $k$ 上的动作序列；$k=K$ 最噪，$k=0$ 为送入环境的序列。环境里实际执行的是 $\mathbf{a}_t^{0}$ 的前 $T_a$ 步。观测在下文写成 $\mathbf{s}_t$，图像观测时它就是 Diffusion Policy 笔记里的 $\mathbf{O}_t$。

---

## 二、为什么梯度放在去噪核上

### 2.1 环境目标

环境 MDP 为 $\mathcal{M}_{\mathrm{Env}}=(\mathcal{S},\mathcal{A},P_0,P,R)$。策略最大化折扣回报

$$
\mathcal{J}(\pi_\theta)
=\mathbb{E}^{\pi_\theta}\!\left[\sum_{t\ge 0}\gamma_{\mathrm{env}}^{t}\,R(\mathbf{s}_t,\mathbf{a}_t^{0})\right].
$$

策略梯度需要 $\nabla_\theta\log\pi_\theta(\mathbf{a}_t^{0}\mid\mathbf{s}_t)$，或它的某个优势加权形式（[`ppo_notes.md`](ppo_notes.md) 第二节）。高斯、GMM 的对数密度可以直接写出来。Diffusion Policy 给出的是反向链

$$
p_\theta(\mathbf{a}^{0:K}\mid\mathbf{s})
=p(\mathbf{a}^{K})\prod_{k=1}^{K}p_\theta(\mathbf{a}^{k-1}\mid\mathbf{a}^{k},\mathbf{s}),
\qquad
p(\mathbf{a}^{K})=\mathcal{N}(0,I),
$$

环境里用到的是边缘 $\pi_\theta(\mathbf{a}^{0}\mid\mathbf{s})=\int p_\theta(\mathbf{a}^{0:K}\mid\mathbf{s})\,\mathrm{d}\mathbf{a}^{1:K}$。这个积分没有闭式。

用 probability-flow ODE 估计边缘似然再反传，是一条可做的对照（论文附录 C.6）。它在 Hopper 上能改进预训练策略，在 Can 上成功率掉到零，而且梯度必须穿过整条去噪链，也不能只微调其中几步。DPPO 优化的是链的联合密度，不估计这条边缘。

### 2.2 联合密度的对数导数

采样动作就是跑这条链。回报只通过最终执行的 $\mathbf{a}_t^{0}$ 进入环境，但 $\mathbf{a}_t^{0}$ 由前面的随机去噪产生。对整条生成过程用对数导数：初始噪声 $\mathbf{a}_t^{K}\sim\mathcal{N}(0,I)$ 与环境转移 $P$ 都不含 $\theta$，因而

$$
\nabla_\theta\log p_\theta(\mathbf{a}_t^{0:K}\mid\mathbf{s}_t)
=\sum_{k=0}^{K-1}\nabla_\theta\log p_\theta(\mathbf{a}_t^{k}\mid\mathbf{a}_t^{k+1},\mathbf{s}_t).
$$

每一项都是高斯核的对数密度梯度。$\gamma_{\mathrm{denoise}}=1$ 时，下面的 PPO 估计量就是把环境优势 $A_t$ 乘到这个和的每一项上。它是环境回报关于**因子化去噪策略**的策略梯度，与「先算出 $\log\pi(\mathbf{a}^{0}\mid\mathbf{s})$ 再求导」不是同一个估计量。

若把 $K$ 步去噪和 $T$ 步环境串成一条链，并且每一步都乘同一个 $\gamma$，则环境步之间相隔 $K$ 个折扣，有效时域约为 $TK$。这是 Q-score matching 认为策略梯度难以用于扩散策略的原因。DPPO 把折扣拆开：$\gamma_{\mathrm{env}}$ 只沿环境步 $t$ 走；同一步里的各去噪核先共享同一个 $A_t$，再用单独的 $\gamma_{\mathrm{denoise}}^{k}$ 重新加权。

---

## 三、两层 Diffusion Policy MDP

### 3.1 状态、动作与奖励

把去噪嵌进环境步，得到 $\mathcal{M}_{\mathrm{DP}}$。指标

$$
\bar{t}(t,k)=tK+(K-k-1)
$$

随 $t$ 增大；同一个 $t$ 内，$k$ 从 $K-1$ 降到 $0$ 时 $\bar{t}$ 也增大。$k=K$ 的纯噪声由转移抽样，不是策略的动作。策略在每个环境步上动作 $K$ 次：$\mathbf{a}^{K}\to\mathbf{a}^{K-1}\to\cdots\to\mathbf{a}^{0}$。

$$
\begin{aligned}
\bar{\mathbf{s}}_{\bar{t}(t,k)}&=(\mathbf{s}_t,\,\mathbf{a}_t^{k+1}),\\
\bar{\mathbf{a}}_{\bar{t}(t,k)}&=\mathbf{a}_t^{k},\\
\bar{R}_{\bar{t}(t,k)}&=
\begin{cases}
0, & k>0,\\
R(\mathbf{s}_t,\mathbf{a}_t^{0}), & k=0.
\end{cases}
\end{aligned}
$$

转移：

$$
\bar{P}(\bar{\mathbf{s}}'\mid\bar{\mathbf{s}},\bar{\mathbf{a}})=
\begin{cases}
\delta_{(\mathbf{s}_t,\,\mathbf{a}_t^{k})}, & k>0,\\[4pt]
P(\mathbf{s}_{t+1}\mid\mathbf{s}_t,\mathbf{a}_t^{0})\otimes\mathcal{N}(0,I), & k=0.
\end{cases}
$$

$k>0$ 时，下一步状态只是把刚采样的动作写回去，观测不变。$k=0$ 时才推进环境，并为本环境步重新抽一份纯噪声，作为下一步去噪的起点。初始分布是 $P_0\otimes\mathcal{N}(0,I)$。

```text
环境步 t
  s_t ,  a_t^K ~ N(0, I)
       │
       ├─ k = K-1, …, 1    高斯去噪，奖励 0，观测不变
       │
       └─ k = 0            得到 a_t^0，执行前 T_a 步
                           奖励 R_t，环境给出 s_{t+1}，再抽 a_{t+1}^K
```

中间奖励为零，所以从 $(t,k)$ 出发、只按 $\gamma_{\mathrm{env}}$ 累计的回报与 $k$ 无关，等于从 $\mathbf{a}_t^{0}$ 出发的环境回报。论文 (4.6) 把折扣写成 $\gamma_{\mathrm{env}}^{t}$；按回报定义，求和里相对当前步的因子是 $\gamma_{\mathrm{env}}^{t'-t}$。

这一构造把 Black et al. (2023) 在文生图里用的 denoising MDP 放进控制的环境 MDP。图像那一侧只有一条去噪链、末端一个奖励；这里每一步环境交互都重新打开一条去噪链。

### 3.2 策略是去噪核

反向核与 Diffusion Policy 笔记第 3.3 节相同，只是把对数密度留下用于策略梯度：

$$
\bar\pi_\theta(\mathbf{a}_t^{k}\mid\mathbf{a}_t^{k+1},\mathbf{s}_t)
=\mathcal{N}\!\big(
\mathbf{a}_t^{k};\,
\mu(\mathbf{a}_t^{k+1},\epsilon_\theta(\mathbf{a}_t^{k+1},k+1,\mathbf{s}_t)),\,
\sigma_{k+1}^{2}I
\big).
$$

网络 $\epsilon_\theta$ 在较噪的那一端求值，时间指标因此是 $k+1$。均值 $\mu$ 由噪声日程和 $\epsilon_\theta$ 写出，DDPM 的形式见 Diffusion Policy 笔记第 2.4.4 节。DDIM 把同一结构写成

$$
\mathbf{a}^{k-1}
\sim\mathcal{N}\!\big(
\mu^{\mathrm{DDIM}}(\mathbf{a}^{k},\epsilon_\theta),\,
\eta\,\sigma_k^{2}I
\big).
$$

$\eta=0$ 时核退化为狄拉克，高斯对数密度无定义。微调时取 $\eta=1$，核与 DDPM 同为非退化高斯；评估时取 $\eta=0$，走确定性轨迹。

各向同性高斯的对数密度为

$$
\log\bar\pi_\theta
=-\frac{1}{2\sigma^{2}}\|\mathbf{a}-\mu_\theta\|^{2}
-\frac{d}{2}\log(2\pi\sigma^{2}).
$$

日程给定的 $\sigma$ 不含 $\theta$（可学习的 DDIM $\eta$ 除外），第二项对 $\theta$ 的梯度为零。于是

$$
\nabla_\theta\log\bar\pi_\theta
=\frac{1}{\sigma^{2}}(\mathbf{a}-\mu_\theta)^{\top}\nabla_\theta\mu_\theta.
$$

$\mu_\theta$ 对 $\epsilon_\theta$ 是仿射的，梯度回到噪声网络。预训练和微调共用这一网络：前者回归注入的 $\epsilon$，后者按优势去增减「采样到的这一跳」的概率。

两层 MDP 上的策略梯度（论文 (4.4)，优势形式）为

$$
\nabla_\theta\bar{\mathcal{J}}
=\mathbb{E}\!\left[
\sum_{\bar{t}}\nabla_\theta\log\bar\pi_\theta(\bar{\mathbf{a}}_{\bar{t}}\mid\bar{\mathbf{s}}_{\bar{t}})\,
\hat{A}_{\bar{t}}
\right].
$$

期望同时覆盖环境步和去噪步。采样这条链就是 Diffusion Policy 原来的前向去噪，没有额外的环境交互。

---

## 四、优势只在环境步上估计

### 4.1 价值函数只读观测

$k>0$ 的奖励为零，中间去噪状态上不必再学一个价值。Critic 记为 $\tilde{V}_\phi(\mathbf{s}_t)$，输入是环境观测，不含 $\mathbf{a}^{k}$。实现里就是 `critic(obs)`。

环境步上的 TD 残差与 GAE 与 PPO 笔记第七节相同，折扣用 $\gamma_{\mathrm{env}}$：

$$
\delta_t=R_t+\gamma_{\mathrm{env}}V(\mathbf{s}_{t+1})-V(\mathbf{s}_t),
\qquad
A_t=\delta_t+(\gamma_{\mathrm{env}}\lambda)\,A_{t+1}.
$$

倒序只沿环境步。终止步把自举项乘上 $1-\mathrm{terminated}$。价值回归的目标是 GAE 回报 $R_t=A_t+V(\mathbf{s}_t)$。论文附录 (A4) 把该目标写成蒙特卡洛回报；仓库与 CleanRL 一样，回归的是上述 $R_t$。

论文比较了三种优势：只依赖 $\mathbf{s}_t$、依赖 $(\mathbf{s}_t,\mathbf{a}^{1})$、以及 $Q(\mathbf{s}_t,\mathbf{a}^{0})$。Hopper 上后两种在微调中拿到更高奖励；Can、Square 以及更长的操作任务上，只依赖 $\mathbf{s}_t$ 更稳。动作块维数高，扩散策略本身又随机，动作条件的价值更难学准。$Q$ 估计偏差会直接改 actor，稀疏奖励下策略容易从预训练的成功率掉到零且回不来。

### 4.2 去噪折扣

环境优势算完之后，分给该步的各去噪核：

$$
\hat{A}_{t,k}=\gamma_{\mathrm{denoise}}^{k}\,A_t.
$$

$k$ 是噪声水平。$k$ 大的核权更小，$k=0$ 的权为 $1$。$\gamma_{\mathrm{denoise}}=1$ 时，同一步的每个核使用同一个 $A_t$，第二节的对数导数原样成立。$\gamma_{\mathrm{denoise}}<1$ 时，更噪的核被降权，估计量不再是 $\mathcal{J}$ 的无偏梯度，而是偏向末端去噪步的重加权。

这个因子来自设计，并不由中间奖励为零推出。中间奖励为零只说明各 $k$ 共享同一份环境回报；$\gamma_{\mathrm{denoise}}^{k}$ 决定这份回报里有多少用来更新第 $k$ 跳。

仓库里微调段的下标 $i=0$ 是该段最噪的一步，$i=K'-1$ 是最干净的一步，乘的权重为

$$
\gamma_{\mathrm{denoise}}^{\,K'-1-i}.
$$

对应代码 `gamma_denoising ** (ft_denoising_steps - i - 1)`。$\gamma_{\mathrm{denoise}}\in\{0.5,0.8,0.9,1\}$ 的消融里，Gym 上 $0.8$ 效率较好，Can 几乎不敏感，Square 上 $0.5$ 略好。它和「少微调几步」是同一件事的软、硬两个版本：降权靠前的去噪步，或干脆冻结它们。

---

## 五、去噪步上的 PPO

旧策略采样的链固定之后，每一跳的似然比为

$$
r_{t,k}(\theta)
=\frac{\bar\pi_\theta(\mathbf{a}_t^{k}\mid\mathbf{a}_t^{k+1},\mathbf{s}_t)}
{\bar\pi_{\theta_{\mathrm{old}}}(\mathbf{a}_t^{k}\mid\mathbf{a}_t^{k+1},\mathbf{s}_t)}.
$$

裁剪目标与 PPO 笔记第五节相同，裁剪宽度可以依赖 $k$：

$$
\mathcal{L}^{\mathrm{CLIP}}(\theta)
=\mathbb{E}\Big[
\min\big(
r_{t,k}(\theta)\,\hat{A}_{t,k},\;
\mathrm{clip}(r_{t,k}(\theta),\,1-\varepsilon_k,\,1+\varepsilon_k)\,\hat{A}_{t,k}
\big)
\Big].
$$

正优势把该跳的概率往上推，推过 $1+\varepsilon_k$ 后这条样本的策略梯度为零；负优势对称。多轮更新、梯度裁剪、价值损失的写法与 PPO 笔记第九、十节一致。

实现把比值做了几处缩放，否则 $T_p\times D_a$ 上的联合密度会让 $r_{t,k}$ 极陡：

- `log_prob` 先截断到 $[-5,2]$。
- 只保留实际执行的前 $T_a$ 步（`reward_horizon`，默认等于 `act_steps`）。预测了但未执行的后缀不进入比值。
- 对这 $T_a\times D_a$ 个坐标的对数密度取**均值**再做 $\exp$。均值使比值的尺度不随动作维和 chunk 长度膨胀；梯度仍通过 $\mu_\theta$ 进入 $\epsilon_\theta$。
- 优势在 batch 内标准化，并可用分位数截断。

裁剪宽度沿微调段从紧到松。记 $i$ 为上一节的去噪下标，

$$
\varepsilon(i)
=\varepsilon_{\mathrm{base}}
+(\varepsilon-\varepsilon_{\mathrm{base}})
\frac{e^{\rho\, i/(K'-1)}-1}{e^{\rho}-1}.
$$

默认 $\varepsilon_{\mathrm{base}}=10^{-3}$（`clip_ploss_coef_base`），$\rho=3$（`clip_ploss_coef_rate`），$\varepsilon$ 为 `clip_ploss_coef`。最噪的一步几乎不允许离开旧核，最干净的一步使用任务里选定的 $\varepsilon$。论文正文写「较早的去噪步用更大的 $\varepsilon$」，同段给出的式子却是 $\varepsilon_{k=K-1}=0.1\,\varepsilon_{k=0}$。代码与后一式一致：更噪的步裁得更紧。

$\varepsilon$ 本身按任务在 $\{0.1,\,0.01,\,0.001\}$ 里取。选取标准是 batch 内落入裁剪区的样本比例大约在 $10\%$ 到 $20\%$ 之间，并取仍满足该比例的最大 $\varepsilon$。稀疏奖励的操作任务对这个比例敏感。

总损失为

$$
L_{\mathrm{total}}
=-\mathcal{L}^{\mathrm{CLIP}}
-c_{\mathrm{ent}}\,H
+c_{\mathrm{vf}}\,L^{\mathrm{VF}}.
$$

固定 $\sigma$ 的高斯熵只依赖日程，对 $\theta$ 没有梯度，熵项不改变噪声网络。DDIM 若把 $\eta$ 设成可学习参数，熵通过 $\eta$ 进入更新；默认实验不学 $\eta$。可选的行为克隆正则默认关闭：论文发现流形上的探索已经把微调后的动作留在自然的演示附近，不必再把策略拉回预训练权重。

新旧策略的距离用 Schulman 的 k3 估计 $\mathbb{E}[(r-1)-\log r]$（PPO 笔记第 10.3 节）。超过 `target_kl` 就结束本批后续轮次。论文写的阈值是 $1$，并说明实践中基本不会触发。

---

## 六、使微调成立的几项设计

### 6.1 只更新最后 $K'$ 步

预训练常取 $K=100$，以便拟合多峰演示。微调时复制一份参数：原网络 $\theta$ 冻结，负责前面的去噪；$\theta_{\mathrm{ft}}$ 只负责最后 $K'$ 步，并接受 PPO 更新。显存和反传深度随 $K'$ 线性增长。

早步决定落到哪一个模态、粗轨迹长什么样；晚步改的是末端精度。只动晚步，等于在预训练已经铺好的流形上改将要执行的动作。$K'$ 太小（例如 $3$）渐近成功率下降，收敛也更慢；$K'$=10 在论文的操作任务上综合最好。状态观测的配置是预训练 $20$ 步、微调最后 $10$ 步；像素观测是预训练 $100$ 步、用 DDIM 微调 $5$ 步。Furniture-Bench 同样用 $100$ 步预训练和 $5$ 步 DDIM。

### 6.2 噪声下界

余弦日程把 $\sigma_k$ 收到约 $10^{-4}$。这个数量级够生成，不够强化学习探索，也会让 $1/\sigma^{2}$ 把对数密度的梯度放大。DPPO 因此用两个不同的下界：

| | 作用位置 | 代码 | 论文中的量级 |
|--|----------|------|----------------|
| $\sigma_{\min}^{\mathrm{exp}}$ | 采样 $\mathbf{a}^{k}=\mu+\sigma z$ | `min_sampling_denoising_std` | $0.01$–$0.1$ |
| $\sigma_{\min}^{\mathrm{prob}}$ | 计算 $\log\bar\pi$ | `min_logprob_denoising_std` | $0.1$ |

采样下界决定探索有多宽。Can 上它几乎不影响最终成功率；Square 上需要大约 $0.1$，否则策略在覆盖不足的样本上过优化然后坍缩。把下界提高之后，轨迹仍然贴着预训练的模态，只是模态周围更宽。似然下界是数值稳定项，可以高于采样时实际使用的标准差，因此比值里的 $\sigma$ 不必等于采样分布的 $\sigma$。

### 6.3 骨干与 chunk 长度

MLP 微调更稳。全连接把 $T_p$ 个时间步绑在同一组权重里，微调时把 $T_a$ 取得比 $T_p$ 小会导致不稳定，因而两者保持相等。UNet 只沿动作序列做卷积，允许预训练用较大的 $T_p$、微调用较小的 $T_a$，例如 $T_p=16$、$T_a=8$。较长的预测有利于行为克隆阶段的时序一致性；较短的执行缩短策略梯度所看到的开环段。像素实验使用 ViT 编码器加 MLP 动作头。

批量按「环境步 $\times$ 微调去噪步」来取，大约是同样任务上高斯 PPO 批量的 $K'$ 倍。Robomimic 上更接近高斯 PPO 的批量有时样本效率更好。更新仍是 on-policy：一批 rollout 用完即弃，不把专家演示放进微调的回放池。

---

## 七、训练流程

```text
0. 预训练
   L_BC = ||ε − ε_θ(s, a^k, k)||²
   余弦噪声日程（Diffusion Policy 笔记第 4.5、8.1 节）
   复制 actor_ft，冻结 actor

1. 用 θ_old 采样，并行 N 个环境
   每个环境步：
     a^K ~ N(0, I)
     对 k = K−1, …, 0：
        最后 K' 步用 actor_ft，其余用冻结的 actor
        a^k = μ_θ + σ z，σ 不低于 σ_exp_min
     执行 a^0 的前 T_a 步
     记录 (s, 去噪链, R, log π_old)
   评估时 DDIM 取 η = 0；DDPM 则只在最后一步去掉噪声

2. V_φ(s) 前向一次
   沿环境步倒序计算 GAE，得到 A_t 与 R_t = A_t + V(s_t)
   中间去噪步不另估价值

3. 把 (环境步, 去噪步) 摊平，共 n_steps × n_envs × K' 条样本
   重复 update_epochs：
     a. 重算该跳的 log π_θ，σ 不低于 σ_prob_min
     b. Â_{t,k} = γ_denoise^k A_t，batch 内标准化
     c. 按第 5 节计算 L^CLIP，ε 从噪到净由紧到松
     d. 价值损失回归 R_t，可选裁剪
     e. 梯度裁剪后更新 actor_ft 与 critic
     f. k3 的 KL 超过 target_kl 则结束本批

4. 丢弃 rollout，回到步骤 1
```

可选的 critic warmup 在前若干迭代只更新 $V_\phi$。采样噪声下界和 $K'$ 的退火接口在代码里留着，论文配置没有打开。

---

## 八、探索、稳定性与部署

第六节的机制实验用 D3IL 的 Avoid：机械臂绕过一组障碍到达桌面另一侧，演示里有多条路径。微调的稀疏奖励只给最上方那条模态。高斯和 GMM 也用 PPO 微调，动作块长度同为 $T_a=4$。

**流形上的探索。** 扩散在每一跳加入噪声，覆盖变宽；学到的去噪方向同时把样本拉回预训练数据附近。多跳之后，轨迹在演示模态周围展开，仍保持绕障的结构。高斯把噪声加在最终动作上，轨迹更散，尤其在模态弯得厉害的时候；GMM 的覆盖更窄。若只在 $k=0$ 加噪声、中间步改为确定性，Avoid 上的轨迹失去这个结构，Can 和 Square 的收敛也更慢。

这一性质依赖预训练已经盖住相关的成功模态。需要跳出演示流形时它会偏保守：Furniture-Bench 的 Lamp、低随机性设定里，高斯略好于 DPPO。从零开始训练时，论文也没有观察到扩散相对高斯的明显探索优势。

**多步修正。** 微调过程中，动作分布沿着去噪步逐渐收拢，粗结构留在早步，末端步继续细化，而不是在第一步或最后一步塌成一个点。往已采样动作里持续加噪声时，高斯和 GMM 的成功率垮掉；DPPO 在至少保留四步去噪时还在。$T_a$ 加大到 $16$ 时，高斯和 GMM 可以完全不提升，DPPO 仍能学。末端控制器不完美时，这个性质也被用来在仿真里加动作噪声，服务于后面的仿真到真机。

**微调后的策略更耐扰动。** 训练结束后再往动作上加噪声，DPPO 的成功率掉得比高斯慢，并且能从更大的初始状态区域走到奖励指定的那条路径。One-leg 上，仿真里训好的策略直接上真机、$10\,\mathrm{Hz}$、不用真实数据：DPPO 为 $16/20$。高斯在仿真里微调到 $88\%$，真机为 $0/20$，动作抖。给高斯加上拉回预训练策略的行为克隆项之后，仿真 $53\%$，真机 $50\%$。

---

## 九、实验里的相对位置

下面只保留和算法选择有关的对照。曲线与种子见论文第五节。

| 对照 | 设定 | 现象 |
|------|------|------|
| DQL、IDQL、DIPO、QSM，以及奖励加权、优势加权回归 | Gym 稠密奖励；Robomimic 稀疏奖励 | Gym 上 IDQL、DIPO 接近 DPPO。Robomimic 上 DPPO 最稳，Transport 上差距最大。依赖 $Q$ 的方法常在微调一开始掉点 |
| RLPD、Cal-QL、IBRL | 微调时仍采样专家数据 | HalfCheetah 上 off-policy 的样本效率更高。Kitchen 与 Robomimic 上 DPPO 的最终成功率更高。DPPO 微调阶段不再使用专家数据，因此更吃预训练是否已经覆盖多峰 |
| 高斯、GMM，骨干为 MLP 或 Transformer，同为 PPO | Robomimic 状态与像素 | 状态下 Lift、Can 都能到接近 $100\%$，Square 与 Transport 上 DPPO 更高，Transport 超过 $90\%$。像素下高斯在 Transport 上停在预训练的 $0\%$；DPPO 把该任务做到 $50\%$ 以上 |
| 高斯 MLP | Furniture-Bench，50 条仿真演示，按阶段给稀疏奖励 | 六种任务与随机性组合里 DPPO 都有提升。高斯在中等随机性下几乎归零。真机结果见第八节 |

从零训练（Gym，$T_a=1$，10 步去噪）的最终回报与高斯相当，墙钟大约慢六倍，多出来的时间主要在多步采样。论文把 DPPO 放在「先行为克隆、再微调」这条管线里；样本效率低于把专家数据留在回放池里的 off-policy 方法，换来的是长时程、稀疏奖励操作上的稳定性。

---

## 十、关键超参数

| 参数 | 代码 | 常见取值 | 作用 |
|------|------|----------|------|
| 去噪步数 $K$ | `denoising_steps` | 状态 20，像素 / 家具 100 | 预训练与采样的总步数；微调时保持不变 |
| 微调步数 $K'$ | `ft_denoising_steps` | 10，或 DDIM 的 5 | 只更新链的末端 |
| $T_p$ / $T_a$ | `horizon_steps` / `act_steps` | MLP 二者相等；UNet 可用 16 / 8 | 预测长度与执行长度 |
| $\gamma_{\mathrm{denoise}}$ | `gamma_denoising` | 0.8–1 | 降低更噪步的优势权重 |
| $\sigma_{\min}^{\mathrm{exp}}$ | `min_sampling_denoising_std` | 0.01–0.1 | 采样时的探索噪声 |
| $\sigma_{\min}^{\mathrm{prob}}$ | `min_logprob_denoising_std` | 0.1 | 计算 $\log\pi$ 时的方差下界 |
| $\varepsilon$ | `clip_ploss_coef` | 0.1、0.01 或 0.001 | 最干净一跳的裁剪宽度 |
| $\varepsilon_{\mathrm{base}}$ | `clip_ploss_coef_base` | $10^{-3}$ | 最噪一跳的裁剪宽度 |
| $\gamma_{\mathrm{env}},\lambda$ | `gamma`, `gae_lambda` | 与 PPO 相同 | 环境步上的 GAE |
| DDIM 的 $\eta$ | `eta` | 训练 1，评估 0 | 训练时保留高斯噪声，评估时确定性 |
| 批量 | `batch_size` | 约为高斯 PPO 的 $K'$ 倍 | 期望同时对 $t$ 与 $k$ 取 |

---

## 十一、参考文献与阅读顺序

1. Ren et al., *Diffusion Policy Policy Optimization*, arXiv:2409.00588, 2024. 代码：[irom-princeton/dppo](https://github.com/irom-princeton/dppo)
2. Chi et al., *Diffusion Policy*, RSS 2023. [`diffusion_policy_notes.md`](diffusion_policy_notes.md)
3. Schulman et al., *Proximal Policy Optimization Algorithms*, 2017. [`ppo_notes.md`](ppo_notes.md)
4. Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, 2016.
5. Black et al., *Training Diffusion Models with Reinforcement Learning*（DDPO）, arXiv:2305.13301, 2023.
6. Ho et al., *Denoising Diffusion Probabilistic Models*, NeurIPS 2020.
7. Song et al., *Denoising Diffusion Implicit Models*, ICLR 2021.
8. Wang et al., *Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning*（Diffusion-QL）, 2022.
9. Hansen-Estruch et al., *IDQL: Implicit Q-Learning as an Actor-Critic Method with Diffusion Policies*, 2023.
10. Psenka et al., *Learning a Diffusion Model Policy from Rewards via Q-Score Matching*, arXiv:2312.11752, 2023.
11. Yang et al., *Policy Representation via Diffusion Probability Model for Reinforcement Learning*（DIPO）, 2023.

建议阅读顺序：Diffusion Policy 笔记第 2.4 与第 3 节（反向核与动作序列）→ PPO 笔记第五、七节（裁剪与 GAE）→ 本文第二至六节 → `diffusion_vpg.py` 的 `p_mean_var`、`get_logprobs` 与 `diffusion_ppo.py` 的 `loss`。

# 分布空间中的策略优化：从指数重加权到自然梯度、镜像下降与扩散策略

> **定位**：本文综述策略学习、零阶优化、采样控制与生成式策略中的分布优化方法，重点讨论 KL 约束、指数重权、重要性采样、加权极大似然与自然梯度之间的关系。  
> **主线**：首先在概率分布空间中求解正则化策略改进，再将所得分布投影到可计算的模型族。具体算法由优化对象、信息来源、散度方向、投影方式和策略参数化共同确定。  
> **相关笔记**：零阶优化与 MPPI/CMA 见 [`zero-order-optimization-study-notes.md`](../../zero-order-robotics/study-notes/zero-order-optimization-study-notes.md)；PPO、SAC、Diffusion Policy 与 DPPO 分别见 [`ppo_notes.md`](ppo_notes.md)、[`sac_notes.md`](sac_notes.md)、[`diffusion_policy_notes.md`](../generative-policies/diffusion_policy_notes.md) 和 [`dppo_notes.md`](../generative-rl/dppo_notes.md)。

---

## 一、分布优化问题

经典优化直接寻找

$$
x^*\in\arg\max_x f(x).
$$

分布优化则在候选解空间 $\mathcal X$ 上维护概率分布 $q$，求

$$
\max_{q\in\mathcal P(\mathcal X)}\;\mathbb E_{x\sim q}[f(x)].
$$

若不加限制，最优 $q$ 会退化为集中在全局最优点上的 Dirac 测度；这个写法本身没有降低求解难度。真正有用的是**相对当前分布作有限改进**：给定参考分布 $p$，只允许新分布 $q$ 在信息意义上移动有限距离，

$$
\begin{aligned}
\max_q\quad & \mathbb E_q[f(x)]\\
\text{s.t.}\quad & D_{\mathrm{KL}}(q\|p)\le \varepsilon,\qquad \int q(x)\,\mathrm dx=1.
\end{aligned}
\tag{1}
$$

相应的拉格朗日惩罚形式为

$$
q^*=\arg\max_q
\left\{
\mathbb E_q[f(x)]-\eta D_{\mathrm{KL}}(q\|p)
\right\},
\qquad \eta>0.
\tag{2}
$$

$\eta$ 是 KL 约束的拉格朗日乘子，也可解释为温度；在镜像下降的常用参数化中，$1/\eta$ 对应步长。$\eta$ 大时更新保守，$\eta$ 小时概率质量迅速集中到高分区域。在适当的可行性与强对偶条件下，若式 (1) 的约束活跃，则存在与给定 $\varepsilon$ 对应的最优乘子 $\eta^*$，使式 (1) 与式 (2) 同解；任意指定的 $\eta$ 并不自动对应同一个 $\varepsilon$。式 (2) 同时是 KL-proximal step、负熵镜像下降的一步，也是带参考测度的自由能最大化。

### 1.1 变分解：指数倾斜

引入归一化约束的乘子 $\lambda$，对密度 $q$ 作变分：

$$
\mathcal L(q,\lambda)
=\int q(x)f(x)\,\mathrm dx
-\eta\int q(x)\log\frac{q(x)}{p(x)}\,\mathrm dx
+\lambda\left(\int q(x)\,\mathrm dx-1\right).
$$

其一阶条件为

$$
f(x)-\eta\left(\log\frac{q(x)}{p(x)}+1\right)+\lambda=0.
$$

因此

$$
\boxed{
q^*(x)=\frac{p(x)\exp(f(x)/\eta)}{Z},
\qquad
Z=\mathbb E_{x\sim p}\left[\exp(f(x)/\eta)\right].
}
\tag{3}
$$

这条 Gibbs/Boltzmann 更新是全文的母式：**保留参考分布的支持，在其上按效用作指数重加权。** 它也给出 Donsker--Varadhan 变分恒等式

$$
\eta\log\mathbb E_p[e^{f/\eta}]
=\sup_q\left\{\mathbb E_q[f]-\eta D_{\mathrm{KL}}(q\|p)\right\}.
\tag{4}
$$

式 (4) 把 log-sum-exp、自由能、风险敏感目标和 KL 正则化连接起来。有限样本下若 $x_i\sim p$，则

$$
w_i=\frac{\exp(f(x_i)/\eta)}{\sum_j\exp(f(x_j)/\eta)}
\tag{5}
$$

就是对 $q^*$ 的自归一化重要性权重。MPPI、REPS、MPO、AWR、reward-weighted regression 与多类进化策略反复出现这一结构，并非偶然。

### 1.2 约束半径、温度与对偶问题

式 (1) 的约束半径 $\varepsilon$ 与式 (2) 的温度 $\eta$ 不是两个彼此独立的超参数。为看清二者的关系，把 KL 约束写进拉格朗日函数：

$$
\mathcal L(q,\eta,\lambda)
=\mathbb E_q[f]
-\eta\big(D_{\mathrm{KL}}(q\|p)-\varepsilon\big)
+\lambda\left(\int q(x)\,\mathrm dx-1\right),
\qquad \eta\ge0.
\tag{6}
$$

固定 $\eta>0$ 对 $q$ 取上确界，仍得到式 (3) 的

$$
q_\eta(x)=\frac{p(x)e^{f(x)/\eta}}{Z(\eta)},
\qquad
Z(\eta)=\mathbb E_p[e^{f/\eta}].
\tag{7}
$$

把它代回拉格朗日函数，得到只含 $\eta$ 的对偶目标

$$
g(\eta)
=\eta\varepsilon+\eta\log Z(\eta),
\qquad
\min_{\eta\ge0}g(\eta).
\tag{8}
$$

推导式 (8) 时用到了式 (4)：

$$
\sup_q\{\mathbb E_q[f]-\eta D_{\mathrm{KL}}(q\|p)\}
=\eta\log Z(\eta).
$$

设 $\eta>0$ 且相关期望有限。对 $g$ 求导：

$$
\begin{aligned}
g'(\eta)
&=\varepsilon+\log Z(\eta)
+\eta\frac{Z'(\eta)}{Z(\eta)},\\
Z'(\eta)
&=\mathbb E_p\left[
e^{f/\eta}\left(-\frac{f}{\eta^2}\right)
\right]
=-\frac{Z(\eta)}{\eta^2}\mathbb E_{q_\eta}[f].
\end{aligned}
$$

因此

$$
g'(\eta)
=\varepsilon+\log Z(\eta)
-\frac{1}{\eta}\mathbb E_{q_\eta}[f].
\tag{9}
$$

另一方面，

$$
\begin{aligned}
D_{\mathrm{KL}}(q_\eta\|p)
&=\mathbb E_{q_\eta}
\left[\log\frac{q_\eta}{p}\right]\\
&=\mathbb E_{q_\eta}
\left[\frac{f}{\eta}-\log Z(\eta)\right]\\
&=\frac{1}{\eta}\mathbb E_{q_\eta}[f]-\log Z(\eta).
\end{aligned}
\tag{10}
$$

结合式 (9)–(10)：

$$
\boxed{
g'(\eta)=\varepsilon-D_{\mathrm{KL}}(q_\eta\|p).
}
\tag{11}
$$

若最优点位于 $\eta^*>0$，则 $g'(\eta^*)=0$，恰有

$$
D_{\mathrm{KL}}(q_{\eta^*}\|p)=\varepsilon.
\tag{12}
$$

这说明实际算法可以不手调温度，而是解一维凸对偶问题，使 KL 半径达到给定值。REPS 与 MPO 的温度更新采用的正是这一思路。有限样本实现把 $Z(\eta)$ 换成样本平均：

$$
\hat g(\eta)
=\eta\varepsilon
+\eta\log\left(
\frac1N\sum_{i=1}^N e^{f_i/\eta}
\right).
\tag{13}
$$

计算时应先减去 $\max_i f_i/\eta$ 再做 log-sum-exp，以避免指数上溢。

### 1.3 极限、支持集与可积性

式 (3) 成立的前提是 $Z(\eta)<\infty$。有限离散空间中该条件自动满足；连续空间中若 $f$ 的正尾部增长快于参考分布 $p$ 的衰减，配分函数可能发散。此时所谓 Gibbs 分布并不存在，必须限制搜索空间、截断效用或更换参考分布。

温度的两个极限给出直观解释。

当 $\eta\to\infty$ 时，

$$
e^{f/\eta}
=1+\frac{f}{\eta}+O(\eta^{-2}),
\qquad
Z(\eta)
=1+\frac{\mathbb E_p[f]}{\eta}+O(\eta^{-2}).
$$

两者相除：

$$
q_\eta(x)
=p(x)\left[
1+\frac{f(x)-\mathbb E_p[f]}{\eta}
\right]+O(\eta^{-2}).
\tag{14}
$$

因此高温更新只是围绕 $p$ 的小步重权，中心化效用 $f-\mathbb E_p[f]$ 自然出现。它是 exponentiated-gradient 更新的局部形式，也是 score-function 梯度中减 baseline 不改变期望的分布版本。

当 $\eta\to0^+$ 时，Laplace 原理给出

$$
\eta\log\mathbb E_p[e^{f/\eta}]
\longrightarrow
\operatorname*{ess\,sup}_{x\sim p} f(x).
\tag{15}
$$

$q_\eta$ 集中到 $p$ 的支持中效用最大的区域。这里是 essential supremum，而不是整个 $\mathcal X$ 上的最大值：若 $p$ 在某一区域没有概率质量，指数重权不能在那里创造质量。这一支持保持性质是离线策略改进保守性的来源，也是其上限。

### 1.4 有限样本重权与有效样本量

给定 $x_i\sim p$，对任意可积函数 $h$，

$$
\mathbb E_{q_\eta}[h(x)]
=\frac{\mathbb E_p[h(x)e^{f(x)/\eta}]}
{\mathbb E_p[e^{f(x)/\eta}]}
\approx
\sum_{i=1}^N\bar w_i h(x_i),
\tag{16}
$$

其中 $\bar w_i$ 为式 (5) 的归一化权重。该比值估计在有限 $N$ 下一般有偏；分子和分母分别无偏，并不意味着二者之比无偏。若二阶矩存在，它仍是一致估计。

权重退化可用有效样本量诊断：

$$
N_{\mathrm{eff}}
=\frac{1}{\sum_{i=1}^N\bar w_i^2},
\qquad
1\le N_{\mathrm{eff}}\le N.
\tag{17}
$$

$\eta$ 很小时，少数高分样本占据几乎全部权重，$N_{\mathrm{eff}}$ 接近 $1$；此时加权最大似然容易过拟合，协方差估计也会退化。自适应温度、权重裁剪、rank transform 和 elite fraction 都在控制这一有限样本问题，但它们对应不同于精确 Gibbs 权重的效用变换。

### 1.5 适用范围与算法实例化

式 (3) 给出理想的非参数分布更新。具体算法还需指定以下要素：

1. $x$ 是动作、轨迹、策略参数，还是状态--动作占用测度？
2. $f$ 是即时回报、累计回报、$Q$、优势，还是排序后的适应度？
3. 样本来自当前策略、旧策略、行为数据，还是动力学模型？
4. $q^*$ 如何投影回高斯、神经网络策略或扩散模型等可计算族？
5. KL 是逐状态策略 KL、轨迹 KL、占用测度 KL，还是搜索分布 KL？

上述要素均一致时，两个更新才具有严格的等价关系。仅共享指数权或 KL 邻域的方法具有相同的变分形式，但对应不同的优化问题和数值实现。

---

## 二、从分布更新到信息几何与镜像下降

### 2.1 KL 的局部二阶展开与 Fisher 度量

设策略属于光滑参数族 $p_\theta$，令 $\theta' = \theta+\delta$。在正则条件下，

$$
D_{\mathrm{KL}}(p_\theta\|p_{\theta+\delta})
=\frac12\delta^\top F(\theta)\delta+o(\|\delta\|^2),
\tag{18}
$$

其中

$$
F(\theta)
=\mathbb E_{x\sim p_\theta}
\left[
\nabla_\theta\log p_\theta(x)
\nabla_\theta\log p_\theta(x)^\top
\right]
\tag{19}
$$

是 Fisher 信息矩阵。将目标作一阶展开、KL 作二阶展开，局部信赖域问题化为

$$
\max_\delta\; g^\top\delta
\quad\text{s.t.}\quad
\frac12\delta^\top F\delta\le\varepsilon,
\tag{20}
$$

方向为

$$
\delta\propto F^{-1}g.
\tag{21}
$$

这就是自然梯度。普通梯度依赖参数坐标；自然梯度用分布可区分性的局部度量修正方向，因此对光滑重参数化不变。

上述结论可由 KL 的逐项求导得到。记

$$
K(\delta)
=D_{\mathrm{KL}}(p_\theta\|p_{\theta+\delta})
=\mathbb E_{p_\theta}
\big[\log p_\theta(x)-\log p_{\theta+\delta}(x)\big].
\tag{22}
$$

在 $\delta=0$ 处，

$$
\nabla_\delta K(0)
=-\mathbb E_{p_\theta}[\nabla_\theta\log p_\theta(x)]
=-\int\nabla_\theta p_\theta(x)\,\mathrm dx
=-\nabla_\theta1
=0.
\tag{23}
$$

二阶导为

$$
\nabla_\delta^2K(0)
=-\mathbb E_{p_\theta}
[\nabla_\theta^2\log p_\theta(x)].
\tag{24}
$$

由

$$
\nabla_\theta^2\log p_\theta
=\frac{\nabla_\theta^2p_\theta}{p_\theta}
-\nabla_\theta\log p_\theta
\nabla_\theta\log p_\theta^\top
$$

并利用 $\int\nabla_\theta^2p_\theta\,\mathrm dx=0$，可得信息矩阵等式

$$
-\mathbb E[\nabla_\theta^2\log p_\theta]
=\mathbb E[
\nabla_\theta\log p_\theta
\nabla_\theta\log p_\theta^\top]
=F(\theta).
\tag{25}
$$

于是 Taylor 展开给出式 (18)。再对式 (20) 引入乘子 $\lambda>0$：

$$
\mathcal L(\delta,\lambda)
=g^\top\delta
-\lambda\left(
\frac12\delta^\top F\delta-\varepsilon
\right).
$$

一阶条件为 $g-\lambda F\delta=0$，故

$$
\delta=\frac1\lambda F^{-1}g.
$$

令约束取等号，

$$
\lambda
=\sqrt{\frac{g^\top F^{-1}g}{2\varepsilon}},
\qquad
\boxed{
\delta^*
=\sqrt{\frac{2\varepsilon}{g^\top F^{-1}g}}\,
F^{-1}g.
}
\tag{26}
$$

自然梯度给出方向，信赖域半径决定长度。TRPO 以共轭梯度近似求 $F^{-1}g$，再用线搜索检查代理目标和实际 KL。

### 2.2 IGO 的准确位置

Information-Geometric Optimization（IGO）在一个搜索分布族 $\{p_\theta\}$ 上，对目标的**分位数变换**作自然梯度流。其贡献不只是“给期望回报加 KL”，还包括：

- 把黑箱搜索写成统计流形上的自然梯度；
- 用基于排序的权重获得对目标单调变换的不变性；
- 从同一框架恢复 CEM 的平滑变体、Gaussian IGO/CMA-ES、NES 与 Bernoulli 族上的 PBIL。

因此，IGO 与式 (2) 共享信息几何和分布更新的精神，但二者不是同义词。式 (2) 的闭式解是 Gibbs 重加权；IGO 的标准表述是自然梯度流及其离散化，而且常用 rank-based utility。把所有 KL 正则化策略优化都称为 IGO，会掩盖它们在目标和离散化上的差别。

### 2.3 镜像下降：非欧氏空间里的 proximal step

对概率向量，取负熵势函数

$$
\psi(q)=\sum_x q(x)\log q(x),
$$

对应的 Bregman 散度正是 $D_{\mathrm{KL}}(q\|p)$。于是式 (2) 是在概率单纯形上的镜像上升：先在对偶坐标中加上 $f/\eta$，再由 softmax 映回概率空间，

$$
\log q_{k+1}(x)=\log q_k(x)+\frac{f_k(x)}{\eta}-\log Z_k.
\tag{27}
$$

这解释了“乘法更新”与“加法梯度”的关系：概率空间中的指数乘权，对应 log-density 空间中的加法步。自然梯度是该几何的局部形式，镜像下降是有限步形式，Gibbs 更新则是无约束分布族中一步精确解。

更一般地，给定严格凸势函数 $\psi$，Bregman 散度为

$$
D_\psi(q,p)
=\psi(q)-\psi(p)-\langle\nabla\psi(p),q-p\rangle.
\tag{28}
$$

镜像上升的一步写成

$$
q_{k+1}
=\arg\max_{q\in\mathcal P}
\left\{
\langle q,f_k\rangle
-\eta D_\psi(q,q_k)
\right\}.
\tag{29}
$$

取负熵

$$
\psi(q)=\int q(x)\log q(x)\,\mathrm dx.
$$

因为 $\nabla\psi(p)=1+\log p$，代入式 (28)：

$$
\begin{aligned}
D_\psi(q,p)
&=\int q\log q-\int p\log p
-\int(1+\log p)(q-p)\\
&=\int q\log\frac qp
-\int q+\int p\\
&=D_{\mathrm{KL}}(q\|p).
\end{aligned}
\tag{30}
$$

概率归一化使最后两项相消。式 (29) 因而退化为式 (2)，其一阶条件在对偶坐标中是

$$
\nabla\psi(q_{k+1})
=\nabla\psi(q_k)+\frac{f_k}{\eta}+c,
$$

即

$$
1+\log q_{k+1}
=1+\log q_k+\frac{f_k}{\eta}+c.
$$

指数化并用归一化确定 $c$，得到式 (27)。这条推导说明 KL 并非为了方便而任意加入：它是负熵几何对应的 Bregman proximity。

---

## 三、序列决策中真正被优化的分布

### 3.1 动作条件分布

给定旧策略 $\pi_k$ 和其优势 $A^{\pi_k}(s,a)$，逐状态理想改进写成

$$
q_k(a\mid s)
\propto
\pi_k(a\mid s)
\exp\left(\frac{A^{\pi_k}(s,a)}{\eta}\right).
\tag{31}
$$

优势可替换为 $Q$，因为状态相关的 $V(s)$ 被归一化常数吸收。式 (31) 是 REPS、MPO、AWR 一类方法的共同核心。

### 3.2 轨迹分布与 control as inference

策略与动力学诱导轨迹分布

$$
p_\pi(\tau)
=p(s_0)\prod_{t=0}^{T-1}
\pi(a_t\mid s_t)p(s_{t+1}\mid s_t,a_t).
\tag{32}
$$

若动力学在新旧分布中相同，轨迹 KL 可分解为沿状态访问分布累积的策略 KL。给轨迹回报 $R(\tau)$ 作指数倾斜，

$$
q^*(\tau)\propto p_{\pi_k}(\tau)e^{R(\tau)/\eta}.
\tag{33}
$$

这就是 control as inference、路径积分控制与部分策略搜索方法的概率推断视角。需要注意：式 (33) 的最优轨迹分布未必能由一个 Markov 策略精确实现，且环境动力学不能任意改变；实际算法必须尊重动力学一致性或作变分近似。

轨迹 KL 的分解说明逐状态策略 KL 从何而来。设 $p_{\pi'}$ 与 $p_\pi$ 共享初始分布和环境动力学，则

$$
\begin{aligned}
\log\frac{p_{\pi'}(\tau)}{p_\pi(\tau)}
&=\log
\frac{
p(s_0)\prod_t\pi'(a_t\mid s_t)p(s_{t+1}\mid s_t,a_t)
}{
p(s_0)\prod_t\pi(a_t\mid s_t)p(s_{t+1}\mid s_t,a_t)
}\\
&=\sum_t\log\frac{\pi'(a_t\mid s_t)}{\pi(a_t\mid s_t)}.
\end{aligned}
\tag{34}
$$

对 $\tau\sim p_{\pi'}$ 取期望：

$$
D_{\mathrm{KL}}(p_{\pi'}(\tau)\|p_\pi(\tau))
=\sum_t
\mathbb E_{s_t\sim d_t^{\pi'}}
\left[
D_{\mathrm{KL}}\big(
\pi'(\cdot\mid s_t)\|\pi(\cdot\mid s_t)
\big)
\right].
\tag{35}
$$

动力学项相消不表示动力学不重要；它仍通过 $d_t^{\pi'}$ 决定哪些状态被加权。将 $d_t^{\pi'}$ 换成旧策略的 $d_t^\pi$ 是 trust-region 方法常用的局部近似。

### 3.3 占用测度

折扣占用测度定义为

$$
d^\pi(s,a)=(1-\gamma)\sum_{t=0}^\infty
\gamma^t\Pr_\pi(s_t=s,a_t=a).
\tag{36}
$$

它满足 Bellman 流约束。REPS 的完整形式优化的常是状态--动作分布或占用测度，而不是对每个状态彼此独立地优化动作分布。忽略流约束会得到漂亮的指数式，却可能不是任何可执行策略的真实占用。价值函数在 REPS 对偶中正是流约束的拉格朗日乘子，由此出现 Bellman residual 的指数权重。

令状态边缘为 $d(s)=\sum_a d(s,a)$。归一化折扣占用测度满足

$$
d(s)
=(1-\gamma)\rho_0(s)
+\gamma\sum_{\bar s,\bar a}
d(\bar s,\bar a)P(s\mid\bar s,\bar a).
\tag{37}
$$

左边是访问状态 $s$ 的折扣质量；右边由初始注入和所有前驱状态动作流入组成。反过来，对任何非负、归一化且满足式 (37) 的 $d$，在 $d(s)>0$ 处定义

$$
\pi_d(a\mid s)=\frac{d(s,a)}{d(s)},
\tag{38}
$$

即可恢复一个实现该占用测度的 Markov 策略。因而策略优化也可以写成带线性流约束的分布优化：

$$
\max_{d\ge0}
\sum_{s,a}d(s,a)r(s,a),
\qquad
d\ \text{满足式 (37)}.
\tag{39}
$$

### 3.4 Performance difference lemma

统一策略改进还依赖一个基础恒等式。对任意两策略 $\pi'$ 与 $\pi$，

$$
J(\pi')-J(\pi)
=\frac{1}{1-\gamma}
\mathbb E_{s\sim d^{\pi'},\,a\sim\pi'}
[A^\pi(s,a)].
\tag{40}
$$

证明从优势的望远镜和开始。沿 $\pi'$ 的轨迹，

$$
A^\pi(s_t,a_t)
=r_t+\gamma\mathbb E[V^\pi(s_{t+1})\mid s_t,a_t]
-V^\pi(s_t).
$$

对轨迹取期望并乘 $\gamma^t$ 求和：

$$
\begin{aligned}
\mathbb E_{\pi'}\sum_{t\ge0}\gamma^t A^\pi(s_t,a_t)
&=\mathbb E_{\pi'}\sum_{t\ge0}\gamma^t r_t\\
&\quad+
\mathbb E_{\pi'}\sum_{t\ge0}
\big(\gamma^{t+1}V^\pi(s_{t+1})
-\gamma^tV^\pi(s_t)\big)\\
&=J(\pi')-\mathbb E_{s_0\sim\rho_0}[V^\pi(s_0)]\\
&=J(\pi')-J(\pi).
\end{aligned}
\tag{41}
$$

再用 $d^{\pi'}$ 的定义把折扣和改写成期望，即得式 (40)。理想的逐状态改进若保证

$$
\mathbb E_{a\sim\pi'(\cdot\mid s)}[A^\pi(s,a)]\ge0
\quad\text{对所有 }s,
$$

则 $J(\pi')\ge J(\pi)$。实际深度 RL 只能在采样状态和近似 critic 上优化，因此 TRPO 用 KL 邻域控制把旧状态分布替换成新状态分布所造成的误差。

### 3.5 REPS 的占用测度原问题与对偶

为突出流约束的作用，考虑有限状态动作空间。给定参考占用测度 $d_0(s,a)>0$，REPS 的一种折扣形式为

$$
\begin{aligned}
\max_d\quad&
\sum_{s,a}d(s,a)r(s,a)\\
\text{s.t.}\quad&
D_{\mathrm{KL}}(d\|d_0)\le\varepsilon,\\
&
\sum_a d(s,a)
=(1-\gamma)\rho_0(s)
+\gamma\sum_{\bar s,\bar a}
d(\bar s,\bar a)P(s\mid\bar s,\bar a),\\
&
\sum_{s,a}d(s,a)=1,\qquad d(s,a)\ge0.
\end{aligned}
\tag{42}
$$

对 KL 约束引入 $\eta\ge0$，对每个状态的流约束引入 $V(s)$，对归一化引入 $\lambda$。将含 $d(s,a)$ 的项收集起来：

$$
\begin{aligned}
\mathcal L
&=\eta\varepsilon
+(1-\gamma)\sum_s\rho_0(s)V(s)+\lambda\\
&\quad+\sum_{s,a}d(s,a)
\bigg[
r(s,a)
+\gamma\mathbb E_{s'\sim P(\cdot\mid s,a)}V(s')
-V(s)\\
&\hspace{42mm}
-\eta\log\frac{d(s,a)}{d_0(s,a)}
-\eta-\lambda
\bigg].
\end{aligned}
\tag{43}
$$

对 $d(s,a)$ 求一阶条件：

$$
r(s,a)+\gamma\mathbb E[V(s')]-V(s)
-\eta\left(\log\frac{d(s,a)}{d_0(s,a)}+1\right)
-\lambda=0.
$$

移项并指数化：

$$
d^*(s,a)
=d_0(s,a)
\exp\left(
\frac{
r(s,a)+\gamma\mathbb E[V(s')]-V(s)
}{\eta}
\right)
\exp\left(-1-\frac{\lambda}{\eta}\right).
\tag{44}
$$

最后一个因子与 $(s,a)$ 无关，由归一化吸收。记 Bellman residual

$$
\delta_V(s,a)
=r(s,a)+\gamma\mathbb E[V(s')]-V(s),
\tag{45}
$$

便得到

$$
\boxed{
d^*(s,a)
=\frac{
d_0(s,a)e^{\delta_V(s,a)/\eta}
}{
\mathbb E_{d_0}[e^{\delta_V/\eta}]
}.
}
\tag{46}
$$

这就是后文 REPS 指数权公式的来源。$V$ 不是为减小方差随意加入的 baseline，而是保证新分布满足 MDP 流约束的对偶变量。将式 (44) 对 $d$ 取上确界，可得关于 $(V,\eta)$ 的对偶目标；数值实现同时优化价值参数和温度，再用样本近似式 (46)。

---

## 四、理想改进分布的参数投影

理想分布 $q^*$ 通常不在参数族 $\Pi=\{\pi_\theta\}$ 内。算法必须做投影。

### 4.1 $D_{\mathrm{KL}}(q^*\|\pi_\theta)$：加权极大似然

最小化

$$
\min_\theta D_{\mathrm{KL}}(q^*\|\pi_\theta)
\quad\Longleftrightarrow\quad
\max_\theta\mathbb E_{q^*}[\log\pi_\theta(x)]
\tag{47}
$$

并以参考分布 $p$ 采样，可写成

$$
\max_\theta
\mathbb E_{x\sim p}
\left[
\frac{e^{f(x)/\eta}}{Z}\log\pi_\theta(x)
\right].
\tag{48}
$$

这就是 reward/advantage-weighted regression 的形状。它是 supervised-learning-friendly 的 M-step：高分样本获得更大的拟合权重。MPO 明确把非参数 E-step 与有约束的参数 M-step 分开；AWR 则直接在回放数据上做优势加权回归。

该方向常被称为“前向 KL 投影”（目标在 KL 第一参数），倾向覆盖 $q^*$ 的多个模态。但离线数据下只能在行为分布的经验支持上重加权；若 critic 对分布外动作过估计，闭式目标本身并不能消除外推误差。

若 $\pi_\theta$ 是全协方差高斯 $\mathcal N(\mu,\Sigma)$，式 (47) 的总体最优参数是 $q^*$ 的一、二阶矩：

$$
\mu^*=\mathbb E_{q^*}[x],
\qquad
\Sigma^*=\mathbb E_{q^*}
[(x-\mu^*)(x-\mu^*)^\top].
\tag{49}
$$

证明来自高斯负对数似然

$$
-\log\pi_{\mu,\Sigma}(x)
=\frac12(x-\mu)^\top\Sigma^{-1}(x-\mu)
+\frac12\log\det\Sigma+\frac d2\log(2\pi).
$$

对 $\mu$ 求导并令其为零：

$$
\Sigma^{-1}(\mu-\mathbb E_{q^*}[x])=0.
$$

代回 $\mu^*$ 后对 $\Sigma^{-1}$ 求导，得到式 (49) 的协方差。用 $p$ 的样本和式 (5) 的权重近似这些矩，便得到加权高斯拟合；CEM、MPO 的 M-step 与若干 CMA 更新都可在这一层比较。

### 4.2 $D_{\mathrm{KL}}(\pi_\theta\|q^*)$：变分策略更新

反向投影给出

$$
\min_\theta D_{\mathrm{KL}}(\pi_\theta\|q^*)
=\min_\theta
\mathbb E_{\pi_\theta}
\left[
\log\pi_\theta(x)-\log p(x)-\frac{f(x)}{\eta}
\right]+\text{const}.
\tag{50}
$$

它更接近最大熵 actor 的目标：提高价值，同时控制相对参考策略的 KL。SAC 在无显式旧策略参考时，以熵项和 soft $Q$ 构造相应的 Boltzmann 改进；其策略更新是在选定参数族内逼近 $\exp(Q/\alpha)$ 诱导的分布。

反向 KL 往往 mode-seeking，但这只是常见行为而非无条件定理；真实结果还取决于模型族、优化和支持集。

### 4.3 双峰分布下两种投影的差异

设目标分布为两个窄高斯的等权混合：

$$
q^*(x)
=\frac12\mathcal N(x;-m,\sigma^2)
+\frac12\mathcal N(x;m,\sigma^2),
\qquad m\gg\sigma,
\tag{51}
$$

而模型族只允许单高斯 $\pi_{\mu,s^2}$。

前向投影最小化 $D_{\mathrm{KL}}(q^*\|\pi)$。由式 (49)，

$$
\mu_{\mathrm F}=0,
\qquad
s_{\mathrm F}^2=m^2+\sigma^2.
\tag{52}
$$

它用一个宽高斯覆盖两个模态，并在两峰之间分配了目标分布本来很少具有的密度。

反向投影最小化 $D_{\mathrm{KL}}(\pi\|q^*)$。当 $m/\sigma$ 很大时，若 $\pi$ 同时覆盖两峰，它的大量质量会落在 $q^*$ 密度极低的中间区域，使

$$
\mathbb E_\pi[-\log q^*(x)]
$$

很大。局部最优因而接近 $\mathcal N(-m,\sigma^2)$ 或 $\mathcal N(m,\sigma^2)$，表现为择取一个模态。这个例子解释了 mode-covering 与 mode-seeking 的常见说法，也说明它们不是散度名称本身的绝对定理：若模型族能精确表示 $q^*$，两种投影都以 $q^*$ 为零散度解。

### 4.4 直接参数梯度：REINFORCE / policy gradient

若直接对

$$
J(\theta)=\mathbb E_{x\sim p_\theta}[f(x)]
$$

求导，得 score-function identity

$$
\nabla_\theta J
=\mathbb E_{p_\theta}
\left[f(x)\nabla_\theta\log p_\theta(x)\right].
\tag{53}
$$

在 MDP 中用因果分解、reward-to-go 和 baseline 化简，就得到策略梯度定理。自然策略梯度进一步左乘 $F^{-1}$。因此 PG 与分布优化有深刻联系，但“PG 等于最小化某个 KL”需要说明所用局部展开、采样分布和参数化；不能仅把 $p_\theta\approx p_{\rm old}$ 代入一个 KL 后便视为严格推导。

---

## 五、代表性策略优化算法

### 5.1 REPS：占用测度上的相对熵约束

Relative Entropy Policy Search 直接约束新旧状态--动作分布的 KL，并保持 Bellman 流一致性。第三节已经从占用测度原问题推出

$$
d_{k+1}(s,a)
\propto d_k(s,a)
\exp\left(\frac{r(s,a)+\gamma\mathbb E[V(s')]-V(s)}{\eta}\right).
\tag{54}
$$

REPS 最接近“式 (3) 在 MDP 中的严谨版本”：$V$ 不是随手加入的 baseline，而是占用测度可实现性约束的对偶变量。

给定从 $d_0$ 抽取的样本 $(s_i,a_i,r_i,s_i')$，可用

$$
\hat\delta_i
=r_i+\gamma V_\nu(s_i')-V_\nu(s_i)
\tag{55}
$$

构造经验对偶：

$$
\hat g(\eta,\nu)
=\eta\varepsilon
+\eta\log\left(
\frac1N\sum_{i=1}^N
e^{\hat\delta_i/\eta}
\right)
+(1-\gamma)\mathbb E_{s_0\sim\rho_0}[V_\nu(s_0)].
\tag{56}
$$

最小化 $\hat g$ 得到温度和价值参数。随后以

$$
\bar w_i
=\frac{e^{\hat\delta_i/\eta}}
{\sum_j e^{\hat\delta_j/\eta}}
\tag{57}
$$

拟合参数策略。原始占用测度更新与参数策略拟合是两个步骤；若策略族有限、状态分布只能由样本近似，投影后的策略不再严格实现式 (46) 的占用测度。

### 5.2 MPO：先非参数改进，再参数投影

MPO 将一次更新明确分成：

1. **E-step**：在旧策略附近求 $q(a\mid s)\propto\pi_k(a\mid s)e^{Q(s,a)/\eta}$；
2. **M-step**：用加权最大似然把 $q$ 投影到 $\pi_\theta$，并对均值和协方差施加 KL 约束。

这种拆分揭示了策略优化的两个误差源：critic/采样造成的**改进误差**，以及模型族/有限训练造成的**投影误差**。

对 replay 中的状态 $s\sim\mu$，MPO 的 E-step 可写为

$$
\begin{aligned}
\max_q\quad&
\mathbb E_{s\sim\mu}
\mathbb E_{a\sim q(\cdot\mid s)}[Q_{\mathrm{old}}(s,a)]\\
\text{s.t.}\quad&
\mathbb E_{s\sim\mu}
D_{\mathrm{KL}}\big(q(\cdot\mid s)\|
\pi_{\mathrm{old}}(\cdot\mid s)\big)
\le\varepsilon.
\end{aligned}
\tag{58}
$$

其解为

$$
q^*(a\mid s)
=\frac{
\pi_{\mathrm{old}}(a\mid s)
e^{Q_{\mathrm{old}}(s,a)/\eta}
}{Z(s)}.
\tag{59}
$$

若每个状态从旧策略采 $M$ 个动作 $a_{ij}$，则

$$
\bar w_{ij}
=\frac{e^{Q(s_i,a_{ij})/\eta}}
{\sum_{j'=1}^M e^{Q(s_i,a_{ij'})/\eta}}.
\tag{60}
$$

这里按状态分别归一化；否则某些整体价值较高的状态会吞掉其他状态的权重，而 E-step 的约束本来是条件动作分布的平均 KL。

M-step 解

$$
\max_\theta
\mathbb E_{s\sim\mu,a\sim q^*}
[\log\pi_\theta(a\mid s)]
\quad\text{s.t.}\quad
\mathbb E_{s\sim\mu}
D_{\mathrm{KL}}(
\pi_{\mathrm{old}}(\cdot\mid s)
\|\pi_\theta(\cdot\mid s))
\le\varepsilon_\pi.
\tag{61}
$$

实际 MPO 对高斯均值和协方差使用分离的 KL 约束，防止 actor 仅通过迅速缩小方差来提高加权样本的似然。E-step 的 $q^*$ 是样本上的非参数分布；M-step 才把它压回神经网络策略族。

### 5.3 AWR/AWAC：把策略改进写成回归

AWR 估计 $V$ 与优势，然后优化

$$
\max_\theta
\mathbb E_{(s,a)\sim\mathcal D}
\left[
\exp\left(\frac{A(s,a)}{\eta}\right)
\log\pi_\theta(a\mid s)
\right].
\tag{62}
$$

其工程价值是将 actor 更新化为稳定的监督学习。代价是策略无法为数据中没有的动作分配“凭空发现”的优势；性能依赖数据覆盖、价值估计和权重截断。

上述优势加权回归目标可直接由行为策略 $\mu(a\mid s)$ 附近的理想改进导出：

$$
q^*(a\mid s)
\propto
\mu(a\mid s)e^{A(s,a)/\eta}.
\tag{63}
$$

将 $q^*$ 正向投影到 $\pi_\theta$：

$$
\begin{aligned}
\arg\min_\theta
D_{\mathrm{KL}}(q^*\|\pi_\theta)
&=\arg\max_\theta
\mathbb E_{s}\mathbb E_{a\sim q^*}
[\log\pi_\theta(a\mid s)]\\
&=\arg\max_\theta
\mathbb E_{s,a\sim\mu}
\left[
\frac{e^{A(s,a)/\eta}}{Z(s)}
\log\pi_\theta(a\mid s)
\right].
\end{aligned}
\tag{64}
$$

$Z(s)$ 对 actor 参数无关，但不能在跨状态的有限样本平均中任意删除；实践通常以 batch 标准化、权重裁剪或每状态采样近似处理。AWR 使用由回报回归得到的优势，AWAC 使用 off-policy critic 的 $Q-V$。二者共享式 (64)，区别在价值估计、数据来源和工程正则。

### 5.4 NPG/TRPO/PPO：局部几何及其近似

- **NPG** 使用 Fisher 逆预条件策略梯度，是 KL 几何下的局部最速上升。
- **TRPO** 最大化旧策略采样的优势代理，同时约束平均策略 KL；共轭梯度近似求自然梯度方向，线搜索控制实际步长。
- **PPO** 用概率比裁剪或 KL penalty 构造易于一阶优化的代理。裁剪并不严格保证策略 KL 落在固定半径内，也不等于解式 (1) 的精确投影。

三者可放在同一 trust-region/mirror-descent 谱系中，但精确约束、局部二阶近似和启发式裁剪是三个不同层次。

具体地，performance difference lemma 给出

$$
J(\pi')-J(\pi)
=\frac1{1-\gamma}
\mathbb E_{s\sim d^{\pi'},a\sim\pi'}
[A^\pi(s,a)].
\tag{65}
$$

TRPO 将未知的新状态分布 $d^{\pi'}$ 换成旧分布 $d^\pi$，构造代理

$$
L_\pi(\pi')
=J(\pi)
+\frac1{1-\gamma}
\mathbb E_{s\sim d^\pi,a\sim\pi'}
[A^\pi(s,a)].
\tag{66}
$$

若数据来自旧策略，重要性采样把动作期望写成

$$
\mathbb E_{a\sim\pi'}
[A^\pi(s,a)]
=\mathbb E_{a\sim\pi}
\left[
\frac{\pi'(a\mid s)}{\pi(a\mid s)}
A^\pi(s,a)
\right].
\tag{67}
$$

记 $r_\theta(s,a)=\pi_\theta(a\mid s)/\pi_{\mathrm{old}}(a\mid s)$，TRPO 的实践问题为

$$
\begin{aligned}
\max_\theta\quad&
\mathbb E_{\pi_{\mathrm{old}}}
[r_\theta A^{\pi_{\mathrm{old}}}]\\
\text{s.t.}\quad&
\mathbb E_{s\sim d^{\pi_{\mathrm{old}}}}
D_{\mathrm{KL}}\big(
\pi_{\mathrm{old}}(\cdot\mid s)
\|\pi_\theta(\cdot\mid s)
\big)\le\delta.
\end{aligned}
\tag{68}
$$

在 $\theta_{\mathrm{old}}$ 附近线性化目标、二次展开 KL，便回到式 (20) 的自然梯度子问题。

PPO-Clip 保留上述重要性比率，却把显式 KL 约束换成逐样本代理

$$
L^{\mathrm{clip}}(\theta)
=\mathbb E\left[
\min\left(
r_\theta A,\,
\operatorname{clip}(r_\theta,1-\epsilon,1+\epsilon)A
\right)
\right].
\tag{69}
$$

若 $A>0$，$r_\theta>1+\epsilon$ 后第二支为常数；若 $A<0$，$r_\theta<1-\epsilon$ 后第二支为常数。共享参数仍会被其他样本更新，所以裁剪目标不保证所有比率留在区间内，也不保证平均 KL 小于固定阈值。它继承了 trust-region 的保守意图，没有继承前述 TRPO 问题的硬约束。

### 5.5 SAC：熵正则下的 soft policy improvement

最大熵 RL 的 soft policy improvement 为

$$
\pi_{k+1}(\cdot\mid s)
=\arg\min_{\pi\in\Pi}
D_{\mathrm{KL}}
\left(
\pi(\cdot\mid s)
\middle\|
\frac{\exp(Q^{\pi_k}(s,\cdot)/\alpha)}{Z(s)}
\right).
\tag{70}
$$

这与反向 KL 投影同型。有限离散动作或有界连续动作上，熵正则也可解释为相对均匀概率分布的 KL 正则；无界连续空间不存在可归一化的均匀概率分布，此时相应项是相对 Lebesgue 基测度定义的微分熵。若将基准换成旧策略或行为先验，则得到显式的 KL-regularized RL。温度 $\alpha$ 控制价值最大化与熵或先验保持之间的权衡。

展开上述 soft policy improvement 的 KL：

$$
\begin{aligned}
D_{\mathrm{KL}}\left(
\pi_\theta
\middle\|
\frac{e^{Q/\alpha}}{Z}
\right)
&=\mathbb E_{a\sim\pi_\theta}
\left[
\log\pi_\theta(a\mid s)
-\frac1\alpha Q(s,a)+\log Z(s)
\right].
\end{aligned}
\tag{71}
$$

$Z(s)$ 不依赖 actor 参数，乘以 $\alpha$ 后得到 SAC actor 损失

$$
\boxed{
J_\pi(\theta)
=\mathbb E_{s\sim\mathcal D,a\sim\pi_\theta}
[\alpha\log\pi_\theta(a\mid s)-Q_\phi(s,a)].
}
\tag{72}
$$

相应的 soft value 为

$$
V^\pi(s)
=\mathbb E_{a\sim\pi}
[Q^\pi(s,a)-\alpha\log\pi(a\mid s)],
\tag{73}
$$

critic 的一步目标是

$$
y
=r+\gamma(1-d)
\left[
\min_iQ_{\bar\phi_i}(s',a')
-\alpha\log\pi_\theta(a'\mid s')
\right],
\qquad a'\sim\pi_\theta(\cdot\mid s').
\tag{74}
$$

因此 SAC 不是简单地给普通 actor loss 加一个熵 bonus：熵同时进入策略改进和 Bellman backup，二者共同定义 soft MDP。

---

## 六、零阶优化、CEM、CMA-ES 与 MPPI

这组方法与策略梯度共享“优化一个分布”，但搜索变量常不同。

同一个符号 $x$ 在不同算法中代表不同对象：

$$
x=
\begin{cases}
u_{0:H-1}, & \text{有限时域开环控制序列},\\
\theta_\pi, & \text{整套反馈策略参数},\\
a, & \text{给定状态下的动作},\\
\tau, & \text{完整状态--动作轨迹},\\
(s,a), & \text{满足流约束的占用测度元素}.
\end{cases}
\tag{75a}
$$

因此，看到相同的权重 $e^{f/\eta}$ 还不足以判定算法等价。必须继续核对参考分布、样本生成方式和投影族。以代表性方法为例：

| 方法 | 随机变量 | 参考分布 | 效用 | 更新后的对象 |
|------|----------|----------|------|--------------|
| CEM | 候选解或控制序列 | 当前搜索分布 | elite 指示函数 | 搜索分布参数 |
| CMA-ES | 候选解或策略参数 | 当前高斯 | rank utility | 均值、协方差及演化路径 |
| MPPI | 控制扰动/控制序列 | 当前控制提议 | 负轨迹代价 | MPC 的开环均值序列 |
| REPS | 状态--动作占用 | 旧占用测度 | Bellman residual | 可实现的新占用测度 |
| MPO/AWR | 给定状态下的动作 | 旧策略或行为策略 | $Q$ 或优势 | 参数化反馈策略 |

前三者不要求学习一个跨状态共享的 actor；后两者必须处理状态条件和 MDP 动力学。参数空间 ES 虽然评估整条环境轨迹，但搜索分布位于 $\theta_\pi$ 上，也不同于轨迹分布 $p_\pi(\tau)$。

### 6.1 参数空间搜索

令 $x=\theta_\pi$ 为整套策略参数，搜索分布 $p_\phi(x)$ 在参数空间采样候选策略，每个候选经 rollout 得到适应度 $f(x)$。则

$$
\nabla_\phi\mathbb E_{x\sim p_\phi}[f(x)]
=\mathbb E[f(x)\nabla_\phi\log p_\phi(x)].
\tag{75}
$$

它与 REINFORCE 公式同型，但随机变量不同：REINFORCE 通常在动作/轨迹随机性上求策略参数梯度；ES 在策略参数的搜索分布上求梯度。

### 6.2 CEM 与 CMA-ES

CEM 选取精英样本，并对精英经验分布作最大似然投影；可视为加权极大似然使用硬阈值权重。Gaussian IGO/NES/CMA-ES 则在高斯搜索族上使用自然梯度或与之密切相关的均值、协方差更新。排序权重带来对适应度严格单调变换的不变性，而 Gibbs 权重一般不具备这一性质。

具体地，从 $p_k(x)=\mathcal N(\mu_k,\Sigma_k)$ 抽取 $N$ 个样本，令 $\mathcal E_k$ 为效用最高的 $\rho N$ 个精英。CEM 解

$$
(\mu_{k+1},\Sigma_{k+1})
=\arg\max_{\mu,\Sigma}
\sum_{i\in\mathcal E_k}
\log\mathcal N(x_i;\mu,\Sigma).
\tag{77a}
$$

由高斯极大似然，

$$
\mu_{k+1}
=\frac1{|\mathcal E_k|}
\sum_{i\in\mathcal E_k}x_i,
\qquad
\Sigma_{k+1}
=\frac1{|\mathcal E_k|}
\sum_{i\in\mathcal E_k}
(x_i-\mu_{k+1})(x_i-\mu_{k+1})^\top.
\tag{78a}
$$

它对应硬权重 $w_i=\mathbf1\{i\in\mathcal E_k\}$。CMA-ES 使用按排序确定的平滑权重，并包含步长控制、演化路径和保证协方差正定的更新结构；将“高斯族自然梯度”称作 CMA-ES 的全部内容会遗漏这些关键机制。

### 6.3 MPPI：控制序列分布的指数加权

MPPI 在控制扰动或控制序列上采样，用轨迹代价 $S_i$ 形成

$$
w_i\propto\exp(-S_i/\lambda),
\qquad
u_t\leftarrow u_t+\sum_iw_i\epsilon_{i,t}.
\tag{76}
$$

这与式 (5) 完全同形，且可由路径积分控制、重要性采样或信息论控制推导。但 MPPI 是每个 MPC 周期在线优化有限时域控制序列；policy optimization 通常离线优化可跨状态泛化的反馈策略。二者共享分布更新，不共享优化变量和部署方式。

若提议为

$$
u^{(i)}=\bar u+\epsilon^{(i)},
\qquad
\epsilon^{(i)}\sim\mathcal N(0,\Sigma),
$$

并把参考分布取为当前高斯，则 KL 正则化的低代价目标为

$$
q^*(u)
\propto
\mathcal N(u;\bar u,\Sigma)
\exp\left(-\frac{S(u)}{\lambda}\right).
\tag{81a}
$$

其均值为

$$
\mathbb E_{q^*}[u]
=\bar u+
\frac{
\mathbb E_{\epsilon}
[\epsilon e^{-S(\bar u+\epsilon)/\lambda}]
}{
\mathbb E_{\epsilon}
[e^{-S(\bar u+\epsilon)/\lambda}]
}.
\tag{82a}
$$

用样本替代期望即得指数加权扰动更新。执行时只施加新序列的第一个控制，再根据下一状态平移 horizon 并重新求解。与 actor 学习相比，它不把不同状态下的解压缩进一个共享神经网络，而是把计算留在每个控制周期。

---

## 七、生成式策略与扩散模型中的 score 表示

### 7.1 理想策略的 score

对连续动作，若

$$
q^*(a\mid s)
\propto p(a\mid s)
\exp(Q(s,a)/\eta),
$$

且各项可微，则对动作求梯度：

$$
\boxed{
\nabla_a\log q^*(a\mid s)
=\nabla_a\log p(a\mid s)
+\frac{1}{\eta}\nabla_aQ(s,a).
}
\tag{77}
$$

这是连接扩散策略与 KL 正则化策略改进的关键式。扩散模型学习的是不同噪声尺度下的 score；价值梯度可以在采样时引导 score，或在训练时成为 score-matching 的目标修正。

式 (77) 只描述干净动作分布。扩散模型在训练和采样时使用的是加噪边缘，不能把同一个 $\nabla_aQ(s,a)$ 原样加到所有噪声尺度。设前向核为

$$
q_t(x_t\mid a)
=\mathcal N(x_t;\alpha_t a,\sigma_t^2I),
\tag{78}
$$

参考策略的干净动作密度为 $p_0(a\mid s)$，理想改进为

$$
p_0^*(a\mid s)
=\frac{
p_0(a\mid s)e^{Q(s,a)/\eta}
}{Z(s)}.
\tag{79}
$$

其噪声边缘为

$$
p_t^*(x_t\mid s)
=\int q_t(x_t\mid a)p_0^*(a\mid s)\,\mathrm da.
\tag{80}
$$

对 $x_t$ 求 score，并使用 Fisher identity：

$$
\begin{aligned}
\nabla_{x_t}\log p_t^*(x_t\mid s)
&=\mathbb E_{a\sim p^*(a\mid x_t,s)}
\left[
\nabla_{x_t}\log q_t(x_t\mid a)
\right]\\
&=-\frac1{\sigma_t^2}
\left(
x_t-\alpha_t
\mathbb E_{p^*(a\mid x_t,s)}[a]
\right).
\end{aligned}
\tag{81}
$$

参考噪声边缘同理：

$$
\nabla_{x_t}\log p_t(x_t\mid s)
=-\frac1{\sigma_t^2}
\left(
x_t-\alpha_t
\mathbb E_{p(a\mid x_t,s)}[a]
\right).
\tag{82}
$$

两者之差取决于价值重权后验与参考后验的均值差：

$$
\nabla_{x_t}\log p_t^*
-\nabla_{x_t}\log p_t
=\frac{\alpha_t}{\sigma_t^2}
\left(
\mathbb E_{p^*(a\mid x_t,s)}[a]
-\mathbb E_{p(a\mid x_t,s)}[a]
\right).
\tag{83}
$$

只有在 $t=0$、或额外近似把带噪状态对应到干净动作时，式 (83) 才退化为直观的 $\eta^{-1}\nabla_aQ$ 修正。一般噪声尺度上的精确 guidance 是一个后验平均问题，这也是 Q-guidance 需要训练 noisy critic、使用重建动作，或直接做 Q-score matching 的原因。

### 7.2 价值引导与 classifier guidance

Classifier guidance 对条件 $c$ 使用

$$
\nabla_x\log p(x\mid c)
=\nabla_x\log p(x)+\nabla_x\log p(c\mid x).
\tag{84}
$$

CFG 用条件与无条件 score 的线性组合近似这一效果。令 $Q/\eta=\log p(c\mid x)$ 时，式 (77) 与 classifier guidance 具有相同形式；在控制问题中，$Q$ 表示未来折扣回报的条件期望，而非标签似然。价值引导、classifier guidance 与能量引导均可写成基分布 score 加势函数梯度，但势函数的语义和估计方法不同。

三者可以按所用势函数区分：

$$
\begin{array}{ll}
\text{classifier guidance:}
&\Phi(x_t,c)=\log p(c\mid x_t),\\[2mm]
\text{energy guidance:}
&\Phi(x_t)=-E(x_t),\\[2mm]
\text{value guidance:}
&\Phi(s,a)=Q^\pi(s,a)/\eta.
\end{array}
\tag{85}
$$

Classifier guidance 的 Bayes 恒等式在同一噪声尺度成立：

$$
\nabla_{x_t}\log p_t(x_t\mid c)
=\nabla_{x_t}\log p_t(x_t)
+\nabla_{x_t}\log p_t(c\mid x_t).
\tag{86}
$$

CFG 则用条件与无条件网络输出之差近似第二项，并常乘 guidance scale $w$：

$$
s_{\mathrm{CFG}}
=s_{\mathrm{uncond}}
+w(s_{\mathrm{cond}}-s_{\mathrm{uncond}}).
\tag{87}
$$

当 $w\ne1$ 时，它甚至不再对应原始条件分布 $p(x\mid c)$ 的精确 score，而是有意改变条件强度。价值引导还多一层困难：$Q^\pi(s,a)$ 定义在可执行动作上，$x_t$ 通常是人工加噪的动作变量，不是环境 MDP 中的动作。

### 7.3 四条 diffusion policy + RL 路线

#### 路线 A：行为建模与价值最大化的联合损失

Diffusion-QL 以扩散行为克隆保持数据支持，再加入最大化 critic 的项。它把“贴近行为”与“朝高价值移动”放在同一次参数更新中。优点是直接；难点是权重标度、critic 外推误差，以及需要穿过采样器反传或使用近似。

其抽象目标可写成

$$
J_{\mathrm{DQL}}(\theta)
=\mathcal L_{\mathrm{denoise}}(\theta)
-\lambda_Q
\mathbb E_{s\sim\mathcal D,\,
a\sim\pi_\theta(\cdot\mid s)}
[Q_\phi(s,a)].
\tag{88}
$$

第一项把策略拉向行为数据，第二项沿 critic 提高动作价值。它一般不是式 (79) 某一 KL 投影的精确离散化：去噪 MSE 定义在随机噪声尺度上，$Q$ 项定义在最终动作上，两项的相对系数还受动作维数、噪声日程和 $Q$ 数值尺度影响。其统一意义在于同时近似“参考分布保持”和“价值倾斜”，而非具有闭式 Gibbs 解。

#### 路线 B：价值引导采样

训练一个行为/轨迹扩散模型，推理时用 $\nabla_aQ$ 或目标代价修正每一步 score。模型本体可以不改，任务通过 guidance 改变。优点是组合灵活；难点是噪声动作上的 $Q$ 或其梯度未必可靠，强引导还会把样本推出数据流形。

若使用重建动作

$$
\hat a_0(x_t,s)
=\frac{x_t-\sigma_t\hat\epsilon_\theta(x_t,t,s)}
{\alpha_t},
\tag{89}
$$

一种常见近似是把

$$
\nabla_{x_t}Q(s,\hat a_0)
=
\left(\frac{\partial\hat a_0}{\partial x_t}\right)^\top
\nabla_aQ(s,\hat a_0)
\tag{90}
$$

加入反向 score。是否对去噪网络的 Jacobian 反传、是否 stop-gradient，以及 guidance scale 如何随 $t$ 变化，都会得到不同算法。早期高噪声处 $\hat a_0$ 误差大，critic 又未在这种重建动作上训练，过强 guidance 容易破坏行为先验。

#### 路线 C：Q-score matching

由式 (77) 直接构造目标，使策略 score 向“行为 score + 价值梯度”对齐。这比把 $Q$ 简单加到去噪 MSE 更贴近扩散模型的概率结构。实际实现仍需处理带噪尺度下的价值梯度与 critic 偏差。

在干净动作层，一个理想目标是

$$
\mathcal L_{\mathrm{QSM}}
=\mathbb E
\left[
\left\|
s_\theta(a,s)
-s_{\mathrm{ref}}(a,s)
-\frac1\eta\nabla_aQ_\phi(s,a)
\right\|^2
\right].
\tag{91}
$$

实际扩散策略只直接提供 $s_\theta(x_t,t,s)$。因此算法必须把式 (91) 搬到噪声边缘，或利用 denoiser 与 clean score 的关系构造可训练目标。式 (83) 表明精确 noisy target 不是简单的 clean $Q$ 梯度；QSM 的具体理论假设和估计器决定了它如何近似这一后验修正。

#### 路线 D：把去噪链视为内层 MDP

DPPO/DDPO 一类方法把每个反向去噪转移视为可计算密度的随机策略。最终环境奖励分配给整条去噪链，再对各转移做 policy gradient/PPO。机器人 DPPO 外面还有环境时间，形成“环境 MDP + 去噪 MDP”两层结构。它不需要最终动作边缘密度的闭式表达，但计算和方差随去噪步数增加。

设反向链为

$$
p_\theta(a^{0:K}\mid s)
=p(a^K)\prod_{k=0}^{K-1}
p_\theta(a^k\mid a^{k+1},s).
\tag{92}
$$

最终动作边缘

$$
\pi_\theta(a^0\mid s)
=\int p_\theta(a^{0:K}\mid s)\,
\mathrm da^{1:K}
\tag{93}
$$

通常无闭式，但联合路径的对数导数可分解：

$$
\nabla_\theta\log p_\theta(a^{0:K}\mid s)
=\sum_{k=0}^{K-1}
\nabla_\theta\log
p_\theta(a^k\mid a^{k+1},s).
\tag{94}
$$

因此对只依赖 $a^0$ 的回报 $R(a^0)$，

$$
\nabla_\theta\mathbb E[R(a^0)]
=\mathbb E\left[
R(a^0)
\sum_{k=0}^{K-1}
\nabla_\theta\log
p_\theta(a^k\mid a^{k+1},s)
\right].
\tag{95}
$$

减去只依赖外层状态的 baseline 后，生成同一最终动作的各去噪步共享外层优势。DPPO 再对每个去噪核构造 PPO 比率。

若新旧联合路径分别为 $P_\theta$ 与 $P_{\mathrm{old}}$，链式 KL 为

$$
D_{\mathrm{KL}}(P_\theta\|P_{\mathrm{old}})
=\sum_k
\mathbb E_{P_\theta}
\left[
D_{\mathrm{KL}}\big(
p_\theta(a^k\mid a^{k+1},s)
\|p_{\mathrm{old}}(a^k\mid a^{k+1},s)
\big)
\right].
\tag{96}
$$

最终动作是联合路径的可测函数。由数据处理不等式，

$$
D_{\mathrm{KL}}\big(
\pi_\theta(a^0\mid s)
\|\pi_{\mathrm{old}}(a^0\mid s)
\big)
\le
D_{\mathrm{KL}}(P_\theta\|P_{\mathrm{old}}).
\tag{97}
$$

控制每一步去噪核因而可以控制一个更强的路径分布距离，但反向不成立：最终动作边缘很接近，并不意味着中间去噪路径接近。PPO 对各核分别裁剪也仍不是对式 (96) 的硬约束。

这四条路线分别对应：**目标值正则、采样时引导、score 投影、路径概率优化**。它们都在逼近高价值的指数倾斜分布，却在何处近似 $q^*$ 上作了不同选择。

### 7.4 四条路线的误差分解

将理想改进记为 $q^*$，实际扩散策略记为 $\pi_\theta$。误差至少来自四层：

1. **价值误差**：$\hat Q$ 与真实 $Q^\pi$ 不同，指数倾斜会放大高估区域；
2. **噪声尺度误差**：clean $Q$ 或 clean score 被近似搬到 $x_t$；
3. **投影误差**：有限扩散网络和给定噪声日程不能精确表示目标 score；
4. **优化误差**：有限数据、有限去噪步和 PPO/score-matching 求解器未达到总体最优。

Diffusion-QL 主要显式控制第一项去噪投影与最终价值之间的折中；guidance 把第二层近似留到推理；QSM 在 score 层处理第二、三项；DPPO 绕开最终边缘 score，转而优化完整路径概率，但承担更长随机计算图的方差和计算成本。这个分解比按 on-policy/off-policy 简单分类更能说明方法之间的实质差别。

---

## 八、算法比较维度

下列维度共同决定分布优化算法的统计性质与数值实现。

| 维度 | 典型选择 | 直接影响 |
|------|----------|----------|
| 优化对象 | 动作、动作序列、轨迹、占用测度、策略参数 | 是否遵守 MDP 流约束；能否跨状态泛化 |
| 参考分布 | 当前策略、行为策略、均匀测度、预训练生成模型 | 支持集、保守性与探索范围 |
| 效用信号 | return、$Q$、advantage、Bellman residual、rank | 偏差、方差与不变性 |
| 分布步长 | KL 约束、温度、熵、ratio clip | 更新稳定性；是否有严格信赖域 |
| 投影方向 | $D_{\rm KL}(q^*\|\pi)$ 或 $D_{\rm KL}(\pi\|q^*)$ | 覆盖/择模倾向；采样方式 |
| 模型族 | 高斯、GMM、能量模型、flow、diffusion | 表达多模态的能力与概率计算代价 |
| 信息来源 | on-policy、replay、静态离线数据、模型 rollout | 重要性校正与分布偏移 |
| 求解器 | 自然梯度、加权 MLE、PPO、score matching、CEM | 数值误差和实现复杂度 |

算法之间的主要差别通常来自模型族、价值估计、投影方向或概率估计方法。按这些维度比较，可以避免仅凭算法名称判断其理论性质。

---

## 九、适用条件与理论边界

### 9.1 指数重权的估计误差

$e^{\hat A/\eta}$ 对优势误差是指数敏感的。小温度提高选择性，也会降低有效样本量。工程上常见减去 batch 最大值、权重裁剪、top-$k$、rank transform 和自适应温度；这些修改提高稳定性，但改变了精确的 Gibbs 解。

### 9.2 参考分布的支持集约束

若 $p(x)=0$，则式 (3) 必有 $q^*(x)=0$。KL 正则更新只能重分配参考分布已有的概率质量。离线 RL 中这是一种安全性，也是一种根本限制；生成模型表达能力再强，也不能从有限数据自动获得未覆盖区域的可靠价值。

### 9.3 策略更新引起的状态分布偏移

逐状态式 (31) 往往用旧策略状态分布估计。新策略改变动作后也改变未来状态访问分布。TRPO 的性能界、占用测度方法和保守策略迭代都在处理这项二阶效应。只优化固定数据上的动作条件分布，不等于完整求解 MDP。

### 9.4 不同层次的 KL 约束

神经网络参数的欧氏距离通常不能控制策略分布距离；单步动作 KL 也未必控制长轨迹 KL。扩散模型中，各去噪转移 KL 的和约束的是扩展路径分布，不等于最终动作边缘 KL，尽管数据处理不等式可以给出方向性的界。

### 9.5 非参数更新与扩散模型参数化

扩散策略仍是由神经网络参数化的生成模型。“非参数更新”在 MPO/REPS 语境中通常指 E-step 先对样本权重或任意分布求解，不受 actor 参数族约束；随后仍需投影。称 diffusion 为非参数分布容易混淆统计学术语。

---

## 十、算法构造的基本要素

一个完整的分布策略优化方法需要依次确定以下要素。

1. **随机变量。** 优化对象可以是逐状态动作、开环控制序列、轨迹、占用测度或策略参数分布。
2. **参考分布。** 在线方法通常采用当前策略，离线方法采用行为策略或数据分布，生成式方法还可采用预训练 diffusion/flow 作为先验。
3. **效用信号。** 可选累计回报、$Q$、优势、reward model 输出或轨迹级排序；不同选择具有不同的偏差与方差。
4. **理想改进分布。** 式 (3) 或带流约束的占用测度问题给出参数化之前的理论目标。
5. **表示族。** 高斯适合近单峰分布；GMM、flow 和 diffusion 可表示更复杂的多峰动作分布。
6. **投影与估计。** 常见方法包括加权 MLE、反向 KL actor、自然梯度、score matching 和去噪链 policy gradient。
7. **支持集与分布偏移。** 价值估计的可信区域及新策略诱导的状态分布决定离线或异策略更新的误差。
8. **有限样本控制。** 温度、KL 半径、权重裁剪、ESS、target KL 和 early stopping 用于控制估计误差与更新幅度。

上述分解将理论目标与参数化实现分开，并区分改进分布误差、投影误差和价值估计误差。

---

## 十一、结论

本文讨论的方法均包含三个相互区分的层次：相对于参考分布定义正则化改进，在分布空间中求得指数倾斜解，再将该解投影到可计算的策略族。

- 在无限维分布空间，理想更新是 Gibbs/Boltzmann 重加权。
- 在概率几何的局部，KL 变成 Fisher 度量，得到自然梯度。
- 在有限步优化中，它表现为负熵镜像下降或 trust-region/proximal update。
- 在样本层面，它变成指数权、精英选择、重要性采样和加权极大似然。
- 在生成式策略中，同一更新可写成 log-density 或 score 的加法修正。

REPS/MPO/AWR、NPG/TRPO/PPO、SAC、CEM/CMA-ES、MPPI 与 diffusion-RL 均可由上述层次分析。它们分别作用于不同的随机对象，并采用不同的数据条件、模型族、投影方向和数值约束。

---

## 十二、文献与代码索引

### 12.1 理论主线

1. Ollivier, Arnold, Auger, Hansen, *Information-Geometric Optimization Algorithms: A Unifying Picture via Invariance Principles*, JMLR 2017（[arXiv](https://arxiv.org/abs/1106.3708)）。
2. Amari, *Natural Gradient Works Efficiently in Learning*, Neural Computation 1998。
3. Kakade, *A Natural Policy Gradient*, NeurIPS 2001。
4. Levine, *Reinforcement Learning and Control as Probabilistic Inference: Tutorial and Review*, 2018（[arXiv](https://arxiv.org/abs/1805.00909)）。
5. Tomar et al., *Mirror Descent Policy Optimization*, 2020（[arXiv](https://arxiv.org/abs/2005.09814)，[代码](https://github.com/manantomar/Mirror-Descent-Policy-Optimization)）。

### 12.2 相对熵与策略改进

6. Peters, Mulling, Altun, *Relative Entropy Policy Search*, AAAI 2010。
7. Abdolmaleki et al., *Maximum a Posteriori Policy Optimisation*, 2018（[arXiv](https://arxiv.org/abs/1806.06920)）。
8. Peng et al., *Advantage-Weighted Regression*, 2019（[arXiv](https://arxiv.org/abs/1910.00177)）。
9. Schulman et al., *Trust Region Policy Optimization*, ICML 2015（[PMLR](https://proceedings.mlr.press/v37/schulman15.html)）。
10. Schulman et al., *Proximal Policy Optimization Algorithms*, 2017（[arXiv](https://arxiv.org/abs/1707.06347)）。
11. Haarnoja et al., *Soft Actor-Critic*, ICML 2018（[PMLR](https://proceedings.mlr.press/v80/haarnoja18b.html)）。

### 12.3 黑箱优化与采样控制

12. Hansen, *The CMA Evolution Strategy: A Tutorial*, 2016（[arXiv](https://arxiv.org/abs/1604.00772)）。
13. Williams, Aldrich, Theodorou, *Model Predictive Path Integral Control using Covariance Variable Importance Sampling*, 2015（[arXiv](https://arxiv.org/abs/1509.01149)）。
14. Williams et al., *Information Theoretic Model Predictive Control*, 2017（[arXiv](https://arxiv.org/abs/1707.02342)）。
15. Jordana et al., *An Introduction to Zero-Order Optimization Techniques for Robotics*, 2025；代码：[ajordana/zoo-rob](https://github.com/ajordana/zoo-rob)。

### 12.4 扩散策略与 RL

16. Wang, Hunt, Zhou, *Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning*（Diffusion-QL）, 2022（[arXiv](https://arxiv.org/abs/2208.06193)，[代码](https://github.com/twitter/diffusion-rl)）。
17. Psenka et al., *Learning a Diffusion Model Policy from Rewards via Q-Score Matching*, 2023（[arXiv](https://arxiv.org/abs/2312.11752)，[代码](https://github.com/escontra/score_matching_rl)）。
18. Ren et al., *Diffusion Policy Policy Optimization*, 2024（[arXiv](https://arxiv.org/abs/2409.00588)，[代码](https://github.com/irom-princeton/dppo)）。
19. Black et al., *Training Diffusion Models with Reinforcement Learning*（DDPO）, 2023（[arXiv](https://arxiv.org/abs/2305.13301)）。
20. Jackson et al., *Policy-Guided Diffusion*, 2024（[arXiv](https://arxiv.org/abs/2404.06356)，[代码](https://github.com/EmptyJackson/policy-guided-diffusion)）。

### 12.5 建议阅读顺序

先读本文第一、二、四节建立“理想分布 + 投影”的骨架；随后按兴趣选择：策略优化读 REPS → MPO → TRPO/PPO → SAC，零阶控制读 IGO → CMA-ES → MPPI，生成式策略读 Diffusion Policy → Diffusion-QL/QSM → DPPO。最后回看第三节的动作、轨迹和占用测度之分，可以避免把只有形式相似的公式误认为严格等价。

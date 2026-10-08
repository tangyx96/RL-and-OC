# 零阶最优控制

> 分节读书笔记的延伸。文献：Jordana et al., *An Introduction to Zero-Order Optimization Techniques for Robotics*（arXiv:2506.22087）。按章节整理见 [`zero-order-optimization-study-notes.md`](zero-order-optimization-study-notes.md)。

## 摘要

从随机预测采样出发，经高斯平滑、CEM、MPPI，再到朗之万动力学、SVGD、扩散退火与 CMA-ES，可以把零阶最优控制里的常用方法放在同一套玻尔兹曼分布语言下。MPPI 估计的是 $p_q\propto q\exp(-J/\lambda)$ 的均值。这一位移等于 log-sum-exp 目标在协方差度量下的自然梯度步，也等于沿高斯平滑后的玻尔兹曼密度、以 $\Sigma$ 为预条件走一步。温度、协方差和噪声日程决定探索与集中；是否更新协方差、协方差绕哪一个均值、是否保留朗之万噪声，则把 CEM、MPPI、CMA 和扩散式 MPC 区分开来。

## 1. 基础：随机预测采样与高斯平滑

### 1.1 问题设定与挑战

有限时域最优控制（只优化开环序列，状态由动力学递推）：

```math
\min_{\mathbf{u}_{0:H-1}} J(\mathbf{u}_{0:H-1}) = \sum_{t=0}^{H-1} c(\mathbf{x}_t, \mathbf{u}_t) + c_f(\mathbf{x}_H)
```

```math
\mathbf{x}_{t+1} = f(\mathbf{x}_t, \mathbf{u}_t)
```

决策维数是 $H\times\dim(\mathbf{u})$。接触会使 $J$ 高度非凸，MPC 还要求在很短时间内给出第一步控制。

### 1.2 随机预测采样

1. 采样：$\mathbf{u}^{(i)}\sim q(\mathbf{u})$，$i=1,\dots,N$
2. 评估：$J^{(i)}=J(\mathbf{u}^{(i)})$
3. 选择：$\mathbf{u}^*=\arg\min_i J^{(i)}$

$q$ 通常取当前均值附近的高斯。维数升高后，要靠均匀铺点覆盖空间，样本需求大致随维数指数增长，因此后面都改为局部采样并迭代更新分布。

### 1.3 高斯平滑

把 $J$ 与高斯卷积，得到光滑代理：

```math
J_\sigma(\bar{\mathbf{u}})
= \mathbb{E}_{\epsilon\sim\mathcal{N}(0,\Sigma)}[J(\bar{\mathbf{u}}+\epsilon)].
```

令 $\mathbf{z}=\bar{\mathbf{u}}+\epsilon$，则 $\mathbf{z}\sim\mathcal{N}(\bar{\mathbf{u}},\Sigma)$。求导只作用在密度上，$\nabla_{\bar{\mathbf{u}}}\log\phi(\mathbf{z};\bar{\mathbf{u}},\Sigma)=\Sigma^{-1}\epsilon$，因此

```math
\nabla J_\sigma(\bar{\mathbf{u}})
= \mathbb{E}_{\epsilon\sim\mathcal{N}(0,\Sigma)}\bigl[J(\bar{\mathbf{u}}+\epsilon)\,\Sigma^{-1}\epsilon\bigr].
```

$J$ 只需在高斯测度下可积，便可交换积分与求导，不必存在 $\nabla J$。$\mathbb{E}[\Sigma^{-1}\epsilon]=0$，减去 $J(\bar{\mathbf{u}})$ 不改变期望，只降低方差。平滑后 $J_\sigma$ 任意阶可微；窄阱会被抹平一些，极小点也可能略有移动。

## 2. 交叉熵方法与重要性采样

### 2.1 CEM 流程

在高斯族 $\mathcal{N}(\boldsymbol{\mu},\Sigma)$ 上迭代：

1. 初始化 $\mathcal{N}(\boldsymbol{\mu}_0,\Sigma_0)$
2. 对 $k=0,1,2,\dots$：采样 $\mathbf{u}^{(i)}\sim\mathcal{N}(\boldsymbol{\mu}_k,\Sigma_k)$，计算 $J^{(i)}$
3. 保留代价最低的约 $\rho\%$ 样本（精英集）
4. 用精英集的样本均值更新 $\boldsymbol{\mu}$，再以这个新均值为中心计算协方差

这是交叉熵法的极大似然更新：高斯族拟合精英的经验分布，协方差围绕更新后的均值。自然梯度形式的 CMA 不同，协方差必须围绕采样时的均值，见 §2.2 末。

### 2.2 玻尔兹曼分布与指数权重

把低代价控制写成玻尔兹曼分布

```math
p^*(\mathbf{u})\propto\exp\bigl(-J(\mathbf{u})/\lambda\bigr).
```

它的均值和协方差是对 $p^*$ 的矩。从当前提议 $q=\mathcal{N}(\boldsymbol{\mu}_k,\Sigma_k)$ 抽样时，$p^*$ 的重要性权重正比于 $\exp(-J/\lambda)/q(\mathbf{u})$，不是 $\exp(-J/\lambda)$ 本身。

信息论目标 $\min_q\mathbb{E}_q[J]+\lambda D_{\mathrm{KL}}(q\|p_0)$ 的解是另一个分布。先验取当前高斯时，

```math
p_q(\mathbf{u})\propto\exp\bigl(-J(\mathbf{u})/\lambda\bigr)\,q(\mathbf{u}).
```

从 $q$ 估计 $p_q$ 的矩，$q$ 与重要性比中的 $q$ 相消，权重才是指数代价：

```math
w_i=\exp\bigl(-J(\mathbf{u}^{(i)})/\lambda\bigr),
\qquad
\boldsymbol{\mu}_{k+1}=\frac{\sum_{i=1}^N w_i\mathbf{u}^{(i)}}{\sum_{i=1}^N w_i}.
```

按 $p_q$ 的矩估计协方差时，中心是这个新均值。$\lambda$ 较小、权重很尖时，接近只留最精英的一小撮；$\rho\%$ 截断则是固定比例的均匀权。温度和分位数是两种把搜索分布收紧的方式。

只更新均值、固定 $\Sigma$，就是下一节的 MPPI。均值与 $\Sigma$ 一起更新时，指数权给出 MPPI-CMA，排序权给出 CMA-ES。长视界下 $\Sigma$ 取块对角，每时刻一块 $n_u\times n_u$。自然梯度 CMA 的协方差一步必须围绕采样时的均值；CEM 与这里的矩匹配相同，围绕更新后的均值。

## 3. MPPI 的两种推导路径

本节固定 $\Sigma$，只更新均值 $\mathbf{v}$。

### 3.1 信息论与 Log-Sum-Exp

目标分布取先验与玻尔兹曼的乘积：

```math
p_q(\mathbf{u})=\frac{1}{Z}\exp\bigl(-J(\mathbf{u})/\lambda\bigr)\,q(\mathbf{u}),
\qquad
q=\mathcal{N}(\mathbf{v},\Sigma)
```

均值的自归一估计为

```math
\mathbb{E}_{p_q}[\mathbf{u}]
=\frac{\mathbb{E}_q\bigl[\mathbf{u}\exp(-J(\mathbf{u})/\lambda)\bigr]}{\mathbb{E}_q\bigl[\exp(-J(\mathbf{u})/\lambda)\bigr]}
```

令 $\mathbf{u}=\mathbf{v}+\epsilon$，$\epsilon\sim\mathcal{N}(0,\Sigma)$：

```math
\mathbf{v}^*
=\mathbf{v}
+\frac{\mathbb{E}_{\epsilon}\bigl[\epsilon\exp(-J(\mathbf{v}+\epsilon)/\lambda)\bigr]}{\mathbb{E}_{\epsilon}\bigl[\exp(-J(\mathbf{v}+\epsilon)/\lambda)\bigr]}
```

有限样本即 MPPI：

```math
\mathbf{v}^* = \mathbf{v} + \frac{\sum_{i=1}^N w_i\epsilon^{(i)}}{\sum_{i=1}^N w_i},
\qquad
w_i = \exp\bigl(-J(\mathbf{v}+\epsilon^{(i)})/\lambda\bigr).
```

同一位移来自 log-sum-exp 平滑

```math
J_{\Sigma,\lambda}(\mathbf{v})
=-\lambda\log\mathbb{E}_{\epsilon\sim\mathcal{N}(0,\Sigma)}\bigl[\exp(-J(\mathbf{v}+\epsilon)/\lambda)\bigr]
```

的自然梯度，而不是它的欧氏梯度。对 $\mathbf{v}$ 求导只作用在高斯密度上，

```math
\nabla J_{\Sigma,\lambda}(\mathbf{v})
=-\lambda\,\Sigma^{-1}
\frac{\mathbb{E}[\epsilon\exp(-J(\mathbf{v}+\epsilon)/\lambda)]}
{\mathbb{E}[\exp(-J(\mathbf{v}+\epsilon)/\lambda)]}.
```

均值参数的 Fisher 信息为 $F=\Sigma^{-1}$。步长取 $1/\lambda$ 时，

```math
\mathbf{v}-\frac{1}{\lambda}F^{-1}\nabla J_{\Sigma,\lambda}(\mathbf{v})
=\mathbf{v}+\frac{\mathbb{E}[w\epsilon]}{\mathbb{E}[w]},
```

正是上面的 $\mathbf{v}^*$。$\lambda\to\infty$ 时 $J_{\Sigma,\lambda}$ 回到 §1.3 的 $J_\sigma$；$\lambda\to 0$ 时接近硬最小。实现上常减 $\min_i J^{(i)}$ 再取指数，只防溢出。样本极多时加权平均几乎没有随机性，需要靠 $\lambda$、$\Sigma$ 或重启保持探索。

### 3.2 朗之万动力学

目标密度 $p$ 的过阻尼朗之万方程是

```math
\mathrm{d}\mathbf{u}=\nabla\log p(\mathbf{u})\,\mathrm{d}t+\sqrt{2}\,\mathrm{d}W.
```

Euler–Maruyama 离散为

```math
\mathbf{u}_{k+1}=\mathbf{u}_k+\alpha\nabla\log p(\mathbf{u}_k)+\sqrt{2\alpha}\,\mathbf{z}_k.
```

$p^*\propto\exp(-J/\lambda)$ 的得分含 $\nabla J$，零阶设定下并不直接可用。改为高斯平滑后的密度

```math
p_\sigma(\mathbf{u})
=\int p^*(\mathbf{u}')\,\mathcal{N}(\mathbf{u}\mid\mathbf{u}',\Sigma)\,d\mathbf{u}'.
```

核的得分为 $\nabla_{\mathbf{u}}\log\mathcal{N}(\mathbf{u}\mid\mathbf{u}',\Sigma)=-\Sigma^{-1}(\mathbf{u}-\mathbf{u}')$，因此

```math
\nabla\log p_\sigma(\mathbf{u})
=-\Sigma^{-1}
\frac{\int p^*(\mathbf{u}')(\mathbf{u}-\mathbf{u}')\,\mathcal{N}(\mathbf{u}\mid\mathbf{u}',\Sigma)\,d\mathbf{u}'}{p_\sigma(\mathbf{u})}.
```

令 $\epsilon=\mathbf{u}'-\mathbf{u}$。高斯核关于 $\epsilon$ 对称，$\mathcal{N}(\mathbf{u}\mid\mathbf{u}+\epsilon,\Sigma)=\phi(\epsilon)$，且 $\mathbf{u}-\mathbf{u}'=-\epsilon$，分子里的负号与前面的负号相消：

```math
\nabla\log p_\sigma(\mathbf{u})
=\Sigma^{-1}
\frac{\mathbb{E}[p^*(\mathbf{u}+\epsilon)\,\epsilon]}{\mathbb{E}[p^*(\mathbf{u}+\epsilon)]}.
```

$p^*\propto\exp(-J/\lambda)$ 时，用 $\epsilon^{(i)}\sim\mathcal{N}(0,\Sigma)$ 估计，

```math
\nabla\log p_\sigma(\mathbf{u})
\approx\Sigma^{-1}\frac{\sum_{i=1}^N w_i\epsilon^{(i)}}{\sum_{i=1}^N w_i},
\qquad
w_i=\exp\bigl(-J(\mathbf{u}+\epsilon^{(i)})/\lambda\bigr).
```

于是 $\Sigma\nabla\log p_\sigma(\mathbf{v})=\sum_i w_i\epsilon^{(i)}/\sum_i w_i$。MPPI 的均值更新就是 $\mathbf{v}+\Sigma\nabla\log p_\sigma(\mathbf{v})$：在 $\Sigma$ 度量下沿平滑得分走一步，步长已经吸收在预条件里。配套的预条件朗之万噪声是 $\sqrt{2}\,\Sigma^{1/2}\mathbf{z}$，不是标量步长下的 $\sqrt{2\alpha}\,\mathbf{z}$。MPPI 丢掉这一项，只保留漂移；随机性来自每步重新采样的 $\epsilon^{(i)}$。

## 4. 朗之万动力学与扩散模型

### 4.1 物理图像

过阻尼朗之万方程：

```math
\gamma\frac{d\mathbf{x}}{dt}=-\nabla U(\mathbf{x})+\sqrt{2\gamma k_B T}\,\boldsymbol{\xi}(t)
```

$U=-\log p$、$\gamma=1$、$k_B T=1$ 时，这就是上一节的过阻尼朗之万。温度降低，样本集中到 $U$ 的低谷，也就是模拟退火。

### 4.2 得分匹配

得分 $s(\mathbf{x})=\nabla_{\mathbf{x}}\log p(\mathbf{x})$。用网络拟合平滑密度的得分，可用去噪得分匹配：

```math
\mathcal{L}(\theta)
=\mathbb{E}_{\mathbf{x}\sim p}
\mathbb{E}_{\tilde{\mathbf{x}}\sim q(\tilde{\mathbf{x}}\mid\mathbf{x})}
\bigl\|s_\theta(\tilde{\mathbf{x}})-\nabla_{\tilde{\mathbf{x}}}\log q(\tilde{\mathbf{x}}\mid\mathbf{x})\bigr\|^2
```

$q(\tilde{\mathbf{x}}\mid\mathbf{x})=\mathcal{N}(\tilde{\mathbf{x}}\mid\mathbf{x},\sigma^2 I)$ 时，

```math
\nabla_{\tilde{\mathbf{x}}}\log q(\tilde{\mathbf{x}}\mid\mathbf{x})=-(\tilde{\mathbf{x}}-\mathbf{x})/\sigma^2
```

### 4.3 退火朗之万

噪声水平 $\sigma_1>\cdots>\sigma_K$ 递减时，

```math
\mathbf{x}_{k+1}=\mathbf{x}_k+\alpha_k s_\theta(\mathbf{x}_k,\sigma_k)+\sqrt{2\alpha_k}\,\mathbf{z}_k
```

先在宽核上混合，再收到尖峰。MPC 里也可以不训练 $s_\theta$，直接按同样精神缩小 MPPI 的 $\Sigma$，见第 6 节。

## 5. Stein 变分梯度下降

用粒子逼近目标密度，每步沿 RKHS 中下降 $D_{\mathrm{KL}}(q\|p)$ 的方向走：

```math
\phi^*(\mathbf{x}')
=\mathbb{E}_{\mathbf{x}\sim q}
\bigl[k(\mathbf{x},\mathbf{x}')\nabla_{\mathbf{x}}\log p(\mathbf{x})+\nabla_{\mathbf{x}}k(\mathbf{x},\mathbf{x}')\bigr]
```

```math
\mathbf{x}_i\leftarrow\mathbf{x}_i
+\alpha\frac{1}{n}\sum_{j=1}^n
\bigl[k(\mathbf{x}_j,\mathbf{x}_i)\nabla_{\mathbf{x}_j}\log p(\mathbf{x}_j)
+\nabla_{\mathbf{x}_j}k(\mathbf{x}_j,\mathbf{x}_i)\bigr]
```

核加权的得分把粒子推向高密度区，核梯度使粒子互相推开，适合多峰。$\nabla\log p$ 在零阶里可用 §3.2 的平滑得分代替。

## 6. 基于扩散的 MPC：DIAL-MPC

生成式扩散把干净样本逐步加噪，再学逆向去噪。前向为

```math
q(\mathbf{u}_t\mid\mathbf{u}_{t-1})=\mathcal{N}(\mathbf{u}_t\mid\sqrt{1-\beta_t}\mathbf{u}_{t-1},\beta_t I)
```

逆向 $p_\theta(\mathbf{u}_{t-1}\mid\mathbf{u}_t)$ 由网络给出。这与「用得分做退火朗之万」是同一家族。

在线 MPC 不训练得分网络，而是按扩散退火缩小 MPPI 的采样协方差。Xue et al. 取各向同性核：扩散阶段 $i$ 从 $N$ 降到 $1$，视界内步数 $h=0,\ldots,H$，

```math
\Sigma^{i}_{t+h}
=\exp\left(-\frac{N-i}{\beta_1 N}-\frac{H-h}{\beta_2 H}\right)I.
```

$i=N$ 时时间项为零，噪声最大；$i$ 降到 $1$ 时时间项约为 $-1/\beta_1$，噪声最小。$h=H$ 时空间项为零，远处更散；$h=0$ 时空间项约为 $-1/\beta_2$，当前步更尖。两项写在同一个指数里，等价于两个标量尺度相乘。

## 7. 主流零阶优化算法全景

| 类别 | 代表算法 | 核心思想 | 适用场景 |
|------|----------|----------|----------|
| 随机搜索 | 预测采样 | 采样后取最小 | 低维、易并行 |
| 交叉熵 | CEM | 精英集拟合高斯 | 中等维数 |
| 路径积分 | MPPI | 指数权重更新均值 | 实时控制 |
| 进化策略 | CMA-ES | 同时适应协方差 | 病态、尺度变化 |
| 变分推断 | SVGD | 带排斥的粒子流 | 多峰 |
| 扩散退火 | DIAL-MPC | 按日程缩小采样噪声 | 复杂地形、需先探索后利用 |

经验上：维数很低可用 CEM、CMA；要滚动实时多用 MPPI 或对其 $\Sigma$ 退火；多峰明显时加 SVGD 或多种群。理论分析则回到朗之万与玻尔兹曼。

## 8. 理论统一框架

### 8.1 信息论目标

带先验的最大熵问题

```math
\min_q\ \mathbb{E}_q[J(\mathbf{u})]+\lambda D_{\mathrm{KL}}(q\|p_0)
```

的解正是 $q\propto p_0 e^{-J/\lambda}$，也就是 MPPI 的 $p_q$。CEM 用精英样本做矩匹配，在高斯族上逼近同一类好控制区域。SVGD 在粒子上下降 $D_{\mathrm{KL}}(q\|p)$。扩散与得分匹配则提供 $\nabla\log p_\sigma$ 的估计或噪声日程。

### 8.2 一条链条

1. 高斯平滑给出不依赖 $\nabla J$ 的梯度。
2. 指数权重把 CEM 的精英更新连到 MPPI 的均值更新。
3. 平滑得分在 $\Sigma$ 预条件下与 MPPI 同一步；配上 $\sqrt{2}\,\Sigma^{1/2}$ 噪声并退火，就进入朗之万和扩散。
4. CMA 在同一高斯上再更新 $\Sigma$。

### 8.3 可延伸的方向

把双重退火与 SVGD 的排斥合在一起；用学习模型提供更好的提议分布 $q$；以及非凸问题中有限步收敛的保证。

## 结论

MPPI 是 $p_q$ 的指数加权均值。它等于 log-sum-exp 目标的自然梯度步 $-\Sigma\nabla J_{\Sigma,\lambda}/\lambda$，也等于预条件朗之万的确定性漂移 $\mathbf{v}+\Sigma\nabla\log p_\sigma$。

```text
随机采样 → CEM（精英或指数权）→ MPPI
              ↗ 信息论 / Log-Sum-Exp
              ↘ 平滑得分 / 朗之万 → 退火与扩散式 MPC
                                    CMA 适应 Σ
```

温度、协方差和是否保留噪声，是在同一套语言里调节探索与集中。

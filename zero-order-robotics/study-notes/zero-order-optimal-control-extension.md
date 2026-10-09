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

提议取当前序列附近的高斯 $q=\mathcal{N}(\mathbf{u},\Sigma)$。采样 $\mathbf{u}^{(i)}\sim q$，$i=1,\dots,N$，候选集包含当前序列：

```math
\mathbf{u}^{+}
\in
\arg\min\bigl\{J(\mathbf{u}),\,J(\mathbf{u}^{(1)}),\dots,J(\mathbf{u}^{(N)})\bigr\}.
```

没有更优样本时保持 $\mathbf{u}$。这是局部搜索，只保留一条最优序列。

与历史无关的全局采样要用样本覆盖决策空间，评估次数随维数指数增长。预测采样把提议限制在当前序列附近，并丢弃最优样本以外的信息。MPPI 与 CEM 使用全部样本的权重，并更新提议分布。

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

该式不依赖 $\nabla J$。交换积分与求导还要求 $|J(\mathbf{z})|\,\|\mathbf{z}\|^k$ 在平移后的高斯下可积，且 $k\ge 1$，以吸收得分中的线性增长；仅对某一个高斯可积并不够。满足该条件时 $J_\sigma$ 无穷次可微。$\mathbb{E}[\Sigma^{-1}\epsilon]=0$，减去 $J(\bar{\mathbf{u}})$ 不改变期望，估计对 $J$ 的加法常数不变。$\Sigma$ 较大时，窄而深的阱会被邻域中的高代价填平，平滑后的极小可以明显偏离原极小。

## 2. 交叉熵方法与重要性采样

### 2.1 CEM 流程

在高斯族上迭代。$k=0,1,2,\dots$ 为轮次，$i=1,\dots,N$ 为该轮样本。

1. 初始化 $\mathcal{N}(\boldsymbol{\mu}_0,\Sigma_0)$。
2. 采样 $\mathbf{u}^{(i)}\sim\mathcal{N}(\boldsymbol{\mu}_k,\Sigma_k)$，计算 $J^{(i)}=J(\mathbf{u}^{(i)})$。
3. 保留 $J^{(i)}$ 最低的约 $\rho\%$ 样本，记为精英集 $\mathcal{E}_k$。
4. $\boldsymbol{\mu}_{k+1}$ 取 $\mathcal{E}_k$ 的样本均值；$\Sigma_{k+1}$ 以 $\boldsymbol{\mu}_{k+1}$ 为中心。

这是交叉熵法的极大似然更新：高斯族拟合精英的经验分布。自然梯度形式的 CMA 不同，协方差绕 $\boldsymbol{\mu}_k$，见 §2.2 末。

### 2.2 玻尔兹曼分布与指数权重

低代价控制的目标分布取归一化的玻尔兹曼密度

```math
p^*(\mathbf{u})=\frac{1}{Z}\exp\bigl(-J(\mathbf{u})/\lambda\bigr),\qquad
Z=\int\exp\bigl(-J(\mathbf{u})/\lambda\bigr)\,d\mathbf{u}.
```

均值与协方差是该密度下的矩：

```math
\boldsymbol{\mu}^*=\int\mathbf{u}\,p^*(\mathbf{u})\,d\mathbf{u},\qquad
\Sigma^*=\int(\mathbf{u}-\boldsymbol{\mu}^*)(\mathbf{u}-\boldsymbol{\mu}^*)^\top p^*(\mathbf{u})\,d\mathbf{u}.
```

\(J\) 由 rollout 计算，\(Z\) 与这两个积分都没有闭式。高斯可由 \((\boldsymbol{\mu}_k,\Sigma_k)\) 直接抽样；\(p^*\) 只给出未归一化的形状，不能交给标准随机数发生器。样本因此仍取自当前提议 \(q=\mathcal{N}(\boldsymbol{\mu}_k,\Sigma_k)\)。

要估计的是 \(p^*\) 下的期望，样本来自 \(q\)。对可积的 \(f\)，

```math
\mathbb{E}_{p^*}[f]
=\int f(\mathbf{u})\frac{p^*(\mathbf{u})}{q(\mathbf{u})}\,q(\mathbf{u})\,d\mathbf{u}
=\mathbb{E}_q\Bigl[f(\mathbf{u})\frac{p^*(\mathbf{u})}{q(\mathbf{u})}\Bigr].
```

权重是 \(p^*/q\)。\(Z\) 在样本间归一化时约去，故

```math
\frac{p^*(\mathbf{u})}{q(\mathbf{u})}
\propto
\frac{\exp(-J(\mathbf{u})/\lambda)}{q(\mathbf{u})}.
```

另一问题是在先验 \(p_0\) 附近最小化期望代价。把待求分布记为 \(\pi\)，与采样提议 \(q\) 区分开：

```math
\min_{\pi}\ \mathbb{E}_{\pi}[J]+\lambda\,D_{\mathrm{KL}}(\pi\|p_0).
```

第一项把质量移向 \(J\) 小的区域，第二项限制 \(\pi\) 偏离 \(p_0\)；\(\lambda\) 越小，解越集中于低代价处。在 \(\int\pi=1\) 下对 \(\pi\) 变分，驻点满足

```math
J(\mathbf{u})+\lambda\log\pi(\mathbf{u})-\lambda\log p_0(\mathbf{u})=\text{常数},
```

因而

```math
\pi(\mathbf{u})\propto p_0(\mathbf{u})\exp\bigl(-J(\mathbf{u})/\lambda\bigr).
```

取 \(p_0=q\)，把这个解记为 \(p_q\)。它比 \(p^*\) 多乘了提议 \(q\)。仍用 \(q\) 估计 \(p_q\) 的矩时，重要性比为 \(p_q/q\propto\exp(-J/\lambda)\)，\(q\) 相消：

```math
w_i=\exp\bigl(-J(\mathbf{u}^{(i)})/\lambda\bigr),
\qquad
\boldsymbol{\mu}_{k+1}=\frac{\sum_{i=1}^N w_i\mathbf{u}^{(i)}}{\sum_{i=1}^N w_i}.
```

用这些权重估计 \(p_q\) 的样本协方差时，中心是 \(\boldsymbol{\mu}_{k+1}\)。CEM 的精英均匀权同样绕更新后的均值。\(\lambda\) 小时指数权集中于少数低代价样本；\(\rho\%\) 截断则在固定比例上取均匀权。二者都收紧搜索分布，权重不同，更新不同。

固定 \(\Sigma\)、只更新均值，即第 3 节的 MPPI。自然梯度同时更新 \(\Sigma\) 时，指数权对应 MPPI-CMA，排序权对应不含 evolution path 的 CMA-ES。该协方差绕采样均值 \(\boldsymbol{\mu}_k\)，绕的不是 \(\boldsymbol{\mu}_{k+1}\)。长视界下 \(\Sigma\) 取块对角，每时刻一块 \(n_u\times n_u\)。

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
| 随机搜索 | 预测采样 | 候选含当前序列，只留最优 | 批量 rollout |
| 交叉熵 | CEM | 精英集拟合高斯 | 中等维数 |
| 路径积分 | MPPI | 指数权重更新均值 | 实时控制 |
| 进化策略 | CMA-ES | 同时适应协方差 | 病态、尺度变化 |
| 变分推断 | SVGD | 带排斥的粒子流 | 多峰 |
| 扩散退火 | DIAL-MPC | 按日程缩小采样噪声 | 复杂地形、需先探索后利用 |

实时控制多用 MPPI，或按第 6 节对 $\Sigma$ 退火。尺度与病态程度随迭代变化时用 CMA-ES。多峰时用 SVGD 或多种群。理论分析回到朗之万与玻尔兹曼。

## 8. 理论统一框架

### 8.1 信息论目标

带先验的最大熵问题

```math
\min_q\ \mathbb{E}_q[J(\mathbf{u})]+\lambda D_{\mathrm{KL}}(q\|p_0)
```

的解正是 $q\propto p_0 e^{-J/\lambda}$，也就是 MPPI 的 $p_q$。CEM 用精英样本做矩匹配，在高斯族上逼近同一类好控制区域。SVGD 在粒子上下降 $D_{\mathrm{KL}}(q\|p)$。扩散与得分匹配则提供 $\nabla\log p_\sigma$ 的估计或噪声日程。

### 8.2 一条链条

1. 高斯平滑给出不依赖 $\nabla J$ 的梯度。
2. 精英均匀权给出 CEM 的均值；指数权在 $\Sigma$ 固定时给出 MPPI。
3. 平滑得分在 $\Sigma$ 预条件下与 MPPI 同一步；配上 $\sqrt{2}\,\Sigma^{1/2}$ 噪声并退火，即进入朗之万与扩散。
4. CMA 在同一高斯上更新 $\Sigma$，中心为采样均值。

### 8.3 可延伸的方向

将第 6 节的轨迹层与动作层噪声日程和 SVGD 的排斥项结合；用学习得到的模型作为提议 $q$；以及非凸问题的有限步收敛保证。

## 结论

MPPI 是 $p_q$ 的指数加权均值。它等于 log-sum-exp 目标的自然梯度步 $-\Sigma\nabla J_{\Sigma,\lambda}/\lambda$，也等于预条件朗之万的确定性漂移 $\mathbf{v}+\Sigma\nabla\log p_\sigma$。

```text
预测采样（含当前序列）
    → CEM（精英权，协方差绕新均值）
    → MPPI（指数权，固定 Σ）
         ↗ Log-Sum-Exp 自然梯度
         ↘ 平滑得分 / 朗之万 → 扩散式退火
    → CMA（自然梯度更新 Σ，绕采样均值）
```

温度、协方差和是否保留噪声，是在同一套语言里调节探索与集中。

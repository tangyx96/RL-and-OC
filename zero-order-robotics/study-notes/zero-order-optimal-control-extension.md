# 零阶最优控制

> 分节读书笔记的延伸。文献：Jordana et al., *An Introduction to Zero-Order Optimization Techniques for Robotics*（arXiv:2506.22087）。按章节整理见 [`zero-order-optimization-study-notes.md`](zero-order-optimization-study-notes.md)。

## 摘要

零阶最优控制更新的是玻尔兹曼目标 $p^*\propto\exp(-J/\lambda)$，或带当前高斯先验的 $p_q\propto q\exp(-J/\lambda)$。MPPI 的均值步等于 log-sum-exp 的自然梯度步 $-\Sigma\nabla J_{\Sigma,\lambda}/\lambda$，也等于平滑密度上的预条件漂移 $\Sigma\nabla\log p_\sigma$。其余算法的差别只有三处：是否更新 $\Sigma$，协方差绕 $\boldsymbol{\mu}_{k+1}$ 还是 $\boldsymbol{\mu}_k$，以及是否保留扩散项。

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

## 2. 玻尔兹曼目标

### 2.1 目标分布与重要性权重

低代价控制的目标分布取归一化的玻尔兹曼密度

```math
p^*(\mathbf{u})=\frac{1}{Z}\exp\bigl(-J(\mathbf{u})/\lambda\bigr),\qquad
Z=\int\exp\bigl(-J(\mathbf{u})/\lambda\bigr)\,d\mathbf{u}.
```

下一轮高斯由该密度的均值与协方差确定，二者即 $p^*$ 下的期望：

```math
\boldsymbol{\mu}^*=\mathbb{E}_{p^*}[\mathbf{u}],\qquad
\Sigma^*=\mathbb{E}_{p^*}\bigl[(\mathbf{u}-\boldsymbol{\mu}^*)(\mathbf{u}-\boldsymbol{\mu}^*)^\top\bigr].
```

$J$ 由 rollout 给出，$Z$ 与上述期望都没有闭式。$p^*$ 只知未归一化密度，样本取自当前提议 $q=\mathcal{N}(\boldsymbol{\mu}_k,\Sigma_k)$。对可积的 $f$，

```math
\mathbb{E}_{p^*}[f]
=\mathbb{E}_q\Bigl[f(\mathbf{u})\frac{p^*(\mathbf{u})}{q(\mathbf{u})}\Bigr].
```

$f(\mathbf{u})=\mathbf{u}$ 与 $f\equiv 1$ 分别给出 $\boldsymbol{\mu}^*=\mathbb{E}_q[\mathbf{u}\,p^*/q]$ 与 $\mathbb{E}_q[p^*/q]=1$。二者相除，

```math
\boldsymbol{\mu}^*
=\frac{\mathbb{E}_q[\mathbf{u}\,p^*/q]}{\mathbb{E}_q[p^*/q]}.
```

用 $q$ 的样本平均代替两个期望，$1/N$ 相消，得到自归一化估计

```math
\hat{\boldsymbol{\mu}}
=\sum_{i=1}^N\bar{w}_i\,\mathbf{u}^{(i)},
\qquad
\bar{w}_i
=\frac{p^*(\mathbf{u}^{(i)})/q(\mathbf{u}^{(i)})}
{\sum_j p^*(\mathbf{u}^{(j)})/q(\mathbf{u}^{(j)})}.
```

分母是权重之和，使 $\sum_i\bar{w}_i=1$，因而 $\hat{\boldsymbol{\mu}}$ 为样本的加权平均。权重

```math
\frac{p^*(\mathbf{u})}{q(\mathbf{u})}
=\frac{\exp(-J(\mathbf{u})/\lambda)}{Z\,q(\mathbf{u})}
```

中的 $1/Z$ 不依赖于样本，在分子与分母中为同一公因子，同时消去。全体权重相等时 $\bar{w}_i=1/N$，估计即样本均值。协方差用同一组 $\bar{w}_i$。计算时取

```math
\frac{p^*(\mathbf{u})}{q(\mathbf{u})}
\propto
\frac{\exp(-J(\mathbf{u})/\lambda)}{q(\mathbf{u})}.
```

### 2.2 带先验的解

在先验 $p_0$ 附近最小化期望代价。待求分布记为 $\pi$，与采样提议 $q$ 区分：

```math
\min_{\pi}\ \mathbb{E}_{\pi}[J]+\lambda\,D_{\mathrm{KL}}(\pi\|p_0),
\qquad\int\pi\,d\mathbf{u}=1.
```

展开为

```math
\int\pi(\mathbf{u})\,J(\mathbf{u})\,d\mathbf{u}
+\lambda\int\pi(\mathbf{u})\log\frac{\pi(\mathbf{u})}{p_0(\mathbf{u})}\,d\mathbf{u}.
```

第一项把质量移向 $J$ 小的区域，第二项限制 $\pi$ 偏离 $p_0$；$\lambda$ 越小，解越集中于低代价处。

乘子 $\nu$ 对应约束 $\int\pi=1$，拉格朗日泛函为

```math
\mathcal{L}
=\int\pi J\,d\mathbf{u}
+\lambda\int\pi\log\frac{\pi}{p_0}\,d\mathbf{u}
+\nu\Bigl(1-\int\pi\,d\mathbf{u}\Bigr).
```

将 $\pi$ 换成 $\pi+\varepsilon\eta$，对 $\varepsilon$ 在 $0$ 处求导。结果是 $\eta$ 与下列系数的积分：$\int\pi J$ 的系数为 $J$；$\int\pi\log\pi$ 的导数为 $\int(\eta\log\pi+\eta)\,d\mathbf{u}$，其中 $\eta$ 来自 $\pi\cdot(\eta/\pi)$，系数为 $\log\pi+1$；$-\int\pi\log p_0$ 的系数为 $-\log p_0$；$-\nu\int\pi$ 的系数为 $-\nu$。驻点要求此导数对任意 $\eta$ 为零，

```math
\int\Bigl(J(\mathbf{u})+\lambda\bigl(\log\pi(\mathbf{u})-\log p_0(\mathbf{u})+1\bigr)-\nu\Bigr)\eta(\mathbf{u})\,d\mathbf{u}=0,
```

因而括号处处为零：

```math
J(\mathbf{u})+\lambda\log\pi(\mathbf{u})-\lambda\log p_0(\mathbf{u})=\nu-\lambda.
```

右端不含 $\mathbf{u}$。$\nu$ 是乘子，$\lambda\cdot 1$ 来自 $\int\pi\log\pi$ 变分中的 $\eta$。移项并除以 $\lambda$，

```math
\log\pi(\mathbf{u})
=\log p_0(\mathbf{u})-\frac{J(\mathbf{u})}{\lambda}+C,
\qquad
C=\frac{\nu-\lambda}{\lambda}.
```

取指数得 $\pi(\mathbf{u})=e^{C}\,p_0(\mathbf{u})\exp(-J(\mathbf{u})/\lambda)$。$e^{C}$ 由 $\int\pi=1$ 确定。记

```math
Z_0
=\int p_0(\mathbf{u})\exp\bigl(-J(\mathbf{u})/\lambda\bigr)\,d\mathbf{u},
```

则 $e^{C}=1/Z_0$，故

```math
\pi(\mathbf{u})
=\frac{1}{Z_0}\,p_0(\mathbf{u})\exp\bigl(-J(\mathbf{u})/\lambda\bigr).
```

未归一化密度比 $p^*$ 多乘先验 $p_0$。取 $p_0=q$，记此解为 $p_q$。重要性比为

```math
\frac{p_q(\mathbf{u})}{q(\mathbf{u})}
=\frac{\exp(-J(\mathbf{u})/\lambda)}{Z_0}.
```

$q$ 相消，$Z_0$ 不依赖于样本。与 §2.1 相同，自归一化只保留样本间的相对权重：

```math
w_i=\exp\bigl(-J(\mathbf{u}^{(i)})/\lambda\bigr),
\qquad
\bar{w}_i=\frac{w_i}{\sum_j w_j},
\qquad
\boldsymbol{\mu}_{k+1}=\sum_{i=1}^N\bar{w}_i\,\mathbf{u}^{(i)}.
```

$1/Z_0$ 在 $\bar{w}_i$ 中消去。协方差用同一组权重，中心为 $\boldsymbol{\mu}_{k+1}$：

```math
\Sigma_{k+1}
=\sum_{i=1}^N\bar{w}_i
(\mathbf{u}^{(i)}-\boldsymbol{\mu}_{k+1})
(\mathbf{u}^{(i)}-\boldsymbol{\mu}_{k+1})^\top.
```

$\lambda$ 小时 $w_i$ 集中于少数低代价样本。$\rho\%$ 截断则在固定比例的样本上取均匀权。

### 2.3 三条开关

各算法共用这一目标，按下表选择权重、协方差和扩散项。协方差更新时，中心是 $\boldsymbol{\mu}_{k+1}$ 或采样时的 $\boldsymbol{\mu}_k$。扩散项为 $\sqrt{2}\,\Sigma^{1/2}\mathbf{z}$；也可以丢掉它，或只按日程缩小 $\Sigma$。

| 方法 | 权重 | $\Sigma$ | 扩散项 |
|------|------|----------|--------|
| 预测采样 | 只留最优，候选含当前序列 | 固定 | 无 |
| CEM | 精英均匀权 | 更新，绕 $\boldsymbol{\mu}_{k+1}$ | 无 |
| MPPI | 指数权 | 固定 | 丢掉，只留漂移 |
| MPPI-CMA | 指数权 | 自然梯度，绕 $\boldsymbol{\mu}_k$ | 无 |
| CMA-ES | 排序权 | 自然梯度，绕 $\boldsymbol{\mu}_k$ | 无 |
| 朗之万、扩散 | $\nabla\log p_\sigma$ | 可缩小 | 保留 |
| SVGD | $\nabla\log p_\sigma$ | 粒子间另有排斥 | 无 |
| DIAL-MPC | 指数权 | 按日程缩小 | 只留漂移 |

表中的自然梯度更新绕采样均值 $\boldsymbol{\mu}_k$。排序权对应不含 evolution path 的 CMA-ES。长视界下 $\Sigma$ 取块对角，每时刻一块 $n_u\times n_u$。

实时控制用 MPPI，或按日程缩小 $\Sigma$。尺度与病态程度随迭代变化时用 CMA-ES。多峰时用 SVGD 或多种群。

## 3. 交叉熵法

精英均匀权，协方差绕 $\boldsymbol{\mu}_{k+1}$。在高斯族上迭代。$k=0,1,2,\dots$ 为轮次，$i=1,\dots,N$ 为该轮样本。

1. 初始化 $\mathcal{N}(\boldsymbol{\mu}_0,\Sigma_0)$。
2. 采样 $\mathbf{u}^{(i)}\sim\mathcal{N}(\boldsymbol{\mu}_k,\Sigma_k)$，计算 $J^{(i)}=J(\mathbf{u}^{(i)})$。
3. 保留 $J^{(i)}$ 最低的约 $\rho\%$ 样本，记为精英集 $\mathcal{E}_k$。
4. 记 $|\mathcal{E}_k|=N_e$。精英上取均匀权，

```math
\boldsymbol{\mu}_{k+1}
=\frac{1}{N_e}\sum_{i\in\mathcal{E}_k}\mathbf{u}^{(i)},
\qquad
\Sigma_{k+1}
=\frac{1}{N_e}\sum_{i\in\mathcal{E}_k}
(\mathbf{u}^{(i)}-\boldsymbol{\mu}_{k+1})
(\mathbf{u}^{(i)}-\boldsymbol{\mu}_{k+1})^\top.
```

精英内 $\bar{w}_i=1/N_e$，其余为 $0$。$N_e$ 约为全部样本数 $N$ 的 $\rho\%$，分母是 $N_e$。这与 §2.2 的加权更新同一形式，权重改为截断后的均匀权。全体 $N$ 个样本都取 $1/N$ 时不再截断，得到普通样本均值。该式是高斯族对精英经验分布的极大似然拟合。

## 4. MPPI

指数权，固定 $\Sigma$，只更新均值 $\mathbf{v}$。

### 4.1 信息论与 Log-Sum-Exp

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

同一位移是 log-sum-exp 平滑

```math
J_{\Sigma,\lambda}(\mathbf{v})
=-\lambda\log\mathbb{E}_{\epsilon\sim\mathcal{N}(0,\Sigma)}\bigl[\exp(-J(\mathbf{v}+\epsilon)/\lambda)\bigr]
```

的下降步。对 $\mathbf{v}$ 求导只作用在高斯密度上，欧氏梯度为

```math
\nabla J_{\Sigma,\lambda}(\mathbf{v})
=-\lambda\,\Sigma^{-1}
\frac{\mathbb{E}[\epsilon\exp(-J(\mathbf{v}+\epsilon)/\lambda)]}
{\mathbb{E}[\exp(-J(\mathbf{v}+\epsilon)/\lambda)]}.
```

均值参数的 Fisher 信息为 $F=\Sigma^{-1}$。自然梯度取 $F^{-1}\nabla J_{\Sigma,\lambda}$，$\Sigma^{-1}$ 与 $F^{-1}$ 相消。步长取 $1/\lambda$ 时，

```math
\mathbf{v}-\frac{1}{\lambda}F^{-1}\nabla J_{\Sigma,\lambda}(\mathbf{v})
=\mathbf{v}+\frac{\mathbb{E}[w\epsilon]}{\mathbb{E}[w]},
```

正是上面的 $\mathbf{v}^*$。$\lambda\to\infty$ 时 $J_{\Sigma,\lambda}$ 回到 §1.3 的 $J_\sigma$；$\lambda\to 0$ 时接近硬最小。实现上常减 $\min_i J^{(i)}$ 再取指数，只防溢出。样本极多时加权平均几乎没有随机性，需要靠 $\lambda$、$\Sigma$ 或重启保持探索。

### 4.2 平滑得分

目标密度 $p$ 的过阻尼朗之万方程是

```math
\mathrm{d}\mathbf{u}=\nabla\log p(\mathbf{u})\,\mathrm{d}t+\sqrt{2}\,\mathrm{d}W,
```

其中 $W$ 为标准维纳过程。步长 $\alpha$ 内其增量服从 $\mathcal{N}(0,\alpha I)$，写成 $\Delta W=\sqrt{\alpha}\,\mathbf{z}_k$，$\mathbf{z}_k\sim\mathcal{N}(0,I)$ 且各步独立。Euler–Maruyama 将漂移冻结在 $\mathbf{u}_k$，扩散项 $\sqrt{2}\,\Delta W$ 即为 $\sqrt{2\alpha}\,\mathbf{z}_k$：

```math
\mathbf{u}_{k+1}=\mathbf{u}_k+\alpha\nabla\log p(\mathbf{u}_k)+\sqrt{2\alpha}\,\mathbf{z}_k.
```

$p^*\propto\exp(-J/\lambda)$ 的得分含 $\nabla J$。零阶情形对 $p^*$ 作与 §1.3 相同的高斯平滑：$p_\sigma(\mathbf{u})$ 是以 $\mathbf{u}$ 为中心的高斯下 $p^*$ 的期望，

```math
p_\sigma(\mathbf{u})
=\int p^*(\mathbf{u}')\,\mathcal{N}(\mathbf{u}'\mid\mathbf{u},\Sigma)\,d\mathbf{u}'
=\mathbb{E}_{\epsilon\sim\mathcal{N}(0,\Sigma)}[p^*(\mathbf{u}+\epsilon)].
```

$\log\mathcal{N}(\mathbf{u}'\mid\mathbf{u},\Sigma)$ 对均值 $\mathbf{u}$ 的导数来自二次型 $-\frac12(\mathbf{u}'-\mathbf{u})^\top\Sigma^{-1}(\mathbf{u}'-\mathbf{u})$，故

```math
\nabla_{\mathbf{u}}\log\mathcal{N}(\mathbf{u}'\mid\mathbf{u},\Sigma)
=\Sigma^{-1}(\mathbf{u}'-\mathbf{u}),
\qquad
\nabla_{\mathbf{u}}\mathcal{N}
=\mathcal{N}\,\nabla_{\mathbf{u}}\log\mathcal{N}.
```

$p^*(\mathbf{u}')$ 不依赖于 $\mathbf{u}$。积分号下求导再除以 $p_\sigma(\mathbf{u})$，

```math
\nabla\log p_\sigma(\mathbf{u})
=\Sigma^{-1}
\frac{\int p^*(\mathbf{u}')(\mathbf{u}'-\mathbf{u})\,\mathcal{N}(\mathbf{u}'\mid\mathbf{u},\Sigma)\,d\mathbf{u}'}{p_\sigma(\mathbf{u})}.
```

令 $\epsilon=\mathbf{u}'-\mathbf{u}$，积分在 $\epsilon\sim\mathcal{N}(0,\Sigma)$ 下写成

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

## 5. 朗之万动力学与扩散模型

保留扩散项，并降低噪声水平。

### 5.1 物理图像

过阻尼朗之万方程：

```math
\gamma\frac{d\mathbf{x}}{dt}=-\nabla U(\mathbf{x})+\sqrt{2\gamma k_B T}\,\boldsymbol{\xi}(t)
```

$U=-\log p$、$\gamma=1$、$k_B T=1$ 时，这就是 §4.2 的过阻尼朗之万。温度降低，样本集中到 $U$ 的低谷，也就是模拟退火。

### 5.2 得分匹配

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

### 5.3 退火朗之万

噪声水平 $\sigma_1>\cdots>\sigma_K$ 递减时，

```math
\mathbf{x}_{k+1}=\mathbf{x}_k+\alpha_k s_\theta(\mathbf{x}_k,\sigma_k)+\sqrt{2\alpha_k}\,\mathbf{z}_k
```

先在宽核上混合，再收到尖峰。第 7 节不训练 $s_\theta$，按日程缩小 MPPI 的 $\Sigma$。

## 6. Stein 变分梯度下降

粒子沿同一得分移动，核梯度提供排斥。

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

核加权的得分把粒子推向高密度区，核梯度使粒子互相推开，适合多峰。零阶情形用 §4.2 的平滑得分。

## 7. 基于扩散的 MPC：DIAL-MPC

指数权保持不变，$\Sigma$ 按轨迹层与动作层两套日程缩小。

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

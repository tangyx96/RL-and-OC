# 零阶最优控制

> 分节读书笔记的延伸。主线文献为 Jordana et al., *An Introduction to Zero-Order Optimization Techniques for Robotics*（arXiv:2506.22087）；PI-MPC 的概率推断组织参考 Honda, *Model Predictive Control via Probabilistic Inference: A Tutorial and Survey*（[arXiv:2511.08019](https://arxiv.org/abs/2511.08019)）。按前者章节整理见 [`zero-order-optimization-study-notes.md`](zero-order-optimization-study-notes.md)。

## 摘要

零阶最优控制可统一理解为：在有限时域控制序列上定义一个由代价和先验共同决定的目标分布，再用可计算的分布族或粒子群逼近该分布。无先验时目标是玻尔兹曼密度 $p^*\propto\exp(-J/\lambda)$；引入控制先验后，目标变为 $\pi^*\propto p_0\exp(-J/\lambda)$。这一区分使随机预测采样、CEM、MPPI、CMA-ES、朗之万方法、SVGD 与扩散式 MPC 可以置于同一条“分布构造—近似推断—滚动执行”的逻辑链上。MPPI 的均值步既可写成 log-sum-exp 目标的自然梯度步 $-\Sigma\nabla J_{\Sigma,\lambda}/\lambda$，也可写成平滑密度上的预条件漂移 $\Sigma\nabla\log p_\sigma$；但其优化对象仍是带 KL 正则的控制分布，而非无约束的原始轨迹代价。

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

下表在共同的采样分布视角下比较这些方法。它们并非严格求解同一个有限样本目标：权重可能来自指数效用、精英截断、排序效用或目标 score；SVGD 还含粒子间排斥。协方差更新时，中心可以是 $\boldsymbol{\mu}_{k+1}$ 或采样时的 $\boldsymbol{\mu}_k$。扩散项写成 $\sqrt{2}\,\Sigma^{1/2}\mathbf{z}$；确定性搜索会将其省略，退火方法还会按日程改变 $\Sigma$。

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

### 2.4 PI-MPC：从最优控制问题到变分推断

令 $\mathbf{u}=\mathbf{u}_{0:H-1}$ 为完整的开环控制序列，给定当前状态后，轨迹由动力学 rollout 唯一确定或诱导出条件分布。把轨迹代价写为 $J_\tau(\mathbf{u})$，并引入“轨迹最优”的虚拟观测事件 $O=1$。若其似然采用玻尔兹曼形式，

```math
p(O=1\mid\mathbf{u})\propto\exp\bigl(-J_\tau(\mathbf{u})/\lambda\bigr),
```

则以控制先验 $p_0(\mathbf{u})$ 为基准的后验为

```math
p(\mathbf{u}\mid O=1)
=\frac{1}{Z_0}p_0(\mathbf{u})
\exp\bigl(-J_\tau(\mathbf{u})/\lambda\bigr).
```

若归一化常数 $Z_0$ 有限，且候选分布相对于 $p_0$ 绝对连续，这就是本笔记中的目标分布 $\pi^*$；它也是下列无限维变分问题的唯一解：

```math
\pi^*
=\arg\min_{\pi}
\left\{
\frac{1}{\lambda}\mathbb{E}_{\pi}[J_\tau]
+D_{\mathrm{KL}}(\pi\|p_0)
\right\}.
```

因此，温度 $\lambda$ 不是单纯的指数权数值超参数：它控制“降低代价”与“保持默认控制行为”之间的权衡。$p_0$ 也不只是方便采样的高斯核；它同时规定了 KL 正则的参考行为、可有效探索的区域以及可达到解的支持集。若 $p_0$ 在某个区域严格为零，有限 KL 的候选 $\pi$ 就不能给该区域分配正质量。

实际算法还必须区分三个容易混淆的分布：

1. **控制先验 $p_0$**：表达 warm start、执行器平滑性、示范、预训练策略或安全偏好；它定义理论目标中的正则项。
2. **采样提议 $q$**：用于以 Monte Carlo 估计目标分布的期望；为了低方差，$q$ 应覆盖 $\pi^*$ 的高概率区域，但它不必等于 $p_0$。
3. **变分族 $\pi_\theta$**：最终允许算法表达和更新的分布，如固定协方差高斯、全协方差高斯、混合模型、粒子经验分布或生成模型。

标准 MPPI 常令三者在一个控制周期内紧密耦合：以前一周期平移后的高斯作为 $p_0$，并从它采样，再以同协方差高斯族表示更新后的均值。这样计算简洁，却把先验偏差、重要性采样误差和表示误差叠加在一起。将三者概念上分开，才能清楚分析学习先验、辅助控制器、SVGD 粒子或 diffusion proposal 分别改善了哪一环。

在 receding-horizon MPC 中，每个时刻只执行 $\mathbf{u}$ 的第一项，然后依据新观测平移控制序列并重建上述问题。故“分布优化”发生在有限时域开环序列上，而闭环鲁棒性来自反复观测和重规划，不能仅由单次分布优化的收敛性推出。

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

### 4.3 先验、温度、有限样本与原问题的差距

MPPI 的加权均值是对 $\pi^*$ 的近似，$\pi^*$ 又是正则化目标的解；因而必须把三个问题分开：是否准确估计了 $\pi^*$，该分布是否可由当前参数族表示，以及它是否与原始的确定性最优控制解一致。即使样本数趋于无穷，前两项误差消失，KL 正则仍可能使解偏离直接最小化 $J_\tau$ 的控制序列。

当 $p_0=\mathcal N(\mathbf{v},\Sigma)$ 时，先验过窄且均值偏离低代价模态，会使所有样本集中在错误区域；先验很宽虽扩大覆盖，却会降低有效样本量。温度降低会强化低代价样本的权重，但也更容易产生权重退化；提高温度则改善数值稳定性，却使解更接近先验。协方差、温度、样本数与代价地形因此共同决定误差，不能只通过调大样本数解决。

有限样本下可用有效样本量监测指数权是否退化：

```math
\operatorname{ESS}
=\frac{1}{\sum_{i=1}^N\bar w_i^2},
\qquad 1\leq\operatorname{ESS}\leq N.
```

ESS 很低表示少数 rollout 主导更新。此时可以增大 $\lambda$、重设或扩张先验协方差、引入多轮退火，或采用多峰分布族；仅对权重截断虽能稳定估计，也会改变原本的变分目标。

在某些特殊结构下，先验正则的偏差可以抵消。例如当控制代价恰与高斯先验协方差诱导的二次项匹配时，信息论 MPPI 与原始控制问题的解可以一致。更一般地，可通过在 rollout 代价中加入与先验和变分族相匹配的校正项来消除或减小该差距；但这依赖模型、噪声解释和参数化假设，不能把一般的指数加权更新自动视为原问题的无偏求解器。

收敛结论也应谨慎解读。二次代价、无限样本或强凸近似下可得到明确结果；一般非凸、非线性、接触丰富的机器人系统中，通常只能讨论目标分布估计误差或局部改进。它们不直接给出经典 MPC 意义的递归可行性和闭环稳定性。对于安全关键系统，约束可行性、模型误差和实时计算预算仍须单独设计与验证。

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

## 8. PI-MPC 的建模与实现设计

上述算法的差异可以归结为目标分布的表示和逼近方式；实际部署还需要同时处理先验、多峰、约束、维度与计算五类问题。

### 8.1 先验设计与 warm start

最直接的先验是以平移后的上一轮解为均值的高斯，这使 MPC 的时间连续性成为概率先验而非额外启发式。时间相关噪声、低频基函数、样条控制点或高斯过程先验可把平滑性直接编码进 $p_0$，从而减少高频、不物理的候选序列。来自示范数据的生成模型、离线训练策略和局部 LQR/iLQG 控制器也可充当先验或 proposal：前者提供任务层面的多模态结构，后者把样本推向局部可控区域。

先验的目标不是盲目缩小方差，而是在主要低代价模态附近保持尽可能小、同时仍覆盖模型误差和任务变化的有效搜索区域。过窄的错误先验会造成不可恢复的 mode miss；过宽的先验则造成维度灾难。在线自适应协方差、辅助反馈修正和基于数据的先验学习分别对应对这项权衡的不同处理。

### 8.2 多峰与约束

单一高斯 MPPI 的加权均值在双峰或多峰问题中可能落在任何一个可行模态之间的低概率区域，例如障碍物左右绕行的两条轨迹平均后直接穿越障碍。CEM 的精英拟合同样可能在合并模态时失效。混合分布、多种群、粒子 SVGD、normalizing flow 或 diffusion 表示可以保留多个模态；执行前再依据风险、连续性或后验质量选择一个模态，而不是机械执行整体均值。

代价惩罚只能软性偏好可行轨迹；在有限样本下，仍可能没有任何安全样本。需要严格满足状态或输入约束时，应将其置于采样支持集、对每个样本作投影/安全过滤，或在变分问题中显式加入约束。投影简洁但可能扭曲提议密度和重要性权重；约束变分推断更一致，却通常增加求解复杂度。无论采用何种方案，都应区分“期望上低违约率”和“每一次控制动作都可行”这两种保证。

### 8.3 高维、模型与实时性

决策维度随视界和动作维数线性增加，而有效覆盖所需的样本数通常急剧上升。常见措施是把逐时刻动作改写为较少的样条/基函数系数，使用块对角或时间相关协方差，并通过退火或自适应重要性采样分阶段集中计算。这些做法提高样本效率，但同时限制了可表示的控制轨迹，应被视为表示偏差与计算预算之间的明确取舍。

PI-MPC 的 rollout、代价评估和权重归一化适合批量并行；GPU 物理仿真、学习动力学模型和向量化代价函数常决定系统能否满足控制周期。快速 rollout 并不能替代模型可信度：接触、延迟、状态估计误差和分布外动作会使模型偏差直接进入指数权重，并被低温放大。实践中应结合模型集成、不确定性惩罚、短视界重规划和真实系统安全层，而不应把高吞吐采样误认为鲁棒性保证。

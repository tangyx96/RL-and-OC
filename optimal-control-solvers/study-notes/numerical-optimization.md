# 数值最优化

## 1. 无约束最优化方法的基本结构

求解最优化问题的基本方法是迭代算法：从初始点 $x_0$ 出发，逐步构造迭代点列 $\{x_k\}$，并在满足终止准则时输出近似解。对极小化问题，常见的两类全局化框架是线搜索与信赖域。

线搜索方法先在当前点选取搜索方向，再沿该方向选择步长：

```math
x_{k+1}=x_k+\alpha_kd_k.
```

信赖域方法则先限定局部模型可信的区域，并在该区域内同时决定方向与步长。两种框架都不改变局部模型的构造原则，区别在于如何把局部模型转化为可接受的实际迭代步。

### 1.1 基础知识

#### 1.1.1 泰勒定理

若 $f:\mathbb R^n\to\mathbb R$ 连续可微，且 $p\in\mathbb R^n$，则存在 $t\in(0,1)$ 使

```math
f(x+p)=f(x)+\nabla f(x+tp)^\top p.
```

若 $f$ 二阶连续可微，则存在 $t\in(0,1)$ 使

```math
f(x+p)
=f(x)+\nabla f(x)^\top p
+\frac12p^\top\nabla^2f(x+tp)p.
```

等价地，当 $p\to0$ 时，

```math
f(x+p)=f(x)+\nabla f(x)^\top p
+\frac12p^\top\nabla^2f(x)p+o(\|p\|^2).
```

泰勒展开是局部一阶、二阶模型以及梯度法、Newton 法、信赖域和 SQP 的共同基础。

将一阶展开应用于 $x+hu$，有

```math
f(x+hu)=f(x)+h\nabla f(x)^\top u+o(h).
```

两边减去 $f(x)$ 并除以 $h$，令 $h\to0$，得到方向导数。对梯度展开则有

```math
\nabla f(x+hu)
=\nabla f(x)+h\nabla^2f(x)u+o(h).
```

再与 $u$ 做内积，可得

```math
D_uf(x+hu)
=D_uf(x)+h\,u^\top\nabla^2f(x)u+o(h),
```

从而得到二阶方向导数公式。

#### 1.1.2 方向导数与二阶方向导数

沿方向 $u$ 的方向导数定义为

```math
D_uf(x)=\lim_{h\to0}\frac{f(x+hu)-f(x)}{h}.
```

由一阶 Taylor 展开可得

```math
D_uf(x)=\nabla f(x)^\top u.
```

二阶方向导数定义为

```math
D_u^2f(x)=\lim_{h\to0}
\frac{D_uf(x+hu)-D_uf(x)}{h},
```

若 $f$ 二阶可微，则

```math
D_u^2f(x)=u^\top\nabla^2f(x)u.
```

因此，梯度描述函数值沿给定方向的一阶变化率，Hessian 二次型描述该方向上的局部曲率。

#### 1.1.3 驻点、极小点与曲率

若 $\nabla f(x)=0$，且对任意非零方向 $u$ 有

```math
u^\top\nabla^2f(x)u>0,
```

即 $\nabla^2f(x)\succ0$，则 $x$ 是严格局部极小点。若 Hessian 不定，则存在正、负曲率方向，$x$ 是鞍点而不是局部极小点。

等式约束问题中，二阶充分条件不能要求所有方向上的 Hessian 都正定，只需在可行切方向上正定。若等式约束为 $c(x)=0$，可行切方向满足

```math
\nabla c(x)d=0.
```

二阶条件应考察 Lagrangian Hessian 在这些方向上的二次型。

#### 1.1.4 特征值、曲率与条件数

对任意单位方向 $u$，对称 Hessian 满足

```math
\lambda_{\min}\bigl(\nabla^2f(x)\bigr)
\le u^\top\nabla^2f(x)u
\le\lambda_{\max}\bigl(\nabla^2f(x)\bigr).
```

最大和最小特征值分别给出最大、最小局部曲率及其对应方向。对于正定 Hessian，谱条件数为

```math
\kappa\bigl(\nabla^2f(x)\bigr)
=\frac{\lambda_{\max}\bigl(\nabla^2f(x)\bigr)}
{\lambda_{\min}\bigl(\nabla^2f(x)\bigr)}.
```

条件数大表示曲率尺度差异大。梯度法会在大曲率方向受步长限制、在小曲率方向收敛缓慢，从而形成之字形迭代；Newton 法通过逆 Hessian 修正尺度，但在 Hessian 接近奇异时会放大导数误差和舍入误差。

一般矩阵的条件数定义为

```math
\kappa(A)=\|A\|\,\|A^{-1}\|.
```

若 $Ax=b$ 且右端项受到扰动，

```math
A(x+\Delta x)=b+\Delta b,
```

则相对误差满足

```math
\frac{\|\Delta x\|}{\|x\|}
\le\kappa(A)\frac{\|\Delta b\|}{\|b\|}.
```

无约束优化通常分析目标函数 Hessian 的条件数；约束优化还必须考虑 KKT 系统及其约化 Hessian 的条件数。

#### 1.1.5 预处理与谱变换

预处理通过变量变换或近似线性求解改善谱性质。其目的不是改变最优解，而是降低有效条件数、平衡不同方向的尺度、提高数值稳定性，并减少迭代线性求解的次数。对大规模问题，预条件器还必须易于应用；一个难以求解的预条件系统并不能降低总计算成本。

对正定二次函数

```math
f(x)=\frac12x^\top Hx-b^\top x,
\qquad H\succ0,
```

设 $H=Q\Lambda Q^\top$，其中 $\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)$。在特征向量坐标 $z=Q^\top(x-x^*)$ 中，各方向彼此解耦，梯度法的误差更新为

```math
z_{k+1,i}=(1-\alpha\lambda_i)z_{k,i}.
```

为了使所有特征方向都稳定，固定步长必须满足

```math
0<\alpha<\frac{2}{\lambda_{\max}}.
```

因此，大曲率方向（$\lambda_{\max}$ 大）决定了步长上限：步长过大时，该方向上的误差会振荡甚至发散。另一方面，小曲率方向（$\lambda_{\min}$ 小）的收缩因子 $1-\alpha\lambda_{\min}$ 接近 $1$，误差衰减很慢。为了在小曲率方向取得明显进展，步长本应较大；但大曲率方向又不允许这样做，这就是梯度法在狭长椭圆谷底中反复横跳、形成之字形轨迹的根本原因。

对强凸二次函数，最优固定步长为

```math
\alpha_*=\frac{2}{\lambda_{\max}+\lambda_{\min}},
```

相应的最坏方向收缩因子为

```math
\frac{\kappa(H)-1}{\kappa(H)+1},
\qquad
\kappa(H)=\frac{\lambda_{\max}}{\lambda_{\min}}.
```

这说明条件数越大，梯度法的线性收敛因子越接近 $1$。所谓“最大曲率方向受限、小曲率方向缓慢”不是两个独立的经验现象，而是同一个谱条件数造成的两个方面。

Newton 法在正定二次函数上直接求解

```math
Hd_k=-\nabla f(x_k),
```

由于 $\nabla f(x)=Hx-b$，有

```math
d_k=-(x_k-x^*),
\qquad
x_{k+1}=x^*.
```

也就是说，精确算术下 Newton 法能够一次消除所有特征方向上的误差；从几何上看，$H^{-1}$ 将不同曲率方向分别缩放为相同尺度，使椭圆等高线变为球面。因此 Newton 法不受梯度法那种单一固定步长的谱限制。

但实际计算中并不显式形成精确的 $H^{-1}$，而是求解线性系统

```math
Hd=-g.
```

当 $\lambda_{\min}(H)$ 很小时，系统对右端扰动和矩阵扰动高度敏感。若梯度含有误差 $\delta g$，则方向误差近似为

```math
\delta d\approx-H^{-1}\delta g.
```

在最小特征值方向上，$H^{-1}$ 的放大因子约为 $1/\lambda_{\min}$；因此微小的导数误差、有限差分误差或舍入误差都可能导致很大的方向误差。若同时存在矩阵近似误差，线性系统的解误差还会受到 $\kappa(H)$ 的放大。实际算法因此使用稳定分解、正则化、修正 Cholesky、信赖域或 Levenberg--Marquardt 移位，而不是无条件地直接采用病态 Newton 步。

### 1.2 最优性条件、下降方向与终止准则

#### 1.2.1 一阶和二阶最优性条件

对无约束问题，若 $x^*$ 是局部极小点且 $f$ 可微，则必须满足

```math
\nabla f(x^*)=0.
```

若 $f$ 二阶连续可微，则二阶必要条件为

```math
\nabla^2f(x^*)\succeq0.
```

若进一步有

```math
\nabla^2f(x^*)\succ0,
```

则 $x^*$ 是严格局部极小点。这些是局部结论；只有在凸性等额外条件下，驻点才自动是全局最优点。

#### 1.2.2 下降方向与充分下降

若

```math
\nabla f(x_k)^\top d_k<0,
```

则 $d_k$ 是 $x_k$ 处的一阶下降方向。由 Taylor 展开可知，存在足够小的 $\alpha>0$，使

```math
f(x_k+\alpha d_k)<f(x_k).
```

若使用二阶 Taylor 展开，则

```math
f(x_k+\alpha d_k)
=f(x_k)+\alpha\nabla f(x_k)^\top d_k
+\frac{\alpha^2}{2}
d_k^\top\nabla^2f(\xi_k)d_k,
```

其中 $\xi_k$ 位于 $x_k$ 与 $x_k+\alpha d_k$ 之间。二阶项是 $O(\alpha^2)$，而下降的一阶项是 $O(\alpha)$；因此一阶下降方向在充分小的步长下必然产生函数下降。

#### 1.2.3 停止准则

常用停止条件包括梯度残差、步长和函数值变化：

```math
\|\nabla f(x_k)\|\le\varepsilon_g,
\qquad
\|x_{k+1}-x_k\|\le\varepsilon_x,
\qquad
|f(x_{k+1})-f(x_k)|\le\varepsilon_f.
```

后两项只表示当前迭代变化很小，可能由病态、过小步长或数值停滞造成；它们不能单独证明接近最优点。因此，实际实现应优先报告一阶残差，并结合相对步长、相对函数变化、迭代上限和数值异常检查。

#### 1.2.4 收敛阶

设 $e_k=\|x_k-x^*\|$。线性、超线性与二次收敛可分别表述为

```math
\limsup_{k\to\infty}\frac{e_{k+1}}{e_k}<1,
\qquad
\frac{e_{k+1}}{e_k}\to0,
\qquad
 e_{k+1}\le C e_k^2.
```

三者都描述局部渐近性质，不能替代全局收敛分析。

### 1.3 线搜索方法

#### 1.3.1 回溯线搜索与单位步

线搜索方法在下降方向 $d_k$ 上选择步长 $\alpha_k$。非精确线搜索的目标是避免步长过大导致模型失真，也避免步长过小导致进展停滞。

一次回溯线搜索可写为：给定 $\alpha\leftarrow\alpha_{\mathrm{init}}$（通常为 $1$），检查 Armijo 条件；若不满足则令 $\alpha\leftarrow\rho\alpha$，其中 $0<\rho<1$；重复直到条件满足或触发最小步长保护，最后取 $\alpha_k=\alpha$。Newton、拟 Newton 和 SQP 方法通常优先尝试全步 $\alpha=1$，因为全步是局部快速收敛的重要组成部分。

#### 1.3.2 Armijo、Wolfe 与强 Wolfe 条件

Armijo 充分下降条件为

```math
f(x_k+\alpha_kd_k)
\le f(x_k)+c_1\alpha_k\nabla f(x_k)^\top d_k,
\qquad 0<c_1<1.
```

Armijo 条件排除了过大的步长，但仍可能接受过小步长。Goldstein 条件、Wolfe 条件和强 Wolfe 条件进一步控制曲率信息。Wolfe 条件为

```math
\nabla f(x_k+\alpha_kd_k)^\top d_k
\ge c_2\nabla f(x_k)^\top d_k,
\qquad 0<c_1<c_2<1.
```

强 Wolfe 条件为

```math
\left|\nabla f(x_k+\alpha_kd_k)^\top d_k\right|
\le c_2\left|\nabla f(x_k)^\top d_k\right|.
```

回溯线搜索从初始步长（通常为 $1$）开始，重复令 $\alpha\leftarrow\rho\alpha$，其中 $0<\rho<1$，直至满足 Armijo 条件。多项式插值可在已知函数值和导数信息时更高效地试探步长，但必须使用保护区间避免插值给出过小或越界步长。

#### 1.3.3 线搜索的收敛作用

在连续可微、方向具有一致下降性、函数值水平集有界等条件下，非精确线搜索可以保证迭代点的梯度残差趋于零。具体结论还取决于方向生成规则和所采用的线搜索条件。

### 1.4 信赖域方法

#### 1.4.1 二次模型与接受准则

在 $x_k$ 处，信赖域方法用二次模型

```math
q_k(d)=f(x_k)+g_k^\top d+\frac12d^\top G_kd
```

近似真实函数，并在半径 $\Delta_k$ 内求解

```math
\min_d\ q_k(d),
\qquad \text{s.t.}\quad \|d\|\le\Delta_k.
```

模型质量由实际下降与预测下降之比衡量：

```math
\rho_k=
\frac{f(x_k)-f(x_k+d_k)}{q_k(0)-q_k(d_k)}.
```

当 $\rho_k$ 接近 $1$ 时，模型预测准确，通常接受该步并扩大信赖域；当 $\rho_k$ 接近零或为负时，模型不可靠，拒绝该步并缩小信赖域。与线搜索不同，信赖域能够在 Hessian 不定时通过边界步和负曲率方向继续产生有意义的模型下降。

#### 1.4.2 信赖域子问题与 Cauchy 步

信赖域子问题不必精确求解。Cauchy 点、Dogleg 和截断共轭梯度只要取得足够的模型下降，即可支撑标准全局收敛分析。信赖域半径不是单纯的步长上界，而是局部模型可信区域的数值表达。

最简单的 Cauchy 步沿负梯度方向求解一维模型。令 $g_k=\nabla f(x_k)$，并取 $d=-\tau g_k$，则在正曲率情况下

```math
\tau_k=
\min\left\{
\frac{\|g_k\|^2}{g_k^\top G_kg_k},
\frac{\Delta_k}{\|g_k\|}
\right\}.
```

若 $g_k^\top G_kg_k\le0$，则沿该方向走到信赖域边界。Cauchy 点不一定接近 Newton 步，但具有可证明的模型下降量，因此足以作为信赖域算法的全局化基础。

> 教材索引：一阶必要条件可参阅高立《数值最优化方法》pp. 9–10，或 Nocedal--Wright, *Numerical Optimization*, pp. 14–15；二阶必要条件对应前者 p. 10、后者 p. 15；二阶充分条件对应前者 pp. 10–11、后者 p. 16。非精确线搜索可参阅 Nocedal--Wright, pp. 35、38–39；信赖域方法建议阅读其 Chapter 4。

## 2. 梯度法、Newton 法与拟 Newton 法

本章讨论无约束光滑优化中最常用的方向生成机制。三类方法的共同形式为 $x_{k+1}=x_k+\alpha_kd_k$，但使用的曲率信息不同：梯度法仅用一阶导数，Newton 法使用完整 Hessian，拟 Newton 法则以低秩更新学习 Hessian 或其逆。

### 2.1 最速下降法

最速下降法取

```math
d_k=-\nabla f(x_k).
```

在欧氏范数下，负梯度方向是单位长度约束下使一阶模型下降最快的方向。若 $f$ 为强凸二次函数，精确线搜索的梯度法线性收敛，速度由 Hessian 条件数控制；条件数大时，相邻搜索方向近似正交，迭代轨迹呈明显之字形。

精确线搜索下有

```math
\nabla f(x_{k+1})^\top d_k=0.
```

该正交关系是二次函数和精确线搜索的性质，不应误认为一般非线性问题或非精确线搜索都满足严格正交。

### 2.2 Newton 法

Newton 法通过二阶 Taylor 模型产生方向：

```math
\nabla^2 f(x_k)d_k=-\nabla f(x_k).
```

若 $\nabla^2f(x_k)\succ0$，则 $d_k$ 是下降方向，并且可写为

```math
d_k=-\nabla^2f(x_k)^{-1}\nabla f(x_k).
```

实际实现只求解线性系统，不显式形成逆矩阵。若最优点满足二阶充分条件、Hessian 在邻域内 Lipschitz 连续，且初值足够接近最优点，则基本 Newton 迭代局部二次收敛。

基本 Newton 法具有局部二次收敛性，但其直接形式不具备全局鲁棒性。其局限性和相应的全局化思路可以分为以下几点。

#### 2.2.1 基本 Newton 法的局限性

**（1）局部二次收敛的适用范围。** 二次收敛结论要求初值已经进入解的局部收敛域。在较远的迭代点，Hessian 可能不定、奇异或病态。Hessian 不定时，Newton 方程虽然可能有解，但所得方向不一定是下降方向；Hessian 奇异时，线性系统可能无解、解不唯一，或对舍入误差极其敏感。因此，迭代可能发散、停滞，或者无法产生可接受的 Newton 步。这里的表述应理解为风险，而不是说远离解时 Hessian 必然不定或奇异。

**（2）单位 Newton 步不保证函数值下降。** 即使 Hessian 正定，也只能保证 Newton 方向是下降方向：


```math
\nabla f(x_k)^\top d_k
=-d_k^\top\nabla^2f(x_k)d_k<0.
```

基本 Newton 法通常直接采用单位步

```math
x_{k+1}=x_k+d_k.
```

但“方向是下降方向”不等于“单位步一定使函数值下降”。二阶模型只在当前点附近可信，单位 Newton 步可能走出该邻域，从而导致实际函数值增加。因此，基本 Newton 法不能保证目标函数序列单调下降；阻尼 Newton、线搜索和信赖域方法正是为了解决这一问题。

**（3）Hessian 形成与线性系统求解的成本。** 基本 Newton 法每步还具有较高的计算成本。对稠密对称 Hessian，独立二阶偏导数约为


```math
\frac{n(n+1)}{2}.
```

因此，形成和存储 Hessian 通常需要 $O(n^2)$ 的成本；对稠密 Hessian 进行 Cholesky 或 $LDL^\top$ 分解，通常需要 $O(n^3)$ 的计算量。实际实现并不显式计算 Hessian 的逆，而是求解线性系统；但对稠密、无特殊结构的问题，该线性系统求解仍具有 $O(n^3)$ 的主导复杂度。

上述复杂度不是所有问题的必然结论。稀疏 Hessian、块结构、Hessian-vector product、截断 Newton 和预条件 Krylov 方法都可以降低存储或计算成本；相应复杂度取决于稀疏模式、填充、条件数和预条件器。因此，Newton 法的实际成本应结合问题结构讨论。

#### 2.2.2 阻尼 Newton 法

阻尼 Newton 法通过线搜索取

```math
x_{k+1}=x_k+\alpha_kd_k,
\qquad 0<\alpha_k\le1.
```

在正定 Hessian、适当线搜索条件和函数值水平集有界等假设下，可以获得全局化的驻点收敛；当迭代进入局部收敛域并逐渐接受全步 $\alpha_k=1$ 时，又能恢复 Newton 法的二次收敛。

### 2.3 修正 Newton 与 Levenberg--Marquardt 型正则化

若 Hessian 不定或病态，可改用

```math
\bigl(\nabla^2f(x_k)+\nu_k I\bigr)d_k=-\nabla f(x_k),
\qquad \nu_k\ge0.
```

当 $\nu_k$ 足够大时，系数矩阵正定，方向成为下降方向；$\nu_k$ 较小时，方向接近 Newton 步；$\nu_k$ 较大时，方向接近缩放的负梯度步。修正 Cholesky、谱修正和信赖域方法也服务于同一目标：在保留有效曲率信息的同时确保子问题可解且模型步可接受。

### 2.4 拟 Newton 法

拟 Newton 法避免每步显式计算 Hessian。令

```math
s_k=x_{k+1}-x_k,
\qquad
y_k=\nabla f(x_{k+1})-\nabla f(x_k).
```

由梯度 Taylor 展开，$y_k\approx\nabla^2 f(x_{k+1})s_k$。因此近似 Hessian $B_{k+1}$ 或逆 Hessian $H_{k+1}$ 应满足割线条件

```math
B_{k+1}s_k=y_k,
\qquad
H_{k+1}y_k=s_k.
```

BFGS 的 Hessian 形式为

```math
B_{k+1}=B_k
-\frac{B_ks_ks_k^\top B_k}{s_k^\top B_ks_k}
+\frac{y_ky_k^\top}{y_k^\top s_k}.
```

若 $B_k\succ0$ 且 $y_k^\top s_k>0$，则 $B_{k+1}\succ0$。精确线搜索或满足曲率条件的 Wolfe 线搜索通常保证该曲率条件。DFP 是另一种正定更新；SR1 可以表达不定曲率，但当更新分母接近零时需要跳过更新。

使用 $H_k$ 时方向为

```math
d_k=-H_k\nabla f(x_k).
```

使用 $B_k$ 时通过解 $B_kd_k=-\nabla f(x_k)$ 得到方向。二者在理论上等价于使用互逆近似，但数值实现中常优先维护方便分解、可检测正定性的形式。

拟牛顿更新并不是任意地修改一个矩阵，而是在已有近似矩阵的基础上，同时满足割线条件和尽可能少修改原矩阵的要求。以逆 Hessian 近似为例，若在第 $k$ 步得到 $s_k$ 和 $y_k$，则更新矩阵应满足

```math
H_{k+1}y_k=s_k.
```

这条关系称为拟牛顿方程。它来自梯度的一阶 Taylor 展开：

```math
\nabla f(x_k+s_k)
=\nabla f(x_k)+\nabla^2f(x_k)s_k+O(\|s_k\|^2),
```

因此

```math
y_k=\nabla f(x_{k+1})-\nabla f(x_k)
=\nabla^2f(x_k)s_k+O(\|s_k\|^2).
```

若维护 Hessian 近似，则割线条件写成 $B_{k+1}s_k=y_k$；若维护逆 Hessian 近似，则写成 $H_{k+1}y_k=s_k$。这两种形式在精确算术下相互对应，但数值上并不要求显式计算 $B_{k+1}^{-1}$。拟牛顿法的核心价值在于：用一次新的位移—梯度差对曲率模型进行低秩修正，避免显式计算全部二阶偏导数，同时保留比纯梯度法更丰富的尺度信息。

#### 2.4.1 变度量意义下的最速下降

最速下降方向依赖于度量。给定对称正定矩阵 $M$，定义

```math
\|d\|_M=(d^\top Md)^{1/2},
\qquad
\|g\|_{M^{-1}}=(g^\top M^{-1}g)^{1/2}.
```

在单位 $M$-范数方向中，一阶模型下降量满足

```math
|\nabla f(x_k)^\top d|
=|(M^{-1}\nabla f(x_k))^\top Md|
\le
\|\nabla f(x_k)\|_{M^{-1}}\|d\|_M.
```

等号成立当且仅当 $d$ 与 $-M^{-1}\nabla f(x_k)$ 同向。因此，$-M^{-1}\nabla f(x_k)$ 是 $M$-度量下的一阶模型最速下降方向。三类方法可以统一理解为：

- 最速下降法取 $M=I$，方向为 $-\nabla f(x_k)$；
- Newton 法取 $M=\nabla^2f(x_k)$，方向为 $-\nabla^2f(x_k)^{-1}\nabla f(x_k)$；
- 拟 Newton 法取 $M=B_k$ 或 $M=H_k^{-1}$，方向为 $-B_k^{-1}\nabla f(x_k)$ 或 $-H_k\nabla f(x_k)$。

所以，拟 Newton 法可以看作在不断学习局部曲率度量下进行最速下降，而不是简单地对梯度方向乘以一个经验矩阵。这也解释了它通常比固定欧氏度量下的梯度法更少受到变量尺度差异的影响。

#### 2.4.2 对称性、正定性与具体更新公式

常用更新都要求保持矩阵的对称性。DFP 的逆 Hessian 更新为

```math
H_{k+1}^{\mathrm{DFP}}
=H_k
-\frac{H_ky_ky_k^\top H_k}{y_k^\top H_ky_k}
+\frac{s_ks_k^\top}{y_k^\top s_k},
```

而 BFGS 的逆 Hessian 更新为

```math
H_{k+1}^{\mathrm{BFGS}}
=
\left(I-\frac{s_ky_k^\top}{y_k^\top s_k}\right)
H_k
\left(I-\frac{y_ks_k^\top}{y_k^\top s_k}\right)
+\frac{s_ks_k^\top}{y_k^\top s_k}.
```

前文给出的 BFGS Hessian 形式与上述逆 Hessian 形式是对偶的。SR1 更新则为

```math
B_{k+1}^{\mathrm{SR1}}
=B_k+
\frac{(y_k-B_ks_k)(y_k-B_ks_k)^\top}
{(y_k-B_ks_k)^\top s_k},
```

它不强制保持正定性，因此可以表达负曲率；但当分母过小时，更新会放大误差，通常应跳过该次更新。更一般地，Broyden 族在满足割线条件的对称更新之间进行参数化，具体选择取决于是否优先保持正定性、表达不定曲率或降低计算成本。

正定性保持是 BFGS 和 DFP 能够生成下降方向的关键。若 $H_k\succ0$ 且 $y_k^\top s_k>0$，则 DFP 和 BFGS 的逆 Hessian 更新均满足

```math
H_{k+1}\succ0.
```

相应地，若 $B_k\succ0$ 且 $y_k^\top s_k>0$，则其 Hessian 形式也保持正定。此时由

```math
d_k=-H_k\nabla f(x_k)
```

可得

```math
\nabla f(x_k)^\top d_k
=-\nabla f(x_k)^\top H_k\nabla f(x_k)<0
```

只要梯度非零，方向就是下降方向。因此，$y_k^\top s_k>0$ 不是形式上的附加条件，而是连接线搜索、矩阵正定性和下降方向的核心曲率条件。

### 2.5 拟 Newton 法的收敛性与实现要点

#### 2.5.1 曲率条件与线搜索

精确线搜索下，若 $d_k$ 是搜索方向，则

```math
\nabla f(x_{k+1})^\top d_k=0.
```

当 $s_k=\alpha_kd_k$ 时，若 $d_k$ 是下降方向，则 $\nabla f(x_k)^\top d_k<0$，从而精确线搜索给出

```math
y_k^\top s_k
=\alpha_k\bigl(\nabla f(x_{k+1})-\nabla f(x_k)\bigr)^\top d_k
=-\alpha_k\nabla f(x_k)^\top d_k>0.
```

非精确线搜索不要求严格满足一维最小化，但满足适当的 Wolfe 条件时同样可以保证曲率条件。强 Wolfe 条件为

```math
\left|\nabla f(x_{k+1})^\top d_k\right|
\le c_2\left|\nabla f(x_k)^\top d_k\right|,
\qquad 0<c_1<c_2<1,
```

结合 Armijo 条件可推出 $y_k^\top s_k>0$。因此，在 BFGS 或 DFP 中通常优先使用 Wolfe 型线搜索；Goldstein 条件本身并不直接保证拟牛顿更新所需的曲率条件。

#### 2.5.2 局部超线性收敛

拟 Newton 法的目标不是仅保证梯度趋于零，还要使近似曲率逐渐接近真实 Hessian。设 $x^*$ 是满足二阶充分条件的严格局部极小点，且 $\nabla^2f(x^*)$ 非奇异。若迭代点收敛到 $x^*$，线搜索最终接受单位步，并且近似矩阵满足 Dennis--Moré 条件

```math
\frac{\|(B_k-\nabla^2f(x^*))s_k\|}{\|s_k\|}
\longrightarrow0,
```

则拟 Newton 迭代具有局部超线性收敛性，即

```math
\frac{\|x_{k+1}-x^*\|}{\|x_k-x^*\|}
\longrightarrow0.
```

这里的“超线性”比固定收敛因子的线性收敛更快，但通常不等同于 Newton 法的二次收敛。BFGS 在标准光滑性、正定性和线搜索条件下具有良好的局部超线性理论；这一定理不能理解为任意非凸问题、任意初值和任意线搜索下都自动成立。

#### 2.5.3 维护 $B_k$ 与维护 $H_k$ 的数值差异

若显式形成密集 $B_k$ 并在每次迭代重新分解，则求解

```math
B_kd_k=-\nabla f(x_k)
```

通常需要 $O(n^3)$ 的分解成本。维护逆 Hessian 近似 $H_k$ 时，BFGS 或 DFP 的矩阵—向量乘法和秩二更新通常为 $O(n^2)$，因而不必每次解新的线性方程组。但这并不意味着逆 Hessian 形式在数值上总是更优：舍入误差可能使显式 $H_k$ 逐渐失去对称正定性，而且这种失效有时不易及时发现。

维护 $B_k$ 的一个重要优点是可以使用稳定分解。若 $B_k\succ0$，可维护 Cholesky 因子；更一般地，可维护

```math
B_k=L_kD_kL_k^\top.
```

通过观察 $D_k$ 的对角元，可以检测病态或非正定情形，并采取跳过更新、阻尼更新、重新初始化或切换到负梯度方向等措施。对于 BFGS 和 DFP，还可以直接更新因子而不重新对整个 $B_k$ 做分解，使单次迭代成本降至与维护 $H_k$ 相近的 $O(n^2)$。因此，选择 $B_k$ 形式并不只是理论偏好，而是与稳定性监测和工程实现直接相关。

#### 2.5.4 拟 Newton 算法流程

一个标准的 BFGS/DFP 型算法可以概括为：

1. 选取初始点 $x_0$ 和对称正定矩阵 $H_0$，通常取 $H_0=I$。
2. 计算 $g_k=\nabla f(x_k)$；若满足梯度或步长停止准则，则终止。
3. 计算搜索方向 $d_k=-H_kg_k$。若 $g_k^\top d_k\ge0$ 或数值上过于接近零，则重置 $H_k$ 或退回 $d_k=-g_k$。
4. 用满足 Armijo--Wolfe 或强 Wolfe 条件的线搜索确定 $\alpha_k$，并令 $x_{k+1}=x_k+\alpha_kd_k$。
5. 计算 $s_k=\alpha_kd_k$ 和 $y_k=g_{k+1}-g_k$。若 $y_k^\top s_k$ 太小或非正，则跳过更新、采用阻尼 BFGS 或重新初始化近似矩阵。
6. 根据所选公式更新 $H_{k+1}$ 或 $B_{k+1}$，并进入下一步。

这一流程体现了拟 Newton 法的完整逻辑：线搜索负责获得可靠的位移—梯度差，割线更新负责学习曲率，正定性检查负责维持下降方向，而停止准则和重置策略负责处理有限精度及非凸结构。

实现时应记录：曲率条件 $y_k^\top s_k$ 是否成立；Hessian 近似是否正定；线搜索是否频繁回溯；以及梯度残差是否实际下降。若连续多步回退到负梯度方向，应检查变量缩放、模型病态、导数误差或局部极小点以外的非凸结构。

> 教材索引：最速下降法的收敛速度可参阅高立《数值最优化方法》pp. 41–46，或 Nocedal--Wright, §3.3；精确线搜索下的正交性见高立 p. 19。Newton 法的二次收敛与阻尼 Newton 法见高立 pp. 50–54；拟 Newton 法和 BFGS/DFP/SR1 可参阅高立第 3.3–3.5 节及 Nocedal--Wright, Chapter 6。

## 3. 共轭梯度与非线性最小二乘

这两类方法都利用特殊结构降低二阶计算成本。共轭梯度处理大规模对称正定线性系统及其二次优化；非线性最小二乘则利用残差平方和结构构造 Gauss--Newton、LM 和 Dogleg 方法。

### 3.1 线性共轭梯度方法

对称正定系统 $Hx=b$ 等价于最小化

```math
q(x)=\frac12x^\top Hx-b^\top x,
\qquad H\succ0.
```

令 $r_k=b-Hx_k$，并取 $d_0=r_0$。线性共轭梯度迭代为

```math
\alpha_k=\frac{r_k^\top r_k}{d_k^\top H d_k},
\qquad x_{k+1}=x_k+\alpha_kd_k,
```

```math
r_{k+1}=r_k-\alpha_kHd_k,
\qquad \beta_k=\frac{r_{k+1}^\top r_{k+1}}{r_k^\top r_k},
```

```math
d_{k+1}=r_{k+1}+\beta_kd_k.
```

精确算术下，$d_i^\top Hd_j=0$（$i\ne j$），算法至多 $n$ 步得到精确解。有限精度下共轭性会丢失，通常通过残差阈值、最大迭代次数、重启与预条件控制误差。经典能量范数界为

```math
\|e_k\|_H
\le2\left(\frac{\sqrt{\kappa(H)}-1}{\sqrt{\kappa(H)}+1}\right)^k\|e_0\|_H.
```

预条件共轭梯度以易于求解的 $M\succ0$ 近似 $H$，其目标是改善 $M^{-1}H$ 的谱分布，而不是形成矩阵逆。

### 3.2 非线性共轭梯度

非线性共轭梯度使用

```math
d_{k+1}=-g_{k+1}+\beta_kd_k,
\qquad g_k=\nabla f(x_k).
```

Fletcher--Reeves、Polak--Ribière--Polyak 和 Hestenes--Stiefel 提供不同的 $\beta_k$。非线性问题中不再有精确共轭性，因此需配合 Wolfe 线搜索和重启；常用 $\beta_k\leftarrow\max(\beta_k,0)$ 防止方向失去下降性。

### 3.3 非线性最小二乘模型

考虑非线性最小二乘问题

```math
\min_x f(x)=\frac12\|r(x)\|_2^2,
\qquad r:\mathbb R^n\to\mathbb R^m.
```

令 $J(x)=\nabla r(x)$，则

```math
\nabla f(x)=J(x)^\top r(x),
```

```math
\nabla^2f(x)=J(x)^\top J(x)+\sum_{i=1}^m r_i(x)\nabla^2r_i(x).
```

第二项是残差加权的二阶曲率。残差小或模型接近线性时该项较小；大残差时不能简单忽略。

### 3.4 Gauss--Newton、LM 与 Dogleg

Gauss--Newton 在线性化残差后解

```math
\min_d\frac12\|r_k+J_kd\|_2^2,
```

其正规方程为

```math
J_k^\top J_kd_k=-J_k^\top r_k.
```

稳定实现通常使用 QR 或 SVD，而不是显式形成 $J_k^\top J_k$，后者会平方条件数。LM 方法引入正则化：

```math
\bigl(J_k^\top J_k+\lambda_kI\bigr)d_k=-J_k^\top r_k,
\qquad \lambda_k>0.
```

$\lambda_k$ 较大时接近最速下降，较小时接近 Gauss--Newton。Dogleg 在 Cauchy 步与 Gauss--Newton 步之间选择信赖域步。若最优点残差较大，应使用完整 Newton、LM、信赖域或结构化拟 Newton，而不能仅依据 $J^\top J$ 的半正定性判断收敛性。

> 教材索引：非线性最小二乘的 Gauss--Newton、LM 和 Dogleg 可结合 Nocedal--Wright 的最小二乘章节阅读；本章将其与一般 Newton、信赖域和正则化统一起来。

### 3.5 推导与算法流程

残差线性化为 $r(x_k+d)=r_k+J_kd+O(\|d\|^2)$，代入平方和得到

```math
\frac12\|r_k+J_kd\|^2
=\frac12\|r_k\|^2+d^\top J_k^\top r_k
+\frac12d^\top J_k^\top J_kd.
```

因此 Gauss--Newton 的梯度项和曲率项分别为 $J_k^\top r_k$ 与 $J_k^\top J_k$。实际流程是计算残差和 Jacobian、求解 Gauss--Newton/LM/Dogleg 子问题、用实际下降与预测下降比接受步长，再更新正则化参数。正规方程会平方条件数，病态问题优先使用 QR 或 SVD。


## 4. 约束最优化与 KKT 条件

本章讨论一般非线性约束优化问题的最优性理论。与无约束情形相比，约束使“梯度为零”不再是必要条件：在边界最优点，目标函数梯度可以非零，但它不能沿任何可行方向继续下降。本章的核心逻辑是先描述可行方向，再用约束资格条件把几何上的“没有可行下降方向”转化为代数上的 KKT 条件，最后在临界方向上考察 Lagrangian 的二阶曲率。这一逻辑也是后续 QP、SQP、主动集法和内点法的理论基础。

全文默认采用 $g(x)\le0$ 的不等式约定，考虑

```math
\min_x f(x),\qquad c(x)=0,\qquad g(x)\le0.
```

定义 Lagrangian

```math
\mathcal L(x,\lambda,\mu)=f(x)+\lambda^\top c(x)+\mu^\top g(x).
```

在 LICQ、MFCQ 等约束资格条件下，局部最优点满足

```math
\nabla_x\mathcal L(x^*,\lambda^*,\mu^*)=0,
\quad c(x^*)=0,
\quad g(x^*)\le0,
\quad \mu^*\ge0,
\quad \mu_i^*g_i(x^*)=0.
```

它们分别是驻点性、原始可行性、对偶可行性与互补性。约束资格条件的作用是保证“没有可行下降方向”可以转化为存在乘子的 KKT 表述。二阶必要与充分条件应在临界锥上考察 Lagrangian Hessian；等式约束下，若 $Z$ 是约束 Jacobian 的零空间基，则考察 $Z^\top\nabla_{xx}^2\mathcal LZ$。对凸问题，在 Slater 条件等适当假设下，KKT 条件也是全局最优性的充分条件。

若不等式写成 $\tilde g(x)\ge0$，则通常将 Lagrangian 写成 $f(x)-\mu^\top\tilde g(x)$，并仍要求 $\mu\ge0$。把不等式乘以 $-1$ 不改变可行域，但会同时改变梯度方向和乘子表达，因此乘子正负不能脱离不等式约定讨论。



### 4.1 可行方向、线性化方向与 Farkas 引理

#### 4.1.1 可行方向与线性化可行方向

设 $x$ 是可行点。真实可行方向可理解为使 $x+td$ 在足够小的 $t>0$ 下仍可行的方向；严格地说，它与可行集在 $x$ 处的切锥有关。对等式约束，一阶可行性要求 $\nabla c_j(x)^\top d=0$。对于不等式约束，只有活跃约束会在一阶上限制方向。记

```math
\mathcal A(x)=\{i:g_i(x)=0\}.
```

则线性化可行方向集合为

```math
\mathcal L_F(x)=
\left\{d:\nabla c(x)d=0,\quad
\nabla g_i(x)^\top d\le0,\ i\in\mathcal A(x)\right\}.
```

不等式方向的符号来自 $g_i(x)\le0$；若采用 $g_i(x)\ge0$ 的记号，方向不等式和乘子表达会同时反向。

下降方向集合为 $\mathcal D(x)=\{d:\nabla f(x)^\top d<0\}$。局部极小点的基本事实是不存在既可行又能使目标函数一阶下降的方向，即

```math
\mathcal F(x^*)\cap\mathcal D(x^*)=\varnothing.
```

这个结论本身只涉及真实可行方向，是约束一阶最优性理论的几何起点。线性化集合通常更容易计算，但它不一定等于真实可行方向集合：线性化约束可能允许一阶上看似可行、实际上被高阶项阻挡的方向。因此，需要约束资格条件才能把真实可行方向的结论转移到线性化方向。

#### 4.1.2 等式约束、活跃不等式与乘子符号

1. **只有等式约束。** 若 $x^*$ 是局部极小点，则目标函数在约束曲面的切空间内不能下降。于是，$\nabla f(x^*)$ 必须属于等式约束梯度所张成的法空间，即存在不受符号限制的乘子 $\lambda^*$ 使

    ```math
    \nabla f(x^*)+\nabla c(x^*)^\top\lambda^*=0.
    ```

   等式约束的方向没有内外之分：$c(x)=0$ 与 $-c(x)=0$ 描述同一个可行集。因此，把约束乘以 $-1$ 后，约束梯度和对应乘子同时变号，最优点和几何条件不变。这正是等式约束乘子不要求非负的原因。

2. **单个不等式约束。** 采用本章的约定 $g(x)\le0$ 时，在最优点有两种情况：

   - **非活跃约束：** 当 $g(x^*)<0$ 时，$x^*$ 周围存在足够小的邻域，使该约束不会限制扰动。因此，就这一约束而言，局部问题与无约束问题没有区别；若不存在其他约束，则必须有 $\nabla f(x^*)=0$。在 Lagrangian 驻点条件中，这对应于

     ```math
     \mu^*=0.
     ```

   - **活跃约束：** 当 $g(x^*)=0$ 时，约束边界在一阶上相当于等式约束。对可行方向必须有

     ```math
     \nabla g(x^*)^\top d\le0.
     ```

     而在局部最优点，所有这些可行方向都不能使目标函数下降。因此，目标梯度必须由活跃约束的法向方向抵消。由于不等式只允许朝可行域内部的一侧运动，分离定理给出的系数具有固定符号；在 $g(x)\le0$ 的约定下，该系数就是满足 $\mu^*\ge0$ 的不等式乘子。若改用 $c(x)\ge0$ 并将 Lagrangian 写成 $f(x)-\lambda c(x)$，则乘子仍为非负，但驻点公式中的符号相应改变。

把非活跃和活跃两种情况合并，就得到

```math
\mu^*\ge0,
\qquad
g(x^*)\le0,
\qquad
\mu^*g(x^*)=0.
```

因此，互补性并不是额外加入的形式条件，而是“非活跃约束不产生法向反作用、活跃约束才可能产生反作用”的代数表达。

#### 4.1.3 约束资格条件

LICQ 要求活跃约束梯度线性无关，它保证局部乘子唯一；MFCQ 较弱，只要求等式 Jacobian 满秩并存在严格减小所有活跃不等式的方向。约束资格条件失效时，常见数值表现包括乘子爆炸、KKT 矩阵秩亏和线性化子问题不稳定。

#### 4.1.4 Farkas 引理与 KKT 乘子

下面分别从等式约束和不等式约束说明 Farkas 引理如何完成“无共同方向”到 KKT 乘子的转换。

这也可以直接按“无共同方向”的方式理解。对于等式约束，不存在方向 $d$ 同时满足

```math
 a_i^\top d=0,\quad i\in\mathcal E,
\qquad
\nabla f(x^*)^\top d<0
```

意味着目标梯度不能在等式约束的切空间中留下下降分量。Farkas 引理给出其等价形式：目标梯度必须属于等式约束梯度的线性张成空间，即

```math
\nabla f(x^*)=-\sum_{i\in\mathcal E}\lambda_i^*a_i.
```

对于不等式约束，若活跃约束采用 $g_i(x)\le0$，则不存在方向 $d$ 同时满足

```math
 a_i^\top d\le0,\quad i\in\mathcal A(x^*),
\qquad
\nabla f(x^*)^\top d<0
```

意味着

```math
\nabla f(x^*)=-\sum_{i\in\mathcal E}\lambda_i^*a_i
-\sum_{i\in\mathcal A(x^*)}\mu_i^*a_i,
\qquad
\mu_i^*\ge0.
```

若将不等式写成 $c_i(x)\ge0$，线性化方向条件变为 $a_i^\top d\ge0$，相应的 Lagrangian 采用减号，结论在几何上完全相同。这一分离视角不仅解释 KKT 条件，也解释了外点罚函数、障碍函数和主动集方法为何分别关注约束违反、边界内侧和活跃约束集合。

### 4.2 KKT 条件的几何解释

可以将 KKT 条件中的各个部分作如下理解：

1. **等式约束：切向方向不能下降。** 等式约束把可行运动限制在约束曲面的切空间内，因此最优点处目标梯度不必为零，只需没有切向下降分量。目标梯度由约束法向量张成，等式乘子描述目标函数沿这些法向方向的平衡强度，且不受正负号限制。

2. **非活跃不等式：没有局部反作用。** 若 $g_i(x^*)<0$，约束有余量，局部小扰动不会立即碰到它，因此该约束对当前局部问题没有一阶影响，必有 $\mu_i^*=0$。

3. **活跃不等式：边界产生单侧反作用。** 若 $g_i(x^*)=0$，约束边界在一阶上限制可行方向；在 $g_i(x)\le0$ 的约定下，允许方向满足 $\nabla g_i(x^*)^\top d\le0$。相应的法向反作用只能指向可行域外侧的抵消方向，因此乘子满足 $\mu_i^*\ge0$。

4. **互补性：只有接触边界才产生反作用。** 非活跃约束满足 $\mu_i^*=0$，活跃约束才可能有 $\mu_i^*>0$，两者统一写为

```math
\mu_i^*g_i(x^*)=0.
```

综合起来，在 LICQ、MFCQ 或其他适当约束资格条件下，局部极小点满足

```math
\nabla_x\mathcal L(x^*,\lambda^*,\mu^*)=0,
\qquad c(x^*)=0,
\qquad g(x^*)\le0,
\qquad \mu^*\ge0,
\qquad \mu_i^*g_i(x^*)=0.
```

反过来，KKT 点不一定就是局部极小点，也可能是局部极大点、鞍点或退化驻点；需要凸性或二阶充分条件才能得到充分性。

### 4.3 二阶条件与临界锥

设 $\mathcal A(x^*)$ 为活跃不等式集合。临界锥由线性化可行方向中一阶上不改变目标函数的方向组成。对给定的 KKT 乘子 $(\lambda^*,\mu^*)$，可写为

```math
\mathcal C(x^*,\lambda^*,\mu^*)=
\left\{d:
\nabla c(x^*)d=0,
\quad
\nabla g_i(x^*)^\top d=0\ \text{if }\mu_i^*>0,
\quad
\nabla g_i(x^*)^\top d\le0\ \text{if }\mu_i^*=0,
\ i\in\mathcal A(x^*)\right\}.
```

二阶必要条件要求

```math
d^\top\nabla_{xx}^2\mathcal L(x^*,\lambda^*,\mu^*)d\ge0
```

对临界锥中的适当非零方向成立；如果存在临界方向使该量为负，则 $x^*$ 不可能是局部极小点。二阶充分条件通常要求对所有非零临界方向有

```math
d^\top\nabla_{xx}^2\mathcal L(x^*,\lambda^*,\mu^*)d>0.
```

只考察临界锥而不是整个空间是关键：约束已经排除了部分方向，真正需要判断的是仍可能保持一阶可行且一阶目标变化为零的方向。对只有等式约束的情形，若 $Z$ 是约束 Jacobian 的零空间基，则条件退化为约化 Hessian

```math
Z^\top\nabla_{xx}^2\mathcal L(x^*,\lambda^*)Z
```

在相应约化空间上的半正定或正定性。严格互补性、LICQ 和二阶充分条件共同构成 Newton--KKT、SQP 与灵敏度分析中常见的局部正则性假设。

一阶充分性也需要限定适用范围。若在某个邻域内所有非零真实可行方向都满足严格的一阶上升条件，则可得到严格局部极小；若只对线性化可行方向作判断，还需要凸性、凸可行集或适当的高阶可行性条件，不能无条件地把线性化结论当作真实问题的充分条件。对于凸优化问题，若 $f$ 和 $g_i$ 为凸函数、$c$ 为仿射函数，并满足 Slater 条件，则 KKT 条件既是必要条件，也是全局最优性的充分条件。


> 教材索引：约束一阶条件可参阅 Nocedal--Wright, §12.1；Farkas 引理可参阅高立《数值最优化方法》引理 6.9、6.10 和 6.12；约束二阶条件见高立第 6.4 节。

## 5. 二次规划

二次规划之所以是约束优化算法的核心子问题，是因为它同时保留了目标函数的曲率、约束的线性结构和 KKT 条件的完整形式。SQP 在每一步产生 QP，障碍法和增广拉格朗日法也会反复求解具有相同线性代数结构的子问题。因此，QP 求解器不仅要计算最优点，还要能够判断不可行性、检测无界性、利用稀疏结构，并在一系列相近问题之间进行 warm start。

标准 QP 为

```math
\min_x\ \frac12x^\top Hx+q^\top x,
\qquad Ax=b,
\qquad Cx\le d.
```

$H\succeq0$ 时为凸 QP，任意满足 KKT 条件的点都是全局最优解；若 $H\succ0$，目标函数严格凸，在可行时原始解唯一。若 $H$ 不定，则问题可能具有多个局部极小点、鞍点或无界方向，不能直接套用凸 QP 中关于 KKT 条件、主动集终止和内点路径的结论。非凸 QP 的算法通常还需要检测负曲率，并区分局部最优、无界和不可行等情形。

等式约束 QP 的 KKT 系统为

```math
\begin{bmatrix}H&A^\top\\A&0\end{bmatrix}
\begin{bmatrix}x\\\lambda\end{bmatrix}
=
\begin{bmatrix}-q\\b\end{bmatrix}.
```

该系统对称但通常不定，可用 $LDL^\top$、Schur 补、零空间法或约化空间 CG。对于含不等式约束的 QP，主动集法反复把当前活跃不等式视为等式，并通过乘子符号决定加入或删除约束；它适合中小规模及 warm start。原始—对偶内点法则通过松弛变量和扰动互补条件

```math
S\Lambda\mathbf{1}=\mu\mathbf{1}
```

追踪中心路径，适合大规模稀疏问题。两者的主要区别在于如何处理未知活跃集，而不是 KKT 最优性概念不同。



### 5.1 等式约束 QP 的解法

#### 5.1.1 问题形式、KKT 条件与 Newton 步

考虑等式约束 QP

```math
\min_x\ \frac12x^\top Hx+q^\top x,
\qquad Ax=b.
```

其 KKT 条件为

```math
\begin{bmatrix}H&A^\top\\A&0\end{bmatrix}
\begin{bmatrix}x^*\\\lambda^*\end{bmatrix}
=
\begin{bmatrix}-q\\b\end{bmatrix}.
```

若从某个估计点 $x$ 出发写成 $x^*=x+p$，则步长满足

```math
\begin{bmatrix}H&A^\top\\A&0\end{bmatrix}
\begin{bmatrix}p\\\Delta\lambda\end{bmatrix}
=
-\begin{bmatrix}Hx+q+A^\top\lambda\\Ax-b\end{bmatrix}.
```

因此，求解等式约束 QP 的 KKT 系统与对一般等式约束问题的 KKT 方程使用 Newton 法具有相同的线性代数形式。

#### 5.1.2 KKT 系统的非奇异性与约化 Hessian

设 $A$ 行满秩，$Z$ 为 $A$ 的零空间基，即

```math
AZ=0,
\qquad
Z\in\mathbb R^{n\times(n-m)}.
```

若约化 Hessian 满足

```math
Z^\top HZ\succ0,
```

则 KKT 矩阵非奇异，且该 QP 存在唯一全局解。这个条件比 $H\succ0$ 弱：$H$ 可以在约束法向方向上不正定，只要在可行的切空间上具有正曲率即可。

#### 5.1.3 等式约束 QP 的主要解法

等式约束 QP 的主要解法包括：

1. **Schur 补法。** 若 $H$ 可逆，由第一行消去 $x$，得到

   ```math
   (AH^{-1}A^\top)\lambda=b+AH^{-1}q.
   ```

   约束数量较少时，Schur 补规模较小；但显式构造 $H^{-1}$ 或 Schur 补可能破坏稀疏性，实际应通过分解和线性系统求解实现。

2. **零空间法。** 先求一个满足 $Ax_p=b$ 的特解，再令

   ```math
   x=x_p+Zp.
   ```

   代入目标函数后得到无约束约化问题

   ```math
   (Z^\top HZ)p=-Z^\top(Hx_p+q).
   ```

   该方法直接显露约化 Hessian 的正定性，但需要构造或隐式表示零空间基，约束很多时可能不经济。

3. **投影共轭梯度法。** 对大规模问题，可以在满足 $Ap=0$ 的切空间内迭代求解约化系统。该方法不要求显式形成 $Z^\top HZ$，只需矩阵—向量乘积和投影操作，适合大规模稀疏或 Hessian 只提供乘积的场景。

4. **增广 KKT 直接法。** 对 $H$ 不可逆或不定的情形，不应直接使用 $H^{-1}$ 的 Schur 补，而应对完整对称不定 KKT 系统进行带主元的 $LDL^\top$ 分解，或使用稳定的正则化和迭代法。

### 5.2 凸 QP 的内点法

#### 5.2.1 松弛变量与扰动 KKT 条件

考虑

```math
\min_x\ q(x)=\frac12x^\top Gx+c^\top x,
\qquad Ax\ge b,
\qquad G\succeq0.
```

引入松弛变量 $y=Ax-b$，得到 $y\ge0$。令 $\lambda\ge0$ 为不等式乘子，并定义

```math
Y=\operatorname{diag}(y_1,\ldots,y_m),
\qquad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_m),
\qquad
e=(1,\ldots,1)^\top.
```

扰动 KKT 方程为

```math
F(x,y,\lambda;\sigma\mu)=
\begin{bmatrix}
Gx-A^\top\lambda+c\\
Ax-y-b\\
Y\Lambda e-\sigma\mu e
\end{bmatrix}=0.
```

其中 $\mu>0$ 为当前互补度，$\sigma\in[0,1]$ 为中心化参数。令 $\mu\to0$，即可逐渐恢复原 QP 的精确互补条件。

对上述方程施加 Newton 线性化，得到关于 $(\Delta x,\Delta y,\Delta\lambda)$ 的原始—对偶系统。消元的不同顺序求的是同一个 Newton 方向，但会产生不同的数值条件和稀疏结构。

#### 5.2.2 Newton 系统与正规方程

对 Newton 系统定义

```math
r_d=Gx-A^\top\lambda+c,
\qquad
r_p=Ax-y-b.
```

原始—对偶系统为

```math
\begin{bmatrix}
G&0&-A^\top\\
A&-I&0\\
0&\Lambda&Y
\end{bmatrix}
\begin{bmatrix}
\Delta x\\\Delta y\\\Delta\lambda
\end{bmatrix}
=
\begin{bmatrix}
-r_d\\-r_p\\-Y\Lambda e+\sigma\mu e
\end{bmatrix}.
```

消去 $\Delta y$ 后得到紧凑系统

```math
\begin{bmatrix}
G&-A^\top\\
A&\Lambda^{-1}Y
\end{bmatrix}
\begin{bmatrix}
\Delta x\\\Delta\lambda
\end{bmatrix}
=
\begin{bmatrix}
-r_d\\
-r_p-y+\sigma\mu\Lambda^{-1}e
\end{bmatrix}.
```

进一步消去 $\Delta\lambda$，得到 normal equation

```math
(G+A^\top Y^{-1}\Lambda A)\Delta x
=-r_d+A^\top Y^{-1}\Lambda
\left(-r_p-y+\sigma\mu\Lambda^{-1}e\right).
```

正规方程矩阵 $G+A^\top Y^{-1}\Lambda A$ 在适当条件下具有正定结构，便于使用改进的 Cholesky 方法求解；当该矩阵较稀疏且约束较多时，正规方程可能更有效。但形成正规方程可能增加填充并放大条件数，因此不能脱离稀疏结构和数值稳定性单独判断优劣。直接对原始—对偶增广系统进行稳定的对称不定分解，则通常更能保留原始稀疏结构。

若内存需求过大而无法直接分解，可以将系统改写为

```math
\begin{bmatrix}
G&0&-A^\top\\
0&Y^{-1}\Lambda&I\\
A&-I&0
\end{bmatrix}
\begin{bmatrix}
\Delta x\\\Delta y\\\Delta\lambda
\end{bmatrix}
=
\begin{bmatrix}
-r_d\\
-\Lambda e+\sigma\mu Y^{-1}e\\
-r_p
\end{bmatrix}.
```

这个系统可以解释为等式约束凸二次规划的 KKT 条件，其中约化变量为 $(\Delta x,\Delta y)$。因此可以通过适当的变量替换使用投影共轭梯度法；该方法不需要形成或分解 $G$，只需矩阵—向量乘积，适合大规模问题。

#### 5.2.3 步长选择与 Mehrotra 预测—校正

步长必须同时保持松弛变量 $y$ 和对偶变量 $\lambda$ 严格为正。设更新为

```math
(x^+,y^+)=(x,y)+\alpha^{\mathrm{pri}}(\Delta x,\Delta y),
\qquad
\lambda^+=\lambda+\alpha^{\mathrm{dual}}\Delta\lambda.
```

应采用 fraction-to-boundary 规则，使 $y^+>0$、$\lambda^+>0$。由 Newton 系统可得更新后的残差关系

```math
r_p^+=(1-\alpha^{\mathrm{pri}})r_p,
```

```math
r_d^+=(1-\alpha^{\mathrm{dual}})r_d
+(\alpha^{\mathrm{pri}}-\alpha^{\mathrm{dual}})G\Delta x.
```

若 $\alpha^{\mathrm{pri}}=\alpha^{\mathrm{dual}}=\alpha$，则对任意 $\alpha\in(0,1)$，两个残差都具有线性缩减关系。若采用不同步长，原始残差和对偶残差可能暂时增大，因此必须同时监控两者；不过数值上不同步长常能加快收敛。

Mehrotra 预测—校正法通常包含两步：

1. 令中心化参数为零，计算仿射方向，预测若不进行中心化时的互补度；
2. 根据预测互补度选择中心化参数，重新构造带校正项的 Newton 系统并求解校正方向。

这种方法在每次内点迭代中增加少量线性代数工作，却通常能显著减少外层迭代次数。内点法一般不会在有限步精确识别活跃集，而是通过 $s_i$ 和 $y_i$ 的互补尺度渐近地揭示活跃结构。


> 教材索引：等式约束 QP 的 Schur 补、零空间法和约化空间方法见 Nocedal--Wright, §§16.1–16.3；主动集法见 §16.5 或高立第 8.3 节；凸 QP 内点法见 Nocedal--Wright, Chapter 16。

### 5.3 KKT 线性系统的消元视角

5.1 节按问题类型介绍了等式约束 QP 的 Schur 补、零空间法和直接分解；5.2 节介绍了凸 QP 内点法中的原始—对偶 Newton 系统。本节不再引入一种新的 QP 算法，而是从线性代数角度说明这些方法的共同本质：它们都是对 KKT 线性系统选择不同的变量消元顺序。这个视角有助于理解正规方程、增广系统、零空间法和后续最优控制中结构化 Riccati 消元之间的关系。

对等式 QP，由

```math
Hx+A^\top\lambda=-q
```

消去 $x$，得到 Schur 补方程

```math
(AH^{-1}A^\top)\lambda=b+AH^{-1}q.
```

这说明 Schur 补、零空间法和 KKT 直接分解求的是同一个步长，差别在于变量消元顺序、稀疏性和数值稳定性。若 $H$ 不可逆或不定，应使用增广 KKT 系统和稳定的 $LDL^\top$ 分解。

### 5.4 主动集法

#### 5.4.1 主动集法的基本思想

若最优点的活跃集合已知，则所有活跃不等式都可以暂时当作等式处理，问题退化为等式约束 QP。实际中最优活跃集未知，但在可行迭代点 $x_k$ 处可以得到当前活跃集 $\mathcal A_k$，并据此求解工作集等式 QP：

1. 判断当前点是否满足 QP 的最优性条件；
2. 若不满足，将当前活跃约束固定为等式，求解一个可行方向及其步长；
3. 沿该方向移动，同时保证未加入工作集的不等式不被违反；
4. 根据工作集乘子和新点的约束状态加入或删除约束。

这样产生可行点序列和工作集序列。若工作集最终等于最优活跃集，并且等式子问题的约化 Hessian 正定，则工作集子问题的解同时满足原 QP 的 KKT 条件。

在当前工作集下，若求得的方向为零，则需要检查工作集乘子：对于采用 $Cx\le d$ 约定的活跃不等式，若某个乘子违反对偶可行性，则删除最不利的约束；若乘子均满足符号条件，则当前点为最优解。若方向非零，则最大步长由即将碰到的非工作集约束决定，碰到约束后将其加入工作集。

#### 5.4.2 可行初值与大 $M$ 方法

主动集法通常要求初始点可行。若无法直接获得 $Ax=b$、$Cx\le d$ 的可行点，可以引入人工变量或松弛变量，构造 phase-I 问题，使约束违反的总量最小；也可以使用大 $M$ 方法，在目标中对人工变量施加强惩罚。若 phase-I 最优值仍为正，则原 QP 不可行；若人工变量被压到零，则得到后续主动集迭代所需的可行初值。大 $M$ 过大又会造成病态，因此实际实现中通常优先采用显式 phase-I 或弹性 QP。

#### 5.4.3 主动集法与内点法的适用性

主动集法与内点法的差异可以概括为：

- 主动集法显式维护一个候选活跃集，通常每步线性代数成本较低，但可能需要较多次迭代；它能识别活跃集，适合中小规模、边界约束较多以及有良好 warm start 的一系列相关 QP。
- 内点法不显式决定哪些约束活跃，而是通过扰动互补条件逼近中心路径，通常迭代次数较少，但每次需要求解较大的线性系统；它更适合大规模稀疏问题。
- 主动集法的矩阵结构可能随工作集改变，因而需要维护或更新分解；内点法每次迭代的非零结构通常不变，更容易复用稀疏分析和符号分解。
- 两者都不能只依据目标函数变化判断收敛，至少应同时检查原始残差、对偶残差、互补残差以及步长。


## 6. 罚函数、障碍函数与增广拉格朗日

等式约束的二次罚函数为

```math
P(x;\sigma)=f(x)+\frac\sigma2\|c(x)\|^2.
```

它以 $\sigma\to\infty$ 迫使可行性，但 Hessian 中的 $\sigma J_c^\top J_c$ 会导致病态。L1 精确罚函数在适当条件下可用有限罚参数恢复局部解，但在存在非零约束乘子的通常情形下必然引入非光滑性。

对 $g_i(x)<0$，对数障碍函数为

```math
\phi(x;\mu)=f(x)-\mu\sum_i\log(-g_i(x)).
```

其解保持严格可行，并在 $\mu\to0$ 时逼近原问题。增广拉格朗日为

```math
\mathcal L_A(x,\lambda;\sigma)=f(x)+\lambda^\top c(x)+\frac\sigma2\|c(x)\|^2,
\qquad
\lambda_{k+1}=\lambda_k+\sigma_kc(x_{k+1}).
```

它以乘子更新减少对无限大罚参数的依赖，但仍需控制内层精度和罚参数更新。ADMM 是变量分裂后的增广拉格朗日交替最小化，不等同于一般 ALM。



### 6.1 二次外点罚函数

#### 6.1.1 等式约束的罚函数与算法框架

对等式约束问题

```math
\min_x f(x),
\qquad c_i(x)=0,\quad i=1,\ldots,m,
```

定义二次外点罚函数

```math
P_E(x,\sigma)
=f(x)+\frac{\sigma}{2}\sum_{i=1}^m c_i(x)^2
=f(x)+\frac{\sigma}{2}\|c(x)\|^2,
\qquad \sigma>0.
```

对固定 $\sigma_k$，先求解无约束子问题

```math
x_{k+1}\approx\arg\min_x P_E(x,\sigma_k),
```

再增大罚参数并把上一轮解作为下一轮初值。一个基本算法流程为：

1. 选择初始点 $x_0$、罚参数 $\sigma_0>0$ 和内层容差 $\varepsilon_0$；
2. 用无约束优化方法近似最小化 $P_E(x,\sigma_k)$；
3. 若约束违反 $\|c(x_{k+1})\|$ 和目标/驻点残差均满足容差，则终止；
4. 否则增大 $\sigma_k$，减小内层容差，并以 $x_{k+1}$ warm start 下一次子问题。

#### 6.1.2 罚参数和内层精度

罚参数不必每一步按固定倍数增长。若当前内层最小化已经很困难，可只取 $\sigma_{k+1}=1.5\sigma_k$ 一类的温和更新；若当前子问题求解顺利，则可采用更大的增长，例如 $\sigma_{k+1}=10\sigma_k$。内层容差通常只需满足

```math
\varepsilon_k\to0,
```

但实践中应同时监控约束违反下降速度。若 $\|c(x_k)\|$ 下降过慢、目标函数发散或内层迭代出现病态，应增大罚参数、恢复较早迭代点或改变缩放，而不能只增加无约束迭代次数。

#### 6.1.3 收敛性质与数值病态

在适当正则性和内层求解精度条件下，随着 $\sigma_k\to\infty$，罚函数极小点序列趋向原约束问题的可行解；若极限点满足约束资格条件，还可得到原问题的 KKT 条件。二次罚函数一般不具有有限参数精确性：对任意有限 $\sigma$，罚函数极小点通常仍有非零约束违反。

其 Hessian 近似为

```math
\nabla^2_{xx}P_E(x,\sigma)
=\nabla^2f(x)
+\sigma\sum_{i=1}^m
\left[\nabla c_i(x)\nabla c_i(x)^\top
+c_i(x)\nabla^2c_i(x)\right].
```

在接近可行解且约束违反较小时，主导项可写成

```math
\nabla^2_{xx}P_E(x,\sigma)
\approx\nabla^2_{xx}\mathcal L(x,\lambda)
+\sigma J_c(x)^\top J_c(x).
```

若 $J_c$ 行满秩且 $m<n$，则 $J_c^\top J_c$ 在约束法向方向上有正曲率，而在切空间中为零。于是，当 $\sigma\to\infty$ 时，约束法向方向的特征值增长，切向方向曲率保持有界，条件数趋于增大。最终，舍入误差和线性系统误差会限制可达到的可行性精度。

#### 6.1.4 不等式约束的外点罚函数

在 $g_j(x)\le0$ 的约定下，可用

```math
P_{EI}(x,\sigma)
=f(x)+\frac{\sigma}{2}\sum_i c_i(x)^2
+\frac{\sigma}{2}\sum_j\bigl[\max\{g_j(x),0\}\bigr]^2.
```

当 $g_j(x)\le0$ 时，不等式违反项为零；当 $g_j(x)>0$ 时，惩罚项推动点回到可行侧。若采用 $c_j(x)\ge0$ 的等价约定，则相应的违反项写成 $\max\{-c_j(x),0\}$。不等式外点罚函数的算法和数值困难与等式情形类似，仍然存在罚参数增大导致的病态问题。

### 6.2 障碍函数方法

#### 6.2.1 倒数障碍与对数障碍

障碍函数从可行域内部逼近边界，适用于不等式约束。对 $g_j(x)<0$，常用对数障碍函数

```math
\Phi(x,\mu)
=f(x)-\mu\sum_{j=1}^p\log\bigl(-g_j(x)\bigr),
\qquad \mu>0.
```

也可以使用倒数障碍函数

```math
\Phi_R(x,\mu)
=f(x)+\mu\sum_{j=1}^p\frac{1}{-g_j(x)}.
```

对数障碍通常具有更好的尺度和理论性质。障碍项在边界附近趋于无穷，因此每个固定 $\mu>0$ 的子问题极小点都严格可行；令 $\mu\to0$，才能允许极小点逐步接近原问题可能位于边界上的解。

#### 6.2.2 障碍法的算法流程与中心路径

障碍法先选择严格可行初值 $x_0$ 和 $\mu_0>0$，近似求解 $\Phi(x,\mu_k)$，然后减小障碍参数并 warm start 下一子问题：

1. 找到满足 $g(x_0)<0$ 的严格可行点；
2. 近似最小化 $\Phi(x,\mu_k)$；
3. 若原始可行性、驻点残差和目标精度满足要求，则终止；
4. 否则令 $\mu_{k+1}<\mu_k$，并以当前解作为下一次内层迭代的初值。

因此，障碍法产生的是从可行域内部逼近原问题解的解序列，而不是从可行域外逐步压回去的序列。

#### 6.2.3 障碍法的性质与数值困难

障碍法有三个基本特点：

1. 固定 $\mu>0$ 时，极小点严格满足所有不等式约束；
2. 当极小点接近边界时，障碍项迅速增大，阻止迭代点越出可行域；
3. 为了逼近边界最优解，必须令 $\mu\to0$。

在适当有界性和内层精度假设下，障碍子问题的解沿中心路径趋向原问题的最优解。以对数障碍为例，其梯度和 Hessian 为

```math
\nabla\Phi(x,\mu)
=\nabla f(x)
-\mu\sum_j\frac{\nabla g_j(x)}{g_j(x)},
```

```math
\nabla^2\Phi(x,\mu)
=\nabla^2f(x)
-\mu\sum_j\frac{\nabla^2g_j(x)}{g_j(x)}
+\mu\sum_j\frac{\nabla g_j(x)\nabla g_j(x)^\top}{g_j(x)^2}.
```

若定义近似乘子 $\lambda_j= -\mu/g_j(x)$，则障碍 Hessian 可以解释为 Lagrangian Hessian 加上与 $\mu/g_j(x)^2$ 相关的法向正则项。当 $\mu\to0$ 且某些约束接近活跃边界时，线性系统同样会出现尺度差异和病态，因此障碍法也需要缩放、正则化和稳定的 KKT 分解。

### 6.3 非光滑精确罚函数

二次罚函数通常需要 $\sigma\to\infty$ 才能逼近原问题。精确罚函数则希望存在有限参数，使一次无约束最小化就能恢复原问题的局部解。典型的 L1 罚函数为

```math
\Phi_1(x,\rho)
=f(x)+\rho\left(\sum_i|c_i(x)|
+\sum_j\max\{g_j(x),0\}\right).
```

在满足适当约束资格条件且 $\rho$ 大于某个最优乘子范数的条件下，原问题的局部解也是 $\Phi_1$ 的局部极小点。因此，L1 罚函数具有有限参数精确性。其代价是非光滑性：原问题最优点通常位于 $c_i(x)=0$ 或 $g_j(x)=0$ 的折点；在存在非零约束乘子的通常情形下，若罚函数仍连续可微，其梯度条件一般无法同时表达法向平衡。

精确罚函数可以推广到其他范数，例如 L∞ 罚函数；但“有限参数精确”与“光滑”通常不能同时保持。实际算法可使用次梯度、近端方法、平滑化或序列二次规划处理其非光滑结构。

### 6.4 增广拉格朗日方法

#### 6.4.1 等式约束增广拉格朗日

对等式约束 $c(x)=0$，增广拉格朗日函数为

```math
\mathcal L_A(x,\lambda,\sigma)
=f(x)+\lambda^\top c(x)+\frac{\sigma}{2}\|c(x)\|^2.
```

与二次罚函数相比，它多了拉格朗日乘子项。若 $(x^*,\lambda^*)$ 满足原问题 KKT 条件，则

```math
\nabla_x\mathcal L_A(x^*,\lambda^*,\sigma)
=\nabla_x\mathcal L(x^*,\lambda^*)=0,
```

因为 $c(x^*)=0$。这说明原问题的 KKT 解可以直接成为增广拉格朗日子问题的一阶驻点，而二次罚函数在有限 $\sigma$ 下通常做不到。

更具体地，二次罚函数的梯度为

```math
\nabla P_E(x,\sigma)
=\nabla f(x)+\sigma J_c(x)^\top c(x).
```

在原问题解 $x^*$ 处，由于 $c(x^*)=0$，有

```math
\nabla P_E(x^*,\sigma)=\nabla f(x^*).
```

对于存在有效等式约束的最优点，通常 $\nabla f(x^*)\ne0$，所以 $x^*$ 一般不是有限 $\sigma$ 二次罚函数的无约束子问题驻点。增广拉格朗日则利用乘子项抵消这个法向梯度：

```math
\nabla_x\mathcal L_A(x^*,\lambda^*,\sigma)
=\nabla f(x^*)+J_c(x^*)^\top\lambda^*=0.
```

#### 6.4.2 二阶曲率与正定性

增广拉格朗日 Hessian 为

```math
\nabla^2_{xx}\mathcal L_A
=\nabla^2_{xx}\mathcal L(x,\lambda)
+\sigma J_c(x)^\top J_c(x).
```

若 Lagrangian Hessian 在 $J_c(x^*)$ 的零空间上正定，则存在 $\sigma_1$，使得当 $\sigma>\sigma_1$ 时，$\nabla^2_{xx}\mathcal L_A(x^*,\lambda^*,\sigma)\succ0$。其几何含义是：Lagrangian Hessian 在切空间提供正曲率，罚项在约束法向方向提供额外正曲率，二者合起来覆盖整个空间。

需要注意，这并不意味着任意点处的增广 Hessian 都正定；该结论是局部的，并且依赖 KKT 点、约束资格条件和二阶充分条件。增广参数过大也会导致法向方向曲率过大，因而产生条件数问题，所以增广拉格朗日法不是简单地不断增大 $\sigma$。

上述正定性可以用一个基本矩阵事实说明：若 $P$、$Q$ 为对称矩阵，$Q\succeq0$，且 $P$ 在 $Q$ 的零空间上正定，即

```math
z\ne0,\quad Qz=0
\quad\Longrightarrow\quad
z^\top Pz>0,
```

则存在 $\sigma_1$，使得当 $\sigma>\sigma_1$ 时

```math
P+\sigma Q\succ0.
```

在增广拉格朗日中取 $P=\nabla^2_{xx}\mathcal L(x^*,\lambda^*)$、$Q=J_c(x^*)^\top J_c(x^*)$。二阶充分条件保证 $P$ 在 $Q$ 的零空间，即约束切空间上正定，于是得到上述结论。

但 $Q$ 通常是秩亏的。若 $z\in\operatorname{Null}(Q)$，则

```math
z^\top(P+\sigma Q)z=z^\top Pz.
```

因此，增大 $\sigma$ 只能改善约束法向方向的曲率，不能修复切空间上的负曲率。即使 $P+\sigma Q$ 在最优点附近最终正定，也不能推出它在远离解的任意迭代点都正定；这也是实际算法需要线搜索、信赖域或修正 Hessian 的原因。

上述讨论可以概括为以下三点数值认识：

1. **增广参数不能无限制增大。** 增大 $\sigma$ 的作用是强化约束法向方向的曲率，但同时会使法向曲率远大于切向曲率，从而恶化增广 Hessian 的条件数。因此，增广拉格朗日法不是单纯地不断增大罚参数，而是通过乘子更新逐步逼近最优乘子，在可接受的 $\sigma$ 下同时改善驻点性和可行性。

2. **正定性结论只在最优点附近成立。** 在 KKT 点 $x^*$ 处，二阶充分条件可以保证 Lagrangian Hessian 在约束切空间上正定，从而通过足够大的 $\sigma$ 得到增广 Hessian 正定。但在任意非最优迭代点，矩阵 $P=\nabla^2_{xx}\mathcal L(x,\lambda)$ 未必在 $Q$ 的零空间上正定，因此不能据此断言 $P+\sigma Q$ 必然正定。

3. **秩亏罚项无法消除切空间负曲率。** 即使写成 $H+\sigma A$，当 $A\succeq0$ 只有秩 $m<n$ 时，它只在 $m$ 个方向上提供正曲率。在任意非零零空间方向 $z$ 上，$Az=0$，于是

   ```math
   z^\top(H+\sigma A)z
   =z^\top Hz+\sigma z^\top Az
   =z^\top Hz.
   ```

   因而，如果 $H$ 在该零空间上存在负曲率，无论 $\sigma$ 多大，$H+\sigma A$ 都可能保持不正定。这正是“增大正则化参数可以修复所有不定性”在秩亏约束罚项下不成立的原因。

在 LICQ、二阶充分条件以及适当的局部正则性假设下，可以进一步得到：存在 $\bar\sigma>0$，使得对任意 $\sigma>\bar\sigma$，原问题的 KKT 对 $(x^*,\lambda^*)$ 是增广拉格朗日函数 $\mathcal L_A(\cdot,\lambda^*,\sigma)$ 的严格局部极小点。该结论说明增广拉格朗日的局部理论，但不意味着远离解时的每个增广子问题都凸，也不意味着数值实现应无条件地选择极大的 $\sigma$。

> 教材索引：增广拉格朗日 Hessian 的局部正定性和严格局部极小点结论可参阅 Nocedal--Wright, §17.3、Theorem 17.5；高立《数值最优化方法》第 7.3 节及定理 7.4。矩阵事实“$P$ 在 $Q$ 的零空间上正定则 $P+\sigma Q$ 充分大时正定”可结合 Bertsekas, *Nonlinear Programming*, §3.2.1 阅读。

#### 6.4.3 乘子更新、算法流程与收敛理解

一个基本等式约束 ALM 流程为：

1. 选择 $x_0$、乘子 $\lambda_0$ 和增广参数 $\sigma_0>0$；
2. 近似最小化 $\mathcal L_A(x,\lambda_k,\sigma_k)$，得到 $x_{k+1}$；
3. 更新乘子和增广参数：

   ```math
   \lambda_{k+1}=\lambda_k+\sigma_k c(x_{k+1});
   ```

4. 根据约束违反、内层精度和线性系统条件数决定是否增大 $\sigma_k$，再进入下一轮。

与二次外点罚函数相比，ALM 的算法框架基本相同，但有两个本质区别：

1. 子问题目标由二次罚函数替换为含乘子的增广拉格朗日函数；
2. 每轮求解后更新拉格朗日乘子，使有效罚力逐渐逼近真实乘子。

因此，ALM 通常不需要像二次罚函数那样把 $\sigma_k$ 无限增大，也能获得高精度可行性。若每个内层问题只近似求解，则需要设置随外层迭代收紧的内层容差，并同时监控驻点残差、约束违反和乘子变化。在标准局部正则性和适当内层精度下，乘子和原始变量可获得局部收敛；简单的乘子更新通常给出线性收敛，而更精确的内层求解和二阶信息可以改善局部速度。

在解附近，由

```math
\nabla_x\mathcal L_A(x_k,\lambda_k,\sigma_k)
=\nabla f(x_k)+J_c(x_k)^\top
\bigl(\lambda_k+\sigma_kc(x_k)\bigr),
```

可见有效乘子近似为

```math
\lambda_k+\sigma_kc(x_k)\approx\lambda^*.
```

因此

```math
c(x_k)\approx\frac{\lambda^*-\lambda_k}{\sigma_k}.
```

当乘子估计已经接近 $\lambda^*$ 时，约束违反可以比单纯二次罚函数的 $1/\sigma_k$ 规律更快地减小。ALM 的核心不是无限增大罚参数，而是同时学习乘子，使罚参数保持在可接受范围内。

若内层子问题精确求解，增广拉格朗日一阶条件给出

```math
\nabla f(x_{k+1})+J_c(x_{k+1})^\top
\bigl(\lambda_k+\sigma_kc(x_{k+1})\bigr)=0.
```

因此自然的乘子更新为

```math
\lambda_{k+1}=\lambda_k+\sigma_kc(x_{k+1}).
```

在局部线性模型下，若 $\lambda_k$ 已接近 $\lambda^*$，则

```math
c(x_{k+1})\approx
\frac{\lambda^*-\lambda_k}{\sigma_k}.
```

这说明乘子更新并非附加的经验技巧，而是将罚项产生的法向梯度解释为对最优乘子的估计。

### 6.5 一般约束、指示函数与 ADMM

一般不等式约束可以通过障碍、外点罚函数或增广拉格朗日处理；也可以将可行集写成指示函数

```math
\delta_{\mathcal C}(x)=
\begin{cases}
0,&x\in\mathcal C,\\
+\infty,&x\notin\mathcal C,
\end{cases}
```

从而把约束问题写成复合目标或变量分裂问题。对 $\min f(x)+g(z)$ 且 $Ax+Bz=c$，ADMM 交替最小化增广拉格朗日的两个块并更新对偶变量。ADMM 可以看作变量分裂后的增广拉格朗日方法，但不等同于一般 ALM：它用交替子问题代替联合内层最小化，收敛性和效率依赖子问题结构、凸性及分块方式。

> 教材索引：二次外点罚函数见 Nocedal--Wright, §17.1 及高立第 7.1 节；障碍法见 Bertsekas, *Nonlinear Programming*, §4.1；L1 精确罚函数见 Nocedal--Wright, §17.2；增广拉格朗日局部理论见 Nocedal--Wright, §17.3、高立第 7.3–7.4 节及 Bertsekas, §4.2。


### 6.6 三类约束处理机制的比较

二次罚函数从不可行点出发但罚参数增大后病态；障碍函数保持严格可行但需要内部点；增广拉格朗日通过乘子更新减少无限增大罚参数的需要。算法选择取决于可行初值、约束结构、内层问题的求解成本和所需可行性精度。


## 7. 序列二次规划

SQP 的基本思想是在当前点用二阶目标模型和一阶约束模型构造 QP 子问题。它直接将原非线性问题的 KKT 方程做 Newton 线性化，因此是连接约束最优性理论与数值算法的核心方法。

### 7.1 SQP 子问题与 Newton--KKT 等价性

#### 7.1.1 二次子问题

在 $x_k$ 处记 $\nabla f_k=\nabla f(x_k)$，并用 $B_k$ 近似 Lagrangian Hessian，得到

```math
\min_d\ \nabla f_k^\top d+\frac12d^\top B_kd,
\qquad c_k+J_{c,k}d=0,
\qquad g_k+J_{g,k}d\le0.
```

#### 7.1.2 与 Newton--KKT 步的等价性

对等式约束问题的 KKT 方程做 Newton 线性化，得到

```math
\begin{bmatrix}
\nabla_{xx}^2\mathcal L_k&J_{c,k}^\top\\
J_{c,k}&0
\end{bmatrix}
\begin{bmatrix}d_k\\\Delta\lambda_k\end{bmatrix}
=-
\begin{bmatrix}
\nabla_x\mathcal L(x_k,\lambda_k)\\c(x_k)
\end{bmatrix}.
```

而 QP 子问题的 KKT 条件为

```math
\begin{bmatrix}
B_k&J_{c,k}^\top\\
J_{c,k}&0
\end{bmatrix}
\begin{bmatrix}d_k\\\hat\lambda_k\end{bmatrix}
=-
\begin{bmatrix}
\nabla f(x_k)\\c(x_k)
\end{bmatrix}.
```

当 $B_k=\nabla_{xx}^2\mathcal L_k$ 且乘子处理一致时，两者给出相同的步长方程。因此，SQP 的局部快速收敛本质上来自对约束优化 KKT 系统使用 Newton 型方法。

#### 7.1.3 约束相容性与 Hessian 近似

原非线性约束可行，并不保证线性化约束

```math
c_k+J_{c,k}d=0,
\qquad
g_k+J_{g,k}d\le0
```

相容。若 QP 子问题不可行，SQP 迭代将无法继续。常用处理方式包括加入弹性变量、求解 feasibility-restoration 子问题，或先最小化约束违反量再恢复目标函数优化。

精确 SQP 使用完整 Lagrangian Hessian；BFGS/DFP 近似更容易维护正定性；Gauss--Newton 适合残差平方和结构，但在残差较大时可能遗漏重要二阶曲率。



### 7.2 Hessian 近似与可行性恢复

精确 SQP 使用 $\nabla_{xx}^2\mathcal L$，但该矩阵可能不定。BFGS-SQP 通过阻尼更新维持约化空间曲率；最小二乘问题中也常用 Gauss--Newton 近似。若 QP 的线性化约束不可行，可加入松弛变量并惩罚其范数，或进入独立的 feasibility-restoration 阶段。仅减少目标函数而不减少不可行性，不能构成可靠的 SQP 全局化。

### 7.3 Merit function、filter 与信赖域

#### 7.3.1 价值函数

价值函数是用来衡量候选步是否取得足够进展的标量函数。在线搜索型无约束方法中，价值函数通常就是目标函数 $f(x)$；只要搜索方向满足 $\nabla f(x_k)^\top d_k<0$，就可以通过选择足够小的步长保证目标函数下降。约束优化中若仍只比较 $f(x_k+\alpha d_k)$，就可能接受目标下降但约束违反显著增加的步长，因而不能可靠地实现全局化。

因此，约束 SQP 的价值函数必须同时反映目标值和约束违反量，使 QP 子问题产生的方向 $d_k$ 在适当罚参数下成为价值函数的下降方向。常用 L1 价值函数为

```math
\Phi_1(x;\rho)
=f(x)+\rho\left(\|c(x)\|_1
+\|\max\{g(x),0\}\|_1\right).
```

也可以使用 L2 型价值函数

```math
\Phi_2(x;\rho)
=f(x)+\frac\rho2\left(\|c(x)\|_2^2
+\|\max\{g(x),0\}\|_2^2\right).
```

在线搜索中，步长通常要求满足

```math
\Phi(x_k+\alpha d_k)
\le\Phi(x_k)+\eta\alpha\Phi'(x_k;d_k),
\qquad 0<\eta<1.
```

其中 $\Phi'(x_k;d_k)$ 是价值函数沿 $d_k$ 的方向导数。步长 $\alpha>0$ 需要足够小，使真实价值函数下降至少达到方向导数预测下降量的一定比例；这正是约束线搜索中“充分下降”条件的含义。

#### 7.3.2 Maratos 效应与二阶修正

Maratos 效应是指：SQP 主 QP 的线性化约束已经满足，但原约束的二阶非线性使全步 $\alpha=1$ 被价值函数线搜索拒绝。以 $h(x)=0$ 为例，若

```math
h(x_k)+\nabla h(x_k)^\top d_k=0,
```

则

```math
h(x_k+d_k)
\approx\frac12d_k^\top\nabla^2h(x_k)d_k,
```

该误差为 $O(\|d_k\|^2)$，会导致局部快速收敛退化。为此引入 $\delta_k$，使

```math
h(x_k+d_k+\delta_k)\approx0.
```

忽略高阶项可得到辅助问题

```math
\min_{\delta}\ \frac12\|\delta\|_2^2,
\qquad
\text{s.t.}\quad
\nabla h(x_k)^\top\delta+h(x_k+d_k)=0.
```

若全步不满足价值函数下降条件，则先计算二阶修正；若修正步仍不可接受，再回溯缩短步长。

#### 7.3.3 信赖域中的价值函数模型

信赖域 SQP 不直接用实际价值函数下降来决定步长，而是在当前点构造价值函数的局部二次模型。例如，对目标函数部分可写为

```math
q_k(p)=f(x_k)+\nabla f(x_k)^\top p
+\frac12p^\top B_kp.
```

若价值函数包含约束违反项，则还需要在模型中加入约束线性化后的违反量，得到 $\Phi_k^{\mathrm{model}}(p)$。信赖域子问题通过限制 $\|p\|\le\Delta_k$，寻找具有足够模型下降的步：

```math
\min_p\ \Phi_k^{\mathrm{model}}(p),
\qquad \|p\|\le\Delta_k.
```

模型下降通常写成

```math
\operatorname{pred}_k
=\Phi_k^{\mathrm{model}}(0)
-\Phi_k^{\mathrm{model}}(p_k),
```

实际下降为

```math
\operatorname{ared}_k
=\Phi(x_k)-\Phi(x_k+p_k).
```

二者之比

```math
\rho_k
=\frac{\operatorname{ared}_k}{\operatorname{pred}_k}
```

用于判断模型是否可信：$\rho_k$ 较大时接受步并可扩大信赖域；$\rho_k$ 较小时拒绝或缩短步并减小信赖域。由此，线搜索通过真实价值函数的方向导数控制步长，信赖域则通过价值函数模型的预测下降和实际下降之比控制步长。

#### 7.3.4 Filter 与非单调线搜索

Filter 不把目标和约束违反合并为单一标量，而是维护“目标值—约束违反量”对，候选点只要在其中一个指标上有足够改善即可接受。非单调线搜索允许目标函数在短期内上升，以换取更大的步长；历史窗口准则可写成

```math
f(x_k+\alpha d_k)
\le\max_{0\le j\le M}f(x_{k-j})
+\gamma\alpha\nabla f(x_k)^\top d_k.
```

信赖域 SQP 进一步限制 $\|d\|\le\Delta_k$。即使线性化约束相容，加入信赖域后也可能无可行步，因此不能简单无限增大 $\Delta_k$；应使用松弛、惩罚或 filter，在信赖域内逐步改善可行性。


> 教材索引：SQP 与 Newton--KKT 的等价性见高立《数值最优化方法》pp. 241–243 及 Nocedal--Wright, §18.1；约束相容性、Hessian 近似和 merit function 见高立第 9.2–9.4 节；Maratos 效应、二阶修正、filter 和 SQP 收敛性见 Nocedal--Wright, §§15.5–15.6、18.3、18.5 和 18.7。

### 7.4 SQP 的完整迭代流程与收敛

标准 SQP 迭代为：

1. 计算目标、约束及其导数，更新 Lagrangian Hessian 近似；
2. 构造并求解 QP 子问题；
3. 检查 QP 相容性，必要时进入可行性恢复阶段；
4. 使用价值函数、filter 或信赖域接受主步；
5. 若全步因 Maratos 效应被拒绝，计算二阶修正；
6. 更新乘子、曲率近似和全局化参数，并检查 KKT 残差。

局部快速收敛需要约束资格条件、二阶充分条件、可靠 Hessian 近似、足够精确的 QP 解以及最终接受全步。远离解时必须依赖价值函数、filter、信赖域或可行性恢复。


## 8. 原始—对偶内点法

第 5.2 节讨论的是凸 QP 的原始—对偶内点法；本章将这一思想推广到一般非线性约束问题，并重点说明非线性 Hessian、原始残差、对偶残差以及大规模稀疏实现所带来的额外问题。

原始—对偶内点法通过同时求解原始可行性、对偶可行性和扰动互补性来逼近 KKT 系统。它不显式枚举活跃集，而是让松弛变量和对偶变量沿中心路径逐渐分化，因此特别适合大规模稀疏约束问题。本章采用内点法文献中常见的不等式约定 $c_I(x)\ge0$；它是全文默认的 $g(x)\le0$ 约定的局部等价写法，只需令 $c_I=-g$，并相应改变 Lagrangian 中的符号。

原始—对偶内点法对原始残差、对偶残差与互补残差同时施加 Newton 法。每次迭代计算方向、用 fraction-to-boundary 规则保持松弛变量与乘子为正，并更新障碍参数。主要成本是反复求解大型对称不定线性系统；可以采用增广系统、正规方程、稀疏 $LDL^\top$、零空间法或预条件 Krylov 方法。$\mu\to0$ 时系统可能病态，但其病态具有互补结构；缩放、正则化和稳定消元不可省略。



### 8.1 松弛变量与原始—对偶 KKT 条件

考虑

```math
\min_x f(x),
\qquad c_E(x)=0,
\qquad c_I(x)\ge0.
```

引入松弛变量 $s=c_I(x)$，并令 $z\ge0$ 为不等式乘子。记 $S=\operatorname{diag}(s)$、$Z=\operatorname{diag}(z)$。

松弛变量形式的问题为

```math
\min_{x,s} f(x),
\qquad c_E(x)=0,
\qquad c_I(x)-s=0,
\qquad s\ge0.
```

对 $s>0$ 引入对数障碍，得到障碍子问题

```math
\min_{x,s}\ f(x)-\mu\sum_{i=1}^m\log s_i,
\qquad c_E(x)=0,
\qquad c_I(x)-s=0.
```

其 KKT 条件为

```math
\nabla f(x)-J_E(x)^\top y-J_I(x)^\top z=0,
\qquad
-\mu S^{-1}\mathbf{1}+z=0,
```

```math
c_E(x)=0,
\qquad
c_I(x)-s=0.
```

```math
SZ\mathbf{1}=\mu\mathbf{1},
\qquad \mu>0,
\qquad s>0,\ z>0.
```

当 $\mu\to0$ 时，中心路径极限满足原问题 KKT 条件。

### 8.2 Newton 系统与步长

对上述 KKT 方程应用 Newton 法，令步为
$(p_x,p_{y_E},p_s,p_z)$，得到原始—对偶系统

```math
\begin{bmatrix}
\nabla_{xx}^2\mathcal L&0&-J_E^\top&-J_I^\top\\
0&Z&0&S\\
J_E&0&0&0\\
J_I&-I&0&0
\end{bmatrix}
\begin{bmatrix}p_x\\p_{y_E}\\p_s\\p_z\end{bmatrix}
=-
\begin{bmatrix}
\nabla f(x)-J_E^\top y-J_I^\top z\\
Sz\mathbf{1}-\mu\mathbf{1}\\
c_E(x)\\
c_I(x)-s
\end{bmatrix}.
```

其中

```math
\mathcal L(x,s,y,z)
=f(x)-y^\top c_E(x)-z^\top\bigl(c_I(x)-s\bigr).
```

该 Newton 系统先消去 $p_s$，再消去 $p_z$，可以得到对称增广系统和浓缩系统；不同消元形式求的是同一个 Newton 方向，但条件数、填充和适合的分解方法不同。

### 8.3 残差、边界分数与算法流程

原始—对偶方法分别控制原始残差、对偶残差和互补度。fraction-to-boundary 规则在保持 $s$ 和 $z$ 严格为正的前提下选择最大步长。给定边界参数 $\tau\in(0,1)$，可采用如下分量步长：

```math
s+\alpha p_s>0,
\qquad z+\alpha p_z>0.
```

更具体地，取

```math
\alpha_s^{\max}
=\max\{\alpha\in(0,1]:s+\alpha p_s\ge(1-\tau)s\},
```

```math
\alpha_z^{\max}
=\max\{\alpha\in(0,1]:z+\alpha p_z\ge(1-\tau)z\}.
```

令 $\alpha_s=\tau\alpha_s^{\max}$、$\alpha_z=\tau\alpha_z^{\max}$，并分别用它们更新 $s,z$；原始变量 $x$ 和等式乘子通常使用相应的线搜索步长。在适当的约束资格条件、严格互补和二阶充分条件下，原始—对偶矩阵在解附近保持非奇异；严格互补意味着每个约束对应的 $s_i$ 或 $z_i$ 至少有一个为正。

定义误差函数

```math
E(x,s,y,z;\mu)
=\max\left\{
\|\nabla f(x)-J_E(x)^\top y-J_I(x)^\top z\|,
\|SZ\mathbf{1}-\mu\mathbf{1}\|,
\|c_E(x)\|,
\|c_I(x)-s\|
\right\}.
```

基本流程可以明确写成：

1. 初始化 $x_0$、$s_0>0$、$z_0>0$ 和障碍参数 $\mu_0$；
2. 计算原始残差、对偶残差和互补残差，构造 Newton 系统；
3. 求解结构化线性系统，必要时采用正则化、缩放或迭代细化；
4. 用 fraction-to-boundary 规则确定步长，并更新 $(x,s,y,z)$；
5. 若残差达到当前中心路径的内层容差，则减小 $\mu_k$，否则继续求解当前系统；
6. 若线性系统分解失败、步长过小或残差不降，则进行正则化、重新缩放或切换全局化策略。

若移除内层循环的严格收敛条件，并在每一步动态更新 $\mu_k$，就得到常见的 infeasible-start 原始—对偶算法。预测—校正法则先计算 $\sigma=0$ 的仿射方向，再根据预测互补度选择中心化参数并求解校正方向。

### 8.4 数值稳定性与大规模实现

障碍参数趋零时，松弛变量和乘子会出现不同尺度，导致线性系统病态。对角正则化、矩阵缩放、稳定 $LDL^\top$ 主元策略和迭代细化用于抑制数值误差。严格互补能改善局部理论，但不意味着有限精度下的系统天然良态。

原始—对偶矩阵的病态通常具有结构性：某些 $s_i$ 或 $z_i$ 趋于零，使互补块尺度极不平衡；当 Hessian 或约束 Jacobian 接近秩亏时，病态会进一步加剧。稳定的稀疏直接分解通常是最可靠的步长计算方法；迭代法则需要能聚集互补对角块特征值的预条件器。内点法的非零结构通常在迭代间保持不变，便于复用稀疏符号分解，但热启动和有限步活跃集识别通常不如主动集法。

为获得对称系统，常令

```math
\Sigma=S^{-1}Z.
```

消去松弛变量步后，可以得到形如

```math
\begin{bmatrix}
\nabla_{xx}^2\mathcal L+A_I^\top\Sigma A_I&A_E^\top\\
A_E&0
\end{bmatrix}
\begin{bmatrix}p_x\\-p_{y_E}\end{bmatrix}
=-\begin{bmatrix}r_d'\\r_E\end{bmatrix}.
```

当不等式约束数量较多时，还可以进一步消去等式乘子，得到浓缩系数矩阵

```math
\nabla_{xx}^2\mathcal L+A_I^\top\Sigma A_I.
```

因此，内点法中主要计算成本不是障碍项本身，而是反复求解这些结构化原始—对偶线性系统。直接分解通常对这种特殊病态最稳健；迭代法则需要能处理 $\Sigma$ 极端尺度的预条件器；零空间法可以在正定约化空间中使用 CG，但需要额外构造稳定的零空间表示。

内点法还需要处理非凸 Hessian、奇异 Jacobian、障碍参数更新和步长接受问题。常见扩展包括信赖域内点法、价值函数或 filter 全局化、拟 Newton 近似以及可行内点法。引入松弛变量后的问题本质上是等式约束问题；因此，用 Newton 法求解其 KKT 条件，最终仍可归结为反复求解具有 QP 结构的线性化子问题。


> 教材索引：原始—对偶内点法的系统消元和预测—校正方法见 Nocedal--Wright, Chapter 19，尤其 §§19.3、19.5–19.8。

## 9. 非光滑优化与近端方法

本章讨论凸但不可微问题及其复合结构，并把它与前文的光滑约束优化联系起来。第一，近端项的二次正则化可以看作信赖域或正则化思想在非光滑问题中的延伸：它用一个稳定的局部子问题替代不可微的直接步长选择。第二，集合的指示函数可以把约束优化写成复合无约束形式，从而用次微分和近端映射统一描述投影与一般约束。第三，ADMM 是变量分裂后的增广拉格朗日方法，与第 6 章的增广拉格朗日框架直接相连，只是将联合内层最小化改为交替子问题。

不可微并不意味着无法建立最优性条件：梯度被次微分替代，显式梯度步被近端映射替代。近端项还提供了稳定性和可分结构，是 L1 正则、约束投影、稀疏优化和 ADMM 的共同基础。

### 9.1 次梯度与不可微最优性

#### 9.1.1 次梯度定义

对凸函数 $h$，向量 $v$ 称为 $x$ 处的次梯度，如果对任意 $y$ 有

```math
h(y)\ge h(x)+v^\top(y-x).
```

所有次梯度构成次微分集合

```math
\partial h(x)=\{v:h(y)\ge h(x)+v^\top(y-x),\ \forall y\}.
```

若 $h$ 在 $x$ 的定义域内可微，则 $\partial h(x)=\{\nabla h(x)\}$。因此，次梯度是梯度概念对凸不可微函数的自然推广。

#### 9.1.2 凸问题的一阶充要条件

对于凸问题

```math
\min_x h(x),
```

点 $x^*$ 为全局最优解，当且仅当

```math
0\in\partial h(x^*).
```

必要性来自最优点的支撑超平面条件，充分性则直接由次梯度不等式得到：若 $0\in\partial h(x^*)$，则对任意 $x$ 有 $h(x)\ge h(x^*)$。

#### 9.1.3 复合问题的一阶条件

对

```math
\min_x F(x)=f(x)+h(x),
```

其中 $f$ 可微凸、$h$ 凸但可以不可微，最优性条件为

```math
0\in\nabla f(x^*)+\partial h(x^*).
```

这条包含关系统一了 L1 正则、集合约束和一般复合优化问题；当 $h$ 是集合 $\mathcal C$ 的指标函数时，$\partial h$ 退化为法锥，条件就变成约束最优性条件。

### 9.2 近端映射与近端梯度

近端映射定义为

```math
\operatorname{prox}_{th}(x)
=\arg\min_u\left\{h(u)+\frac1{2t}\|u-x\|^2\right\}.
```

其子问题为强凸问题，最优性条件为

```math
0\in\partial h(u)+\frac1t(u-x),
```

等价于

```math
\frac{x-u}{t}\in\partial h(u).
```

因此近端点法

```math
x_{k+1}=\operatorname{prox}_{t_kh}(x_k)
```

可以看作隐式次梯度迭代。与显式次梯度法相比，近端子问题中的二次项能够稳定大步长并吸收非光滑结构。

对复合问题 $F=f+h$，其中 $f$ 光滑，近端梯度法先对 $f$ 做显式梯度步，再对 $h$ 做近端步：

```math
x_{k+1}=\operatorname{prox}_{t_kh}\bigl(x_k-t_k\nabla f(x_k)\bigr).
```

若 $t_k\le1/L$，其中 $L$ 是 $\nabla f$ 的 Lipschitz 常数，则近端梯度具有标准下降性质。回溯线搜索可从较大初始步长开始，并通过

```math
F(x_{k+1})
\le F(x_k)+\nabla f(x_k)^\top(x_{k+1}-x_k)
+\frac{1}{2t_k}\|x_{k+1}-x_k\|^2
```

检查模型充分下降。加速近端梯度通过外推点改善凸问题的函数值收敛阶，但外推会造成非单调性，实际常配合回溯、单调化或重启。

当 $h=0$ 时退化为梯度法；当 $h$ 是集合指标函数时退化为投影梯度法；当 $h$ 是 L1 范数时，近端映射为软阈值算子。非凸情形不能直接使用凸问题的全局结论，需要额外的下界、半连续性和 KL 性质等假设。



### 9.3 次梯度、近似点与近端算法流程

#### 9.3.1 次梯度算法

次梯度法使用

```math
x_{k+1}=x_k-\alpha_kg_k,
\qquad g_k\in\partial h(x_k).
```

由于次梯度不唯一，显式迭代方向也不唯一。在有界次梯度等标准假设下，采用适当的递减步长，例如

```math
\alpha_k\to0,
\qquad
\sum_{k=0}^{\infty}\alpha_k=\infty,
\qquad
\sum_{k=0}^{\infty}\alpha_k^2<\infty,
```

可以得到相应的凸问题收敛结论；固定步长通常只能把迭代点带到最优值附近的邻域。

#### 9.3.2 近似点法与近端梯度展开

近端点法是次梯度法的隐式版本：

```math
x_{k+1}=\operatorname{prox}_{t_kh}(x_k)
\quad\Longleftrightarrow\quad
x_{k+1}\in x_k-t_k\partial h(x_{k+1}).
```

对复合问题，近端梯度更新等价于求解

```math
x_{k+1}
=\arg\min_u\left\{
h(u)+\nabla f(x_k)^\top(u-x_k)
+\frac1{2t_k}\|u-x_k\|^2
\right\}.
```

因此，它是在当前点线性化光滑项、保留非光滑项，并加入二次模型后求得下一步。其最优性条件还可以写成

```math
x_{k+1}
=x_k-t_k\nabla f(x_k)-t_kg_k,
\qquad g_k\in\partial h(x_{k+1}),
```

这说明近端梯度对光滑项是显式梯度下降，对非光滑项是隐式次梯度下降。

#### 9.3.3 Nesterov 加速与算法选择

加速近端梯度在外推点 $y_k$ 处计算近端步，例如

```math
x_{k+1}=\operatorname{prox}_{t_kh}
\bigl(y_k-t_k\nabla f(y_k)\bigr),
```

再用动量参数构造 $y_{k+1}$。在光滑凸和复合凸问题中，加速可以改善函数值收敛阶，但外推会破坏单调性，实际常配合回溯、单调化或重启。特殊情形包括：$h=0$ 时退化为梯度法；$h=\delta_{\mathcal C}$ 时退化为投影梯度法；$h=\|x\|_1$ 时近端映射为软阈值。

### 9.4 ADMM 和分裂

对

```math
\min_{x,z}\ f(x)+g(z),
\qquad Ax+Bz=c,
```

增广拉格朗日函数为

```math
\mathcal L_\rho(x,z,y)
=f(x)+g(z)+y^\top(Ax+Bz-c)
+\frac\rho2\|Ax+Bz-c\|^2.
```

ADMM 的基本流程为：

1. 初始化 $x_0$、$z_0$、$y_0$ 和罚参数 $\rho>0$；
2. 固定 $(z^k,y^k)$，求解 $x$ 子问题；
3. 固定 $(x^{k+1},y^k)$，求解 $z$ 子问题；
4. 根据原始残差和对偶残差更新 $y$，必要时调整 $\rho$；
5. 若残差满足容差则终止，否则进入下一轮；若子问题难以求解或残差停滞，则改变分块、预条件或罚参数。

对应的交替更新为

```math
x^{k+1}=\arg\min_x\mathcal L_\rho(x,z^k,y^k),
```

```math
z^{k+1}=\arg\min_z\mathcal L_\rho(x^{k+1},z,y^k),
```

```math
y^{k+1}=y^k+\rho(Ax^{k+1}+Bz^{k+1}-c).
```

其优势来自子问题可分、投影或近端映射易算；若每个子问题本身很难，ADMM 不会自动降低总难度。标准双块凸情形可通过原始残差、对偶残差和增广拉格朗日值分析收敛；多块和非凸 ADMM 需要更强的正则性、强凸性或步长假设。


> 教材索引：次梯度、近端映射和近端梯度可参阅高立《数值最优化方法》第 5 章；凸不可微最优性和近端梯度证明可参阅《最优化建模、算法与理论》pp. 165–166。

### 9.5 近端方法的推导与实现

近端映射满足

```math
u=\operatorname{prox}_{th}(x)
\quad\Longleftrightarrow\quad
\frac{x-u}{t}\in\partial h(u).
```

近端梯度先对光滑项做显式梯度步，再对非光滑项做隐式近端步。若 $h$ 是集合指标函数，近端映射就是投影；若 $h$ 有闭式近端映射，算法每步成本较低，否则仍需求解内层子问题。ADMM 的效率同样取决于分块子问题是否真正容易。



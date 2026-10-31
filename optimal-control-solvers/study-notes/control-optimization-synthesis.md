# 最优控制与数值优化

> 主干文献：Kelly, *An Introduction to Trajectory Optimization*（2017）；Giftthaler 等, *A Family of Iterative Gauss--Newton Shooting Methods*（2018）；Jordana 等, *Stagewise Implementations of SQP for MPC*（2023）与 *Structure-Exploiting SQP for MPC*（2025）；Abhijeet 与 Chakravorty, *A Sequential Quadratic Programming Perspective on Optimal Control*（2025）；Sousa-Pinto 与 Orban, *Primal-Dual iLQR*（2024）。

## 摘要

最优控制要成为可计算算法，通常需回答四个层次的问题：连续轨迹如何转录为有限维非线性规划；非线性规划如何构造局部二次模型；所得结构化 OCP-QP 如何利用时间结构高效求解；外层迭代如何在有限计算预算内落实为可行、稳定的滚动闭环控制。单次打靶、多重打靶和直接配点首先决定变量与约束；拉格朗日 Hessian 的精确值或近似值决定局部模型，精确 Hessian 含目标曲率与约束曲率，Gauss–Newton 型近似则忽略部分或全部曲率项；Riccati 递推、凝聚和稀疏矩阵分解用于利用时间/稀疏结构做消元，内点法、主动集和 ADMM 则在 QP 算法层求解所得子问题；线搜索、过滤器、信赖域、可行性恢复以及实时迭代/提前终止策略，则决定局部模型如何转化为实际算法。

iLQR、DDP、GNMS、FDDP、SQP 和 SQP-RTI 因而不是互不相干的名称，而是上述四层中不同设计选择的组合；例如 iLQR 与 DDP 的差别主要落在 Hessian 近似层次（Gauss–Newton 近似 vs 精确曲率）。在有效集固定或忽略不等式时，OCP-QP 的 KKT 系统具有时间带状结构；动态规划中的 Riccati 递推就是对该系统作阶段消元。近端点与算子分裂进一步说明，正则化二次子问题不仅是数值补丁，也可视为求解一阶最优性条件（KKT 包含）的隐式步。这个等价关系是本文的主线，但它只统一局部模型及其结构化求解，并不消除离散误差、非凸性、约束资格条件、混杂接触和闭环稳定性等问题。当可靠的局部导数模型无法获得时，问题还会转入仅依赖函数值的优化；这一信息边界构成本文与零阶优化、策略学习笔记之间的接口。

## 各章关系

| 部分 | 章 | 内容 | 主要视角 | 承接 |
|------|----|------|----------|------|
| 一、问题建模与转录 | 0 | 连续 OCP、直接转录与算法设计轴 | 建模与离散化 | 说明离散问题从何而来 |
| 一、问题建模与转录 | 1 | 离散 OCP、单次打靶、多重打靶与直接配点 | NLP 建模与转录误差 | 给出后文求解的问题 |
| 二、局部模型与结构化子问题 | 2 | SQP 的局部二次模型 | 数值优化与 KKT 线性化 | 区分精确 Hessian、目标 Hessian 与广义高斯--牛顿 |
| 二、局部模型与结构化子问题 | 3 | iLQR、DDP 与局部动态规划 | 最优控制与 Bellman 递推 | 从线性--二次模型导出反馈步和值函数递推 |
| 二、局部模型与结构化子问题 | 4 | OCP-QP 的结构化求解 | 结构化线性代数 | 说明 Riccati、Schur 补与 KKT 分解的等价 |
| 三、求解器机制与实时实现 | 5 | 从局部模型到可靠求解器 | 全局化、约束处理与近端方法 | 补入正则化、约束、QP 后端与 RTI |
| 三、求解器机制与实时实现 | 6 | 算法设计维度 | 算法组合与比较 | 定位 iLQR、DDP、GNMS、FDDP、SQP、ALTRO 与 SQP-RTI |
| 三、求解器机制与实时实现 | 7 | 代表性实现与数值案例 | 工程实现与实验解释 | 按设计维度分析具体论文和软件 |
| 四、推广、信息边界与统一性 | 8 | 欧氏光滑框架之外 | 流形、接触与可微优化 | 讨论流形、接触、混杂系统与可微最优控制 |
| 四、推广、信息边界与统一性 | 9 | 导数信息、函数值优化与策略学习 | oracle 与策略优化 | 按信息来源连接轨迹优化、黑箱 MPC 与策略优化 |
| 四、推广、信息边界与统一性 | 10 | 统一性的范围与边界 | 等价关系与适用条件 | 说明等价关系的适用条件 |
| 四、推广、信息边界与统一性 | 11 | 开放问题 | 理论与工程问题 | 汇总尚未闭合的理论与工程问题 |
| 四、推广、信息边界与统一性 | 12 | 结论与阅读路线 | 总结与延伸阅读 | 总结主线并给出进阶顺序 |

## 0. 连续 OCP、直接转录与算法设计轴

### 0.1 连续问题不是一个有限维优化问题

一般有限时域最优控制写成

```math
\begin{aligned}
\min_{x(\cdot),u(\cdot),p,t_f}\quad
&\Phi\bigl(x(t_f),p,t_f\bigr)
+\int_{t_0}^{t_f}L\bigl(t,x(t),u(t),p\bigr)\,\mathrm dt,\\
\text{s.t.}\quad
&\dot x(t)=F\bigl(t,x(t),u(t),p\bigr),\\
&b\bigl(x(t_0),x(t_f),p,t_f\bigr)=0,\\
&g\bigl(t,x(t),u(t),p\bigr)\le 0.
\end{aligned}
\tag{0.1}
```

决策变量包含函数 $x(\cdot),u(\cdot)$，还可能包含参数和终止时间。数值算法必须先选择有限维参数化与离散化。这个步骤称为直接转录；它不仅产生近似误差，也决定非线性规划的稀疏结构和条件数。

本文用 $F$ 表示连续动力学右端；后文一般等式约束的 KKT 残差记为 $\mathcal R$，以避免与动力学映射混淆。

### 0.2 三种基本转录

单次打靶只把控制参数 $U=(u_0,\ldots,u_{T-1})$ 作为决策变量，状态由数值积分递推：

```math
x_{k+1}=\Psi_k(x_k,u_k),\qquad J(U)=\Phi(x_T)+\sum_{k=0}^{T-1}\ell_k(x_k,u_k).
\tag{0.2}
```

它自动满足离散动力学，但长时域敏感度由 Jacobian 连乘给出。记 $A_i=\partial\Psi_i/\partial x_i$、$B_i=\partial\Psi_i/\partial u_i$，则对 $j<k$ 有

```math
\frac{\partial x_k}{\partial u_j}
=A_{k-1}\cdots A_{j+1}B_j.
```

在不稳定或强收缩动力学下，这条连乘链可能随步数放大或衰减，导致约化 Hessian 稠密并出现病态；状态约束也因此变成控制序列上的非局部约束。

多重打靶同时优化各段端点状态与控制参数，只在每个区间内部积分，并用间隙约束连接：

```math
x_{k+1}-\Psi_k(x_k,u_k)=0.
\tag{0.3}
```

它增加变量与等式，却保留阶段稀疏性，允许不可行状态轨迹作为初值，并避免把全部敏感度压进一个稠密约化 Hessian。

直接配点不把完整积分器当作黑箱，而将状态和控制表示为分段多项式，在配点处施加微分方程。节点状态、控制以及必要的中间点均可作为 NLP 决策变量，动力学通过缺陷约束进入问题。梯形配点的缺陷为

```math
d_k=x_{k+1}-x_k-
\frac{h_k}{2}\left(F_k+F_{k+1}\right)=0.
\tag{0.4}
```

Hermite--Simpson 方法先构造中点

```math
x_{k+\frac12}
=\frac{x_k+x_{k+1}}{2}
+\frac{h_k}{8}(F_k-F_{k+1}),
\tag{0.5}
```

其中 $F_k=F(t_k,x_k,u_k)$，$F_{k+1}=F(t_{k+1},x_{k+1},u_{k+1})$。中点控制 $u_{k+\frac12}$ 可以作为额外决策变量，也可以由端点控制插值得到；相应地，$F_{k+\frac12}=F(t_{k+\frac12},x_{k+\frac12},u_{k+\frac12})$。

再施加 Simpson 缺陷

```math
d_k=x_{k+1}-x_k-
\frac{h_k}{6}\left(F_k+4F_{k+\frac12}+F_{k+1}\right)=0.
\tag{0.6}
```

配点约束可以理解为隐式 Runge--Kutta 公式。局部配点保持带状稀疏结构；全局伪谱法用高阶全局多项式与 Lobatto、Radau 等节点，在光滑问题上精度高，但导数矩阵更稠密，对不连续控制和局部快速变化更敏感。

### 0.3 离散可行不等于连续可行

求得 NLP 的 KKT 点，只说明离散问题得到满足。对由方法一致插值得到的候选轨迹，连续动力学残差为

```math
\varepsilon(t)=\dot x(t)-F\bigl(t,x(t),u(t)\bigr),
\qquad
\eta_k=\int_{t_k}^{t_{k+1}}\|\varepsilon(\tau)\|\,\mathrm d\tau.
\tag{0.7}
```

$\eta_k$ 与 NLP 的 KKT 残差是不同误差。前者由网格和多项式阶数控制，后者由优化器容差控制。$h$-refinement 增加区间，$p$-refinement 提高多项式阶数，$hp$-refinement 联合调整二者。控制近似不连续时，应在切换附近细化网格或使用多阶段问题，而不能只把 SQP 容差继续收紧。

### 0.4 七个相互独立的设计轴

一个局部最优控制算法可以按下列坐标描述：

1. 转录：单次打靶、多重打靶、配点或伪谱；
2. 局部曲率：精确拉格朗日 Hessian、目标 Hessian、广义高斯--牛顿或拟牛顿；
3. QP 线性代数：Riccati、稀疏分解、凝聚、内点法、主动集或 ADMM；
4. 约束：直接 QP、不等式障碍、增广拉格朗日、投影或可行性恢复；
5. 前向更新：线性试探步、非线性 rollout 或缺陷收缩；
6. 全局化：线搜索、merit function、filter 或信赖域；
7. 实时化：求至收敛、固定迭代预算、RTI 或 advanced-step RTI。

算法名称只是这些选择的常见组合。后文先固定一个离散 OCP，推导其局部二次子问题和结构化解，再回到其余设计轴。

## 1. 离散 OCP、单次打靶、多重打靶与直接配点

### 1.1 离散 OCP 与单次打靶

经过时间离散后，有限时域问题可写成

```math
\begin{aligned}
\min_{x,u}\quad &\Phi(x_T)+\sum_{k=0}^{T-1}\ell_k(x_k,u_k),\\
\text{s.t.}\quad &x_{k+1}=f_k(x_k,u_k),\qquad x_0=\bar x_0,\\
&c_k(x_k,u_k)\ge 0,\qquad c_T(x_T)\ge 0.
\end{aligned}
```

单次打靶把状态递推消去，只保留控制序列 $U$ 作为决策变量。它适合状态维数较大而控制维数较小、或需要保持 rollout 始终动力学可行的场景；相应代价是约化导数在时间上稠密，状态约束需要通过控制序列间接表达。

### 1.2 多重打靶

时域 $k=0,\ldots,T-1$，状态与控制都是决策变量。Jordana 求解

```math
\min_{x,u}\ \sum_{k=0}^{T-1}\ell_k(x_k,u_k)+\ell_T(x_T)
\tag{1}
```

```math
x_{k+1}=f_k(x_k,u_k),\qquad
c_k(x_k,u_k)\ge 0,\qquad
c_T(x_T)\ge 0,\qquad
x_0=\bar x_0.
\tag{2}
```

动力学间隙为

```math
\gamma_{k+1}=f_k(x_k,u_k)-x_{k+1}.
\tag{3}
```

迭代过程中允许 $\gamma\neq 0$，因此初值不必是一条可行轨迹，上一拍的解平移后也可以直接作为热启动。收敛时把 $\|\gamma\|$ 收到停机容差之内。

拉格朗日函数把间隙约束和不等式分别配上乘子 $\lambda_{k+1}$、$\mu_k$：

```math
\mathcal L
=\ell_T-\mu_T^\top c_T
+\sum_{k=0}^{T-1}\Bigl[
\ell_k+\lambda_{k+1}^\top\bigl(f_k(x_k,u_k)-x_{k+1}\bigr)-\mu_k^\top c_k
\Bigr].
\tag{4}
```

$\lambda_{k+1}$ 是间隙约束的乘子。后文 DDP 的协态与这一乘子在最优轨迹上对齐；离开最优轨迹时，二者的取法不同，见 §3。

### 1.3 单次打靶的局部分析

Chakravorty 为陈述下降性，在本节的假设下把状态用动力学消去，只优化开环序列。本节沿用该文的 $c$ 与 $C_T$ 记号；全文的一般离散 OCP 记号为阶段代价 $\ell_k$ 与终端代价 $\Phi$：

```math
\min_{\{u_t\}}\ \sum_{t=0}^{T-1}c(x_t,u_t)+C_T(x_T),
\qquad
x_{t+1}=f(x_t,u_t),\quad x_0\ \text{给定}.
\tag{5}
```

两条结构假设贯穿其命题。阶段代价分离为 $c(x,u)=\ell(x)+\frac12 u^\top R u$，其中 $R\succ 0$、$\ell_{xx}\succeq 0$。动力学对控制仿射，$x_{t+1}=\bar f(x_t)+\bar g(x_t)u_t$。基础命题还假定当前迭代已经可行，即间隙为零；步长接受之后用非线性模型重新积分，以保持这一可行性。

无不等式时，单次打靶是无约束非线性规划，多重打靶仍是等式约束问题。前者的 Hessian 经整条轨迹复合后一般稠密；后者的 KKT 矩阵保持块三对角。不等式在多重打靶中进入二次子问题；单次打靶一侧的实现多依赖罚函数或增广拉格朗日。

### 1.4 直接配点的离散 NLP

直接配点把节点状态、节点控制和必要的中间状态一并作为决策变量。以梯形配点为例，离散动力学约束为

```math
d_k=x_{k+1}-x_k-
\frac{h_k}{2}\left(F(x_k,u_k)+F(x_{k+1},u_{k+1})\right)=0.
```

Hermite--Simpson 配点还引入中点状态和控制，并以 Simpson 缺陷约束连接端点与中点。路径约束可以直接施加在节点和中间点上；局部配点的约束 Jacobian 保持时间带状结构，适合稀疏 NLP 和结构化 SQP。与多重打靶相比，配点把积分器内部阶段显式展开为优化变量，因而可以在更密集的时间位置施加约束，但变量数和约束数相应增加。

## 2. SQP 的局部二次模型

### 2.1 等式约束上的两个子问题

先看一般等式约束问题 $\min_x c(x)$，约束 $h(x)=0$。拉格朗日函数 $\mathcal L=c+\lambda^\top h$。一阶必要条件是

```math
\mathcal R(x,\lambda)
=\begin{bmatrix}c_x+h_x^\top\lambda\\ h\end{bmatrix}
=0.
\tag{6}
```

$\mathcal R$ 对 $(x,\lambda)$ 非线性。在当前点 $(\bar x,\bar\lambda)$ 作 Newton 线性化，步 $(\delta x,\delta\lambda)$ 由 $D\mathcal R(\bar x,\bar\lambda)\,(\delta x,\delta\lambda)=-\mathcal R(\bar x,\bar\lambda)$ 确定：

```math
\begin{bmatrix}
c_{xx}+\bar\lambda\cdot h_{xx} & h_x^\top \\
h_x & 0
\end{bmatrix}
\begin{bmatrix}\delta x\\ \delta\lambda\end{bmatrix}
=
 -\begin{bmatrix}c_x+h_x^\top\bar\lambda\\ \bar h\end{bmatrix}.
\tag{7}
```

左上块是 $\nabla_x\mathcal L$ 对 $x$ 的导数。第一种子问题即 Newton 子问题，二次项保留乘子与约束曲率 $\bar\lambda\cdot h_{xx}$：

```math
\min_{\delta x}\ 
c_x^\top\delta x+\frac12\delta x^\top(c_{xx}+\bar\lambda\cdot h_{xx})\,\delta x
\quad\text{s.t.}\quad
h_x\delta x+\bar h=0.
\tag{8}
```

第二种子问题即修正子问题，去掉 $\bar\lambda\cdot h_{xx}$，只保留 $c_{xx}$：

```math
\min_{\delta x}\ 
c_x^\top\delta x+\frac12\delta x^\top c_{xx}\,\delta x
\quad\text{s.t.}\quad
h_x\delta x+\bar h=0.
\tag{9}
```

式 (8) 对 $\delta x$ 的驻点条件是 $(c_{xx}+\bar\lambda\cdot h_{xx})\delta x+c_x+h_x^\top\mu=0$，约束仍是 $h_x\delta x+\bar h=0$。与式 (7) 比较，$\mu=\bar\lambda+\delta\lambda$。线性项改用 $\nabla_x\mathcal L=c_x+h_x^\top\bar\lambda$ 时，同一驻点的乘子改为 $\delta\lambda$。在该约束上，两个目标相差常数 $-\bar\lambda^\top\bar h$，极小点相同。式 (9) 的乘子关系相同，只是 Hessian 不含 $\bar\lambda\cdot h_{xx}$。

### 2.2 下降方向

设当前点可行，$\bar h=0$，因而可行方向满足 $h_x\delta x=0$。第二种子问题的驻点条件为

```math
c_x+c_{xx}\delta x+h_x^\top\mu=0.
\tag{10}
```

左乘 $\delta x^\top$。约束项 $(\,h_x\delta x\,)^\top\mu=0$，于是

```math
\delta x^\top c_x=-\delta x^\top c_{xx}\,\delta x.
\tag{11}
```

若 $c_{xx}\succ 0$ 且 $\delta x\neq 0$，右端为负，$\delta x$ 是 $c$ 的下降方向。有限步长不一定保持非线性约束，需要线搜索；单次打靶用重新积分恢复可行，多重打靶则把间隙留在迭代中，由后续子问题继续消除。

第一种子问题的同一计算给出

```math
\delta x^\top c_x=-\delta x^\top(c_{xx}+\bar\lambda\cdot h_{xx})\,\delta x.
\tag{12}
```

对当前非零步，右端为负的充要条件是 $\delta x^\top(c_{xx}+\bar\lambda\cdot h_{xx})\delta x>0$。若要求所有非零可行方向都是下降方向，则需要拉格朗日 Hessian 在切空间 $\{\delta x:h_x\delta x=0\}$ 上正定；等价地，对任一切空间基 $Z$ 有 $Z^\top\nabla^2_{xx}\mathcal L Z\succ0$。$\bar\lambda$ 偏离最优乘子时该二次型可以不定。

原问题的解由式 (6) 决定。$\delta x=0$ 时，两种子问题接受同一个点。最优乘子一般不为零，$\bar\lambda\cdot h_{xx}$ 在解上仍然存在，因此第二种子问题一般不能继承精确 Newton--SQP 的局部二次收敛；仿射约束、零残差广义高斯--牛顿或约化空间中被忽略曲率恰好消失，是可能的例外。精确 SQP 的局部速率还需要约束资格条件、二阶充分条件和足够准确的子问题解。Nganga（2023）在人形机器人上观察到，精确 Hessian 的额外计算可能由外层迭代数的减少抵消；这是一项经验结果，不构成从第一次迭代使用不定精确 Hessian 的一般保证。

### 2.3 动力学约束上的两个子问题

式 (8) 与式 (9) 的变量在这里是整条 $(\delta x,\delta u)$。为避免乘子符号混乱，后文固定使用动力学约束 $g_{k+1}=f_k-x_{k+1}=\gamma_{k+1}=0$，拉格朗日函数取 $\mathcal L=J+\sum_k\lambda_{k+1}^\top g_{k+1}$。它与一般形式中取 $h_{k+1}=x_{k+1}-f_k$ 只差乘子重定义 $\lambda_g=-\lambda_h$。无不等式时，式 (4) 去掉不等式乘子，正是 $\sum_{k=0}^{T-1}\ell_k+\ell_T+\sum_{k=0}^{T-1}\lambda_{k+1}^\top(f_k-x_{k+1})$。

$\lambda_{k+1}^\top f_k$ 对 $(x_k,u_k)$ 的导数是 $f^\top\lambda_{k+1}$。前一段约束里的 $-\lambda_k^\top x_k$ 对 $x_k$ 的导数是 $-\lambda_k$。$x_0$ 给定、不作为变量时，没有 $-\lambda_0$。在当前乘子 $\bar\lambda$ 处，

```math
\nabla_{x_k}\mathcal L=\ell_x+f_x^\top\bar\lambda_{k+1}-\bar\lambda_k,
\qquad
\nabla_{u_k}\mathcal L=\ell_u+f_u^\top\bar\lambda_{k+1},
\qquad
\nabla_{x_T}\mathcal L=\ell_{T,x}-\bar\lambda_T.
\tag{13}
```

$-\bar\lambda_k^\top x_k$ 对 $x_k$ 是线性的，不进入 Hessian。第一种子问题保留 $\bar\lambda_{k+1}\cdot\nabla^2 f$，阶段 Hessian 为

```math
Q_k=\ell_{xx}+\nabla^2_{xx}(\bar\lambda_{k+1}^\top f),
\qquad
S_k=\ell_{xu}+\nabla^2_{xu}(\bar\lambda_{k+1}^\top f),
\qquad
R_k=\ell_{uu}+\nabla^2_{uu}(\bar\lambda_{k+1}^\top f).
\tag{14}
```

$S_k$ 是 $n_x\times n_u$ 的混合块。第二种子问题去掉含 $\bar\lambda_{k+1}$ 的三项。§1.2 中 $\ell_u=Ru$、$\ell_{xu}=0$、$\ell_{uu}=R$，且 $f$ 对 $u$ 仿射，因而 $f_{uu}=0$。第二种子问题化为 $Q_k=\ell_{xx}$、$S_k=0$、$R_k=R$；第一种子问题的 $R_k$ 同样等于 $R$。两个子问题的差别在 $Q_k$ 与 $S_k$。

式 (7) 的约束线性化为

```math
\delta x_{k+1}=f_x\delta x_k+f_u\delta u_k+\gamma_{k+1},
\qquad
\gamma_{k+1}=f_k-x_{k+1}.
\tag{15}
```

式 (13) 比 $\nabla\ell$ 多出的部分与增量作内积，得到 $\sum_{k=0}^{T-1}\bar\lambda_{k+1}^\top(f_x\delta x_k+f_u\delta u_k-\delta x_{k+1})$，其中 $-\delta x_{k+1}$ 来自下一拍梯度中的 $-\bar\lambda_{k+1}$。$x_0$ 固定，故没有 $-\bar\lambda_0^\top\delta x_0$。在式 (15) 上，括号等于 $-\gamma_{k+1}$，这个和变成常数 $-\sum_k\bar\lambda_{k+1}^\top\gamma_{k+1}$。因此以 $\nabla\ell$ 为线性项，与以式 (13) 为线性项，两个二次规划的极小点相同。前者的动力学乘子等于后者的乘子加上 $\bar\lambda$。Sousa-Pinto 与 Orban（2024）的第四章采用式 (13)，得到的仍是线性二次调节问题。第一种子问题与第二种子问题只由式 (14) 是否保留 $\bar\lambda_{k+1}\cdot\nabla^2 f$ 区分。

### 2.4 不等式、目标 Hessian 与广义高斯--牛顿近似

Jordana 的第 $n$ 次外层迭代求解

```math
\min_{\Delta x,\Delta u}\ 
\sum_{k=0}^{T-1}\frac12
\begin{bmatrix}\Delta x_k\\ \Delta u_k\end{bmatrix}^\top
\begin{bmatrix}Q_k& S_k\\ S_k^\top& R_k\end{bmatrix}
\begin{bmatrix}\Delta x_k\\ \Delta u_k\end{bmatrix}
+\begin{bmatrix}q_k\\ r_k\end{bmatrix}^\top
\begin{bmatrix}\Delta x_k\\ \Delta u_k\end{bmatrix}
+\frac12\Delta x_T^\top Q_T\Delta x_T+q_T^\top\Delta x_T
\tag{16}
```

```math
\Delta x_{k+1}=A_k\Delta x_k+B_k\Delta u_k+\gamma_{k+1},
\qquad
D_k\Delta x_k+E_k\Delta u_k+\hat c_k\ge 0,
\qquad
D_T\Delta x_T+\hat c_T\ge 0.
\tag{17}
```

$A_k,B_k$ 是 $f_k$ 的雅可比，$\gamma_{k+1}$ 是当前间隙。$Q_k,S_k,R_k$ 取拉格朗日函数对 $(x_k,u_k)$ 的完整 Hessian 时，其中包含 $\lambda^{[n]},\mu^{[n]}$ 与 $f_k,c_k$ 的二阶项。无不等式时这就是式 (14)，即第一种子问题；这里同时保留路径不等式。

忽略动力学和一般约束的二阶项，只保留目标函数的 Hessian，得到第二种子问题：

```math
Q_k=\nabla^2_{xx}\ell_k,\qquad
S_k=\nabla^2_{xu}\ell_k,\qquad
R_k=\nabla^2_{uu}\ell_k.
\tag{18}
```

子问题不再依赖乘子。若阶段代价具有加权最小二乘形式 $\ell_k(z)=\frac12\|r_k(z)\|_{W_k}^2$，还可以把目标 Hessian

```math
\nabla^2\ell_k
=J_{r_k}^\top W_kJ_{r_k}
+\sum_i(W_kr_k)_i\nabla^2r_{k,i}
```

近似为 $J_{r_k}^\top W_kJ_{r_k}$；这才是通常意义上的广义高斯--牛顿近似。文献有时把“忽略约束曲率、保留代价 Hessian”也简称为高斯--牛顿，二者在非最小二乘代价下不应混用。无不等式时，多重打靶与广义高斯--牛顿组合得到 GNMS；单次打靶、线性化动力学和相同曲率模型对应 iLQR。内层均可由 §4 的 Riccati 递推求解。外层更新

```math
x^{[n+1]}=x^{[n]}+\alpha\Delta x,\qquad
u^{[n+1]}=u^{[n]}+\alpha\Delta u,
\tag{19}
```

新乘子取该二次子问题的 KKT 乘子。无不等式时，这对乘子可由 Riccati 的协态直接读出。

## 3. iLQR、DDP 与局部动态规划

### 3.1 与 Bellman 方程

本节沿用动态规划文献的阶段代价 $c$、终端代价 $C_T$ 和值函数 $V_t$ 记号；它们分别对应前文的 $\ell_t$、$\Phi$ 和离散值函数。

Bellman 方程为

```math
V_t^*(x_t)=\min_u Q_t(x_t,u),
\qquad
Q_t(x_t,u)=c(x_t,u)+V_{t+1}^*\bigl(f(x_t,u)\bigr).
\tag{20}
```

记最优反馈为 $u_t^*(x)$。只在名义点满足 $\bar u_t=u_t^*(\bar x_t)$ 时，对 $Q_t$ 的局部展开才能直接解释为最优 Bellman 方程在该点的展开。此时 $Q_u=0$，包络定理给出 $V_x^*=Q_x$；反馈斜率不显式进入一阶导数。

一般名义控制 $\bar u_t\neq u_t^*(\bar x_t)$ 处，算法构造的是沿当前名义轨迹的局部二次改进模型，不是未知最优值函数 $V_t^*$ 的 Taylor 展开。下一节最小化阶段代价与后一拍二次模型之和，仍然可以得到合法的局部反馈步。若名义点满足局部最优性且 $Q_{uu}\succ0$，隐函数定理给出 $u_x^*=-Q_{uu}^{-1}Q_{ux}$，于是

```math
V_{xx}^*=Q_{xx}-Q_{xu}Q_{uu}^{-1}Q_{ux}.
```

反馈敏感度通过 Schur 补进入二阶导数。Chakravorty 将非最优轨迹上的 DDP 解释为对 Bellman 展开的启发式近似，这一解释用于分析 DDP 与精确 SQP 的差别。DDP 反向递推在非最优轨迹上仍有明确定义，并可产生局部改进步。

### 3.2 控制增量与递推

间隙为零时，式 (15) 化为 $\delta x_{t+1}=f_x\delta x_t+f_u\delta u_t$，且 $\delta x_0=0$。$V_{t+1}(x)$ 是 $t+1$ 之后的值函数（Tassa、Erez 与 Todorov，2012）。常数 $V_{t+1}(x_{t+1})$ 不依赖 $\delta x_{t+1}$。记 $V_{x,t+1}=\nabla_x V_{t+1}$、$V_{xx,t+1}=\nabla^2_{xx}V_{t+1}$，进入最小化的增量是

```math
V_{x,t+1}^\top\delta x_{t+1}+\frac12\delta x_{t+1}^\top V_{xx,t+1}\delta x_{t+1}.
\tag{21}
```

阶段代价丢掉常数 $c(x_t,u_t)$ 后，增量为

```math
\ell_x^\top\delta x_t+(Ru_t)^\top\delta u_t+\frac12\delta x_t^\top\ell_{xx}\delta x_t+\frac12\delta u_t^\top R\delta u_t.
\tag{22}
```

终端为 $V_{x,T}=\nabla C_T$、$V_{xx,T}=C_{T,xx}$。记 $V_x=V_{x,t+1}$、$V_{xx}=V_{xx,t+1}$。把 $\gamma_{t+1}=0$ 时的式 (15) 代入式 (21)，得到

```math
\begin{aligned}
&V_x^\top f_x\delta x_t+V_x^\top f_u\delta u_t\\
&+\frac12\delta x_t^\top f_x^\top V_{xx}f_x\delta x_t+\delta x_t^\top f_x^\top V_{xx}f_u\delta u_t+\frac12\delta u_t^\top f_u^\top V_{xx}f_u\delta u_t.
\end{aligned}
\tag{23}
```

与式 (22) 相加。$(Ru_t)^\top\delta u_t$ 与 $V_x^\top f_u\delta u_t$ 是 $\delta u_t$ 的一次项，$\frac12\delta u_t^\top R\delta u_t$ 与 $\frac12\delta u_t^\top f_u^\top V_{xx}f_u\delta u_t$ 是二次项，$\delta x_t^\top f_x^\top V_{xx}f_u\delta u_t$ 是交叉项。第二种子问题的 $S_t=0$，交叉项只有最后这一项，系数即为

```math
q_{u,t}=Ru_t+f_u^\top V_x,
\qquad
H_t=R+f_u^\top V_{xx}f_u,
\qquad
Q_{ux,t}=f_u^\top V_{xx}f_x.
\tag{24}
```

含 $\delta u_t$ 的部分是

```math
q_{u,t}^\top\delta u_t+\delta x_t^\top Q_{ux,t}^\top\delta u_t+\frac12\delta u_t^\top H_t\delta u_t.
\tag{25}
```

$R$ 与 $V_{xx}$ 对称，故式 (24) 的 $H_t$ 对称。对式 (25) 关于 $\delta u_t$ 求导，得 $q_{u,t}+Q_{ux,t}\delta x_t+H_t\delta u_t$。令其为零，

```math
H_t\delta u_t=-(q_{u,t}+Q_{ux,t}\delta x_t).
\tag{26}
```

$H_t\succ 0$ 时，控制增量为

```math
\delta u_t=-H_t^{-1}q_{u,t}-H_t^{-1}Q_{ux,t}\,\delta x_t.
\tag{27}
```

再代回式 (22) 与式 (23) 之和。令 $g_t=q_{u,t}+Q_{ux,t}\delta x_t$，则式 (27) 即 $\delta u_t=-H_t^{-1}g_t$，式 (25) 化为

```math
g_t^\top\delta u_t+\frac12\delta u_t^\top H_t\delta u_t=-\frac12 g_t^\top H_t^{-1}g_t.
\tag{28}
```

展开右端，

```math
-\frac12 q_{u,t}^\top H_t^{-1}q_{u,t}-q_{u,t}^\top H_t^{-1}Q_{ux,t}\delta x_t-\frac12\delta x_t^\top Q_{ux,t}^\top H_t^{-1}Q_{ux,t}\delta x_t.
\tag{29}
```

只含 $\delta x_t$ 的项还有 $\ell_x^\top\delta x_t+V_x^\top f_x\delta x_t$ 与 $\frac12\delta x_t^\top(\ell_{xx}+f_x^\top V_{xx}f_x)\delta x_t$。合并后，值函数的梯度与 Hessian 为

```math
V_{x,t}=\ell_x+f_x^\top V_x-Q_{ux,t}^\top H_t^{-1}q_{u,t},
\qquad
V_{xx,t}=\ell_{xx}+f_x^\top V_{xx}f_x-Q_{ux,t}^\top H_t^{-1}Q_{ux,t}.
\tag{30}
```

下一拍把式 (30) 的 $V_{x,t},V_{xx,t}$ 当作这里的 $V_x,V_{xx}$。间隙不为零时的消元是式 (37)–(41)。

第一种子问题在阶段模型中还有式 (14) 的曲率，

```math
\frac12\delta x_t^\top\nabla^2_{xx}(\bar\lambda_{t+1}^\top f)\,\delta x_t+\delta x_t^\top\nabla^2_{xu}(\bar\lambda_{t+1}^\top f)\,\delta u_t+\frac12\delta u_t^\top\nabla^2_{uu}(\bar\lambda_{t+1}^\top f)\,\delta u_t.
\tag{31}
```

这些项对 $\delta u_t$ 没有一次贡献，式 (24) 的 $q_{u,t}$ 不变。其中的 $V_x$ 是值函数梯度；$\bar\lambda$ 只经式 (14) 进入 Hessian。交叉项对 $\delta u_t$ 的导数是 $\nabla^2_{ux}(\bar\lambda_{t+1}^\top f)\,\delta x_t$，故 $Q_{ux,t}$ 加上这个转置。二次项给 $H_t$ 加上 $\nabla^2_{uu}(\bar\lambda_{t+1}^\top f)$。控制仿射时后一项为零。代回式 (30) 时，$\ell_{xx}$ 换成 $\ell_{xx}+\nabla^2_{xx}(\bar\lambda_{t+1}^\top f)$，并且 $Q_{ux,t}$ 与 $H_t$ 取加上这些增量之后的值。

### 3.3 正定性

第二种子问题还要保证每一拍 $H_t\succ 0$，式 (27) 才有定义，并与 §2.2 的下降方向相接。终端 Hessian 取 $V_{xx,T}=C_{T,xx}\succeq 0$。若 $V_{xx,t+1}\succeq 0$，第二种子问题的二次项为

```math
\frac12
\begin{bmatrix}\delta x_t\\ \delta u_t\end{bmatrix}^\top
\left(
\begin{bmatrix}\ell_{xx}&0\\ 0&R\end{bmatrix}
+\begin{bmatrix}f_x^\top\\ f_u^\top\end{bmatrix}
V_{xx}
\begin{bmatrix}f_x&f_u\end{bmatrix}
\right)
\begin{bmatrix}\delta x_t\\ \delta u_t\end{bmatrix}.
\tag{32}
```

前一块中 $\ell_{xx}\succeq 0$、$R\succ 0$，因而半正定。后一块由 $V_{xx}\succeq 0$ 得到，也半正定。和半正定，控制对角块满足

```math
H_t=R+f_u^\top V_{xx}f_u\succeq R\succ 0.
\tag{33}
```

对 $\delta u_t$ 最小化等于取式 (32) 关于 $H_t$ 的 Schur 补，即式 (30) 的 $V_{xx,t}$。右下块正定时 Schur 补半正定，故 $V_{xx,t}\succeq 0$。从终端推到 $t=0$，每一拍 $H_t\succ 0$。若 $q_{u,t}\neq 0$，式 (35) 在足够小的 $\alpha>0$ 下为负。

第一种子问题多出的 $\nabla^2(\bar\lambda_{t+1}^\top f)$ 不必半正定，二次型之和可以不定，归纳到 $V_{xx,t}$ 即停止。$f_{uu}=0$ 时 $H_t$ 仍写成 $R+f_u^\top V_{xx,t+1}f_u$；$V_{xx,t+1}$ 不定时，$f_u^\top V_{xx,t+1}f_u$ 可以抵消 $R$，使 $H_t$ 不定。

### 3.4 乘子的取法

由 §3.1，二次模型只在名义控制为最优反馈时等于 Bellman 方程在最优轨迹上的展开。DDP 的反向使用该消元，但式 (14) 中的 $\bar\lambda_{t+1}$ 换成这次递推得到的 $V_{x,t+1}$。Newton 步的乘子是上一轮二次规划的解。二者只在驻点残差 $R\bar u_t+f_u^\top V_{x,t+1}$ 接近零时一致；残差大时，二次模型偏离 Newton 步。

Pantoja（1988）的阶段 Newton 法使用当前伴随。Chakravorty 的 Newton LQR 使用上一轮乘子 $\bar\lambda$。二者与 DDP 同样只在最优附近一致。

### 3.5 预测下降与线搜索

记后一拍值函数的二次模型已经得到，阶段对控制的导数与 Hessian 为 $Q_{u,t}$、$Q_{uu,t}$。前馈为 $\kappa_t=-Q_{uu,t}^{-1}Q_{u,t}$。步长 $\alpha\in(0,1]$ 下，二次模型预测的代价变化为

```math
\Delta J(\alpha)
=\alpha\sum_{t=0}^{T-1}\kappa_t^\top Q_{u,t}
+\frac{\alpha^2}{2}\sum_{t=0}^{T-1}\kappa_t^\top Q_{uu,t}\,\kappa_t.
\tag{34}
```

代入 $\kappa_t$，

```math
\Delta J(\alpha)
=\Bigl(-\alpha+\frac{\alpha^2}{2}\Bigr)
\sum_{t=0}^{T-1}Q_{u,t}^\top Q_{uu,t}^{-1}Q_{u,t}.
\tag{35}
```

$Q_{uu,t}\succ 0$ 且 $Q_{u,t}\neq 0$ 时，足够小的 $\alpha$ 使式 (35) 为负。iLQR 的 $Q_{uu,t}$ 即式 (24) 的 $H_t$，其正定性由式 (32)、式 (33) 的归纳给出。DDP 的 $Q_{uu}$ 可以不定，预测变化的符号随之失效。

Cart-pole 摆起的第一步，DDP 给出 $J=8.9\times 10^5$、$\Delta J=-8.2\times 10^5$，预测值 $7.0\times 10^4$；后续迭代中 DDP 的 $J+\Delta J$ 会低于代价下界 $0$。另一组记录为 DDP 的 $\Delta J=+3.0\times 10^7$、iLQR 的 $\Delta J=-5.6\times 10^5$。$Q_{uu}$ 在摆与 Cart-pole 上对 iLQR 保持正定，DDP 则出现负值。

对单次打靶的约化目标 $J(U)$，线搜索可使用 Armijo 条件。若 $p$ 是控制序列方向且 $\nabla J(U)^\top p<0$，则要求

```math
J(U+\alpha p)-J(U)
\le \sigma\alpha\nabla J(U)^\top p,
\qquad 0<\sigma<1.
\tag{36}
```

等价地，在分母为负时，实际变化与一阶预测变化之比应不小于 $\sigma$。这里的方向是约化控制方向，不是独立状态方向。iLQR 的预测与一阶模型一致，回退少；摆起实验中步长可以保持为 $1$。未正则化 DDP 的二阶预测与积分得到的代价可能不一致，Chakravorty 的 Cart-pole 实验中步长被压到接近 $0$。该结果隔离了曲率模型的影响，但不能外推为带 Levenberg--Marquardt 正则、信赖域或修正 Cholesky 的一般 DDP 实现。

### 3.6 近最优与远离最优

在 Chakravorty 的未正则化实验中，初值靠近最优时 DDP 的迭代数更少：摆起为 $5$ 对 iLQR 的 $8$，Cart-pole 摆起为 $5$ 对 $17$，与更完整曲率模型在局部可能收敛更快的判断一致。初值远离最优时，在 §1.2 的凸性、控制仿射和可行轨迹假设下，iLQR 子问题给出下降方向；该文的未正则化 DDP 则可能因 $Q_{uu}$ 不定而使线搜索失效。这些数字不构成对所有正则化 DDP 与一般 SQP 实现的排序。

## 4. OCP-QP 的结构化求解

### 4.1 间隙不为零时的同一消元

无不等式时，§2.4 的子问题沿用 §3 的消元。记 $v_{k+1}=V_{x,k+1}$、$V_{k+1}=V_{xx,k+1}$，$A_k,B_k$ 即 $f_x,f_u$。相对式 (24) 与式 (30)，多出三项：间隙 $\gamma_{k+1}$ 进入前馈，交叉项 $S_k$ 进入 $Q_{ux}$，线性项 $q_k,r_k$ 替换 $\ell_x$ 与 $Ru$。当 $\gamma_{k+1}=0$、$S_k=0$、$Q_k=\ell_{xx}$、$R_k=R$、$q_k=\ell_x$、$r_k=Ru_k$ 时，下面各式回到式 (24)、式 (27) 与式 (30)。

```math
H_k=R_k+B_k^\top V_{k+1}B_k,
\tag{37}
```

```math
Q_{ux,k}=S_k^\top+B_k^\top V_{k+1}A_k,
\qquad
q_{u,k}=r_k+B_k^\top(v_{k+1}+V_{k+1}\gamma_{k+1}).
\tag{38}
```

$H_k\succ 0$ 时，控制增量为 $K_k\Delta x_k+\kappa_k$，

```math
K_k=-H_k^{-1}Q_{ux,k},
\qquad
\kappa_k=-H_k^{-1}q_{u,k}.
\tag{39}
```

值函数的 Hessian 与梯度为

```math
V_k=Q_k+A_k^\top V_{k+1}A_k-K_k^\top H_k K_k,
\tag{40}
```

```math
v_k=q_k+A_k^\top(v_{k+1}+V_{k+1}\gamma_{k+1})-K_k^\top H_k\kappa_k.
\tag{41}
```

其中 $K_k^\top H_k K_k=Q_{ux,k}^\top H_k^{-1}Q_{ux,k}$，故 $S_k$ 只出现在 $K_k$ 中。终端取子问题自己的 $Q_T,q_T$。每一拍只分解 $n_u\times n_u$ 的 $H_k$；状态维上的乘法使整段递推对时域线性、对 $n_x$ 为三次。

动力学乘子由同一二次模型读出，

```math
\lambda_k=V_k\Delta x_k+v_k.
\tag{42}
```

它是值函数模型对 $\Delta x_k$ 的梯度，也是式 (3) 的乘子在该子问题里的解。

### 4.2 块三对角系统

补充材料把同一 KKT 系统按阶段排列。第 $k$ 段的未知量合成 $s_k$，包含该段的控制增量、状态增量与乘子。无不等式时系统为

```math
\begin{bmatrix}
\Gamma_1 & M_1^\top & & \\
M_1 & \Gamma_2 & M_2^\top & \\
& \ddots & \ddots & \ddots \\
& & M_{T-1} & \Gamma_T
\end{bmatrix}
\begin{bmatrix}s_1\\ \vdots\\ s_T\end{bmatrix}
=
\begin{bmatrix}g_1\\ \vdots\\ g_T\end{bmatrix}.
\tag{43}
```

对角块含该段的代价 Hessian 和动力学等式，

```math
\Gamma_k=
\begin{bmatrix}
R_{k-1}& 0& -B_{k-1}^\top\\
0& Q_k& I\\
-B_{k-1}& I& 0
\end{bmatrix},
\qquad
M_k=
\begin{bmatrix}
0& S_k^\top& 0\\
0& 0& 0\\
0& -A_k& 0
\end{bmatrix}.
\tag{44}
```

$M_k$ 只连接相邻阶段。Thomas 消元从终端向起点做，

```math
\bar\Gamma_k=\Gamma_k-M_k^\top\bar\Gamma_{k+1}^{-1}M_k,
\qquad
\bar g_k=\bar\Gamma_k^{-1}(g_k-M_k^\top\bar g_{k+1}),
\tag{45}
```

再正向回代 $s_1=\bar g_1$、$s_{k+1}=\bar g_{k+1}-\bar\Gamma_{k+1}^{-1}M_k s_k$。

归纳假设是消元后的对角块仍保持动力学 KKT 的形状，只是状态上的 $Q_k$ 换成值函数 Hessian $V_k$，右端的状态分量换成 $-v_k$。把 $\bar\Gamma_{k+1}^{-1}$ 的 Schur 补算出来，需要分解的正是式 (37) 的 $H_k$，消元公式化为式 (39)--(41)。因此 Riccati 递推就是式 (43) 的 Thomas 消元，动态规划与 KKT 是同一线性系统的两种写法。正向回代对应线性 rollout。循环约化和并行前缀扫描可以把理想并行条件下的关键路径由 $O(T)$ 降为 $O(\log T)$；总工作量通常仍为 $O(T)$，并且需要额外通信与中间存储。

构造上述局部 KKT 系统需要足够准确的一阶导数和所选的曲率模型，但不要求手工解析导数。有限差分、复步微分、自动微分以及 BFGS、SR1 等拟牛顿近似都可以提供所需信息。只有在无法获得或可靠估计局部导数模型时，才需要转向纯函数值方法；即便如此，也可以先由采样拟合局部线性--二次模型，再形成近似的结构化子问题。

## 5. 从局部模型到可靠求解器

### 5.1 曲率模型与正则化

局部二次模型不只有“代价 Hessian”和“完整 Hessian”两种。常见选择包括：

- 精确拉格朗日 Hessian，局部模型最忠实，但可能不定且二阶动力学导数昂贵；
- 广义高斯--牛顿，舍去残差二阶项，最小二乘代价下通常半正定；
- BFGS、L-BFGS 与 SR1，用梯度差更新曲率；
- convex-over-nonlinear 或 sequential convex quadratic approximation，保留已知凸结构；
- 由自动微分得到的 Hessian--vector product，再用迭代线性求解器处理。

即使理论模型是正定的，有限精度、非凸代价和远离解的线性化仍可能使 $Q_{uu}$ 奇异或不定。iLQR/DDP 常用 Levenberg--Marquardt 修正

```math
Q_{uu}^{\mathrm{reg}}=Q_{uu}+\rho I,
\qquad \rho>0,
\tag{46}
```

或先正则值函数曲率，再形成 $Q_{uu}$。两者给出的反馈增益并不完全相同。其他选择包括特征值截断、modified Cholesky 和显式信赖域

```math
\|D\Delta z\|\le \Delta.
\tag{47}
```

正则化的作用不只是让矩阵可逆：它在模型不可信时缩短步长，并在接受良好步骤后逐渐减弱。因而“未正则化 DDP 的 $Q_{uu}$ 可能不定”不能推出工程 DDP 必然不稳定；同样，广义高斯--牛顿的半正定性也不自动保证全局收敛。

### 5.2 凝聚、稀疏分解与 Riccati

线性化动力学可以递归消去状态：

```math
\Delta X=\mathcal A\Delta x_0+\mathcal B\Delta U+\mathcal G.
\tag{48}
```

代回目标和不等式得到只含 $\Delta U$ 的稠密 QP，这称为 full condensing。它减少变量和等式，但破坏时间稀疏性，适合时域较短或控制维数较小的情形。partial condensing 只合并若干连续阶段，在稠密规模和稀疏深度之间折中。

保留全部状态和控制时，可以直接对稀疏 KKT 系统作 $LDL^\top$ 分解，也可以用 Riccati 递推完成等价的阶段消元。两者的差别主要在变量排列、数值线性代数实现和是否方便纳入不等式，而不是在所求 Newton 步本身。对树状场景 MPC，相同思想推广为 tree Riccati 或 tree-structured interior-point method。

结构化 QP 后端主要有三类：

1. 主动集法维护当前活跃约束，热启动良好时每拍只改变少量工作集；
2. 原始--对偶内点法通过障碍参数逼近互补条件，每次解结构化 KKT 系统；
3. ADMM 和其他算子分裂方法把动力学 LQ 步与局部投影分开，容易并行和热启动，但中间迭代一般不可行。

HPIPM 同时提供 dense、OCP 和 tree-OCP QP 的内点求解及凝聚；acados 则把 SQP、RTI、积分器、condensing 和不同 QP 后端做成可组合模块。它们体现的正是本文的观点：转录、外层 NLP 算法和内层线性代数不应混为一个不可拆分的算法名。

### 5.3 一般约束：直接处理、增广拉格朗日与投影

控制盒约束可以在每一阶段解 box-QP，得到分段仿射的局部反馈；一般状态约束却跨阶段耦合，不能靠逐分量截断保持一致。常见路线如下。

直接 SQP 在每个 QP 中线性化约束，并用主动集、内点法或 ADMM 求解。这一路线能直接监控 KKT 残差，但子问题可能昂贵或不可行。

增广拉格朗日把等式约束 $h=0$ 加入目标：

```math
\mathcal L_\rho(z,\nu)
=J(z)+\nu^\top h(z)+\frac{\rho}{2}\|h(z)\|^2,
\tag{49}
```

不等式则通过投影乘子或正部函数处理。ALTRO 用增广拉格朗日 iLQR 先获得粗略可行解，再用主动集投影 Newton 阶段提高约束精度。它说明 DDP 式反向递推可以嵌入标准约束优化框架，但罚参数、乘子更新和最终可行性必须单独分析。

障碍法把严格不等式 $c_i(z)>0$ 写成

```math
J_\tau(z)=J(z)-\tau\sum_i\log c_i(z),
\tag{50}
```

并令 $\tau\downarrow0$。它保持迭代在内部，却要求严格可行初值或额外的可行性阶段。软罚函数只减少违反量，不能把“罚值很大”当作硬安全保证。

### 5.4 Merit、filter、信赖域与可行性恢复

局部 QP 步必须同时改善目标与约束。精确罚 merit function 的典型形式是

```math
\phi_\rho(z)=J(z)+\rho\|h(z)\|_1+\rho\|[c(z)]_+\|_1.
\tag{51}
```

$\rho$ 足够大时，merit 的下降可以排除只降低目标却扩大约束违反的步骤，但罚权选择困难。Filter 不把两者压成一个标量，而拒绝同时在目标和可行性上被已有迭代支配的候选。信赖域则限制只在局部模型可信的区域内求步，并根据实际下降与预测下降的比值调整半径。

SQP 即使在解附近也可能受 Maratos effect 影响：一阶线性化约束使全步被 merit function 拒绝。二阶修正沿约束法向补偿曲率，可以恢复快速局部收敛。若 QP 本身不可行，还需要 elastic variables、feasibility restoration phase 或增大信赖域罚项。仅写出一个线搜索公式不足以给出全局收敛结论。

### 5.5 MPC、RTI 与闭环问题

离线轨迹优化追求一个 NLP 的高精度局部解；MPC 每拍改变初始状态并平移时域，更关心固定时间内的解质量。标准流程是：

1. 平移上一拍轨迹与乘子，补齐末端作为 warm start；
2. 用新状态更新初始条件；
3. 完成若干次线性化、QP 求解与全局化；
4. 只施加第一段控制，再在下一拍重求。

SQP-RTI 每拍只执行一次 SQP 步，并把计算分成 preparation phase 与 feedback phase。前者在测量到来前完成动力学线性化和大部分因子分解，后者只更新初始状态相关项并快速解 QP。Advanced-step RTI 进一步根据预测状态提前准备问题。

RTI 的依据不是“单步已经收敛”，而是相邻 MPC 问题变化较小、warm start 落在局部收缩区域内。采样时间、扰动、线性化误差和 QP 求解误差共同决定跟踪误差。优化器返回一个低 KKT 残差也不自动给出闭环稳定性；通常还需要终端代价/终端集、递归可行性、鲁棒管束或独立稳定性论证。

### 5.6 近端点、复合目标与算子分裂

正则化还可以从近端点方法理解。对正常（proper）、下半连续的凸函数 $\varphi$，带度量 $M_k\succ0$ 的近端点迭代为

```math
z^{k+1}
=\operatorname{prox}^{M_k}_{\eta_k \varphi}(z^k)
:=\arg\min_z\left\{
\varphi(z)+\frac{1}{2\eta_k}\|z-z^k\|_{M_k}^2
\right\}.
```

它的最优性条件是

```math
0\in \partial \varphi(z^{k+1})
+\eta_k^{-1}M_k(z^{k+1}-z^k),
```

即 $z^{k+1}=(I+\eta_kM_k^{-1}\partial \varphi)^{-1}z^k$。该式在新迭代点满足隐式最优性条件。凸目标允许不可微，但相应的近端子问题必须能够精确或以受控误差求解。

若目标具有复合结构 $\varphi(z)=f(z)+g(z)$，其中 $f$ 光滑而 $g$ 可非光滑，前向--后向分裂给出

```math
z^{k+1}
=\operatorname{prox}_{\eta_k g}^{M_k}
\!\left(z^k-\eta_kM_k^{-1}\nabla f(z^k)\right).
```

$g$ 可以是控制盒约束的指标函数、$\ell_1$ 稀疏正则或某些可分阶段的非光滑罚项；此时近端映射分别退化为投影、软阈值或局部可分的隐式更新。若非光滑性来自碰撞事件、模式切换或黑箱仿真器，而不是一个具有可计算近端映射的显式 $g$，上述分裂并不会自动解决问题。

SQP 子问题中的正则项

```math
\frac{1}{2\eta_k}\|\Delta z\|_{M_k}^2
```

使曲率变为 $B_k+\eta_k^{-1}M_k$，可视为局部二次模型上的变度量近端步。信赖域的拉格朗日乘子也会产生形如 $B_k+\lambda D^\top D$ 的移位矩阵，但其中的 $\lambda$ 由半径约束和互补条件决定，近端参数则直接定义隐式步。SQP 还同时线性化约束，其收敛性质取决于局部模型、约束资格条件和全局化机制。

更一般地，KKT 条件可写成单调或非单调算子包含式 $0\in T(w)$。预条件近端点使用 resolvent $(I+M^{-1}T)^{-1}$；Douglas--Rachford splitting 对算子和进行反射分裂，ADMM 在标准凸条件下可由其在对偶问题上的应用导出；增广拉格朗日法在凸性及适当正则条件下可解释为对偶变量上的近端点方法。这些联系说明了 §5.3 与 §7 中罚项、近端项和对偶更新的共同算子结构。非凸 OCP 中的 ALM、ADMM 与原始近端点仍具有不同的收敛条件。

## 6. 算法设计维度

### 6.1 局部模型、打靶方式与前向更新

表中的完整拉格朗日 Hessian 是第一种子问题，目标 Hessian 是第二种子问题；最小二乘代价下还可进一步采用广义高斯--牛顿 Hessian。iLQR、GNMS 与 Jordana 的相应形式忽略动力学约束曲率。二次项、打靶方式和前向更新互相独立，常用算法是它们的组合。

| 算法 | 二次项 | 打靶 | 接受步之后的前向 |
|------|--------|------|------------------|
| iLQR | 代价 Hessian | 单次 | 非线性积分 |
| Newton LQR | 完整拉格朗日 Hessian | 单次 | 非线性积分或线性更新 |
| DDP | 完整形式，但乘子换成当前 $V_x$ | 单次 | 非线性积分 |
| GNMS | 代价 Hessian | 多重 | 线性间隙更新 |
| FDDP | DDP 式局部模型 | 允许动力学间隙的混合打靶 | feasibility-driven rollout，逐步收缩间隙 |
| Jordana，目标/GN Hessian | 目标或广义高斯--牛顿 Hessian | 多重 | 线性间隙更新，含不等式 |
| Jordana，精确 Hessian | 完整拉格朗日 Hessian | 多重 | 线性间隙更新，含不等式 |

无不等式时，若二次项都用上一轮乘子 $\bar\lambda$，相应的阶段 Newton 反向与精确 SQP 的 KKT 系统相同。标准 DDP 实际把式 (14) 中的 $\bar\lambda$ 换成正在反向递推的值函数梯度 $V_x$，二次项因而不再是同一个 Newton--KKT 系统。FDDP 不是普通多重打靶加非线性积分：它允许初始动力学间隙，并在 feasibility-driven rollout 中控制间隙收缩；间隙闭合后通常不再重新打开，算法随之接近可行 DDP。

线性更新 $\Delta x_{k+1}=A_k\Delta x_k+B_k\Delta u_k+\gamma_{k+1}$ 是标准 SQP 试探步的动力学部分，因此可以纳入 SQP 的收敛分析框架。具体的局部或全局结论仍要求约束资格条件、二阶充分条件、适当的 Hessian 修正、足够准确的 QP 解，以及满足接受准则的 merit、filter 或信赖域机制。非线性积分是 DDP 的传统前向，接受步之后的状态离开了二次模型所预测的点，预测下降与真实代价可以不一致。过滤器与停机对齐、只保留这一差别时，Humanoid 上的迭代次数见 §7。非线性前向与多重打靶并用时，不能直接套用标准线搜索 SQP 的全局收敛定理。

### 6.2 常见算法在完整坐标系中的位置

下面用文字而不是单一谱系排列常见方法。

- iLQR：单次打靶，线性化动力学，通常采用目标或广义高斯--牛顿曲率，Riccati 反向，非线性 rollout，线搜索与正则化。
- DDP：单次打靶，保留动力学二阶项，Riccati 式反向，非线性 rollout；实际实现通常需要正则化。
- GNMS：多重打靶，广义高斯--牛顿局部模型，线性间隙更新，结构化 LQ 子问题。
- FDDP：允许动力学不可行初值，以 DDP 式反向和 feasibility-driven rollout 收缩间隙，闭合后接近可行 DDP。
- direct-collocation SQP：配点转录，SQP 局部 QP，通常由稀疏直接法、内点法或凝聚后的 QP 后端求解。
- ALTRO：增广拉格朗日 iLQR，加主动集 projected Newton 精修。
- SQP-RTI：多重打靶或配点转录，每个采样周期执行一个准备充分的 SQP 步。

没有一种方法在所有维度占优。长时域敏感系统通常受益于多重打靶；高精度连续轨迹需要配点和网格误差控制；控制维数小且需要反馈增益时，Riccati 消元很有吸引力；复杂硬约束则常把选择推向结构化 SQP/IPM。算法比较必须同时报告转录、网格、导数、正则化、全局化、QP 容差和 warm start，不能只比较名称。

## 7. 代表性实现与数值案例

本章不再引入新的统一原理，而是把 Jordana、Chakravorty 等工作的实现选择和实验结果放回前述坐标系中。具体参数和性能数字只适用于相应问题、正则化与停止准则，不能直接解释为算法名称之间的普遍排序。

### 7.1 分裂

标准 OSQP 把全部线性约束写成 $Pv\in\mathcal C$。动力学若与不等式放在同一次投影里，ADMM 的中间迭代不满足线性化动力学。OSQP_OCP 把变量 $v=(\Delta x,\Delta u)$ 的定义域取为线性化动力学、二次代价和近端项，集合 $\mathcal C$ 只保留分阶段的不等式。

一轮 ADMM 因而分成四步。第一步解带 $\rho,\sigma$ 正则的扩展线性二次问题，仍用反向 Riccati 和正向回代，间隙进入前馈修正。第二步对辅助变量作超松弛。第三步把不等式残差投影回 $\mathcal C$，各阶段独立，界约束就是逐分量截断。第四步更新对偶。参数取 $\sigma=10^{-6}$、超松弛系数 $1.6$，$\rho$ 每 $25$ 步更新一次，因此反向分解不必每步重做。从动力学可行的 Riccati 解热启动，可减少 ADMM 步数。

Jordana 的实验在所测试规模上观察到 OSQP_OCP 对时域长度近似线性，并对状态维数呈现良好经验缩放；这不是对 $n_x$ 的一般线性复杂度结论。稠密阶段矩阵的 Riccati 或内点分解仍含关于 $n_x,n_u$ 的二次和三次运算。小规模上 HPIPM 的结构内点法略快。通用 OSQP 在 Solo 与 Taichi 上比这两种结构求解器慢一个数量级以上。不等式的乘子取 ADMM 的对偶变量，与式 (42) 使用同一 Riccati 骨架。

### 7.2 过滤器与停机

试探步的步长由过滤器决定。过滤器保存以往迭代的阶段代价、动力学间隙和不等式违反。试探点在这一偏序下改进已有记录时被接受，比较本身不带罚权重。

停机看拉格朗日梯度、间隙和不等式残差的无穷范数，基准里阈值取 $10^{-4}$。Jordana 在每个控制周期把外层迭代收到该阈值或迭代上限，当拍的 KKT 残差在施加控制之前下降。实时迭代每拍只接受一步，残差留到后续时刻，每拍计算量更小。

### 7.3 约束闭环

Solo12 的摩擦锥为 $\|F_T\|_2\le \mu F_N$，$\mu=0.8$。时域 $250$ 步、周期 $20\,\mathrm{ms}$，质心跟踪直径 $13\,\mathrm{cm}$、角速度 $0.2\,\mathrm{rad/s}$ 的圆。无约束用 $34$ 次 SQP 迭代，带锥约束用 $31$ 次。无约束的接触力穿出锥外，约束解贴在锥边界上。

KUKA 的约束实验为 $100\,\mathrm{Hz}$、$10$ 个节点、时域 $50\,\mathrm{ms}$，每拍 $4$ 次 SQP 与 $50$ 步内层迭代。关节 $q_1$ 限制在 $[-0.05,0.05]\,\mathrm{rad}$ 时，无约束末端误差 $3.3\,\mathrm{cm}$，约束下为 $8.5\,\mathrm{cm}$，约束本身被满足。末端半空间、四块半平面围成的方形区域，以及平面等式，都由预测时域提前改道；参考加快后，方形区域的角被走成圆弧。这些实验里，约束作为二次子问题的不等式进入 KKT 系统。

### 7.4 数值对照及其解释边界

Chakravorty 用摆与 Cart-pole 检查 §3 的机制：iLQR 的 $Q_{uu}$ 保持正定，步长维持在 $1$ 或 $0.1$；DDP 出现负的 $Q_{uu}$，步长落到接近 $0$，预测代价可以低于下界或远大于当前代价。近最优时 DDP 迭代更少。实验没有给 DDP 加正则。

Jordana 的基准为 KUKA（$n_x=14$）、四旋翼（$13$）、双摆（$4$）和 Humanoid 太极（$n_x=77$，$n_u=32$），各 $100$ 组随机初值与目标，KKT 阈值 $10^{-4}$。多重打靶在四旋翼和双摆上好于单次打靶。统一过滤器之后，相同迭代预算下的求解比例从 SQP、带过滤器的 FDDP 到 DDP 递减。Humanoid 上 SQP 约在 $50$ 次迭代达到 $80\%$ 的求解比例，带过滤器的 FDDP 约需要 $200$ 次。KUKA 画圆的 $500\,\mathrm{Hz}$ 闭环里，二者跟踪接近；SQP 三次迭代达到阈值，FDDP 在五次上限处仍未达到。

前一组实验解释下降方向为何依赖二次项，后一组在统一线搜索和停机之后比较打靶与前向。维度、约束和正则都不同，数字不能逐项相减。

## 8. 欧氏光滑框架之外

### 8.1 流形状态

浮动基座机器人的姿态不应作为普通欧氏向量直接相加。若 $x\in\mathcal M$，局部增量位于切空间 $\delta x\in T_x\mathcal M$，需要选定局部的加法与差分算子。本文采用

```math
x_{\mathrm{new}}
=x\boxplus\delta x
:=\operatorname{Retr}_x(\delta x),
\qquad
\delta x
=x_{\mathrm{new}}\boxminus x
:=\operatorname{Diff}(x,x_{\mathrm{new}})
=\operatorname{Retr}_x^{-1}(x_{\mathrm{new}}).
\tag{52}
```

这里 $\operatorname{Retr}_x$ 表示以 $x$ 为基点的局部回缩，$\operatorname{Retr}_x^{-1}$ 只在相应局部邻域内定义；采用指数映射时，它才是严格的黎曼对数映射 $\operatorname{Log}_x$。不同软件可能把 `difference` 或 $\boxminus$ 的两个参数按相反顺序书写，因此公式必须和具体接口约定一并使用。动力学和代价导数均相对于这些局部坐标计算。Riccati 递推仍作用在切空间上的线性--二次模型，但 rollout 用流形的 `integrate`，状态差用 `difference`。在 $SO(3)$、$SE(3)$ 上忽略这一点会引入坐标奇异性、不一致的 Hessian 和无意义的欧氏误差。Crocoddyl 与 Pinocchio 的状态抽象正是为此设计。

### 8.2 接触、混杂动力学与非光滑性

固定接触序列时，可以为每个模式建立光滑动力学，并在切换处加入 reset map。若接触时刻和接触模式也是决策变量，问题会出现互补约束、非光滑冲量和离散组合选择。常见方法包括：

- 预先给定模式序列，只优化连续轨迹；
- contact-implicit transcription，用互补条件连接间隙与接触力；
- 对互补条件作松弛、平滑或罚函数近似；
- 混合 DDP、分支搜索或分阶段 NLP。

此时经典 SQP 的光滑性和约束资格条件可能失效，Riccati 递推只能解固定局部模式下的子问题。碰撞 indicator、库仑摩擦切换和仿真器事件也不能因为使用自动微分就自动变得光滑。

### 8.3 灵敏度与可微最优控制

设参数化 NLP 的 KKT 条件写成

```math
\mathcal K(w^*(\theta),\theta)=0,
\tag{53}
```

其中 $w$ 包含原始变量和对偶变量。若活跃集固定且 $\mathcal K_w$ 非奇异，隐函数定理给出

```math
\frac{\mathrm dw^*}{\mathrm d\theta}
=-\mathcal K_w^{-1}\mathcal K_\theta.
\tag{54}
```

$\mathcal K_w$ 正是求解器已经因子分解过的 KKT 矩阵，因此反向灵敏度可以复用 Riccati 或稀疏线性求解。由此可以学习动力学、代价、终端价值和约束参数，也可以把 MPC 作为策略层嵌入端到端训练。

这项等式有明确边界：活跃集切换时解映射通常只分段光滑；NLP 多解或 KKT 矩阵奇异时导数不唯一或不存在；未收敛迭代的反向传播得到的是算法映射的导数，不一定是最优解映射的导数。因而 differentiable MPC/DDP 连接了最优控制与策略学习，却没有绕过可辨识性、分布外误差和闭环安全问题。

## 9. 导数信息、函数值优化与策略学习

数值方法不仅取决于优化变量，也取决于求解器能够查询的信息。设决策变量为 $y$。在轨迹优化中，$y$ 通常是状态--控制轨迹或消去状态后的控制序列；在策略学习中，$y$ 可以是策略参数，也可以是动作分布、轨迹分布或占用测度。

### 9.1 局部模型的信息来源

直接转录、SQP 和 DDP 需要一阶局部模型及相应的曲率近似。这些信息可以来自解析微分、自动微分、伴随法、复步微分或有限差分；BFGS、SR1 和局部系统辨识还可由迭代数据构造曲率或动力学模型。因此，解析导数是否可用与局部导数模型是否可构造是两个不同问题。

当目标写成 $J(y)=f(y)+g(y)$，且 $f$ 光滑、$g$ 具有可计算的投影或近端映射时，可以采用近端梯度、proximal SQP 或算子分裂。此类方法利用非光滑项的显式结构。若求解器只能执行候选解并取得代价、回报或约束违反，则可用信息退化为函数值或 rollout。随机方向有限差分、高斯平滑、SPSA、CEM、CMA-ES 和采样式 MPC 均以这类查询为基础。

不同信息来源可以组合使用。有限差分得到的 Jacobian 可进入结构化 SQP；零阶近端梯度用随机平滑估计 $\nabla f$，再对 $g$ 计算近端映射；采样生成的初值也可由 SQP 或 DDP 精修。求解器的分类因而应同时注明局部模型的来源及其后续更新方式。

### 9.2 控制序列与策略参数

令 $y=\mathbf u=(u_0,\ldots,u_{T-1})$。一次函数值查询对应一条候选控制序列的 rollout。Predictive Sampling、MPPI、CEM 与 CMA-ES 更新控制序列或其搜索分布，嵌入滚动时域后形成采样式 MPC。与 SQP-MPC 相比，这类方法通常以批量 rollout 代替显式动力学导数；两者在约束处理、样本开销和线性代数开销上具有不同的计算特征。

令 $y=\theta$，则函数值查询对应参数化策略 $\pi_\theta$ 的一组完整 rollout。Evolution Strategies、ARS 和参数空间 CMA-ES 直接根据累计回报更新参数，属于黑箱策略搜索。参数化策略将单条有限时域控制序列替换为可跨状态复用的反馈映射，同时引入高维参数搜索和策略诱导状态分布的变化。

### 9.3 策略梯度的信息结构

基于 rollout 的策略学习还可以利用策略和价值函数的可微结构。REINFORCE 通过似然比恒等式使用 score $\nabla_\theta\log\pi_\theta(a\mid s)$ 估计策略梯度；环境只需返回样本和回报。确定性策略梯度及 SAC 的 actor 更新使用 critic 关于动作的导数，并通过策略网络反向传播。TRPO 与 PPO 根据优势估计和概率比构造局部策略更新，REPS、MPO 与 AWR/AWAC 则采用相对熵约束或优势加权投影。

这些方法都可能使用环境 rollout，但其更新信息分别来自参数扰动、似然比、critic 导数或分布投影。比较策略优化算法时，需要同时说明优化对象、数据来源、梯度或改进量的估计方式以及更新所采用的几何。相关的函数值方法见[零阶优化笔记](../../zero-order-robotics/study-notes/zero-order-optimization-study-notes.md)，KL 正则化和分布投影见[分布空间中的策略优化](../../policy-learning/foundations/distribution-space-policy-optimization.md)。

## 10. 统一性的范围与边界

本文的统一性建立在三个等价关系上：

```math
\text{线性--二次子问题的 Bellman 反向递推（Riccati 递推）}
\equiv
\text{OCP-QP 的 Schur 补}
\equiv
\text{时间带状 KKT 系统的结构化分解},
\tag{55}
```

```math
\text{iLQR、DDP、GNMS、SQP}
=
\text{转录}+\text{曲率}+\text{rollout}+\text{globalization},
\tag{56}
```

```math
\text{轨迹优化器}
=
\text{NLP 外层}+\text{OCP-QP 后端}+\text{实时调度}.
\tag{57}
```

式 (55) 的等价只针对同一个无约束或等式约束线性--二次子问题：Bellman 最小化中的 Schur 补产生 Riccati 递推，按时间排列的 KKT 系统则通过块消元得到同一组解。一般非线性 Bellman 方程、带任意不等式的动态规划和一次 KKT 分解并不具有这种直接等价。这套视角也没有把 Pontryagin 极大值原理、Hamilton--Jacobi--Bellman 方程、直接法和间接法宣称为无条件相同。

## 11. 开放问题

1. 非线性前向与多重打靶并用时，尚无与线搜索 SQP 平行的全局收敛证明。
2. Riccati 已经给出反馈增益，约束闭环实验仍只施加开环的第一段控制，增益的稳定性没有在同一实验里检验。
3. 由代价 Hessian 改用完整 Hessian 的时机还没有定量准则。
4. 每拍收到 KKT 阈值，与每拍只做一步实时迭代，相应的跟踪误差和频率还缺少在同一约束问题上的定量比较。
5. 内层 ADMM 的中间迭代可以违反不等式。若要把它用作安全控制的证书，需要单独的可行性论证。

以上问题主要来自所综述论文。放到全文的统一框架中，还需要继续研究离散误差与优化误差的联合控制、混杂接触下的收敛性、近似 QP 解对闭环性能的影响、可微求解器在活跃集切换处的广义导数，以及学习模型误差下的递归可行性与安全保证。

## 12. 结论与阅读路线

本文把最优控制数值算法分解为转录、局部曲率、结构化线性代数、约束处理、前向更新、全局化和实时调度七个层次。具体算法是这些选择的组合；真正可以严格等同的是同一线性--二次子问题的不同求解表述，而不是所有最优控制理论之间的无条件等价。

第一次阅读可按 Kelly 的直接配点教程、Tassa 等的 iLQR/DDP、Giftthaler 等的 GNMS、Mastalli 等的 FDDP、Jordana 等的结构化 SQP、acados/HPIPM 的工程实现依次推进。随后可按信息条件分流：显式非光滑结构读近端点、Douglas--Rachford/ADMM 与增广拉格朗日；仅有 rollout 时转入仓库的[零阶优化笔记](../../zero-order-robotics/study-notes/zero-order-optimization-study-notes.md)和[零阶最优控制扩展](../../zero-order-robotics/study-notes/zero-order-optimal-control-extension.md)；优化对象由控制序列转为策略或动作分布后，再读[分布空间中的策略优化](../../policy-learning/foundations/distribution-space-policy-optimization.md)。这条阅读线连接了数值最优控制、黑箱 MPC 与策略学习，但保留了导数 oracle、score-function 梯度和纯参数黑箱搜索之间的区别。

## 参考文献

- M. Kelly (2017). An introduction to trajectory optimization: How to do your own direct collocation. *SIAM Review*, 59(4), 849--904. 配套代码：[OptimTraj](https://github.com/MatthewPeterKelly/OptimTraj)。
- D. H. Jacobson and D. Q. Mayne (1970). *Differential Dynamic Programming*. Elsevier.
- W. Li and E. Todorov (2004). Iterative linear quadratic regulator design for nonlinear biological movement systems. *ICINCO*.
- Y. Tassa, T. Erez, and E. Todorov (2012). Synthesis and stabilization of complex behaviors through online trajectory optimization. *IROS*.
- M. Giftthaler et al. (2018). A family of iterative Gauss--Newton shooting methods for nonlinear optimal control. *IROS*. [arXiv:1711.11006](https://arxiv.org/abs/1711.11006)
- C. Mastalli et al. (2022). A feasibility-driven approach to control-limited DDP. *Autonomous Robots*. [arXiv:2010.00411](https://arxiv.org/abs/2010.00411)
- T. Howell, B. Jackson, and Z. Manchester (2019). ALTRO: A fast solver for constrained trajectory optimization. *IROS*. 配套代码：[Altro.jl](https://github.com/RoboticExplorationLab/Altro.jl)。
- A. Jordana et al. (2023). Stagewise implementations of sequential quadratic programming for model-predictive control.
- A. Jordana et al. (2025). Structure-exploiting sequential quadratic programming for model-predictive control. *IEEE Transactions on Robotics*.
- Abhijeet and S. Chakravorty (2025). A sequential quadratic programming perspective on optimal control. [arXiv:2510.03475](https://arxiv.org/abs/2510.03475)
- H. G. Bock and K.-J. Plitt (1984). A multiple shooting algorithm for direct solution of optimal control problems.
- J. Nocedal and S. J. Wright (1999). *Numerical Optimization*.
- R. T. Rockafellar (1976). Monotone operators and the proximal point algorithm. *SIAM Journal on Control and Optimization*, 14(5), 877--898.
- P. L. Combettes and J.-C. Pesquet (2011). Proximal splitting methods in signal processing. In *Fixed-Point Algorithms for Inverse Problems in Science and Engineering*.
- J. Eckstein and D. P. Bertsekas (1992). On the Douglas--Rachford splitting method and the proximal point algorithm for maximal monotone operators. *Mathematical Programming*, 55, 293--318.
- J. F. A. D. O. Pantoja (1988). Differential dynamic programming and Newton's method. *International Journal of Control*.
- J. Sousa-Pinto and D. Orban (2024). Primal-dual iLQR. arXiv:2403.00748.
- R. Verschueren et al. (2022). acados: A modular open-source framework for fast embedded optimal control. *Mathematical Programming Computation*. 代码：[acados](https://github.com/acados/acados)。
- G. Frison and M. Diehl (2020). HPIPM: A high-performance quadratic programming framework for model predictive control. *IFAC-PapersOnLine*. 代码：[HPIPM](https://github.com/giaf/hpipm)。
- T. Dinev et al. (2022). Differentiable optimal control via differential dynamic programming. [arXiv:2209.01117](https://arxiv.org/abs/2209.01117)
- C. Mastalli et al. (2020). Crocoddyl: An efficient and versatile framework for multi-contact optimal control. *ICRA*. 代码：[Crocoddyl](https://github.com/loco-3d/crocoddyl)。

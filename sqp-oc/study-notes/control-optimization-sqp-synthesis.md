# 序列二次规划与有限时域最优控制

> 文献：Jordana 等, *Stagewise Implementations of SQP for MPC*（2023）；Jordana 等, *Structure-Exploiting SQP for MPC*（*IEEE TRO*, 2025）及其补充材料；Abhijeet 与 Chakravorty, *A Sequential Quadratic Programming Perspective on Optimal Control*（[arXiv:2510.03475](https://arxiv.org/abs/2510.03475)）。显示公式按出现顺序编号。

## 摘要

离散有限时域最优控制是带动力学等式和路径不等式的非线性规划，模型预测控制则在每个采样时刻滚动求解该问题。iLQR、DDP、GNMS 与 Jordana 的 SQP 解的是同一类二次子问题：差别在于二次项用代价 Hessian 还是拉格朗日 Hessian，状态是由动力学消去还是与控制一并优化，以及接受步长后用线性化动力学更新还是重新积分非线性模型。代价 Hessian 在当前轨迹可行且正定时给出目标的下降方向；DDP 的反向递推只在名义控制已是最优反馈时与 Bellman 方程的二阶展开一致。无不等式时，该二次子问题的 KKT 系统为块三对角，Thomas 消元即 Riccati 递推。

## 各章关系

| 章 | 内容 | 承接 |
|----|------|------|
| 1 | 有限时域非线性规划，以及单次打靶与多重打靶 | 给出后文求解的问题 |
| 2 | 两个二次子问题在动力学上的形式；不等式进入同一规划 | 把第1章的规划写成迭代中求解的二次子问题 |
| 3 | 值函数推导，以及微分动态规划、预测下降与线搜索 | 第2章给出子问题之后，导出零间隙下的控制增量与值函数递推，并比较微分动态规划与两个子问题 |
| 4 | 间隙不为零时的消元，及其与块三对角 KKT 系统的关系 | 把第3章的零间隙消元推广到一般间隙 |
| 5 | 不等式的内层求解，以及外层过滤与停机 | §2.4 写出带不等式的子问题，这里讨论如何求解和接受步 |
| 6 | 按二次项、打靶方式和前向更新对应常见算法 | 前述差别都已出现之后，命名这些算法 |
| 7 | 数值对照 | 用实验检查第3章的下降方向，以及第5章的打靶与前向 |
| 8 | 尚未解决的问题 | 列出上文没有闭合的问题 |

## 1. 问题

### 1.1 多重打靶

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
\ell_k-\lambda_{k+1}^\top\bigl(x_{k+1}-f_k(x_k,u_k)\bigr)-\mu_k^\top c_k
\Bigr].
\tag{4}
```

$\lambda_{k+1}$ 是间隙约束的乘子。后文 DDP 的协态与这一乘子在最优轨迹上对齐；离开最优轨迹时，二者的取法不同，见 §3。

### 1.2 单次打靶

Chakravorty 为陈述下降性，把状态用动力学消去，只优化开环序列：

```math
\min_{\{u_t\}}\ \sum_{t=0}^{T-1}c(x_t,u_t)+C_T(x_T),
\qquad
x_{t+1}=f(x_t,u_t),\quad x_0\ \text{给定}.
\tag{5}
```

两条结构假设贯穿其命题。阶段代价分离为 $c(x,u)=\ell(x)+\frac12 u^\top R u$，其中 $R\succ 0$、$\ell_{xx}\succeq 0$。动力学对控制仿射，$x_{t+1}=\bar f(x_t)+\bar g(x_t)u_t$。基础命题还假定当前迭代已经可行，即间隙为零；步长接受之后用非线性模型重新积分，以保持这一可行性。

无不等式时，单次打靶是无约束非线性规划，多重打靶仍是等式约束问题。前者的 Hessian 经整条轨迹复合后一般稠密；后者的 KKT 矩阵保持块三对角。不等式在多重打靶中进入二次子问题；单次打靶一侧的实现多依赖罚函数或增广拉格朗日。

## 2. 二次子问题

### 2.1 等式约束上的两个子问题

先看一般等式约束问题 $\min_x c(x)$，约束 $h(x)=0$。拉格朗日函数 $\mathcal L=c+\lambda^\top h$。一阶必要条件是

```math
F(x,\lambda)
=\begin{bmatrix}c_x+h_x^\top\lambda\\ h\end{bmatrix}
=0.
\tag{6}
```

$F$ 对 $(x,\lambda)$ 非线性。在当前点 $(\bar x,\bar\lambda)$ 作 Newton 线性化，步 $(\delta x,\delta\lambda)$ 由 $DF(\bar x,\bar\lambda)\,(\delta x,\delta\lambda)=-F(\bar x,\bar\lambda)$ 确定：

```math
\begin{bmatrix}
c_{xx}+\bar\lambda\cdot h_{xx} & h_x^\top \\
h_x & 0
\end{bmatrix}
\begin{bmatrix}\delta x\\ \delta\lambda\end{bmatrix}
=
-\begin{bmatrix}c_x+\bar\lambda\, h_x\\ \bar h\end{bmatrix}.
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

右端为负，当且仅当 $c_{xx}+\bar\lambda\cdot h_{xx}$ 在切空间 $\{\delta x:h_x\delta x=0\}$ 上正定。$\bar\lambda$ 偏离最优乘子时该二次型可以不定。

原问题的解由式 (6) 决定。$\delta x=0$ 时，两种子问题接受同一个点。最优乘子一般不为零，$\bar\lambda\cdot h_{xx}$ 在解上仍然存在，第二种子问题在该邻域没有二次收敛。仅当 $\bar\lambda$ 已接近最优乘子、且拉格朗日 Hessian 在切空间上正定时，恢复这一项才加快局部收敛。Nganga（2023）在人形机器人上看到，精确 Hessian 的计算时间可以被外层迭代的减少抵消。Jordana 的 Remark 5 据此说明，第一种子问题也可以从第一次迭代使用。

### 2.3 动力学约束上的两个子问题

式 (8) 与式 (9) 的变量在这里是整条 $(\delta x,\delta u)$。等式约束取式 (3)，写成 $h_{k+1}=x_{k+1}-f_k=-\gamma_{k+1}$。无不等式时，式 (4) 去掉不等式乘子，为 $\sum_{k=0}^{T-1}\ell_k+\ell_T+\sum_{k=0}^{T-1}\lambda_{k+1}^\top(f_k-x_{k+1})$。

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

### 2.4 不等式与高斯–牛顿近似

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

高斯–牛顿近似丢掉约束的二阶项，得到第二种子问题：

```math
Q_k=\nabla^2_{xx}\ell_k,\qquad
S_k=\nabla^2_{xu}\ell_k,\qquad
R_k=\nabla^2_{uu}\ell_k.
\tag{18}
```

子问题不再依赖乘子。无不等式时它就是多重打靶上的 iLQR，也就是 GNMS；内层由 §4 的 Riccati 闭式解出。外层更新

```math
x^{[n+1]}=x^{[n]}+\alpha\Delta x,\qquad
u^{[n+1]}=u^{[n]}+\alpha\Delta u,
\tag{19}
```

新乘子取该二次子问题的 KKT 乘子。无不等式时，这对乘子可由 Riccati 的协态直接读出。

## 3. 微分动态规划与两个子问题

### 3.1 与 Bellman 方程

Bellman 方程为

```math
V_t^*(x_t)=\min_u Q_t(x_t,u),
\qquad
Q_t(x_t,u)=c(x_t,u)+V_{t+1}^*\bigl(f(x_t,u)\bigr).
\tag{20}
```

记最优反馈为 $u_t^*(x)$。只在名义点满足 $\bar u_t=u_t^*(\bar x_t)$ 时，对 $Q_t$ 的二阶展开才是这条方程在最优轨迹上的展开。此时 $Q_u=0$，包络定理去掉最优反馈斜率对值函数梯度的贡献。

一般名义控制 $\bar u_t\neq u_t^*(\bar x_t)$ 处，展开中心是该名义控制。下一节最小化阶段代价与后一拍二次模型之和，得到控制增量与值函数递推。仅当名义控制已是最优反馈时，这个模型才等于 $V_t^*(\bar x_t+\delta x)$；展开 $V^*$ 还必须保留 $\mathrm du^*/\mathrm dx$。

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

线搜索使用 Armijo 条件，分母是一阶预测下降：

```math
\frac{J_{k+1}-J_k}{\alpha\,\delta x^\top\nabla J_k}>\sigma>0.
\tag{36}
```

iLQR 的预测与一阶模型一致，回退少；摆起实验中步长可以保持为 $1$。DDP 的二阶预测与积分得到的代价不一致，Cart-pole 上步长被压到接近 $0$。Chakravorty 的混合实验在步长崩溃处改用 iLQR 的二次项，迭代得以继续。

### 3.6 近最优与远离最优

初值已经靠近最优时，DDP 的迭代数更少：摆起为 $5$ 对 iLQR 的 $8$，Cart-pole 摆起为 $5$ 对 $17$，与 §2.2 中恢复完整 Hessian 后的局部速率一致。初值远离最优时，iLQR 与高斯–牛顿 SQP 仍有 §2.2 的下降方向；文中的 DDP 不附加正则，步长搜索会失效。

## 4. Riccati 递推

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

归纳假设是消元后的对角块仍保持动力学 KKT 的形状，只是状态上的 $Q_k$ 换成值函数 Hessian $V_k$，右端的状态分量换成 $-v_k$。把 $\bar\Gamma_{k+1}^{-1}$ 的 Schur 补算出来，需要分解的正是式 (37) 的 $H_k$，消元公式化为式 (39)–(41)。因此 Riccati 递推就是式 (43) 的 Thomas 消元，动态规划与 KKT 是同一线性系统的两种写法。正向回代对应线性 rollout。循环约化把消元分成可并行的阶段，时域因子由 $T$ 降为 $\log T$。

消元要求 $f_k$ 与 $\ell_k$ 可微。导数不可用时，同一有限时域问题只能由代价的函数值更新，式 (43) 不再形成。

## 5. 不等式与外层迭代

### 5.1 分裂

标准 OSQP 把全部线性约束写成 $Pv\in\mathcal C$。动力学若与不等式放在同一次投影里，ADMM 的中间迭代不满足线性化动力学。OSQP_OCP 把变量 $v=(\Delta x,\Delta u)$ 的定义域取为线性化动力学、二次代价和近端项，集合 $\mathcal C$ 只保留分阶段的不等式。

一轮 ADMM 因而分成四步。第一步解带 $\rho,\sigma$ 正则的扩展线性二次问题，仍用反向 Riccati 和正向回代，间隙进入前馈修正。第二步对辅助变量作超松弛。第三步把不等式残差投影回 $\mathcal C$，各阶段独立，界约束就是逐分量截断。第四步更新对偶。参数取 $\sigma=10^{-6}$、超松弛系数 $1.6$，$\rho$ 每 $25$ 步更新一次，因此反向分解不必每步重做。从动力学可行的 Riccati 解热启动，可减少 ADMM 步数。

大规模问题上 OSQP_OCP 的时间对 $T$ 与 $n_x$ 接近线性。小规模上 HPIPM 的结构内点法略快。通用 OSQP 在 Solo 与 Taichi 上比这两种结构求解器慢一个数量级以上。不等式的乘子取 ADMM 的对偶变量，与式 (42) 使用同一 Riccati 骨架。

### 5.2 过滤器与停机

试探步的步长由过滤器决定。过滤器保存以往迭代的阶段代价、动力学间隙和不等式违反。试探点在这一偏序下改进已有记录时被接受，比较本身不带罚权重。

停机看拉格朗日梯度、间隙和不等式残差的无穷范数，基准里阈值取 $10^{-4}$。Jordana 在每个控制周期把外层迭代收到该阈值或迭代上限，当拍的 KKT 残差在施加控制之前下降。实时迭代每拍只接受一步，残差留到后续时刻，每拍计算量更小。

### 5.3 约束闭环

Solo12 的摩擦锥为 $\|F_T\|_2\le \mu F_N$，$\mu=0.8$。时域 $250$ 步、周期 $20\,\mathrm{ms}$，质心跟踪直径 $13\,\mathrm{cm}$、角速度 $0.2\,\mathrm{rad/s}$ 的圆。无约束用 $34$ 次 SQP 迭代，带锥约束用 $31$ 次。无约束的接触力穿出锥外，约束解贴在锥边界上。

KUKA 的约束实验为 $100\,\mathrm{Hz}$、$10$ 个节点、时域 $50\,\mathrm{ms}$，每拍 $4$ 次 SQP 与 $50$ 步内层迭代。关节 $q_1$ 限制在 $[-0.05,0.05]\,\mathrm{rad}$ 时，无约束末端误差 $3.3\,\mathrm{cm}$，约束下为 $8.5\,\mathrm{cm}$，约束本身被满足。末端半空间、四块半平面围成的方形区域，以及平面等式，都由预测时域提前改道；参考加快后，方形区域的角被走成圆弧。这些实验里，约束作为二次子问题的不等式进入 KKT 系统。

## 6. 算法对应

表中的完整拉格朗日 Hessian 是第一种子问题，代价 Hessian 是第二种子问题。iLQR、GNMS 与 Jordana 的高斯–牛顿形式在全部外层迭代中使用第二种子问题。二次项、打靶方式和前向更新互相独立，常用算法是它们的组合。

| 算法 | 二次项 | 打靶 | 接受步之后的前向 |
|------|--------|------|------------------|
| iLQR | 代价 Hessian | 单次 | 非线性积分 |
| Newton LQR | 完整拉格朗日 Hessian | 单次 | 非线性积分或线性更新 |
| DDP | 完整形式，但乘子换成当前 $V_x$ | 单次 | 非线性积分 |
| GNMS | 代价 Hessian | 多重 | 线性间隙更新 |
| FDDP | DDP 的二次项 | 多重 | 非线性积分 |
| Jordana，高斯–牛顿 | 代价 Hessian | 多重 | 线性间隙更新，含不等式 |
| Jordana，精确 Hessian | 完整拉格朗日 Hessian | 多重 | 线性间隙更新，含不等式 |

无不等式时，若二次项都用上一轮乘子 $\bar\lambda$，DDP 的反向与精确 SQP 的 KKT 系统相同。DDP 实际把式 (14) 中的 $\bar\lambda$ 换成正在反向递推的值函数梯度 $V_x$，二次项因而离开该系统，见 §3。FDDP 在间隙闭合之后若不再重新打开，后续迭代的行为介于两种打靶之间。

线性更新 $\Delta x_{k+1}=A_k\Delta x_k+B_k\Delta u_k+\gamma_{k+1}$ 就是 SQP 的试探步，Nocedal 与 Wright 关于线搜索 SQP 的局部与全局结论可以沿用。非线性积分是 DDP 的传统前向，接受步之后的状态离开了二次模型所预测的点，预测下降与真实代价可以不一致。过滤器与停机对齐、只保留这一差别时，Humanoid 上的迭代次数见 §7。非线性前向与多重打靶并用时，尚无对应的全局收敛证明。

## 7. 数值对照

Chakravorty 用摆与 Cart-pole 检查 §3 的机制：iLQR 的 $Q_{uu}$ 保持正定，步长维持在 $1$ 或 $0.1$；DDP 出现负的 $Q_{uu}$，步长落到接近 $0$，预测代价可以低于下界或远大于当前代价。近最优时 DDP 迭代更少。实验没有给 DDP 加正则。

Jordana 的基准为 KUKA（$n_x=14$）、四旋翼（$13$）、双摆（$4$）和 Humanoid 太极（$n_x=77$，$n_u=32$），各 $100$ 组随机初值与目标，KKT 阈值 $10^{-4}$。多重打靶在四旋翼和双摆上好于单次打靶。统一过滤器之后，相同迭代预算下的求解比例从 SQP、带过滤器的 FDDP 到 DDP 递减。Humanoid 上 SQP 约在 $50$ 次迭代达到 $80\%$ 的求解比例，带过滤器的 FDDP 约需要 $200$ 次。KUKA 画圆的 $500\,\mathrm{Hz}$ 闭环里，二者跟踪接近；SQP 三次迭代达到阈值，FDDP 在五次上限处仍未达到。

前一组实验解释下降方向为何依赖二次项，后一组在统一线搜索和停机之后比较打靶与前向。维度、约束和正则都不同，数字不能逐项相减。

## 8. 尚未解决的问题

1. 非线性前向与多重打靶并用时，尚无与线搜索 SQP 平行的全局收敛证明。
2. Riccati 已经给出反馈增益，约束闭环实验仍只施加开环的第一段控制，增益的稳定性没有在同一实验里检验。
3. 由代价 Hessian 改用完整 Hessian 的时机还没有定量准则。
4. 每拍收到 KKT 阈值，与每拍只做一步实时迭代，相应的跟踪误差和频率还缺少在同一约束问题上的定量比较。
5. 内层 ADMM 的中间迭代可以违反不等式。若要把它用作安全控制的证书，需要单独的可行性论证。

## 参考文献

- Abhijeet and S. Chakravorty (2025). A sequential quadratic programming perspective on optimal control. arXiv:2510.03475.
- A. Jordana et al. (2023). Stagewise implementations of sequential quadratic programming for model-predictive control.
- A. Jordana et al. (2025). Structure-exploiting sequential quadratic programming for model-predictive control. *IEEE Transactions on Robotics*.
- H. G. Bock and K.-J. Plitt (1984). A multiple shooting algorithm for direct solution of optimal control problems.
- M. Giftthaler et al. (2018). A family of iterative Gauss–Newton shooting methods for nonlinear model predictive control. IROS.
- J. Nocedal and S. J. Wright (1999). *Numerical Optimization*.
- J. F. A. D. O. Pantoja (1988). Differential dynamic programming and Newton's method. *International Journal of Control*.
- J. Sousa-Pinto and D. Orban (2024). Primal-dual iLQR. arXiv:2403.00748.
- Y. Tassa, T. Erez, and E. Todorov (2012). Synthesis and stabilization of complex behaviors through online trajectory optimization. IROS.

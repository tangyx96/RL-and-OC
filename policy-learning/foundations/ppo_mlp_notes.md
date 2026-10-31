# CleanRL 连续动作 PPO 中的 MLP

对应实现见带注释的[`ppo_continuous_action.py`](https://github.com/tangyx96/cleanrl/blob/master/cleanrl/ppo_continuous_action.py)（`Agent`）。
算法层面见 [`ppo_notes.md`](ppo_notes.md)。本文说明该实现所采用的函数近似器。

---

## 一、MLP 原理

多层感知机（multilayer perceptron, MLP）是一类前馈映射 $f_\theta:\mathbb{R}^{d_{\mathrm{in}}}\to\mathbb{R}^{d_{\mathrm{out}}}$，由仿射变换与逐点非线性交替复合而成。给定输入 $h^{(0)}=x$，第 $\ell$ 层为

$$
z^{(\ell)}=W^{(\ell)}h^{(\ell-1)}+b^{(\ell)},
\qquad
h^{(\ell)}=\phi\bigl(z^{(\ell)}\bigr),
$$

其中 $W^{(\ell)}\in\mathbb{R}^{n_\ell\times n_{\ell-1}}$，$b^{(\ell)}\in\mathbb{R}^{n_\ell}$，$\phi$ 按坐标独立作用。信息沿层序单向传播；相邻两层之间每个输出坐标均依赖全部输入坐标，故称全连接。本实现不含卷积、循环或残差连接，隐藏层取 $\phi=\tanh$，输出层取 $\phi=\mathrm{id}$，即

$$
\mathrm{Linear}\circ\tanh\circ\mathrm{Linear}\circ\tanh\circ\mathrm{Linear}.
$$

### 1. 仿射层

映射 $z=Wh+b$ 为仿射：先以 $W$ 作线性变换，再以 $b$ 平移。$W$ 的第 $i$ 行与 $h$ 的内积给出 $z$ 的第 $i$ 个坐标，因此 $n_{\ell-1}$ 维输入被映射为 $n_\ell$ 维预激活。可训练参数集中于此；$ \phi$ 在本实现中无参数。

`nn.Linear(in, out)` 对行向量 batch 实现 $y=xW^\top+b$，$W\in\mathbb{R}^{\mathrm{out}\times\mathrm{in}}$，与列向量写法 $z=Wh+b$ 为同一类映射。层间维数须满足 $n_\ell$ 与下一层的输入维一致。

仿射映射在复合下封闭：

$$
W_2(W_1 x+b_1)+b_2=\widetilde W x+\widetilde b.
$$

因而仅由仿射层堆叠所表示的函数类与单层仿射相同，深度不增加表达能力。

### 2. 非线性；$\tanh$

$\phi$ 一般不是仿射。于是 $W_2\phi(W_1 x+b_1)+b_2$ 通常不再仿射，复合映射可以表示弯曲的水平集。若 $\phi$ 连续且非多项式，则存在充分宽的两层网络，可在紧集上一致逼近任意连续函数（Hornik, 1991 等）。该结论保证表示类足够大，并不保证给定宽度与有限样本下的优化可达该近似。增加深度则以较少参数实现层级复合。

本脚本隐藏层取

$$
\tanh z=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}}=2\sigma(2z)-1,
\qquad
\sigma(u)=(1+e^{-u})^{-1}.
$$

$\tanh$ 为奇函数，$\tanh 0=0$，值域 $(-1,1)$，且 $\tanh'(0)=1$，故在原点附近一阶等价于恒等。导数为

$$
\tanh' z=1-\tanh^{2}z\in(0,1].
$$

反向传播中该层的局部雅可比为对角阵，对角元 $1-{h'_i}^{2}$。当 $|z|\to\infty$ 时 $\tanh z\to\pm 1$ 且 $\tanh'z\to 0$，相应坐标上的梯度衰减，称为饱和。$\tanh$ 属于 $C^\infty$，输出近似零中心，与值域为 $(0,1)$ 的 $\sigma$ 相比，其后继仿射层的偏置与权重较少因输入恒正而耦合。ReLU 在正半轴导数恒为 $1$，仅负半轴截断，故更常见于深层卷积；在浅层、输入已标准化的向量任务上，$\tanh$ 仍为 Mujoco 连续控制中的惯用选择。

输出层保持线性。状态价值 $V(s)$ 与策略均值 $\mu(s)$ 因此可取任意实数值：$V$ 须匹配回报的尺度，$\mu$ 亦不必事先落在动作盒 $[-1,1]$ 内。环境对执行动作施加 `ClipAction`，即对已采样的 $a$ 作硬投影；该投影不进入 $\log\pi(a\mid s)$。这与 SAC 中常见的做法不同：后者对潜变量 $u\sim\mathcal{N}(\mu,\sigma)$ 取 $a=\tanh u$，并在对数密度中减去 $\sum_i\log(1-\tanh^2 u_i)$。本脚本按未变换的正态密度计 $\log\pi$，以包装器约束动作盒。

观察经 running 标准化后裁剪至 $[-10,10]$，再配合正交初始化，使训练初期的预激活多集中于原点附近。此处 $\tanh z\approx z$，$\tanh'\approx 1$。初始化增益 $\sqrt{2}$ 放大 $\|W\|$，用以补偿 $\tanh$ 导致的坐标方差收缩；映射 $\tanh$ 本身不变。

### 3. 宽度、深度与参数量

记第 $\ell$ 层宽度为 $n_\ell$。宽度决定该层线性泛函的维数；深度决定非线性复合的次数；参数个数约为 $\sum_\ell(n_\ell n_{\ell-1}+n_\ell)$，随宽度近似平方增长。宽度或深度不足则逼近能力受限；过大则在有限轨迹上易于过拟合，且优化更困难。隐藏层 $64\times 64$ 是连续控制文献中的经验配置，并非由逼近定理给出的最优结构。

### 4. 链式法则与反向传播

设标量损失 $L$ 通过网络输出依赖于 $\theta=\{W^{(\ell)},b^{(\ell)}\}_\ell$。复合函数的导数由链式法则给出；按层递推计算即反向传播。`loss.backward()` 实现该计算。损失的具体形式由算法规定：PPO 中为裁剪策略目标与价值误差，$x$ 为观察。

若 $L=f(g(\theta))$ 且 $f,g$ 可微，则

$$
\frac{\mathrm{d}L}{\mathrm{d}\theta}=f'(g(\theta))\,g'(\theta).
$$

多层复合对应各层导数在相应中间点处的乘积。

考虑一层 $z=Wh+b$，$h'=\tanh(z)$（逐元）。由 $\partial h'_i/\partial z_i=1-\tanh^2 z_i$ 得

$$
\frac{\partial L}{\partial z}=\frac{\partial L}{\partial h'}\odot\bigl(1-\tanh^{2}z\bigr).
$$

再由 $z_i=\sum_j W_{ij}h_j+b_i$，

$$
\frac{\partial L}{\partial W_{ij}}=\frac{\partial L}{\partial z_i}h_j,
\qquad
\frac{\partial L}{\partial b_i}=\frac{\partial L}{\partial z_i},
\qquad
\frac{\partial L}{\partial h_j}=\sum_i W_{ij}\frac{\partial L}{\partial z_i}.
$$

列向量约定下

$$
\frac{\partial L}{\partial W}=\frac{\partial L}{\partial z}\,h^\top,
\qquad
\frac{\partial L}{\partial b}=\frac{\partial L}{\partial z},
\qquad
\frac{\partial L}{\partial h}=W^\top\frac{\partial L}{\partial z}.
$$

$\partial L/\partial h$ 传入前一层并重复。输出层 $\phi=\mathrm{id}$ 时省略 $\odot(1-\tanh^2 z)$。

一维示意：$z_1=w_1 x$，$h=\tanh z_1$，$z_2=w_2 h$，$L=\tfrac12(z_2-y)^2$。则

$$
\frac{\partial L}{\partial z_2}=z_2-y,\quad
\frac{\partial L}{\partial w_2}=(z_2-y)h,\quad
\frac{\partial L}{\partial z_1}=(z_2-y)w_2(1-h^2),\quad
\frac{\partial L}{\partial w_1}=\frac{\partial L}{\partial z_1}\,x.
$$

当 $|z_1|$ 充分大时 $1-h^2\approx 0$，即使 $\partial L/\partial h$ 非零，$\partial L/\partial w_1$ 仍趋于零。Adam 沿上述梯度更新参数。

### 5. 归纳偏置

对网络而言，观察只是 $x=(x_1,\ldots,x_d)$。仿射层

$$
z_i=\sum_{j=1}^{d} W_{ij}x_j+b_i
$$

将第 $j$ 个坐标与 $W$ 的第 $j$ 列对齐；权重不携带「该维是关节还是像素」的语义。置换 $x$ 的坐标而不相应置换 $W$ 的列，一般改变 $z$。MLP 对输入置换不等变，也不内置空间局部性或时间因果。何种结构合适，取决于各维含义是否固定、相邻维是否相关。

**向量观察。** 关节角、角速度在环境接口中按固定语义排列：第 $j$ 维始终对应同一物理量。$W$ 的第 $j$ 列可稳定承担该量的贡献。真实环境不会任意对调关节次序，故 MLP 与此类已对齐的特征向量相合。本脚本即此形：

```text
x ──► Linear ──► tanh ──► Linear ──► tanh ──► Linear ──► y
      全连接               全连接               输出头
```

**图像。** 语义在平移下近似不变，展平后的欧氏向量则几乎逐坐标变化。以 $3\times 3$ 灰度为例，亮点右移一格：

$$
\begin{pmatrix}0&0&0\\0&1&0\\0&0&0\end{pmatrix}
\;\mapsto\;
\begin{pmatrix}0&1&0\\0&0&0\\0&0&0\end{pmatrix}.
$$

展平后由 $(0,0,0,0,1,0,0,0,0)$ 变为 $(0,1,0,0,0,0,0,0,0)$。对人仍是「同一亮点」；对 MLP 则几乎每个坐标都已更换。若要以全连接学会平移不变，须为每一可能位置各备一套权重，参数与样本需求随位置数增长。

卷积将两条假定写入结构：核只作用于局部窗口；同一组权重在空间上平移（权值共享）。「检测局部模式」只需学习一次，响应随模式平移。参数量由核尺寸决定，不随可能位置数线性膨胀。深层常用残差：跳跃连接服务优化，不改变平移局部性。

```text
图像 ──► [ 同一核 W 滑过各窗口 ] ──► [ 核 W' … ] ──► 展平 ──► Linear ──► y

深层（可选残差）：
        ┌──────── 跳跃 ────────┐
        │                      ▼
   h ───┴──► Conv ──► … ──► ⊕ ──►
```

**时间序列。** 决策若依赖历史，输入是 $(x_{t-k},\ldots,x_t)$ 而非单帧 $x_t$。将整段历史沿时间维拼接成一个长向量再送入 MLP，则第 $\tau$ 步被钉在固定槽位：同一事件出现在 $t-1$ 与出现在 $t-3$，对应 $W$ 的不同列。例如长度 $3$ 的标量序列，脉冲右移一步：

$$
(0,1,0)\;\mapsto\;(0,0,1).
$$

语义是「脉冲刚发生」相对「脉冲早一步」；展平后的坐标几乎全变，MLP 须为每一滞后各学一套权重。循环网络以同一转移 $(h_{\tau},x_{\tau})\mapsto h_{\tau+1}$ 沿时间迭代，将「后步依赖前步、规则不随时标改变」写入结构；注意力则以一套查询–键权重检索任意时刻，不要求把历史摊成无序的超长坐标。本脚本假定观察已近似马尔可夫，仅以当前 $s$ 为输入，故不用循环或注意力；部分可观测任务中的 LSTM-PPO 属于后一类。

```text
循环（同一 f_θ 沿时间共用）：
  x_{t-1} ──►┐              x_t ──►┐
             ├► f_θ ──► h_t ───────┤► f_θ ──► h_{t+1}
  h_{t-1} ──►┘                     ┘

注意力（各时刻并行，同一套 Q,K,V）：
  x_1, x_2, …, x_T  ──►  Q,K,V  ──►  softmax(QK^⊤) V  ──►  …
```

**表示与样本。** 充分宽的 MLP 仍可逼近「图像 $\mapsto$ 标签」或序列映射（万能逼近）。实践中轨迹有限：卷积、循环与注意力已将相应对称性作为先验，权重主要学习「检测何种模式」；MLP 尚须从数据中同时恢复邻域、平移与时间对齐规则，参数更多、样本效率通常更低。归纳偏置约束的是假设类的几何，而非该映射在存在性意义上可否被逼近。

| 观察 | 写入结构的假定 | 默认骨干 |
|------|----------------|----------|
| 固定语义的向量 | 第 $j$ 维始终为同一物理量 | MLP |
| 网格图像 | 局部相关，平移下语义近似不变 | 卷积 |
| 时间序列 | 后步依赖前步 | 循环 / 注意力 |

本脚本的观察为已对齐的向量，故以两套 MLP 分别表示 $V_\psi(s)$ 与 $\mu_\theta(s)$。更换骨干时，若仍输出 $\mu(s)$ 与 $V(s)$，GAE 与裁剪目标保持不变。

---

## 二、智能体中的函数分解

`Agent` 由两个独立 MLP 与一组与状态无关的对数标准差组成，而非共享主干再分叉。

| 模块 | 类型 | 输入 | 输出 |
|------|------|------|------|
| `critic` | MLP | 观察 $s$ | $V_\psi(s)\in\mathbb{R}$ |
| `actor_mean` | MLP | 观察 $s$ | $\mu_\theta(s)\in\mathbb{R}^{d_a}$ |
| `actor_logstd` | 参数 | — | $\log\sigma\in\mathbb{R}^{d_a}$ |

Actor 与 Critic 不共享权重。单一 Adam 优化器更新 `agent.parameters()`，即两套 MLP 与 $\log\sigma$。

条件策略取对角高斯

$$
\pi_\theta(a\mid s)=\mathcal{N}\bigl(a;\,\mu_\theta(s),\,\mathrm{diag}(\sigma^2)\bigr).
$$

均值由状态经 MLP 给出；$\sigma=\exp(\log\sigma)$ 在动作维上可学习，但不依赖 $s$。该设定沿用 OpenAI Baselines / CleanRL 的连续 PPO：探索尺度为全局参数，而非状态条件方差。

---

## 三、维数与实现

记展平后观察维为 $d_s$，动作维为 $d_a$。默认环境 `HalfCheetah-v4` 上 $d_s=17$，$d_a=6$。

```text
critic:       d_s → 64 → tanh → 64 → tanh → 1
actor_mean:   d_s → 64 → tanh → 64 → tanh → d_a
actor_logstd: (1, d_a)，初值 0，故 σ = 1
```

```python
self.critic = nn.Sequential(
    layer_init(nn.Linear(d_s, 64)),
    nn.Tanh(),
    layer_init(nn.Linear(64, 64)),
    nn.Tanh(),
    layer_init(nn.Linear(64, 1), std=1.0),
)
self.actor_mean = nn.Sequential(
    layer_init(nn.Linear(d_s, 64)),
    nn.Tanh(),
    layer_init(nn.Linear(64, 64)),
    nn.Tanh(),
    layer_init(nn.Linear(64, d_a), std=0.01),
)
self.actor_logstd = nn.Parameter(torch.zeros(1, d_a))
```

`FlattenObservation` 将字典型观察（如部分 dm_control）展为向量后再输入 MLP。

---

## 四、正交初始化

$\tanh$ 的饱和要求预激活落在导数尚未消失的区域。正交初始化与观察归一化共同约束该尺度。

```python
torch.nn.init.orthogonal_(layer.weight, std)
torch.nn.init.constant_(layer.bias, bias_const)  # 默认 0
```

**正交初始化。** `orthogonal_(W, std)` 先取满足 $W^\top W=I$ 或 $WW^\top=I$ 的矩阵（行或列单位正交，视形状较矮的一边而定），再乘增益 $\mathrm{std}$。正交使初始化处线性映射的奇异值接近 $1$，前向范数与反向梯度不易指数膨胀或衰减，从而保持前向激活与反向梯度的范数。$\tanh$ 的饱和仍要求预激活不要过大，故与观察归一化一并使用。

**偏置。** $z=Wx+b$ 中 $b$ 平移超平面。`constant_(bias, 0)` 使 $x=0$ 时 $z=0$，初始阈值中性，不对称性来自 $W$ 而非人为偏置。

增益 $\mathrm{std}$ 只缩放 $W$，不改变正交性：

| 位置 | `std` | 作用 |
|------|-------|------|
| 隐藏层 | $\sqrt{2}$（默认） | 补偿 $\tanh$ 引起的方差收缩 |
| Critic 输出 | $1.0$ | 价值初值尺度适中 |
| Actor 均值输出 | $0.01$ | 初始 $\mu\approx 0$，策略接近 $\mathcal{N}(0,I)$ |

均值头若增益过大，初期 $|\mu|$ 偏大，`ClipAction` 将样本推至动作边界，早期梯度质量下降。小增益约束的是训练起点，并不迫使收敛后的策略趋于确定性。$\log\sigma$ 初值为 $0$，即 $\sigma=1$，在已归一化的观察与裁剪动作下作为默认探索尺度。

---

## 五、前向计算

`get_action_and_value(x, action=None)` 计算

1. $\mu=\mathrm{actor\_mean}(x)$；
2. $\sigma=\exp(\mathrm{logstd})$，广播至与 $\mu$ 相同的 batch 形状；
3. 分布 $\mathrm{Normal}(\mu,\sigma)$：采集时 `action is None`，由 `sample()` 得 $a$；更新时传入轨迹中的 $a$，仅计算当前参数下的 $\log\pi(a\mid s)$；
4. $V=\mathrm{critic}(x)$。

多维对角高斯的对数密度与熵对动作维求和：

$$
\log\pi(a\mid s)=\sum_{i=1}^{d_a}\log\mathcal{N}(a_i;\mu_i,\sigma_i).
$$

采集在 `torch.no_grad()` 下进行；更新对同一批 $(s,a)$ 再求 `newlogprob` 与 `newvalue`，供裁剪目标使用。

`Normal.sample()` 不经 $\tanh$。环境侧 `ClipAction` 将执行动作投影到盒约束，故 $\log\pi$ 对应未投影样本的密度，与投影后动作的密度不必一致。此为实现上的近似，有别于 SAC 的 $\tanh$ 变换及雅可比修正。

---

## 六、与 PPO 更新的接口

网络提供 $\log\pi_\theta(a\mid s)$、$V_\psi(s)$ 与熵 $\mathcal{H}(\pi(\cdot\mid s))$。广义优势 $\hat A$ 与回报 $R$ 由 rollout 中的价值与奖励按 GAE 计算；策略损失使用新旧对数密度之比的裁剪目标；价值损失拟合 $R$（可再裁剪）。默认 `ent_coef=0`，熵进入前向返回值但不进入损失。梯度经 `clip_grad_norm_(..., 0.5)` 后由 Adam 更新全部参数。无目标网络。

---

## 七、若干区分

1. MLP 描述函数类的结构；Actor–Critic 描述策略与价值的分工。本实现中二者均为 MLP。
2. 连续动作脚本中 Actor 与 Critic 不共享隐层。共享卷积主干出现于部分像素 PPO，与此不同。
3. $\log\sigma$ 为与 $s$ 无关的参数。状态条件方差 $\sigma(s)$ 需另设输出头。
4. 隐藏宽度 $64$ 为惯例。更困难的控制任务可增加宽度、深度或层归一化。
5. 观察与奖励的 running 标准化由 `NormalizeObservation`、`NormalizeReward` 完成，网络输入为已标准化并裁剪至 $[-10,10]$ 的向量。

---

## 八、观察模态与骨干

| 观察 | 常用骨干 | CleanRL 中的对应 |
|------|----------|------------------|
| 向量（本文件） | MLP，$64$–$64$，$\tanh$ | `ppo_continuous_action.py` |
| 像素 | CNN | `ppo_atari.py` 一类 |
| 混合 | 分模态提取后拼接 | 需改写 `Agent` |

更换骨干时，若仍输出 $\mu(s)$ 与 $V(s)$，GAE 与裁剪目标保持不变。

# Vision-Language-Action（VLA）算法笔记

下文把 VLA 写成语言条件下的模仿学习问题，并比较该条件分布的几种参数化。极大似然与正向 KL 的等价见 [`diffusion_policy_notes.md`](diffusion_policy_notes.md) 第 2.2 节；连续动作块上的 DDPM 与 flow matching 见同文与 [`flow_matching_notes.md`](flow_matching_notes.md)；显式未来观测见 [`wam_notes.md`](wam_notes.md)。离散自回归参数化以 Kim 等（2024）的 [OpenVLA](https://arxiv.org/abs/2406.09246) 为可核对实例，实现见 [openvla/openvla](https://github.com/openvla/openvla)。

---

## 一、算法定位

Vision-Language-Action（VLA）是一类 visuomotor 策略：条件由图像观测与自然语言指令组成，输出为机器人动作。训练范式是离线模仿学习——演示提供样本，策略拟合专家条件分布 $p^*(\mathbf{A}\mid o,l)$，不使用环境奖励，也不维护价值函数。本质是行为克隆，但两项因素使其区别于标准 BC。其一，语言是组合性的任务说明符：训练时见过「拿起红杯子」和「推开蓝方块」，测试时「拿起蓝杯子」可由语义组合导出，而模仿学习本身不保证这种零样本泛化。其二，参数从预训练的视觉–语言模型（VLM）出发：VLM 提供视觉与语言的强先验，使数十至数百条机器人演示即可引导出合理行为，远少于从零训练的 BC 所需。简言之，训练公式是 BC，泛化的来源是语言的组合性与预训练知识。

与 [`diffusion_policy_notes.md`](diffusion_policy_notes.md) 中的 visuomotor 策略相比，VLA 的区别正在于语言条件与 VLM 初始化。动作头仍是参数化选择——离散 token 上的自回归范畴分布、连续空间中的扩散或 flow、以及对动作块的回归——优化目标同为式 (2)。

| 特性 | 高斯 / MSE 行为克隆 | Diffusion Policy | 离散 VLA | Flow VLA（π₀） |
|------|---------------------|------------------|----------|----------------|
| 学习范式 | 离线模仿 | 离线模仿 | 离线模仿 | 离线模仿 |
| 条件 | 图像或低维状态 | 图像，可含本体感觉与历史 | 图像 + 语言 | 多视角图像 + 语言 + 本体感觉 |
| 动作对象 | 单步 $\mathbf{a}$ | 动作块 $\mathbf{A}$ | 单步动作的离散码 | 动作块 $\mathbf{A}$ |
| 训练目标 | 负对数似然或平方误差 | 去噪回归 | 动作 token 交叉熵 | 条件速度场回归 |
| 初始化 | 通常随机 | 视觉编码器通常从零训练 | 预训练 VLM | 预训练 VLM + 动作专家 |

---

## 二、问题形式化

### 2.1 语言条件的 visuomotor 策略

记图像观测为 $o$，语言指令为 $l$，动作为 $\mathbf{a}\in\mathbb{R}^{D_a}$。需要预测未来 $H$ 步时，记动作块 $\mathbf{A}\in\mathbb{R}^{H\times D_a}$。专家诱导条件分布

$$
p^{\ast}(\mathbf{A}\mid o,l).
\tag{1}
$$

$l$ 与本体感觉、观测历史处于同一位置：它们都是条件，不参与被拟合的动作密度。$H=1$ 时 $\mathbf{A}$ 退化为单步 $\mathbf{a}$。OpenVLA 的默认接口取 $H=1$、$D_a=7$（末端相对位姿 6 维与夹爪 1 维），输入为当前帧的单张图像和指令。

### 2.2 行为克隆目标

设演示 $\mathcal{D}=\{(o_j,l_j,\mathbf{A}_j)\}$ 独立取自专家联合分布。策略 $\pi_\theta(\mathbf{A}\mid o,l)$ 的负对数似然为

$$
\mathcal{L}_{\mathrm{NLL}}(\theta)
=\mathbb{E}_{(o,l,\mathbf{A})\sim\mathcal{D}}
\big[-\log\pi_\theta(\mathbf{A}\mid o,l)\big].
\tag{2}
$$

由 [`diffusion_policy_notes.md`](diffusion_policy_notes.md) 第 2.2 节，对专家分布有

$$
\mathbb{E}_{(o,l)\sim p^{\ast}}
\Big[
D_{\mathrm{KL}}\big(p^{\ast}(\cdot\mid o,l)\,\big\|\,\pi_\theta(\cdot\mid o,l)\big)
\Big]
=C+\mathbb{E}_{(o,l,\mathbf{A})\sim p^{\ast}}
\big[-\log\pi_\theta(\mathbf{A}\mid o,l)\big],
$$

其中 $C$ 不依赖 $\theta$。用 $\mathcal{D}$ 的经验分布代替 $p^{\ast}$，右端第二项就是式 (2)。极大似然因此等价于在专家的条件分布上最小化正向 KL。VLA 没有在此之外另给一个最优性原理。后续各节的差别，是 $\pi_\theta$ 用哪一种密度或哪一种替代损失来实现式 (2)。

高斯策略且方差固定时，式 (2) 与最小二乘同最优，网络输出条件均值。同一观测下若存在多种合理动作，该均值会落在各模态之间。离散范畴分布与连续生成模型不把策略限制成这个单峰点估计。

---

## 三、离散动作参数化

本节给出式 (2) 的一种实现：把连续动作映成有限字母表上的序列，再令 $\pi_\theta$ 为该序列的自回归范畴分布。RT-2 与 OpenVLA 采用这一实现。映射是固定的分箱与查表，训练不更新格子边界。

### 3.1 分箱

对每个动作维，先用训练集的分位数做仿射，再在归一化坐标上分箱。记 $q_{d,0.01}$、$q_{d,0.99}$ 为第 $d$ 维的 $1\%$ 与 $99\%$ 分位数，

$$
\tilde a_d
=\mathrm{clip}\!\left(
2\cdot\frac{a_d-q_{d,0.01}}{q_{d,0.99}-q_{d,0.01}}-1,\;
-1,\;1
\right).
$$

两个分位数分别被送到 $-1$ 与 $1$，落在区间外的值被裁剪。OpenVLA 在 $[-1,1]$ 上取 $256$ 个等距边界

$$
e_j=-1+\frac{2j}{255},\qquad j=0,\ldots,255.
$$

相邻边界相距 $\Delta=2/255$，共 $255$ 个区间。$\mathrm{digitize}(\tilde a_d;\{e_j\}_{j=0}^{255})$ 给出整数 $\iota_d$：当 $e_{k-1}\le\tilde a_d<e_k$ 时 $\iota_d=k$（$k=1,\ldots,255$）；$\tilde a_d=1$ 时 $\iota_d=256$。量化映射 $Q$ 逐维取值 $\iota_d$。$Q$ 固定，同一区间内的数值对应同一个 $\iota_d$。右端点单独得到 $\iota_d=256$。

### 3.2 量化误差

第 $b$ 个区间 $[e_b,e_{b+1})$ 的中心为 $c_b=e_b+\Delta/2$，$b=0,\ldots,254$。解码先把档位映回中心下标，

$$
\hat{\tilde a}_d=c_{\mathrm{clip}(\iota_d-1,\,0,\,254)},
$$

再做仿射的逆，

$$
\hat a_d
=q_{d,0.01}+\frac{\hat{\tilde a}_d+1}{2}\,(q_{d,0.99}-q_{d,0.01}).
$$

$\iota_d=1,\ldots,255$ 分别对应 $c_0,\ldots,c_{254}$。$\iota_d=256$ 与 $\iota_d=255$ 共用最后一个中心，因此 $256$ 个档位只有 $255$ 个重建值。落在某一区间内的 $\tilde a_d$ 满足 $|\tilde a_d-\hat{\tilde a}_d|\le\Delta/2$。分位数之外的动作先被裁剪，原单位下的误差可以大于半个区间在原单位下的宽度。式 (2) 拟合的是 $Q(\mathbf{a})$ 的条件分布，区间内部的连续差异不进入似然。正向 KL 因此是对量化后的专家分布计算的。

### 3.3 动作字母表

语言模型不直接读字符串。Llama 2 有一张 $32000$ 行的片段表 $\mathcal{V}$，行号称为 token。这张表从字节造起：在英文语料里把最常相邻的两段合成一个新片段，反复合并，直到行数达到词表规模。常见词因此占单独一行，例如 `pick`；空格不单独成行，而粘在后一个词上，带前导空格的 ` pick` 与行首的 `pick` 是不同的行。

分词器用这张固定的表，从左向右切字符串，每切出表中的一段就换成它的行号。行号个数记为 $T$。`pick up the cup` 有 4 个词；四个词都在表中时 $T=4$，某个词不在表中就会被切成更短的几段，此时 $T$ 既不等于词数，也不等于字符数。该切分不随式 (2) 更新。

OpenVLA 不把任务句单独送入。任务 `pick up the cup` 先写成

```text
In: What action should the robot take to pick up the cup?
Out:
```

$w_{1:T}$ 是这两行切出的行号，止于 `Out:`，里面还没有动作。`Out:` 之后由模型逐个写出 $u_{1:D_a}$。表主要在英文上合并而成，汉字通常没有单独一行，而落回组成它的三个 UTF-8 字节，每个字节一个 token。每个行号对应 $\mathbb{R}^{4096}$ 中的嵌入。网络对这串向量做运算，每一步在 $\mathcal{V}$ 中选取下一个行号。

档位 $\iota_d\in\{1,\ldots,256\}$ 还不是行号。这套 BPE 把使用频率最低的片段排在词表末尾。RT-2 与 OpenVLA 都用这些行作为动作标签，不另扩词表。OpenVLA 覆盖末尾 $256$ 行，记为 $\mathcal{V}_{\mathrm{act}}\subset\mathcal{V}$，行号为

$$
u_d=|\mathcal{V}|-\iota_d=|\mathcal{V}|-Q(\mathbf{a})_d.
$$

$u_d$ 仍是 $\mathcal{V}$ 中的整数：$\iota_d=1$ 对应 $|\mathcal{V}|-1$，$\iota_d=256$ 对应 $|\mathcal{V}|-256$。这些行原有的文本片段不再使用，其嵌入随式 (2) 更新。这个查表不改变 $Q$ 已经做出的量化。

### 3.4 自回归似然

在离散码上，式 (2) 的密度取链式法则

$$
\pi_\theta(u_{1:D_a}\mid o,l)
=\prod_{i=1}^{D_a}\pi_\theta(u_i\mid o,l,u_{<i}).
\tag{3}
$$

代入式 (2)，

$$
\mathcal{L}_{\mathrm{NLL}}(\theta)
=\mathbb{E}\Big[-\sum_{i=1}^{D_a}\log\pi_\theta(u_i\mid o,l,u_{<i})\Big].
\tag{4}
$$

式 (3) 是 $u_{1:D_a}$ 上联合分布的链式法则。同一区间内的不同 $\mathbf{a}$ 映到同一个 $u$，区间内部的差别不进入式 (4)；这一步由 $Q$ 完成。

从式 (3) 去掉 $u_{<i}$，得到 $\prod_d\pi_\theta(u_d\mid o,l)$。这个分布只有各维边缘。OpenVLA 按末端位移、姿态、夹爪的顺序写 $u_{1:D_a}$，夹爪位于位移之后。设竖直位移取「不动 / 向下」，夹爪取「开 / 合」，专家只有（向下，合）与（不动，开）。两维边缘都为正时，独立模型会给「向下且开」正概率。式 (3) 写夹爪时以已经写出的位移 token 为条件，可以指定 $\pi(\text{合}\mid\text{向下})=1$、$\pi(\text{合}\mid\text{不动})=0$。如此写出的是档位与档位的联合；同一区间内的连续值仍由 $Q$ 等同。式 (4) 中的 $u_{<i}$ 取自专家动作经 $u_d=|\mathcal{V}|-Q(\mathbf{a})_d$ 得到的行号。

### 3.5 序列上的监督位置

图像不是 token。$224\times 224$ 的输入按 $14\times 14$ 切块，得到 $M=256$ 块。每块经视觉编码器再投影到嵌入维数，记为 $v_i\in\mathbb{R}^{4096}$。$v_i$ 没有词表行号。指令经第 3.3 节的分词得到行号 $w_{1:T}$，动作经 $u_d=|\mathcal{V}|-Q(\mathbf{a})_d$ 得到行号 $u_{1:D_a}$。行号 $k$ 在 $32000\times 4096$ 的嵌入表中对应一行 $e_k\in\mathbb{R}^{4096}$。注意力读入的序列是这些向量

$$
(v_1,\ldots,v_M,\; e_{w_1},\ldots,e_{w_T},\; e_{u_1},\ldots,e_{u_{D_a}}),
$$

图像块与文本、动作共用同一套注意力。

$\mathrm{LM}$ 即该语言模型，其中包含上述查表。预测第 $i$ 个动作 token 时，它已看见 $v_{1:M}$、$e_{w_{1:T}}$ 与 $e_{u_{<i}}$，并给 $\mathcal{V}$ 中每一个行号一个分数。这些分数组成

$$
z=\mathrm{LM}(v_{1:M},w_{1:T},u_{<i})\in\mathbb{R}^{|\mathcal{V}|}.
$$

$z_k$ 是「下一个 token 为第 $k$ 号」的分数，不要求和为 $1$。$|z|=|\mathcal{V}|$，因为这一步的候选是整个词表，而不是当前句子的长度。条件概率取正确行号上的 softmax 分量：

$$
\pi_\theta(u_i\mid o,l,u_{<i})
=\frac{\exp z_{u_i}}{\sum_{v\in\mathcal{V}}\exp z_v}.
\tag{5}
$$

分母把普通词与 $\mathcal{V}_{\mathrm{act}}$ 加在同一次归一化里。监督取 $u_i=|\mathcal{V}|-Q(\mathbf{a})_i$，故它是 $\mathcal{V}_{\mathrm{act}}$ 中的一个行号。

计算该分数时，$v_{1:M}$ 与 $e_{w_{1:T}}$ 留在注意力可见的前缀里，式 (5) 才依赖 $(o,l)$。式 (4) 只对 $u_{1:D_a}$ 求和：图像块与指令不必被再次预测，这些位置没有自己的 $-\log\pi$ 项。动作项的梯度仍沿前缀的隐藏状态回传，视觉编码器随式 (4) 更新。若冻结编码器，$v_{1:M}$ 保持为预训练网络对 $o$ 的输出，训练只更新语言模型如何读取这组向量。

### 3.6 解码

推理时按同一顺序自回归。第 $i$ 步只把已经写出的 $\hat u_{<i}$ 嵌入后接在图像与指令后面。生成在整个 $\mathcal{V}$ 上选取下一个行号，不把候选限制在 $\mathcal{V}_{\mathrm{act}}$。得到 $\hat u_{1:D_a}$ 后取 $\iota_d=|\mathcal{V}|-\hat u_d$，再按第 3.2 节读出区间中心并做仿射的逆。该映射记为 $\delta$，即 $\hat{\mathbf{a}}=\delta(\hat u_{1:D_a})$。$\hat u_d$ 不在这 $256$ 行之内时，$\iota_d$ 仍被 clip 进 $\{0,\ldots,254\}$。每个区间中心对应的输出是常数。

---

## 四、同一目标的其他参数化

式 (1) 不变。下面几种写法改变的是 $\pi_\theta$ 的样本空间和推理过程。

### 4.1 连续联合密度

Diffusion Policy 用条件去噪过程参数化 $p(\mathbf{A}\mid o)$，训练损失是噪声回归，作为式 (2) 的变分替代，推导见 [`diffusion_policy_notes.md`](diffusion_policy_notes.md) 第四节。π₀ 把同一动作块写成条件流的推前分布，用速度场回归实现，见 [`flow_matching_notes.md`](flow_matching_notes.md)。二者都不对 $\mathbf{A}$ 做分箱，联合结构直接定义在 $\mathbb{R}^{H\times D_a}$ 上。

在此参数化中，预训练 VLM 是条件编码器。π₀ 将图像与语言放在前缀，动作块的速度场由单独的 action expert 在该前缀条件下计算。动作不再占用语言模型的词表。

### 4.2 动作块与重规划

单步因子 $\pi(\mathbf{a}_t\mid o_t,l)$ 把各控制周期的动作在给定新观测后分开建模，相邻动作之间没有联合密度。将输出改为 $\mathbf{A}_t\in\mathbb{R}^{H\times D_a}$ 后，一段未来动作在同一次生成中取出，时间上的平滑来自这个联合分布。执行时通常只使用前 $T_a$ 步，然后用新观测重新规划。$T_a$ 小则闭环更勤，$T_a$ 大则更依赖开环的动作块。该接口与 [`diffusion_policy_notes.md`](diffusion_policy_notes.md) 第六节相同。

离散自回归若要覆盖同一动作块，token 数为 $H\cdot D_a$，解码步数随 $H$ 与控制频率上升。这是高频、长时域任务上离开逐维分箱的直接原因。

### 4.3 并行回归

OpenVLA-OFT 将动作头写成 $f_\theta(o,l)\in\mathbb{R}^{H\times D_a}$，各坐标在一次前向中同时读出。训练损失为

$$
\mathbb{E}\big[\lVert\mathbf{A}-f_\theta(o,l)\rVert_1\big].
$$

该损失对每个坐标的总体极小是条件中位数。策略给出的是点估计，同一 $(o,l)$ 下的多种动作收成这一中位数。整段 $\mathbf{A}$ 一次取出，似然不按式 (3) 分解。

### 4.4 压缩后的离散码

π₀-FAST 用 π₀ 的视觉–语言前缀，把式 (3) 用在动作块的频谱码上。动作块先按维做仿射，使训练集该维的 $1\%$ 与 $99\%$ 分位数分别映到 $-1$ 与 $1$。对第 $d$ 维时间序列做离散余弦变换，系数为 $C_{d,j}$，$j=0,\ldots,H-1$。量化取

$$
\bar C_{d,j}=\mathrm{round}(\gamma C_{d,j}),
$$

其中 $\gamma>0$ 固定。Pertsch 等在单数据集实验中取 $\gamma=10$，字节对编码的词表大小为 $1024$。取整后多数高频系数为 $0$。将 $\bar C$ 按频率从低到高排列，同一频率上各维相邻，得到整数序列，再经字节对编码压缩为 token $u_{1:L}$。合并表在量化后的系数序列上拟合，拟合后固定。压缩本身可逆。$L$ 随动作块变化，通常为数十，低于逐时刻分箱的 $H\cdot D_a$。

式 (3) 与式 (4) 的连乘改到 $i=1,\ldots,L$。$u_{<i}$ 取自专家经上述映射得到的前缀。连续信息的损失在 $\mathrm{round}$：系数除以 $\gamma$ 后做逆离散余弦变换，回到的是量化后的频谱。解码按逆字节对编码、除以 $\gamma$、逆变换、分位数仿射之逆的顺序进行。自回归先写低频系数，再写高频系数。

---

## 五、条件编码器

式 (5) 中的 $v_{1:M}$ 决定图像以何种特征进入条件。OpenVLA 沿用 Prismatic-7B：SigLIP 与 DINOv2 分别编码 patch，特征沿通道维拼接，经两层 MLP 投到 Llama 2 7B 的嵌入空间。输入分辨率为 $224\times 224$。SigLIP 来自图文对比预训练，DINOv2 来自自监督稠密特征。拼接之后，条件同时带有语义对齐与空间结构。

在 BridgeData V2 上，作者用同一套动作离散化比较视觉–语言骨干。单物体任务上 LLaVA 与 IDEFICS-1 接近；五个语言接地任务上，LLaVA 的绝对成功率比 IDEFICS-1 高约 $35\%$。Prismatic 在单物体任务和多物体语言接地任务上，绝对成功率再比 LLaVA 高约 $10\%$。论文将该差距归因于融合编码器的空间推理。这个比较说明条件特征 $v_{1:M}$ 会改变式 (4) 所能达到的策略，它不改变式 (4) 本身。

Patch-as-token 把视觉特征排进与文本相同的序列。交叉注意力式 VLM 用另一套融合模块读取图像；在式 (5) 的写法里，图像就是前缀的一部分。

---

## 六、边界

把上述参数化放在一起，VLA 的边界可以写成几条对式 (1) 的限制。

1. **量化。** 第 3 节在归一化坐标的每个区间内输出常数，区间宽度为 $\Delta=2/255$。右端点与最后一个区间共用一个重建值。
2. **时间联合。** $H=1$ 的离散接口不约束相邻控制周期的动作联合分布。动作块、重规划与时间平滑见第 4.2 节。
3. **观测。** 条件里可以只有当前图像和指令。本体感觉、多视角与历史是条件变量的扩充，不改变式 (2)。
4. **解码长度。** 式 (3) 的采样步数等于动作 token 数。并行回归与压缩码分别从并行化和缩短码长处理这一问题。
5. **图文预训练。** 只在机器人数据上最小化式 (2) 时，图文下一词目标不再进入训练。是否与互联网图文联合训练，OpenVLA 留为未决。
6. **转移。** 式 (1) 的条件是当前 $(o,l)$。未来视觉进入生成模型的写法见 [`wam_notes.md`](wam_notes.md)。

---

## 参考文献

- Brohan, A., et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. [arXiv:2307.15818](https://arxiv.org/abs/2307.15818).
- Kim, M. J., Pertsch, K., Karamcheti, S., et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. [arXiv:2406.09246](https://arxiv.org/abs/2406.09246).
- Kim, M. J., et al. (2025). Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success. [arXiv:2502.19645](https://arxiv.org/abs/2502.19645).
- Pertsch, K., et al. (2025). FAST: Efficient Action Tokenization for Vision-Language-Action Models. [arXiv:2501.09747](https://arxiv.org/abs/2501.09747).
- Black, K., et al. (2024). π₀: A Vision-Language-Action Flow Model for General Robot Control. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164).
- Karamcheti, S., et al. (2024). Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models. [arXiv:2402.07865](https://arxiv.org/abs/2402.07865).
- Chi, C., et al. (2023). Diffusion Policy. RSS. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137).
- Open X-Embodiment Collaboration (2023). Open X-Embodiment: Robotic Learning Datasets and RT-X Models. [arXiv:2310.08864](https://arxiv.org/abs/2310.08864).
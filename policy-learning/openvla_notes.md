# Vision-Language-Action（VLA）与 OpenVLA 算法笔记

下文以 Kim 等（2024）的 [OpenVLA](https://arxiv.org/abs/2406.09246) 为主线，梳理将视觉–语言模型直接微调为 visuomotor 策略的形式化与训练目标。官方实现：[openvla/openvla](https://github.com/openvla/openvla)。连续动作上的生成式模仿见 [`diffusion_policy_notes.md`](diffusion_policy_notes.md)（DDPM）与 [`flow_matching_notes.md`](flow_matching_notes.md)（CFM / π₀），视频联合建模见 [`wam_notes.md`](wam_notes.md)。

---

## 一、算法定位

Vision-Language-Action（VLA）模型将预训练的视觉–语言模型（VLM）直接微调为机器人策略：输入图像观测与语言指令，输出可执行的控制量。动作写入语言模型词表，与下一词预测共用同一自回归目标。

OpenVLA 是该路线的开源实例：7B 参数，基于 Prismatic-7B VLM（SigLIP + DINOv2 视觉编码、Llama 2 骨干），在 Open X-Embodiment 约 $9.7\times 10^{5}$ 条操作轨迹上微调。相对 RT-2 等闭源 VLA，其贡献在于公开架构、数据配比与训练流程，并系统考察向新本体、新任务的参数高效微调。

同属 Open X-Embodiment 上的通用策略，Octo 走另一条路：Transformer 在机器人数据上从零训练，动作由独立的扩散头输出，不经过预训练 LLM。Diffusion Policy 则是单任务 visuomotor 扩散，通常不依赖大规模跨本体预训练。

| 特性 | Octo | Diffusion Policy | OpenVLA |
|------|------|------------------|---------|
| 骨干 | Transformer，机器人数据上从零训练 | 视觉编码器 + 扩散 | 预训练 VLM 微调 |
| 训练目标 | 动作扩散 | 动作序列去噪 | 动作 token 交叉熵 |
| 动作表示 | 连续扩散 | 连续扩散 | 各维独立分箱后的范畴分布 |
| 语言先验 | 有限（指令编码器） | 通常外接或无 | Internet 规模 VLM |
| 推理 | 少量去噪步 | $K$ 步去噪 | 自回归解码 $N$ 个动作 token |

OpenVLA 仍是专家条件似然（行为克隆），但把连续动作映入 LLM 词表，从而继承 VLM 的语义与空间表征。相对 Diffusion Policy 与 Octo 的连续扩散头，离散分箱引入量化误差，换取与 LLM 训练栈的兼容。

---

## 二、问题形式化

### 2.1 条件策略

记图像观测 $o$、语言指令 $l$、连续动作 $\mathbf{a}\in\mathbb{R}^{N}$（OpenVLA 默认 $N=7$：末端相对位姿与夹爪）。目标是专家条件

$$
p^{\ast}(\mathbf{a}\mid o,l).
$$

标准 visuomotor 模仿直接参数化 $\pi_{\theta}(\mathbf{a}\mid o,l)$。VLA 先将 $\mathbf{a}$ 映为离散序列 $u_{1:N}$，再在 VLM 上建模

$$
p_{\theta}(u_{1:N}\mid o,l)
=\prod_{i=1}^{N}p_{\theta}(u_{i}\mid o,l,u_{<i}).
\tag{1}
$$

解码 $\delta$ 把 token 还原为连续动作，$\mathbf{a}=\delta(u_{1:N})$。策略接口在部署时仍是 $\mathbf{a}=\delta(\hat u_{1:N})$，$\hat u$ 由自回归采样或贪心解码得到。

### 2.2 与行为克隆的关系

对演示 $\mathcal{D}=\{(o_j,l_j,\mathbf{a}_j)\}$，行为克隆最小化负对数似然（见 [`diffusion_policy_notes.md`](diffusion_policy_notes.md) 第 2.2 节）。将 $\mathbf{a}$ 换为 $u_{1:N}$ 后，

$$
\mathcal{L}_{\mathrm{NLL}}(\theta)
=\mathbb{E}_{(o,l,u_{1:N})\sim\mathcal{D}}
\Big[-\sum_{i=1}^{N}\log p_{\theta}(u_{i}\mid o,l,u_{<i})\Big].
\tag{2}
$$

这与 LLM 的下一词预测同一形式，仅监督动作位置上的交叉熵，提示中的图像与指令 token 不计入损失。极大似然、经验平均与正向 KL 的推导与连续 BC 相同，因子分解由自回归范畴分布承担。

---

## 三、动作离散化

### 3.1 分箱

对每一动作维 $d=1,\ldots,N$，取训练集该维的 $1\%$ 与 $99\%$ 分位数 $q_{d,0.01}$、$q_{d,0.99}$，在区间上均匀划分 $B=256$ 个箱。用分位数而非最小–最大，是为避免离群点拉宽区间、降低有效分辨率。将 $a_d$ 裁剪到该区间后，

$$
b_d=\mathrm{digitize}(a_d;\,\{q_{d,0.01}+j\Delta_d\}_{j=0}^{B}),
\qquad
\Delta_d=\frac{q_{d,0.99}-q_{d,0.01}}{B},
$$

得到 $b_d\in\{0,\ldots,B-1\}$（实现中 digitize 的边界约定需将越界索引夹到有效箱）。$N$ 维动作对应长度 $N$ 的整数序列。

### 3.2 词表覆盖

Llama tokenizer 预留的 special token 不足 $256$。OpenVLA 沿用 RT-2：用词表末尾 $256$ 个最低频 token 覆盖动作箱，建立双射

$$
\phi:\{0,\ldots,255\}\to\mathcal{V}_{\mathrm{act}}\subset\mathcal{V}_{\mathrm{Llama}}.
$$

于是 $u_d=\phi(b_d)$。推理时 $\phi^{-1}$ 还原箱索引，再取箱中心得到连续控制量。量化误差来自箱宽 $\Delta_d$，是离散 VLA 相对扩散策略的精度代价。

### 3.3 自回归分解的含义

式 (1) 对各维依次条件化：预测第 $d$ 维时可见 $o$、$l$ 及已生成的 $u_{<d}$。维间依赖由 LLM 隐状态传递，而非独立因子 $\prod_d p(u_d\mid o,l)$。这与将 $\mathbf{a}$ 视为对角高斯或各维独立 MSE 不同；与 Diffusion Policy 的联合连续密度也不同：后者一次生成整段动作，OpenVLA 一次生成当前步的 $N$ 个 token（默认无 action chunk）。

---

## 四、VLM 骨干作为策略网络

### 4.1 三件套

当代 VLM（LLaVA、Prismatic 等）由三部分组成：

1. **视觉编码器** $\mathrm{Enc}$：图像切分为 patch，输出 $M$ 个视觉向量；
2. **投影** $W$：将视觉向量映入 LLM 输入空间；
3. **语言模型** $\mathrm{LM}$：自回归预测下一 token。

OpenVLA 取 Prismatic-7B：视觉约 600M，投影为两层 MLP，骨干为 Llama 2 7B。输入图像 $224\times 224$。训练时视觉编码器**不解冻限制**：与多数 VLM 冻结视觉骨干相反，VLA 需适配精细空间结构，冻结会导致控制精度不足。

### 4.2 Patch-as-token

视觉 patch 经投影后与指令 token 拼接，作为 LLM 的前缀。记视觉前缀为 $v_{1:M}$、指令为 $w_{1:T}$，则

$$
p_{\theta}(u_i\mid o,l,u_{<i})
=\mathrm{softmax}\big(\mathrm{LM}(v_{1:M},w_{1:T},u_{<i})\big)_{u_i}.
$$

无需为机器人另设交叉注意力模块，从而直接复用 LLM 的分布式训练栈（混合精度、FlashAttention、FSDP）。

### 4.3 融合视觉编码

图像分别通过 SigLIP 与 DINOv2，特征在通道维拼接。SigLIP 提供语义对齐，DINOv2 提供无监督空间结构；对操作中的位姿与接触，后者往往更关键。BridgeData 上的骨干消融：相对 LLaVA / IDEFICS，Prismatic 在多物体语言接地任务上成功率更高，作者将其归因于该融合。

---

## 五、训练

### 5.1 目标

对每个样本构造序列 $(v_{1:M},w_{1:T},u_{1:N})$，最小化式 (2)。实现上为标准交叉熵，mask 掉非动作位置。优化器沿用 VLM 预训练学习率 $2\times 10^{-5}$，无 warmup。

### 5.2 数据

从 Open X-Embodiment 筛选：第三人称相机、单臂末端控制，以保证输入–输出空间一致。配比沿用 Octo 的启发式：上调场景与任务多样的子集，下调重复子集。最终约 $9.7\times 10^{5}$ 条轨迹。DROID 曾以 $10\%$ 权重加入，动作 token 准确率始终偏低，训练后三分之一阶段移除，以免拖累整体拟合。

与 LLM 通常不足两个 epoch 不同，VLA 需反复遍历：实机表现随动作 token 训练准确率上升，直至超过约 $95\%$。最终约 $27$ 个 epoch。直观上，机器人数据相对 Internet 文本更少、更偏分布，需更多轮次才能把控制映射写入词表覆盖的那 $256$ 个 token。

### 5.3 计算

$64$ 张 A100，batch $2048$，约 $14$ 天（约 $2.15\times 10^{4}$ A100-hour）。推理 bfloat16 约 $15$ GB，RTX 4090 上约 $6$ Hz（无编译与投机解码）。单图、单步相对动作，无本体感觉历史，也无 action chunk。

---

## 六、向新任务微调

### 6.1 全量微调

在新本体、新任务上用 $10$–$150$ 条演示全量更新。相对从零训练的 Diffusion Policy：窄任务、单指令上扩散策略更平滑；多物体、需语言接地时 OpenVLA 与 Octo 更强。相对从 Prismatic 直接在目标数据上训（无 OpenX 预训练），OpenX 预训练提高多样指令任务的适应。OpenVLA 是唯一在所测任务上成功率均不低于 $50\%$ 的方法。

### 6.2 LoRA

对线性层引入低秩增量 $\Delta W=BA$，$B\in\mathbb{R}^{d\times r}$，$A\in\mathbb{R}^{r\times k}$，$r\ll\min(d,k)$。可训练参数约为全量的 $1.4\%$（$r=32$）。实验中 LoRA 与全量微调成功率接近，冻结视觉或只训末层则明显下降，说明目标场景仍需更新视觉特征。$r=32$ 与 $r=64$ 差异可忽略。单卡 A100 约 $10$–$15$ 小时，相对全量约 $8$ 倍算力下降。

### 6.3 量化推理

bfloat16 为默认。int4 在 Bridge 代表任务上与 bfloat16 成功率接近，显存约减半；int8 因量化开销降低频率（评测 GPU 上约 $1.2$ Hz），与采集时 $5$ Hz 非阻塞控制不匹配，成功率下降。离线 token 准确率在 4/8 bit 上仍接近，故该跌幅主要来自动力学失配而非表示崩溃。

---

## 七、实验要点

**开箱跨本体。** WidowX（BridgeData V2）与 Google Robot 上，OpenVLA 总体优于 RT-1-X、Octo，并在多数类别上达到或超过 55B 的 RT-2-X。语义泛化（未见物体与指令）上 RT-2-X 仍略强，与其更大规模 Internet 共训、而非仅在机器人数据上微调有关。数据量（$97$ 万对 $35$ 万）、清洗（如去掉 Bridge 中的全零动作）与融合视觉编码是作者给出的主要解释。

**与 Diffusion Policy。** 输入–输出对齐后（单图、无本体、单步相对动作），扩散策略在窄技能上仍更稳；OpenVLA 在多指令、干扰物场景占优。论文指出：引入 action chunk 与时间平滑可能缩小灵巧度差距——后续 OpenVLA-OFT 与 $\pi_0$ 正是沿连续动作 / chunk 这一方向。

---

## 八、局限与后续路线

OpenVLA 的离散自回归接口带来与 LLM 工具链的兼容，也带来量化误差、逐步解码延迟、以及单步动作缺乏时序平滑。原文已指出：仅单图、无本体与历史；频率难以支撑 ALOHA 一类 $50$ Hz 双臂；未与 Internet 图文共训以保持语义。

后续开源 VLA 多在动作参数化上离开纯离散 token：$\pi_0$ 用 flow matching 生成连续 chunk；OpenVLA-OFT 改为并行解码与 L1 回归。二者仍建立在「VLM 前缀 + 动作头」之上，OpenVLA 提供的是这一接口的离散、可推导原型。

---

## 参考文献

- Kim, M. J., Pertsch, K., Karamcheti, S., et al. (2024). OpenVLA: An Open-Source Vision-Language-Action Model. [arXiv:2406.09246](https://arxiv.org/abs/2406.09246).
- Brohan, A., et al. (2023). RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. [arXiv:2307.15818](https://arxiv.org/abs/2307.15818).
- Karamcheti, S., et al. (2024). Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models. [arXiv:2402.07865](https://arxiv.org/abs/2402.07865).
- Open X-Embodiment Collaboration (2023). Open X-Embodiment: Robotic Learning Datasets and RT-X Models. [arXiv:2310.08864](https://arxiv.org/abs/2310.08864).
- Chi, C., et al. (2023). Diffusion Policy. RSS. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137).

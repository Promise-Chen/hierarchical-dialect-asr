# 分层方言自适应 ASR 架构总结

## 1. 整体思想

本文方法面向**已知方言类别的 ASR 场景**，核心思想是对方言差异进行由粗到细的分层建模：

$$
\boxed{
\text{全局方言校准}
\rightarrow
\text{跨方言共享}
\rightarrow
\text{方言内个性化残差}
}
$$

整体由三部分组成：

1. **全局模块（Global Dialect Calibration）**
   - 建模方言层面的整体分布偏移；
   - 采用低秩 FiLM 风格的条件调制；
   - 负责粗粒度、整句级的方言校准。

2. **方言簇模块（Latent Dialect Cluster Adaptation）**
   - 不人工指定方言归属，而是让网络自动学习方言之间的潜在共享关系；
   - 采用可学习方言表示、簇原型、Top-2 路由和多个 Cluster Adapter；
   - 负责建模多个方言之间可共享的中等粒度发音和音系模式。

3. **单方言动态残差模块（Dynamic Dialect Residual Adaptation）**
   - 在全局和方言簇共享之后，进一步建模单个方言仍未被解释的细粒度差异；
   - 每个方言拥有低秩参数残差；
   - 当前语音与方言标签共同生成 Dynamic Gate，动态控制该方言残差在当前语句中的使用强度。

整体可以抽象表示为：

$$
H
\rightarrow
H^G
\rightarrow
H^C
\rightarrow
H^D
$$

其中：

- $H$：原始 Encoder 表示；
- $H^G$：经过全局方言校准后的表示；
- $H^C$：经过方言簇共享适配后的表示；
- $H^D$：进一步加入单方言动态残差后的表示。

---

# 2. 第一部分：全局方言校准

## 2.1 设计目的

全局模块用于建模一个方言在整个语句范围内较稳定的整体差异，例如：

- 全局声学分布偏移；
- 隐藏通道重要性的变化；
- 音高、韵律、语速等整体倾向；
- 粗粒度频谱或发音分布差异。

这一模块不负责具体音素级变化，也不负责单个方言的细粒度独有特征。

它回答的问题是：

> 已知当前语句属于方言 $d$，共享 Paraformer Encoder 的隐藏表示应该整体如何校准？

因此，同一句话的所有时间帧共享相同的方言调制参数。

---

## 2.2 低秩 FiLM 参数生成

假设 Paraformer Encoder hidden size 为：

$$
D=512
$$

对于每个方言 $d$，直接学习两组低维系数：

$$
c_d^\gamma\in\mathbb{R}^{r_G}
$$

$$
c_d^\beta\in\mathbb{R}^{r_G}
$$

其中：

$$
r_G\ll D
$$

第一版可以设置：

$$
r_G=16
$$

同时学习两组所有方言共享的调制基底：

$$
U^\gamma\in\mathbb{R}^{r_G\times D}
$$

$$
U^\beta\in\mathbb{R}^{r_G\times D}
$$

完整的 FiLM 参数由低维系数与共享基底组合得到：

$$
\gamma_d
=
c_d^\gamma U^\gamma
$$

$$
\beta_d
=
c_d^\beta U^\beta
$$

展开后：

$$
\gamma_d
=
\sum_{j=1}^{r_G}
c_{d,j}^\gamma u_j^\gamma
$$

$$
\beta_d
=
\sum_{j=1}^{r_G}
c_{d,j}^\beta u_j^\beta
$$

因此，全局模块的核心结构为：

$$
\boxed{
\text{方言特定低维系数}
+
\text{所有方言共享的高维调制基底}
}
$$

也就是说，不允许每个方言自由学习完整的 512 维调制方向，而是让所有方言共享少量基础调制方向，每个方言只决定如何组合这些方向。

---

## 2.3 全局模块前向传播

对于第 $l$ 层 Encoder 的隐藏表示：

$$
H_l
\in
\mathbb{R}^{B\times T\times D}
$$

首先进行归一化：

$$
\widehat H_l
=
\operatorname{LN}(H_l)
$$

然后进行方言条件调制：

$$
\widetilde H_l
=
\left(
1+a_l^G\gamma_d
\right)
\odot
\widehat H_l
+
a_l^G\beta_d
$$

其中：

$$
a_l^G
=
\tanh(\alpha_l^G)
$$

为可学习调制强度。

Encoder Block 的输出为：

$$
H_{l+1}
=
H_l
+
F_l(\widetilde H_l)
$$

其中保留原始残差路径 $H_l$，方言信息只影响当前 Block 的增量部分。

初始化时：

$$
\alpha_l^G=0
$$

因此：

$$
a_l^G=0
$$

使模型训练初始状态尽可能保持原始预训练 Paraformer 的行为。

---

## 2.4 全局模块数据流

整体前向流程可以表示为：

$$
d
\rightarrow
[c_d^\gamma,c_d^\beta]
$$

$$
c_d^\gamma U^\gamma
\rightarrow
\gamma_d
$$

$$
c_d^\beta U^\beta
\rightarrow
\beta_d
$$

然后：

$$
H_l
\rightarrow
\operatorname{LN}
\rightarrow
\operatorname{FiLM}(\gamma_d,\beta_d)
\rightarrow
F_l
\rightarrow
H_{l+1}
$$

最终得到经过全局方言条件校准后的表示：

$$
H^G
$$

---

# 3. 第二部分：潜在方言簇适配

## 3.1 设计目的

方言簇模块负责建模多个方言之间可以共享的中等粒度结构，例如：

- 相似的声母变化；
- 相似的韵母变化；
- 相似的音系规则；
- 相似的发音偏移模式；
- 可跨方言迁移的共享声学规律。

该模块不提前规定每个方言属于哪个方言簇，而是让网络在 ASR 目标驱动下自行学习：

> 哪些方言应该共享哪些 Cluster Adapter。

因此，方言簇本身被视为潜在变量。

---

## 3.2 方言簇 Embedding

对于每个方言 $d$，方言簇模块单独学习：

$$
e_d^C
\in
\mathbb{R}^{D_C}
$$

第一版可以设置：

$$
D_C=32
$$

该 Embedding 与全局模块中的低维 FiLM 系数独立。

原因在于两者承担不同职责：

- 全局模块表示“该方言应该如何整体校准隐藏空间”；
- 方言簇模块表示“该方言在潜在跨方言共享空间中的位置”。

因此，簇级 Embedding 专门服务于路由与方言关系学习。

---

## 3.3 可学习方言簇原型

假设预设最大潜在方言簇数量为：

$$
K
$$

为每个簇设置一个可学习原型：

$$
\mu_k
\in
\mathbb{R}^{D_C},
\qquad
k=1,\ldots,K
$$

计算方言表示与每个簇原型之间的相似度：

$$
s_{d,k}
=
\frac{
\cos(e_d^C,\mu_k)
}{
\tau
}
$$

其中：

$$
\tau
$$

为 Softmax temperature。

然后得到方言 $d$ 对各潜在簇的软归属概率：

$$
\pi_{d,k}
=
\frac{
\exp(s_{d,k})
}{
\sum_{j=1}^{K}
\exp(s_{d,j})
}
$$

记为：

$$
\boldsymbol{\pi}_d
=
[
\pi_{d,1},
\ldots,
\pi_{d,K}
]
$$

满足：

$$
\sum_{k=1}^{K}
\pi_{d,k}=1
$$

当前主结构采用静态方言路由：

$$
\boxed{
P(C\mid d)
}
$$

即同一个方言的所有语句共享同一套潜在簇归属。

这样可以保证学习到的主要是方言类别之间的稳定共享关系，而不是说话人、噪声、语速或当前语句内容等动态因素。

---

## 3.4 Top-2 方言簇路由

对于方言 $d$，选择概率最大的两个簇：

$$
\mathcal K_d
=
\operatorname{Top2}
(
\boldsymbol{\pi}_d
)
$$

对选中的两个概率重新归一化：

$$
\widetilde\pi_{d,k}
=
\frac{
\pi_{d,k}
}{
\sum_{j\in\mathcal K_d}
\pi_{d,j}
},
\qquad
k\in\mathcal K_d
$$

因此，一个方言不需要被强制划入唯一一个簇，而可以同时共享两个潜在方言结构。

---

## 3.5 Cluster Adapter

每个潜在方言簇对应一个 Bottleneck Adapter：

$$
A_k^C(H)
=
W_{\mathrm{up},k}^{C}
\,
\sigma
\left(
H
W_{\mathrm{down},k}^{C}
\right)
$$

其中：

$$
W_{\mathrm{down},k}^{C}
\in
\mathbb{R}^{D\times r_C}
$$

$$
W_{\mathrm{up},k}^{C}
\in
\mathbb{R}^{r_C\times D}
$$

对于：

$$
D=512
$$

第一版可以设置：

$$
r_C=64
$$

方言 $d$ 的簇级共享修正为：

$$
\Delta H_d^C
=
\sum_{k\in\mathcal K_d}
\widetilde\pi_{d,k}
A_k^C(H^G)
$$

最终：

$$
H^C
=
H^G
+
g^C
\Delta H_d^C
$$

其中：

$$
g^C
$$

为 Cluster Adapter 的残差强度门控。

---

# 4. 方言簇路由损失

方言簇路由需要避免两个典型退化问题：

1. 每个方言平均使用所有 Cluster Adapter；
2. 所有方言都集中进入相同的少数 Cluster Adapter。

因此使用两个互补的路由正则项。

---

## 4.1 Top-2 集中约束

对于每个方言，惩罚 Top-2 以外的概率质量：

$$
\mathcal L_{\mathrm{outside}}
=
\frac{1}{N_d}
\sum_{d=1}^{N_d}
\sum_{k\notin\operatorname{Top2}(\boldsymbol{\pi}_d)}
\pi_{d,k}
$$

其作用是：

$$
\boxed{
\text{让单个方言主要集中使用少数几个潜在方言簇}
}
$$

在当前 Top-2 设计下，该损失比直接使用强熵最小化更合适，因为不会进一步强迫 Top-2 退化成 Top-1。

---

## 4.2 Cluster Utilization Loss

首先计算所有方言类别对各簇的平均软概率：

$$
\bar\pi_k
=
\frac{1}{N_d}
\sum_{d=1}^{N_d}
\pi_{d,k}
$$

定义均匀目标分布：

$$
u_k
=
\frac{1}{K}
$$

然后使用 KL 散度：

$$
\mathcal L_{\mathrm{util}}
=
D_{\mathrm{KL}}
\left(
\bar{\boldsymbol{\pi}}
\Vert
\boldsymbol u
\right)
$$

展开为：

$$
\mathcal L_{\mathrm{util}}
=
\sum_{k=1}^{K}
\bar\pi_k
\log
\frac{
\bar\pi_k
}{
1/K
}
$$

其目标为：

$$
\boxed{
\text{防止所有方言整体坍缩到相同的少数 Cluster Adapter}
}
$$

这里按照**方言类别**平均，而不是按照 batch 中的样本数量平均，因此高资源方言不会因为数据更多而主导整体簇结构。

---

## 4.3 总路由损失

最终路由正则为：

$$
\boxed{
\mathcal L_{\mathrm{route}}
=
\lambda_{\mathrm{out}}
\mathcal L_{\mathrm{outside}}
+
\lambda_{\mathrm{util}}
\mathcal L_{\mathrm{util}}
}
$$

两者分别约束：

- $\mathcal L_{\mathrm{outside}}$：每一行，即单个方言自身的路由集中程度；
- $\mathcal L_{\mathrm{util}}$：整体列利用率，即多个潜在簇是否都得到使用。

因此最终希望形成：

$$
\boxed{
\text{单个方言集中使用一到两个簇}
+
\text{不同方言整体分散到多个簇}
}
$$

---

# 5. 第三部分：动态单方言残差

## 5.1 设计目的

经过 Global 和 Cluster 模块后，大部分方言相关信息已经由：

$$
\text{全局校准}
+
\text{跨方言共享}
$$

进行建模。

第三部分只负责处理剩余的细粒度单方言差异。

其核心问题是：

> 对于已知方言 $d$，当前这条语音在共享 Cluster Adapter 基础上，还需要多少额外的方言特定修正？

因此第三部分不再建立一个完整独立的 Dialect Adapter，而是在选中的 Cluster Adapter 参数上增加：

$$
\boxed{
\text{Dialect-specific Low-Rank Parameter Residual}
}
$$

同时利用当前语音动态决定该 residual 在本句话中的使用强度。

---

## 5.2 Dialect-Specific Low-Rank Residual

Cluster Adapter 的 down projection 为：

$$
W_{\mathrm{down},k}^{C}
\in
\mathbb{R}^{D\times r_C}
$$

对于方言 $d$，学习一个低秩参数残差：

$$
\Delta W_{\mathrm{down},d}^{D}
=
A_{\mathrm{down},d}^{D}
B_{\mathrm{down},d}^{D}
$$

其中：

$$
A_{\mathrm{down},d}^{D}
\in
\mathbb{R}^{D\times r_D}
$$

$$
B_{\mathrm{down},d}^{D}
\in
\mathbb{R}^{r_D\times r_C}
$$

并满足：

$$
r_D\ll r_C
$$

第一版可以设置：

$$
r_D=4
$$

同理，对 up projection：

$$
\Delta W_{\mathrm{up},d}^{D}
=
A_{\mathrm{up},d}^{D}
B_{\mathrm{up},d}^{D}
$$

其中：

$$
A_{\mathrm{up},d}^{D}
\in
\mathbb{R}^{r_C\times r_D}
$$

$$
B_{\mathrm{up},d}^{D}
\in
\mathbb{R}^{r_D\times D}
$$

这些低秩 residual 参数只由方言标签决定，因此代表该方言稳定的个性化参数方向。

---

# 6. Dynamic Gate

## 6.1 当前语音特征提取

从 Encoder 中间层得到：

$$
H_m^C
\in
\mathbb{R}^{B\times T\times D}
$$

使用 Attentive Pooling 获得 utterance-level representation：

$$
u_x
=
\operatorname{AttentivePool}(H_m^C)
$$

其中：

$$
u_x
\in
\mathbb{R}^{B\times D}
$$

然后采用：

$$
\widetilde u_x
=
\operatorname{StopGrad}(u_x)
$$

阻止 Dynamic Gate 分支反向改变前面的 ASR Encoder 表示。

再通过线性投影压缩：

$$
v_x
=
W_x\widetilde u_x
$$

例如：

$$
v_x
\in
\mathbb{R}^{32}
$$

---

## 6.2 方言条件

第三部分单独使用一个方言 embedding：

$$
e_d^D
\in
\mathbb{R}^{D_D}
$$

第一版可以设置：

$$
D_D=16
$$

将方言信息和当前语音表示拼接：

$$
z_{d,x}
=
[e_d^D;v_x]
$$

然后经过一个小型 Gate Network：

$$
g_{d,x}
=
\sigma
\left(
f_G(z_{d,x})
\right)
$$

第一版使用 scalar gate：

$$
g_{d,x}\in(0,1)
$$

它表示：

$$
\boxed{
\text{当前语句对单方言 residual 的需求强度}
}
$$

它不严格等价于“口音强度”，而是当前语句在共享结构之外仍需要多少方言特定修正。

---

# 7. Dynamic Gate × Dialect Residual

当前语句的有效方言残差为：

$$
\Delta W_{\mathrm{down},d,x}^{D}
=
g_{d,x}
\Delta W_{\mathrm{down},d}^{D}
$$

即：

$$
\Delta W_{\mathrm{down},d,x}^{D}
=
g_{d,x}
A_{\mathrm{down},d}^{D}
B_{\mathrm{down},d}^{D}
$$

对应的有效 Cluster Adapter 参数为：

$$
\widetilde W_{\mathrm{down},k}^{d,x}
=
W_{\mathrm{down},k}^{C}
+
g_{d,x}
A_{\mathrm{down},d}^{D}
B_{\mathrm{down},d}^{D}
$$

同理：

$$
\widetilde W_{\mathrm{up},k}^{d,x}
=
W_{\mathrm{up},k}^{C}
+
g_{d,x}
A_{\mathrm{up},d}^{D}
B_{\mathrm{up},d}^{D}
$$

因此，经过单方言个性化后的 Cluster Adapter 为：

$$
A_k^{d,x}(H)
=
\sigma
\left(
H
\widetilde W_{\mathrm{down},k}^{d,x}
\right)
\widetilde W_{\mathrm{up},k}^{d,x}
$$

对于 Top-2 Cluster：

$$
\Delta H_{d,x}
=
\sum_{k\in\mathcal K_d}
\widetilde\pi_{d,k}
A_k^{d,x}(H^G)
$$

最终：

$$
H^D
=
H^G
+
\Delta H_{d,x}
$$

同一个方言 residual 可以作用于当前 Top-2 选中的 Cluster Experts，而不同 Cluster Experts 仍保留各自不同的基础参数。

---

## 7.1 Rank-wise Gate 备注

后续可以将 scalar gate：

$$
g_{d,x}\in\mathbb{R}
$$

升级为 rank-wise gate：

$$
\boldsymbol g_{d,x}
=
[
g_{d,x,1},
\ldots,
g_{d,x,r_D}
]
$$

此时低秩 residual 写为：

$$
\Delta W_{d,x}^{D}
=
A_d^D
\operatorname{diag}
(
\boldsymbol g_{d,x}
)
B_d^D
$$

这样当前语音可以分别控制不同低秩 residual direction 的使用强度，而不是对整个 residual 统一缩放。

---

# 8. 完整前向传播流程

对于输入语音：

$$
X
$$

及已知方言标签：

$$
d
$$

完整流程如下。

---

## Step 1：共享声学编码

前端得到初始表示：

$$
H_0
=
\operatorname{Frontend}(X)
$$

---

## Step 2：全局方言校准

根据方言标签查找：

$$
c_d^\gamma,c_d^\beta
$$

计算：

$$
\gamma_d
=
c_d^\gamma U^\gamma
$$

$$
\beta_d
=
c_d^\beta U^\beta
$$

然后在 Encoder 内进行：

$$
\widetilde H_l
=
(1+a_l^G\gamma_d)
\odot
\operatorname{LN}(H_l)
+
a_l^G\beta_d
$$

得到全局校准表示：

$$
H^G
$$

---

## Step 3：方言簇路由

根据方言标签查找簇级表示：

$$
d
\rightarrow
e_d^C
$$

与所有簇原型计算相似度：

$$
s_{d,k}
=
\frac{
\cos(e_d^C,\mu_k)
}{
\tau
}
$$

得到：

$$
\boldsymbol{\pi}_d
=
\operatorname{softmax}(\boldsymbol{s}_d)
$$

选择：

$$
\mathcal K_d
=
\operatorname{Top2}(\boldsymbol{\pi}_d)
$$

并得到重新归一化后的：

$$
\widetilde\pi_{d,k}
$$

---

## Step 4：生成当前语句的 Dynamic Gate

从中间 Encoder 表示：

$$
H_m^C
$$

提取当前语句信息：

$$
u_x
=
\operatorname{StopGrad}
\left(
\operatorname{AttentivePool}(H_m^C)
\right)
$$

投影得到：

$$
v_x
=
W_xu_x
$$

方言标签同时得到：

$$
e_d^D
$$

拼接：

$$
z_{d,x}
=
[e_d^D;v_x]
$$

然后：

$$
g_{d,x}
=
\sigma
\left(
f_G(z_{d,x})
\right)
$$

---

## Step 5：构造当前语句的 Dialect Residual

方言 $d$ 的基础低秩残差为：

$$
\Delta W_d^D
=
A_d^DB_d^D
$$

当前语句的有效残差为：

$$
\Delta W_{d,x}^{D}
=
g_{d,x}
\Delta W_d^D
$$

因此选中的 Cluster Adapter 参数变为：

$$
W_k^{d,x}
=
W_k^C
+
\Delta W_{d,x}^{D}
$$

---

## Step 6：Top-2 Cluster Expert 融合

分别计算两个被选中的动态个性化 Cluster Adapter：

$$
A_k^{d,x}(H^G)
$$

然后按路由权重融合：

$$
\Delta H_{d,x}
=
\sum_{k\in\mathcal K_d}
\widetilde\pi_{d,k}
A_k^{d,x}(H^G)
$$

得到：

$$
H^D
=
H^G
+
\Delta H_{d,x}
$$

---

## Step 7：Paraformer 后续模块

最终表示继续送入 Paraformer 后续结构：

$$
H^D
\rightarrow
\operatorname{Predictor}
\rightarrow
\operatorname{Decoder}
\rightarrow
\widehat Y
$$

---

# 9. 总损失

主 ASR 损失记为：

$$
\mathcal L_{\mathrm{ASR}}
$$

方言簇路由损失为：

$$
\mathcal L_{\mathrm{route}}
=
\lambda_{\mathrm{out}}
\mathcal L_{\mathrm{outside}}
+
\lambda_{\mathrm{util}}
\mathcal L_{\mathrm{util}}
$$

为了限制单方言 residual 只承担小幅个性化修正，可以增加 residual 参数正则：

$$
\mathcal L_{\mathrm{res}}
=
\frac{1}{N_d}
\sum_{d=1}^{N_d}
\left\|
\Delta W_d^D
\right\|_F^2
$$

因此总损失可以写为：

$$
\boxed{
\mathcal L
=
\mathcal L_{\mathrm{ASR}}
+
\lambda_{\mathrm{out}}
\mathcal L_{\mathrm{outside}}
+
\lambda_{\mathrm{util}}
\mathcal L_{\mathrm{util}}
+
\lambda_{\mathrm{res}}
\mathcal L_{\mathrm{res}}
}
$$

---

# 10. 训练流程

整体训练遵循：

$$
\boxed{
\text{先学习共享结构}
\rightarrow
\text{再学习单方言残差}
\rightarrow
\text{最后联合微调}
}
$$

---

## Stage 1：训练 Global + Cluster

首先训练：

- Global Dialect Calibration；
- Cluster Embedding；
- Cluster Prototype；
- Cluster Router；
- Cluster Adapter。

关闭单方言残差：

$$
\Delta W_d^D=0
$$

训练目标：

$$
\mathcal L_{\mathrm{Stage1}}
=
\mathcal L_{\mathrm{ASR}}
+
\lambda_{\mathrm{out}}
\mathcal L_{\mathrm{outside}}
+
\lambda_{\mathrm{util}}
\mathcal L_{\mathrm{util}}
$$

该阶段先建立稳定的全局校准能力和跨方言共享结构。

---

## Stage 2：训练 Dynamic Dialect Residual

开启：

- Dialect-specific Low-Rank Residual；
- Dialect + Audio Dynamic Gate。

此阶段首先保持 Cluster Router 稳定，Global 和 Cluster 主体固定或采用明显更小的学习率。

训练目标：

$$
\mathcal L_{\mathrm{Stage2}}
=
\mathcal L_{\mathrm{ASR}}
+
\lambda_{\mathrm{res}}
\mathcal L_{\mathrm{res}}
$$

使第三层只学习前两层共享结构无法解释的剩余单方言差异。

---

## Stage 3：联合微调

最后联合优化：

- Global；
- Cluster Router；
- Cluster Adapter；
- Dynamic Dialect Residual。

使用较小学习率进行整体协调，避免破坏已经学习到的潜在方言共享结构。

---

# 11. 三层架构职责总结

| 层次 | 主要机制 | 条件信息 | 核心作用 |
|---|---|---|---|
| Global | Low-Rank FiLM | 方言标签 | 粗粒度方言整体分布校准 |
| Dialect Cluster | Prototype Router + Top-2 Cluster Adapter | 方言标签 | 学习多个方言之间的潜在共享结构 |
| Dialect Residual | Low-Rank Parameter Residual × Dynamic Gate | 方言标签 + 当前语音 | 建模单方言内部的细粒度语句级个性化差异 |

最终整体思想可以概括为：

$$
\boxed{
\text{Global Calibration}
\rightarrow
\text{Latent Cross-Dialect Sharing}
\rightarrow
\text{Utterance-Adaptive Dialect Residual}
}
$$

即：

$$
\boxed{
\text{方言整体分布偏移}
+
\text{跨方言共享潜在结构}
+
\text{当前语句相关的单方言残差}
}
$$

#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Latent Dialect Cluster Adaptation —— global_cluster_single_v2.md 第二部分

与 Global 的区别：
  - Global：纯方言条件，per-dialect 低秩 FiLM，整句共享
  - Cluster：学习方言间的潜在共享关系，每方言 → Top-2 簇 → 加权融合多个 adapter

设计要点（计划 §3）：
  - 方言簇 embedding e_d^C ∈ R^{D_C}（与 Global 的 c_d 独立）
  - K 个可学簇原型 μ_k，cos 相似度 + softmax + Top-2 路由
  - 每簇一个 bottleneck adapter (512→64→512)
  - 静态路由 P(C|d)：同方言所有语句共享同一套簇归属
  - 路由损失：Top-2 外惩罚 + 簇利用率 KL（按方言类别平均，不被高资源方言主导）

注入点：encoder 输出 H^G（经 Global 校准后）→ Cluster Adapter → H^C
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== 全局诊断收集器 =====
_GX_COLLECTOR = {'enabled': False, 'data': []}

def enable_gx_collection():
    """启用g_x收集"""
    global _GX_COLLECTOR
    _GX_COLLECTOR['enabled'] = True
    _GX_COLLECTOR['data'] = []

def get_collected_gx():
    """获取收集的g_x数据"""
    global _GX_COLLECTOR
    return _GX_COLLECTOR['data']

def clear_gx_collection():
    """清空收集的数据"""
    global _GX_COLLECTOR
    _GX_COLLECTOR['data'] = []
# ===== 结束 =====


class ClusterAdapter(nn.Module):
    """
    单个 Cluster 的 Bottleneck Adapter（计划 §3.5）

    A_k^C(H) = W_up · σ(H · W_down)

    down: 512 → r_C (64)
    up:   r_C → 512
    """
    def __init__(self, hidden_size: int = 512, bottleneck_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_dim, bias=True)
        self.up = nn.Linear(bottleneck_dim, hidden_size, bias=True)
        self.act = nn.ReLU()

        # down：Xavier（给方向）
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        # up：ZERO_INIT_UP=1 时零初始化（初始 delta=0，严格恒等），否则 Xavier
        if os.environ.get("ZERO_INIT_UP", "0") == "1":
            nn.init.zeros_(self.up.weight)
            nn.init.zeros_(self.up.bias)
        else:
            nn.init.xavier_uniform_(self.up.weight)
            nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, D=512]
        Returns:
            delta: [B, T, D]
        """
        return self.up(self.act(self.down(x)))


class AcousticClusterAdapter(nn.Module):
    """
    声学条件化 Cluster Adapter（完整声学信号版）

    与 ClusterAdapter 的区别：
      - ClusterAdapter：A_k(x) = up(relu(down(x)))，delta 无条件，只依赖输入 x
      - AcousticClusterAdapter：在 bottleneck 层注入声学 FiLM 调制，
        delta 依赖当前语句的声学内容 u（utterance 表示）

    公式：
      h = relu(down(x))                          # [B,T,rC]
      γ, β = gamma_proj(u), beta_proj(u)         # u=[B,D] → [B,rC]
      h = (1 + γ) ⊙ h + β                        # 声学 FiLM 调制
      delta = up(h)                              # [B,T,D]

    零启动：up / gamma_proj / beta_proj 全零初始化 → 初始 delta = 0（严格恒等），
            gate 从 0 学起，扰动渐进可控（对齐 FiLM 的零启动设计）。
    """
    def __init__(self, hidden_size: int = 512, bottleneck_dim: int = 64,
                 cond_dim: int = 512):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_dim, bias=True)
        self.up = nn.Linear(bottleneck_dim, hidden_size, bias=True)
        self.act = nn.ReLU()
        # 声学条件 → γ/β
        self.gamma_proj = nn.Linear(cond_dim, bottleneck_dim)
        self.beta_proj = nn.Linear(cond_dim, bottleneck_dim)

        # 反死锁初始化：
        #   down Xavier（给方向）
        #   up 小随机 std=0.01（打破"up全零→delta=0→gate梯度=0"的死锁）
        #   gamma/beta 全零（gate 动起来后，它们经 up^T 收到梯度，自然跟着学）
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.normal_(self.up.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.gamma_proj.weight)
        nn.init.zeros_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, x: torch.Tensor, acoustic_cond: torch.Tensor):
        """
        Args:
            x: [B, T, D] 输入隐藏表示
            acoustic_cond: [B, D] 当前语句的声学 utterance 表示
        Returns:
            delta: [B, T, D]
        """
        h = self.act(self.down(x))                     # [B,T,rC]
        gamma = self.gamma_proj(acoustic_cond)         # [B,rC]
        beta = self.beta_proj(acoustic_cond)           # [B,rC]
        h = (1.0 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)  # 声学调制
        return self.up(h)                              # [B,T,D]


class ClusterFiLMBank(nn.Module):
    """
    诊断用：per-cluster 低秩 FiLM（替代 ClusterAdapter 的 bottleneck 残差）

    目的：把 Cluster 的调制机制换成与 GlobalFiLM **完全同款**的低秩仿射，
    只保留"条件来自簇路由 vs 方言 id"这一个差异，从而隔离：
      - 若 cluster-FiLM 不再有害 → A2 的害来自 adapter 残差机制（腐蚀已收敛表示）
      - 若 cluster-FiLM 仍 ≈ Global 上限、加不出增益 → 粒度冗余（信息论约束）

    参数化（对齐 modules/global_film_layer.py::GlobalFiLM）：
      gamma_k = c_k^gamma @ U^gamma      # [K, D]，K 个簇各一组低秩系数
      beta_k  = c_k^beta  @ U^beta       # [K, D]
      A_k(x)  = gamma_k ⊙ x + beta_k     # 仿射调制（delta 形式见下）

    注意：这里返回的是"delta"以套进 DialectClusterModule 的 x + g_c·delta 结构。
      FiLM 调制 x' = (1+gamma)⊙x + beta，故 delta = x' - x = gamma⊙x + beta。
      gate g_c=0 时 x_out=x 严格恒等（与 adapter 版零启动一致）。
    """
    def __init__(self, hidden_size: int = 512, num_clusters: int = 4, rank: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_clusters = num_clusters
        self.rank = rank

        # per-cluster 低秩系数（对齐 GlobalFiLM 的 c_d，这里是 c_k）
        self.c_gamma = nn.Parameter(torch.zeros(num_clusters, rank))
        self.c_beta = nn.Parameter(torch.zeros(num_clusters, rank))
        # 全簇共享调制基底（对齐 GlobalFiLM 的 U）
        self.U_gamma = nn.Parameter(torch.empty(rank, hidden_size))
        self.U_beta = nn.Parameter(torch.empty(rank, hidden_size))

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.c_gamma, std=0.02)
        nn.init.normal_(self.c_beta, std=0.02)
        nn.init.xavier_uniform_(self.U_gamma)
        nn.init.xavier_uniform_(self.U_beta)

    def all_gamma_beta(self):
        """返回全部簇的 gamma/beta：[K, D] 各一。"""
        gamma = self.c_gamma @ self.U_gamma   # [K, D]
        beta = self.c_beta @ self.U_beta      # [K, D]
        return gamma, beta

    def weighted_delta(self, x: torch.Tensor, batch_weights: torch.Tensor):
        """
        稠密加权的 FiLM delta（DDP 安全，与 adapter 版 _cluster_delta 同一融合口径）。

        delta = Σ_k w_k · (gamma_k ⊙ x + beta_k)
              = (Σ_k w_k·gamma_k) ⊙ x + (Σ_k w_k·beta_k)

        Args:
            x: [B, T, D]
            batch_weights: [B, K]  每样本的稠密簇权重（Top-2 位置非零）
        Returns:
            delta: [B, T, D]
        """
        gamma, beta = self.all_gamma_beta()          # [K, D], [K, D]
        # 每样本按簇权重线性组合 gamma/beta
        gamma_b = batch_weights @ gamma              # [B, D]
        beta_b = batch_weights @ beta                # [B, D]
        delta = gamma_b.unsqueeze(1) * x + beta_b.unsqueeze(1)  # [B, T, D]
        return delta


class DialectClusterRouter(nn.Module):
    """
    方言簇路由 + Top-2 选择（计划 §3.2-3.4）

    流程：
      e_d^C → cos(e_d, μ_k) / τ → softmax → Top-2 → 重归一化权重

    返回：
      - top2_indices: [num_dialects, 2]  每方言选中的 2 个簇索引
      - top2_weights: [num_dialects, 2]  重归一化后的权重（和为 1）
      - full_probs:   [num_dialects, K]  完整 softmax 分布（用于计算路由损失）
    """
    def __init__(
        self,
        num_dialects: int = 9,
        num_clusters: int = 4,
        cluster_emb_dim: int = 32,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.num_dialects = num_dialects
        self.num_clusters = num_clusters
        self.temperature = temperature

        # 方言簇 embedding（每方言一个，独立于 Global 的 c_d）
        self.dialect_emb = nn.Parameter(torch.randn(num_dialects, cluster_emb_dim))

        # 簇原型（K 个可学习）
        self.cluster_prototypes = nn.Parameter(torch.randn(num_clusters, cluster_emb_dim))

        self._init_parameters()

    def _init_parameters(self):
        """小随机初始化，避免一开始就 one-hot 坍缩"""
        nn.init.normal_(self.dialect_emb, std=0.02)
        nn.init.normal_(self.cluster_prototypes, std=0.02)

    def compute_routing_probs(self):
        """
        计算所有方言对所有簇的软归属概率（静态路由 P(C|d)）

        Returns:
            probs: [num_dialects, num_clusters]  每行和为 1
        """
        # L2 归一化
        e_d_norm = F.normalize(self.dialect_emb, p=2, dim=1)      # [N_d, D_C]
        mu_norm = F.normalize(self.cluster_prototypes, p=2, dim=1)  # [K, D_C]

        # cos 相似度 / temperature
        sim = torch.matmul(e_d_norm, mu_norm.t()) / self.temperature  # [N_d, K]

        # softmax
        probs = F.softmax(sim, dim=1)  # [N_d, K]
        return probs

    def forward(self):
        """
        前向传播：计算静态路由分配（训练时每次调用都重新计算，推理时可缓存）

        Returns:
            top2_indices: [num_dialects, 2] long
            top2_weights: [num_dialects, 2] float，重归一化后和为 1
            full_probs:   [num_dialects, K] float，完整分布（用于损失）
        """
        full_probs = self.compute_routing_probs()  # [N_d, K]

        # Top-2
        top2_probs, top2_indices = torch.topk(full_probs, k=2, dim=1)  # [N_d, 2]

        # 重归一化（让 Top-2 和为 1）
        top2_weights = top2_probs / top2_probs.sum(dim=1, keepdim=True)  # [N_d, 2]

        return top2_indices, top2_weights, full_probs


class DynamicDialectClusterRouter(nn.Module):
    """
    动态路由：P(C|x) 学习"声学→簇"映射（诊断：破静态路由的信息冗余）

    与静态 DialectClusterRouter 的区别：
      - 静态：dialect_emb[d] @ cluster_prototypes → 每方言固定 top2，查表
      - 动态：pool(H) @ W → 每句独立 top2，依赖声学特征

    目的：让 cluster 访问 Global(方言 id)访问不到的声学维度 → 理论上能破信息瓶颈。
    但在收敛主干上可能仍缺正向梯度 → 预期大概率还是撞天花板，但这是唯一理论可行的改法。

    返回：与静态版接口一致，但 top2_indices/weights 是 [B,2]（每句独立），full_probs 是 [B,K]
    """
    def __init__(
        self,
        hidden_size: int = 512,
        num_clusters: int = 4,
        pool_att_dim: int = 128,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.temperature = temperature

        # 复用 AttentivePool 提取 utterance 表示
        self.pool = AttentivePool(hidden_size, pool_att_dim)

        # 学习"声学 utterance 表示 → 簇 logits"
        self.to_logits = nn.Linear(hidden_size, num_clusters)

        # Xavier 初始化：让初始路由随机但平衡，避免一开始就坍缩到某一簇
        nn.init.xavier_uniform_(self.to_logits.weight)
        nn.init.zeros_(self.to_logits.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: [B, T, D] encoder 输出
            mask: [B, 1, T] or [B, T]  padding mask (True=有效)
        Returns:
            top2_indices: [B, 2] long  每句的 top2 簇索引
            top2_weights: [B, 2] float  重归一化权重
            full_probs:   [B, K] float  完整 softmax 分布（路由损失用）
        """
        # Utterance-level 表示
        u = self.pool(x, mask)  # [B, D]

        # 簇 logits → softmax
        logits = self.to_logits(u) / self.temperature  # [B, K]
        full_probs = F.softmax(logits, dim=1)          # [B, K]

        # Top-2 + 重归一化
        top2_probs, top2_indices = torch.topk(full_probs, k=2, dim=1)  # [B, 2]
        top2_weights = top2_probs / top2_probs.sum(dim=1, keepdim=True)

        return top2_indices, top2_weights, full_probs


class DialectResidualBank(nn.Module):
    """
    第三部分：Dynamic Dialect Residual（global_cluster_single_v2.md §5-7）

    对每个方言 d 学一组低秩参数残差，动态扰动被选中 Cluster Adapter 的投影权重：
      ΔW_down,d = A_down_d · B_down_d   (D×r_D · r_D×r_C = D×r_C，与 W_down 同形)
      ΔW_up,d   = A_up_d   · B_up_d     (r_C×r_D · r_D×D = r_C×D，与 W_up 同形)
    有效权重（当前语句、乘 dynamic gate g_x）：
      W̃_down = W_down,k + g_x · ΔW_down,d
      W̃_up   = W_up,k   + g_x · ΔW_up,d

    零启动 / 反死锁初始化：
      B_* 零初始化 → 初始 ΔW=0 → 严格恒等（起始 CER 与 Stage 1 相同）
      A_* Xavier   → 打破对称：∂L/∂B ∝ g_x·A ≠ 0（g_x=sigmoid≈0.5）→ B 先动，
                     B≠0 后 ∂L/∂A ∝ g_x·B ≠ 0 → A 再动。避免上次 gate=0/up=0 双零死锁。

    DDP 安全：全部 num_dialects 组残差每步都用 einsum 稠密计算（one-hot×g_x 加权），
              forward 图各 rank 一致，不因 batch 内方言集合不同产生 unused 参数分歧。
    """
    def __init__(self, hidden_size: int = 512, cluster_bottleneck: int = 64,
                 num_dialects: int = 9, residual_rank: int = 4):
        super().__init__()
        D, rC, rD, Nd = hidden_size, cluster_bottleneck, residual_rank, num_dialects
        self.D, self.rC, self.rD, self.Nd = D, rC, rD, Nd

        # down 残差：ΔW_down = A_down @ B_down，形状 [Nd,D,rC]
        self.A_down = nn.Parameter(torch.empty(Nd, D, rD))
        self.B_down = nn.Parameter(torch.empty(Nd, rD, rC))
        # up 残差：ΔW_up = A_up @ B_up，形状 [Nd,rC,D]
        self.A_up = nn.Parameter(torch.empty(Nd, rC, rD))
        self.B_up = nn.Parameter(torch.empty(Nd, rD, D))

        # A Xavier（逐方言），B 小随机初始化（打破挂主干版梯度死锁）
        for d in range(Nd):
            nn.init.xavier_uniform_(self.A_down[d])
            nn.init.xavier_uniform_(self.A_up[d])
            # B 用小 std 正态分布初始化（std=0.01）
            # 目的：挂主干版需要 B≠0 打破梯度死锁（B=0 → h=0 → relu'(0)=0 → ∂L/∂B=0）
            # 代价：初始 ΔW≠0，不再严格从 Stage1 起步（但影响<0.01pp可接受）
            nn.init.normal_(self.B_down[d], mean=0.0, std=0.01)
            nn.init.normal_(self.B_up[d], mean=0.0, std=0.01)

    def delta_down(self, batch_dialect_onehot: torch.Tensor) -> torch.Tensor:
        """按 batch 每样本方言选出 ΔW_down：[B,D,rC]。onehot: [B,Nd]"""
        dW = torch.einsum("ndr,nrc->ndc", self.A_down, self.B_down)  # [Nd,D,rC]
        return torch.einsum("bn,ndc->bdc", batch_dialect_onehot, dW)  # [B,D,rC]

    def delta_up(self, batch_dialect_onehot: torch.Tensor) -> torch.Tensor:
        """按 batch 每样本方言选出 ΔW_up：[B,rC,D]。"""
        dW = torch.einsum("ncr,nrd->ncd", self.A_up, self.B_up)  # [Nd,rC,D]
        return torch.einsum("bn,ncd->bcd", batch_dialect_onehot, dW)  # [B,rC,D]

    def res_loss(self) -> torch.Tensor:
        """
        残差 Frobenius 正则（v2 文档 §9）：L_res = (1/Nd)·Σ_d(‖ΔW_down_d‖_F² + ‖ΔW_up_d‖_F²)
        限制单方言残差只做小幅个性化修正。带梯度返回，供 total loss 相加。
        """
        dWd = torch.einsum("ndr,nrc->ndc", self.A_down, self.B_down)  # [Nd,D,rC]
        dWu = torch.einsum("ncr,nrd->ncd", self.A_up, self.B_up)      # [Nd,rC,D]
        per_d = dWd.pow(2).sum(dim=(1, 2)) + dWu.pow(2).sum(dim=(1, 2))  # [Nd]
        return per_d.mean()


class AttentivePool(nn.Module):
    """
    Attentive pooling → utterance-level 表示（v2 文档 §6.1）
    u_x = Σ_t softmax(w·tanh(V·h_t)) · h_t，带 padding mask。
    """
    def __init__(self, hidden_size: int = 512, att_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(hidden_size, att_dim)
        self.score = nn.Linear(att_dim, 1, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """x: [B,T,D]; mask: [B,1,T] or [B,T] (True=有效) → u: [B,D]"""
        e = self.score(torch.tanh(self.proj(x))).squeeze(-1)  # [B,T]
        if mask is not None:
            m = mask.squeeze(1) if mask.dim() == 3 else mask   # [B,T]
            e = e.masked_fill(~m.bool(), float("-inf"))
        a = torch.softmax(e, dim=1).unsqueeze(-1)              # [B,T,1]
        return (a * x).sum(dim=1)                              # [B,D]


class DialectClusterModule(nn.Module):
    """
    完整的方言簇模块 = Router + K 个 Adapter + 残差门控

    输入：H^G（经 Global 校准后）
    输出（Stage 1）：H^C = H^G + g^C * Σ_{k in Top2} π_k * A_k(H^G)
    输出（Stage 2 v2，enable_dialect_residual=True）：在选中 cluster 的投影权重上
      叠加 per-dialect 低秩残差 × dynamic gate g_x（见 DialectResidualBank）。
    """
    def __init__(
        self,
        hidden_size: int = 512,
        num_dialects: int = 9,
        num_clusters: int = 4,
        bottleneck_dim: int = 64,
        cluster_emb_dim: int = 32,
        temperature: float = 0.1,
        gate_init: float = 0.0,  # 残差门控初值（0 = 恒等启动）
        # 诊断：cluster 调制机制 adapter(默认,A2/A3) | film(per-cluster 低秩 FiLM)
        cluster_delta_type: str = "adapter",
        cluster_film_rank: int = 16,
        # 诊断：cluster 路由 static(默认,P(C|d)) | dynamic(P(C|x),破信息冗余)
        cluster_routing_type: str = "static",
        cluster_routing_pool_dim: int = 128,
        # Stage 2 v2 —— Dynamic Dialect Residual
        enable_dialect_residual: bool = False,
        residual_rank: int = 4,
        residual_dialect_emb_dim: int = 16,
        residual_pool_proj_dim: int = 32,
        residual_att_dim: int = 128,
    ):
        super().__init__()
        self.num_clusters = num_clusters
        self.enable_dialect_residual = enable_dialect_residual
        self.num_dialects = num_dialects
        self.cluster_delta_type = str(cluster_delta_type).lower()
        self.cluster_routing_type = str(cluster_routing_type).lower()

        # 动态路由缓存(供 get_routing_loss/dump_routing_matrix 读,它们在 forward 外调用)
        self._last_routing_cache = None

        # 路由（静态 vs 动态）
        if self.cluster_routing_type == "dynamic" or self.cluster_delta_type == "acoustic":
            self.router = DynamicDialectClusterRouter(
                hidden_size=hidden_size,
                num_clusters=num_clusters,
                pool_att_dim=cluster_routing_pool_dim,
                temperature=temperature,
            )
        else:
            self.router = DialectClusterRouter(
                num_dialects=num_dialects,
                num_clusters=num_clusters,
                cluster_emb_dim=cluster_emb_dim,
                temperature=temperature,
            )

        # Cluster 调制机制
        if self.cluster_delta_type == "film":
            # 诊断版：per-cluster 低秩 FiLM（对齐 GlobalFiLM 机制/容量）
            self.cluster_film = ClusterFiLMBank(
                hidden_size=hidden_size, num_clusters=num_clusters, rank=cluster_film_rank)
            self.adapters = None
            if enable_dialect_residual:
                raise NotImplementedError(
                    "cluster_delta_type='film' 仅用于 Stage1 诊断，不支持 dialect_residual")
        elif self.cluster_delta_type == "acoustic":
            # 声学信号版：声学条件化 adapter + 动态路由
            self.cluster_film = None
            self.adapters = nn.ModuleList([
                AcousticClusterAdapter(hidden_size, bottleneck_dim, cond_dim=hidden_size)
                for _ in range(num_clusters)
            ])
        else:
            # 默认版（A2/A3）：K 个 bottleneck adapter
            self.cluster_film = None
            self.adapters = nn.ModuleList([
                ClusterAdapter(hidden_size, bottleneck_dim)
                for _ in range(num_clusters)
            ])

        # 残差门控 g^C（标量，初始 0 → 恒等）
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        # 声学信号版：额外的 utterance pool（提取声学条件 u）
        if self.cluster_delta_type == "acoustic":
            self.acoustic_pool = AttentivePool(hidden_size, cluster_routing_pool_dim)

        # ── Stage 2 v2：Dynamic Dialect Residual ──
        if enable_dialect_residual:
            self.dialect_residual = DialectResidualBank(
                hidden_size=hidden_size, cluster_bottleneck=bottleneck_dim,
                num_dialects=num_dialects, residual_rank=residual_rank,
            )
            # 方言 embedding e_d^D（第三部分独立，§6.2）
            self.residual_dialect_emb = nn.Parameter(
                torch.randn(num_dialects, residual_dialect_emb_dim) * 0.02)
            # attentive pooling + 语句投影 W_x（§6.1）
            self.residual_pool = AttentivePool(hidden_size, residual_att_dim)
            self.residual_Wx = nn.Linear(hidden_size, residual_pool_proj_dim)
            # gate network f_G：[e_d^D ; v_x] → scalar（§6.3）
            self.residual_gate_net = nn.Linear(
                residual_dialect_emb_dim + residual_pool_proj_dim, 1)

    def _cluster_delta(self, x, batch_weights, acoustic_cond=None):
        """Stage 1 基础簇残差：Σ_k w_k · A_k(x)。x:[B,T,D] w:[B,K] → [B,T,D]"""
        if self.cluster_delta_type == "film":
            # per-cluster 低秩 FiLM：delta = (Σ w_k·gamma_k)⊙x + (Σ w_k·beta_k)
            return self.cluster_film.weighted_delta(x, batch_weights)
        delta = torch.zeros_like(x)
        for k in range(self.num_clusters):
            if self.cluster_delta_type == "acoustic":
                adapter_out = self.adapters[k](x, acoustic_cond)  # [B,T,D]
            else:
                adapter_out = self.adapters[k](x)          # [B,T,D]
            w_k = batch_weights[:, k].view(-1, 1, 1)   # [B,1,1]
            delta = delta + w_k * adapter_out
        return delta

    def _cluster_delta_residual(self, x, batch_weights, onehot, g_x):
        """
        Stage 2 v2：每个 cluster adapter 用「基础权重 + g_x·per-dialect 低秩残差」重算。
        A_k^{d,x}(x) = up_eff( relu( down_eff(x) ) )，其中
          down_eff(x) = down_k(x) + g_x·(x @ ΔW_down_d)
          up_eff(h)   = up_k(h)   + g_x·(h @ ΔW_up_d)
        全部 num_dialects 组残差经 einsum 稠密参与（DDP 安全）。
        """
        dW_down = self.dialect_residual.delta_down(onehot)  # [B,D,rC]
        dW_up = self.dialect_residual.delta_up(onehot)      # [B,rC,D]
        g = g_x.view(-1, 1, 1)                              # [B,1,1]

        delta = torch.zeros_like(x)
        for k in range(self.num_clusters):
            ad = self.adapters[k]
            h = ad.down(x) + g * torch.bmm(x, dW_down)      # [B,T,rC]
            h = ad.act(h)
            out = ad.up(h) + g * torch.bmm(h, dW_up)        # [B,T,D]
            w_k = batch_weights[:, k].view(-1, 1, 1)
            delta = delta + w_k * out
        return delta

    def _backbone_delta_residual(self, x, onehot, g_x):
        """
        残差直接挂主干（并联，不寄生 cluster adapter，不受 g_c/路由压制）：
          ΔhR = g_x · up( relu( down(x) ) )
          down(x) = x @ ΔW_down_d   [B,T,rC]
          up(h)   = h @ ΔW_up_d     [B,T,D]
        B 零初始化 → 初始 ΔW=0 → ΔhR=0 → 起始严格 = Stage1（6.65%）。
        返回 ΔhR:[B,T,D]，由 forward 直接加到 x_out（与 g_c·cluster 平级）。
        """
        dW_down = self.dialect_residual.delta_down(onehot)  # [B,D,rC]
        dW_up = self.dialect_residual.delta_up(onehot)      # [B,rC,D]
        g = g_x.view(-1, 1, 1)                              # [B,1,1]

        h = torch.bmm(x, dW_down)                           # [B,T,rC]
        h = torch.relu(h)
        delta_r = torch.bmm(h, dW_up)                       # [B,T,D]
        return g * delta_r

    def forward(self, x: torch.Tensor, dialect_ids: torch.Tensor,
                mask: torch.Tensor = None, gate_feat: torch.Tensor = None,
                return_diag: bool = False):
        """
        Args:
            x: [B, T, D] H^G（Global 校准后的表示）
            dialect_ids: [B] long
            mask: [B,1,T] padding mask（True=有效），仅 Stage 2 v2 attentive pooling 用
            gate_feat: [B,T,D] 中间层 H_m^C（§6.1），供 dynamic gate 提取语句特征。
                       已在 encoder 里 detach。None 时回退到 h_c_base（向后兼容）。
            return_diag: 是否返回诊断信息

        Returns:
            x_out: [B, T, D] H^C（Stage1）或 H^D（Stage2 v2）
            (可选) diag
        """
        # ⭐ 主干隔离：CLUSTER_DETACH_INPUT=1 时，cluster 分支输入 detach，
        #    切断路由/声学条件/adapter 三条路径回传主干的梯度（adapter-tuning 范式）。
        #    前向残差仍用原始 x（主干吃到修正），cluster 自身照常学习。
        if os.environ.get("CLUSTER_DETACH_INPUT", "0") == "1":
            x_cluster = x.detach()
        else:
            x_cluster = x

        # 路由（静态 vs 动态）
        if self.cluster_routing_type == "dynamic" or self.cluster_delta_type == "acoustic":
            # 动态路由：P(C|x)，每句独立，依赖声学
            top2_indices, top2_weights, full_probs = self.router(x_cluster, mask)  # [B,2], [B,2], [B,K]
            # 缓存给 get_routing_loss/dump_routing_matrix 用(它们在 forward 外被调)
            self._last_routing_cache = (top2_indices.detach(), top2_weights.detach(), full_probs.detach())
            # 直接用 batch 级路由，构造稠密权重 [B, K]
            B, K = full_probs.shape
            batch_weights = torch.zeros_like(full_probs)  # [B, K]
            batch_weights.scatter_(1, top2_indices, top2_weights)  # Top-2 位置填权重
        else:
            # 静态路由：P(C|d)，每方言固定 top2，查表
            top2_indices, top2_weights, full_probs = self.router()  # [N_d,2], [N_d,2], [N_d,K]
            # 构造每方言的稠密路由权重 [N_d, K]（DDP 安全：稠密调用全部 adapter）
            N_d, K = full_probs.shape
            dense_weights = torch.zeros_like(full_probs)  # [N_d, K]
            dense_weights.scatter_(1, top2_indices, top2_weights)
            # 当前 batch 按 dialect_ids 查表
            batch_weights = dense_weights[dialect_ids]  # [B, K]

        # 残差门控：FIXED_GATE>0 时固定为常数（不学 gate，绕过 gate 自锁），否则用可学习 tanh(gate)
        _fixed_gate = os.environ.get("FIXED_GATE", "")
        if _fixed_gate != "":
            # 乘0技巧：让 self.gate 仍出现在计算图（避免 DDP unused 报错），但梯度为0，等价于不学
            g_c = float(_fixed_gate) + 0.0 * torch.tanh(self.gate)
        else:
            g_c = torch.tanh(self.gate)  # Stage1 簇残差门控

        if not self.enable_dialect_residual:
            # ── Stage 1 路径 ──
            if self.cluster_delta_type == "acoustic":
                acoustic_cond = self.acoustic_pool(x_cluster, mask)  # [B,D] 声学条件
                delta = self._cluster_delta(x_cluster, batch_weights, acoustic_cond)
            else:
                delta = self._cluster_delta(x_cluster, batch_weights)
            x_out = x + g_c * delta
            g_x_mean = None
        else:
            # ── Stage 2 v2 路径 ──
            # 1) dynamic gate 的输入特征（§6.1）：
            #    优先用 encoder 中间层 H_m^C（gate_feat，已 detach）——符合文档设计；
            #    回退：无 gate_feat 时用最后层算出的 h_c_base（StopGrad，解循环依赖）。
            if gate_feat is not None:
                pool_input = gate_feat                             # 中间层 H_m^C
            else:
                with torch.no_grad():
                    base_delta = self._cluster_delta(x, batch_weights)
                    pool_input = (x + g_c * base_delta).detach()
            # 2) dynamic gate g_x（§6）
            u_x = self.residual_pool(pool_input, mask)             # [B,D] StopGrad
            v_x = self.residual_Wx(u_x)                            # [B,proj]
            e_d = self.residual_dialect_emb[dialect_ids]           # [B,emb]
            z = torch.cat([e_d, v_x], dim=-1)
            g_x = torch.sigmoid(self.residual_gate_net(z)).squeeze(-1)  # [B] in (0,1)

            # ===== gate 干预：FORCE_GX 覆盖 dynamic gate（推理诊断用）=====
            _force_gx = os.environ.get("FORCE_GX", "")
            if _force_gx != "":
                g_x = torch.full_like(g_x, float(_force_gx))
            # ===== 结束 =====

            # ===== 收集g_x用于诊断 =====
            global _GX_COLLECTOR
            if _GX_COLLECTOR['enabled']:
                _GX_COLLECTOR['data'].append({
                    'g_x': g_x.detach().cpu().numpy().tolist(),
                    'dialect_ids': dialect_ids.detach().cpu().numpy().tolist(),
                })
            # ===== 结束 =====

            # 3) 用 g_x 缩放的低秩残差
            onehot = F.one_hot(dialect_ids, self.num_dialects).float()  # [B,Nd]
            if os.environ.get("RESIDUAL_ON_BACKBONE", "0") == "1":
                # 残差直接挂主干（并联）：H^D = x + g_c·cluster(x) + ΔhR
                # 残差与 g_c·cluster 平级，不受 g_c/路由权重压制
                base_delta = self._cluster_delta(x, batch_weights)
                delta_r = self._backbone_delta_residual(x, onehot, g_x)
                x_out = x + g_c * base_delta + delta_r
            else:
                # 原版：残差寄生 cluster adapter 投影权重
                delta = self._cluster_delta_residual(x, batch_weights, onehot, g_x)
                x_out = x + g_c * delta
            g_x_mean = g_x

        if not return_diag:
            return x_out

        # 诊断信息
        with torch.no_grad():
            diag = {
                "gate": g_c.item(),
                "delta_norm": delta.pow(2).mean().sqrt().item(),
                "delta_rel": (delta.pow(2).sum() / x.pow(2).sum().clamp(min=1e-8)).sqrt().item(),
                "full_probs": full_probs.detach().cpu(),  # [N_d, K] 用于路由损失计算
            }
            if g_x_mean is not None:
                diag["g_x_mean"] = g_x_mean.mean().item()
                diag["g_x_std"] = g_x_mean.std().item()
        return x_out, diag

    def get_last_routing(self):
        """
        获取最后一次 forward 的路由结果（供 encoder 的 get_routing_loss/dump 调用）。

        - 静态路由：直接调 router() 返回固定的 [N_d,2]/[N_d,K]
        - 动态路由：返回缓存的最后一次 batch 路由 [B,2]/[B,K]

        Returns:
            (top2_indices, top2_weights, full_probs)
        """
        if self.cluster_routing_type == "dynamic":
            if self._last_routing_cache is None:
                raise RuntimeError("动态路由：forward 未调用前无法获取路由结果")
            return self._last_routing_cache
        else:
            return self.router()


def compute_routing_loss(
    full_probs: torch.Tensor,
    top2_indices: torch.Tensor,
    lambda_outside: float = 0.01,
    lambda_util: float = 0.01,
):
    """
    计算路由正则损失（计划 §4）

    Args:
        full_probs: [num_dialects, K] 所有方言对所有簇的 softmax 分布
        top2_indices: [num_dialects, 2] 每方言的 Top-2 簇索引
        lambda_outside: Top-2 外惩罚权重
        lambda_util: 簇利用率 KL 权重

    Returns:
        loss_dict: {"outside": ..., "util": ..., "total": ...}
    """
    num_dialects, K = full_probs.shape

    # Loss 1: Top-2 外概率惩罚（计划 §4.1）
    # 构造 Top-2 mask
    mask_top2 = torch.zeros_like(full_probs, dtype=torch.bool)
    mask_top2.scatter_(1, top2_indices, True)  # [N_d, K]

    outside_probs = full_probs.masked_fill(mask_top2, 0.0)  # Top-2 置零
    loss_outside = outside_probs.sum(dim=1).mean()  # 按方言平均

    # Loss 2: 簇利用率 KL（计划 §4.2）
    # 按方言类别平均（不被高资源方言主导）
    avg_probs = full_probs.mean(dim=0)  # [K]
    uniform = torch.ones_like(avg_probs) / K  # [K]

    # KL(avg_probs || uniform)
    loss_util = (avg_probs * (avg_probs.log() - uniform.log())).sum()

    # 总损失
    loss_total = lambda_outside * loss_outside + lambda_util * loss_util

    return {
        "outside": loss_outside.item(),
        "util": loss_util.item(),
        "total": loss_total.item(),
        "loss": loss_total,  # 用于 backward
    }

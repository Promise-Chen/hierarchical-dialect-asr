#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Global Dialect Calibration (Low-Rank FiLM) —— global_cluster_single_v2.md 第一部分

与 modules/film_layer.py 的区别（后者是 Samsung ICASSP utterance-conditioned FiLM）：
  - 纯方言条件，无 utterance 依赖：gamma/beta 只由 dialect id 决定，整句共享。
  - 低秩共享参数化：每方言只学低维系数 c_d ∈ R^{r_G}，乘全方言共享基底
    U ∈ R^{r_G×D} 得到完整 gamma_d/beta_d。不允许每方言自由学 512 维方向。
  - 注入点：norm1 之后、attention 之前（pre-norm SANM block 内），逐层独立强度。

公式（计划 §2.2 / §2.3）：
  gamma_d = c_d^gamma @ U^gamma        # [D]
  beta_d  = c_d^beta  @ U^beta         # [D]
  a_l     = tanh(alpha_l)              # 逐层可学标量强度
  H~_l    = (1 + a_l * gamma_d) ⊙ LN(H_l) + a_l * beta_d

零启动：alpha_l = 0 → a_l = 0 → H~_l = LN(H_l)，严格恒等于原始 pre-norm 行为。
因此插桩后 encoder 输出与原版 diff = 0，可先做 Stage 0 恒等验证再放开训练。
"""
import torch
import torch.nn as nn


class GlobalFiLM(nn.Module):
    """
    全局方言校准参数库 + 逐层调制强度。

    一个 GlobalFiLM 实例服务整个 encoder（所有插入层共享同一套 c_d / U，
    但每层有独立的 alpha_l 强度）。这样：
      - 方言层面的"整体分布偏移方向"由共享的 c_d/U 表达（跨层一致）；
      - 每层用多大力度施加这个方向，由 alpha_l 逐层学习。
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_dialects: int = 9,
        rank: int = 16,
        num_layers: int = 50,
        alpha_init: float = 0.0,
    ):
        """
        Args:
            hidden_size:  encoder 隐藏维 D (512)
            num_dialects: 方言数 (KeSpeech = 9)
            rank:         低秩系数维度 r_G (计划第一版 = 16)
            num_layers:   注入 FiLM 的层数（每层一个 alpha_l）
            alpha_init:   alpha_l 初值，0 → 恒等启动（保证 diff=0）
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_dialects = num_dialects
        self.rank = rank
        self.num_layers = num_layers

        # 方言特定低维系数 c_d^gamma / c_d^beta ∈ R^{num_dialects × r_G}
        self.c_gamma = nn.Parameter(torch.zeros(num_dialects, rank))
        self.c_beta = nn.Parameter(torch.zeros(num_dialects, rank))

        # 全方言共享调制基底 U^gamma / U^beta ∈ R^{r_G × D}
        self.U_gamma = nn.Parameter(torch.empty(rank, hidden_size))
        self.U_beta = nn.Parameter(torch.empty(rank, hidden_size))

        # 逐层调制强度 alpha_l（tanh 前）
        self.alpha = nn.Parameter(torch.full((num_layers,), float(alpha_init)))

        self._reset_parameters()

    def _reset_parameters(self):
        """
        初始化策略（保证零启动恒等）：
          - c_gamma / c_beta: 小随机（std=0.02），给方言方向一点初始区分度，
            但因 alpha=0 时不影响输出。
          - U_gamma / U_beta: Xavier，作为共享基底方向。
          - alpha: 已在 __init__ 用 alpha_init(=0) 填充。

        只要 alpha=0，无论 c/U 取何值，a_l*gamma=0、a_l*beta=0 → 输出恒等。
        """
        nn.init.normal_(self.c_gamma, std=0.02)
        nn.init.normal_(self.c_beta, std=0.02)
        nn.init.xavier_uniform_(self.U_gamma)
        nn.init.xavier_uniform_(self.U_beta)

    def compute_gamma_beta(self, dialect_ids: torch.Tensor):
        """
        Args:
            dialect_ids: [B] long
        Returns:
            gamma_d: [B, D]
            beta_d:  [B, D]
        """
        # c_d: [B, r_G] ；U: [r_G, D] → gamma_d: [B, D]
        c_g = self.c_gamma[dialect_ids]          # [B, r_G]
        c_b = self.c_beta[dialect_ids]           # [B, r_G]
        gamma_d = c_g @ self.U_gamma             # [B, D]
        beta_d = c_b @ self.U_beta               # [B, D]
        return gamma_d, beta_d

    def forward(
        self,
        x_ln: torch.Tensor,
        dialect_ids: torch.Tensor,
        layer_idx: int,
        return_diag: bool = False,
    ):
        """
        对 norm1 之后的表示施加全局方言调制。

        Args:
            x_ln:        [B, T, D] LN(H_l) 归一化后的表示
            dialect_ids: [B] long
            layer_idx:   int，当前是第几个 FiLM 插入点（索引 self.alpha）
            return_diag: True 时额外返回诊断字典（no-op 监控用）
        Returns:
            x_mod: [B, T, D]
            (可选) diag
        """
        gamma_d, beta_d = self.compute_gamma_beta(dialect_ids)   # [B, D]
        a_l = torch.tanh(self.alpha[layer_idx])                  # scalar

        gamma_eff = 1.0 + a_l * gamma_d      # [B, D]，初始 = 1
        beta_eff = a_l * beta_d              # [B, D]，初始 = 0

        x_mod = gamma_eff.unsqueeze(1) * x_ln + beta_eff.unsqueeze(1)

        if not return_diag:
            return x_mod

        with torch.no_grad():
            diff = x_mod - x_ln
            num = diff.pow(2).sum().sqrt()
            den = x_ln.pow(2).sum().sqrt().clamp(min=1e-8)
            diag = {
                "layer_idx": layer_idx,
                "a_l": a_l.item(),
                "rel_perturb": (num / den).item(),
                "gamma_abs_mean": gamma_d.abs().mean().item(),
                "beta_abs_mean": beta_d.abs().mean().item(),
            }
        return x_mod, diag

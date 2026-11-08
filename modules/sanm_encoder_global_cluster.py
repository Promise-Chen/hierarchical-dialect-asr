#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
SANMEncoder + Global FiLM + Cluster Adaptation —— Stage 1 完整版

架构（计划 §2-3）：
  1. 每层内 norm1 之后插入 Global FiLM（per-dialect 低秩调制）
  2. Encoder 输出后叠加 Cluster（方言簇路由 + Top-2 adapter 加权融合）

前向流程：
  H_0 → [norm1 → Global → attn → +res → norm2 → FFN → +res] × 49 → H^G
  H^G → Cluster Router → Top-2 Adapter → H^C

损失：
  L = L_ASR + λ_out * L_outside + λ_util * L_util

零启动：
  - Global: alpha=0 → a_l=0
  - Cluster: gate=0 → g^C=0
  → 初始严格等价于原始 SANMEncoder
"""
import os
import torch
import torch.nn as nn
from typing import Tuple, Optional
from funasr.models.sanm.encoder import SANMEncoder
from funasr.register import tables
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.models.transformer.utils.subsampling import (
    Conv2dSubsampling,
    Conv2dSubsampling2,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    check_short_utt,
    TooShortUttError,
)
from modules.global_film_layer import GlobalFiLM
from modules.cluster_layer import DialectClusterModule


@tables.register("encoder_classes", "SANMEncoderGlobalCluster")
class SANMEncoderGlobalCluster(SANMEncoder):
    def __init__(
        self,
        input_size: int,
        output_size: int = 512,
        # Global FiLM 参数
        num_accents: int = 9,
        global_film_rank: int = 16,
        global_film_layers: str = "all",
        global_film_alpha_init: float = 0.0,
        # Cluster 参数
        num_clusters: int = 4,
        cluster_bottleneck: int = 64,
        cluster_emb_dim: int = 32,
        cluster_temperature: float = 0.1,
        cluster_gate_init: float = 0.0,
        cluster_delta_type: str = "adapter",   # 诊断：adapter(默认) | film
        cluster_film_rank: int = 16,
        cluster_routing_type: str = "static",  # 诊断：static(默认) | dynamic
        cluster_routing_pool_dim: int = 128,
        # Residual 参数（Stage2）
        enable_dialect_residual: bool = False,
        residual_rank: int = 4,
        residual_dialect_emb_dim: int = 16,
        residual_pool_proj_dim: int = 32,
        **kwargs,
    ):
        if input_size is None:
            input_size = 560

        self._num_accents = num_accents
        self._global_film_rank = global_film_rank
        self._num_clusters = num_clusters

        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        super().__init__(input_size=input_size, output_size=output_size, **kwargs)

        n_enc = len(self.encoders)

        # Global FiLM 注入层（复用 GlobalFiLM 逻辑）
        env_idx = os.environ.get("GLOBAL_FILM_LAYERS", "").strip() or str(global_film_layers).strip()
        if env_idx.lower() == "all" or env_idx == "":
            self.film_layer_indices = list(range(n_enc))
        else:
            idx = [int(x) for x in env_idx.split(",") if x.strip() != ""]
            self.film_layer_indices = [i for i in idx if 0 <= i < n_enc]

        self.global_film = GlobalFiLM(
            hidden_size=output_size,
            num_dialects=num_accents,
            rank=global_film_rank,
            num_layers=len(self.film_layer_indices),
            alpha_init=global_film_alpha_init,
        )
        self._film_pos = {lidx: i for i, lidx in enumerate(self.film_layer_indices)}

        # Cluster 模块（作用在 encoder 输出后）
        self.cluster = DialectClusterModule(
            hidden_size=output_size,
            num_dialects=num_accents,
            num_clusters=num_clusters,
            bottleneck_dim=cluster_bottleneck,
            cluster_emb_dim=cluster_emb_dim,
            temperature=cluster_temperature,
            gate_init=cluster_gate_init,
            cluster_delta_type=cluster_delta_type,
            cluster_film_rank=cluster_film_rank,
            cluster_routing_type=cluster_routing_type,
            cluster_routing_pool_dim=cluster_routing_pool_dim,
            enable_dialect_residual=enable_dialect_residual,
            residual_rank=residual_rank,
            residual_dialect_emb_dim=residual_dialect_emb_dim,
            residual_pool_proj_dim=residual_pool_proj_dim,
        )

        self._last_film_diag = {}
        self._last_cluster_diag = {}

        # Dynamic gate 输入特征来源层（v2 文档 §6.1：H_m^C 来自中间层）
        # 指 self.encoders 循环内的 layer_idx（0-based，0~n_enc-1），可用 env 覆盖。
        # 设为 -1 表示不使用gate_feat（回退到h_c_base，最后层）
        gf = int(os.environ.get("GATE_FEAT_LAYER", "25"))
        if gf < 0:
            self._gate_feat_layer = None  # None = 不用gate_feat
        else:
            self._gate_feat_layer = max(0, min(gf, n_enc - 1))

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            gate_info = f"gate_feat_layer={self._gate_feat_layer}" if self._gate_feat_layer is not None else "gate_feat=disabled(用最后层)"
            print(
                f"[GlobalCluster-Encoder] encoders={n_enc}, "
                f"Global: rank={global_film_rank} @ {self.film_layer_indices}, "
                f"Cluster: K={num_clusters} bottleneck={cluster_bottleneck}, "
                f"{gate_info}"
            )

    def _run_layer(self, layer, x, mask, dialect_ids, film_idx, diag=False):
        """与 SANMEncoderGlobalFiLM 相同，内联 layer 计算插入 Global FiLM"""
        stoch_layer_coeff = 1.0
        skip_layer = False
        if layer.training and layer.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < layer.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - layer.stochastic_depth_rate)
        if skip_layer:
            return x, mask

        residual = x
        if layer.normalize_before:
            x = layer.norm1(x)

        # Global FiLM: norm1 之后、attention 之前
        if film_idx is not None:
            if diag:
                x, d = self.global_film(x, dialect_ids, film_idx, return_diag=True)
                self._last_film_diag[film_idx] = d
            else:
                x = self.global_film(x, dialect_ids, film_idx)

        if layer.concat_after:
            x_concat = torch.cat((x, layer.self_attn(x, mask)), dim=-1)
            if layer.in_size == layer.size:
                x = residual + stoch_layer_coeff * layer.concat_linear(x_concat)
            else:
                x = stoch_layer_coeff * layer.concat_linear(x_concat)
        else:
            if layer.in_size == layer.size:
                x = residual + stoch_layer_coeff * layer.dropout(layer.self_attn(x, mask))
            else:
                x = stoch_layer_coeff * layer.dropout(layer.self_attn(x, mask))
        if not layer.normalize_before:
            x = layer.norm1(x)

        residual = x
        if layer.normalize_before:
            x = layer.norm2(x)
        x = residual + stoch_layer_coeff * layer.dropout(layer.feed_forward(x))
        if not layer.normalize_before:
            x = layer.norm2(x)

        return x, mask

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        accent_ids: torch.Tensor,
        prev_states: torch.Tensor = None,
        ctc=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if accent_ids.dim() == 2:
            accent_ids = accent_ids.squeeze(1)
        accent_ids = accent_ids.long()

        bypass = int(os.environ.get("GLOBAL_CLUSTER_BYPASS", "0")) == 1
        diag = int(os.environ.get("GLOBAL_CLUSTER_DIAG", "0")) == 1
        self._last_film_diag = {}
        self._last_cluster_diag = {}

        masks = (~make_pad_mask(ilens)[:, None, :]).to(xs_pad.device)
        xs_pad = xs_pad * self.output_size() ** 0.5

        if self.embed is None:
            xs_pad = xs_pad
        elif isinstance(
            self.embed,
            (Conv2dSubsampling, Conv2dSubsampling2, Conv2dSubsampling6, Conv2dSubsampling8),
        ):
            short_status, limit_size = check_short_utt(self.embed, xs_pad.size(1))
            if short_status:
                raise TooShortUttError(
                    f"has {xs_pad.size(1)} frames and is too short for subsampling "
                    + f"(it needs more than {limit_size} frames), return empty results",
                    xs_pad.size(1),
                    limit_size,
                )
            xs_pad, masks = self.embed(xs_pad, masks)
        else:
            xs_pad = self.embed(xs_pad)

        # encoders0（首层，不注入 FiLM）
        encoder_outs = self.encoders0(xs_pad, masks)
        xs_pad, masks = encoder_outs[0], encoder_outs[1]

        # encoders（逐层内联，按需注入 Global FiLM）
        intermediate_outs = []
        gate_feat = None  # 中间层 H_m^C（供 dynamic gate 用，§6.1）
        for layer_idx, encoder_layer in enumerate(self.encoders):
            film_idx = None
            if (not bypass) and (layer_idx in self._film_pos):
                film_idx = self._film_pos[layer_idx]
            xs_pad, masks = self._run_layer(
                encoder_layer, xs_pad, masks, accent_ids, film_idx, diag=diag
            )

            # 捕获中间层输出作为 gate 输入特征（StopGrad：不让 gate 分支改主干）
            if self._gate_feat_layer is not None and layer_idx == self._gate_feat_layer:
                gate_feat = xs_pad.detach()

            if len(self.interctc_layer_idx) > 0 and (layer_idx + 1) in self.interctc_layer_idx:
                encoder_out = xs_pad
                if self.normalize_before:
                    encoder_out = self.after_norm(encoder_out)
                intermediate_outs.append((layer_idx + 1, encoder_out))
                if self.interctc_use_conditioning:
                    ctc_out = ctc.softmax(encoder_out)
                    xs_pad = xs_pad + self.conditioning_layer(ctc_out)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        # ⭐ Cluster 模块（作用在 encoder 输出 H^G 上）
        # 传 masks 供 Stage2 v2 attentive pooling；Stage1 cluster 忽略该参数
        if not bypass:
            if diag:
                xs_pad, cluster_diag = self.cluster(
                    xs_pad, accent_ids, mask=masks, gate_feat=gate_feat, return_diag=True)
                self._last_cluster_diag = cluster_diag
            else:
                xs_pad = self.cluster(xs_pad, accent_ids, mask=masks, gate_feat=gate_feat)

        olens = masks.squeeze(1).sum(1)
        if len(intermediate_outs) > 0:
            return (xs_pad, intermediate_outs), olens, None
        return xs_pad, olens, None

    def get_routing_loss(self, lambda_outside: float = 0.01, lambda_util: float = 0.01):
        """
        计算 Cluster 路由正则损失（供训练循环调用）

        Returns:
            loss_dict: {"outside": ..., "util": ..., "total": ..., "loss": Tensor}
        """
        from modules.cluster_layer import compute_routing_loss
        top2_indices, top2_weights, full_probs = self.cluster.get_last_routing()
        return compute_routing_loss(full_probs, top2_indices, lambda_outside, lambda_util)

    @torch.no_grad()
    def dump_routing_matrix(self, path: str, step: int = -1):
        """
        Dump 当前路由矩阵到 jsonl 文件（每行一条快照，供分析簇坍缩）。
        静态路由：[N_d, K]；动态路由：[B, K] 最后一个 batch。
        """
        import json
        top2_indices, top2_weights, full_probs = self.cluster.get_last_routing()
        rec = {
            "step": int(step),
            "gate": float(torch.tanh(self.cluster.gate).item()),
            "full_probs": full_probs.detach().cpu().tolist(),
            "top2_indices": top2_indices.detach().cpu().tolist(),
            "top2_weights": top2_weights.detach().cpu().tolist(),
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

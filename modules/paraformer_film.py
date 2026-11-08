#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Paraformer with FiLM (Feature-wise Linear Modulation)

基于 BiCifParaformer，要求输入 accent_id，透传给 FiLM encoder

Global+Cluster 训练时（encoder 提供 get_routing_loss）：
  在 ASR loss 之上叠加 Cluster 路由正则 L = L_ASR + λ_out·L_outside + λ_util·L_util
  由环境变量 ROUTING_LOSS=1 开启，λ 由 ROUTING_LAMBDA_OUTSIDE / ROUTING_LAMBDA_UTIL 控制。
  对不含 cluster 的老 FiLM 模型无影响（get_routing_loss 不存在时自动跳过）。
"""
import os
import torch
from torch.cuda.amp import autocast
from funasr.register import tables
from funasr.models.bicif_paraformer.model import BiCifParaformer


@tables.register("model_classes", "ParaformerFiLM")
class ParaformerFiLM(BiCifParaformer):
    """
    Paraformer with FiLM (Feature-wise Linear Modulation)

    继承 BiCifParaformer（与预训练 paraformer_zh 一致）。
    最小侵入：forward 暂存 accent_id → encode 透传给 encoder。
    """

    def forward(self, speech, speech_lengths, text, text_lengths, accent_id=None, **kwargs):
        """
        Args:
            accent_id: [B] or [B, 1] 方言标签（必需）
        """
        if accent_id is None:
            raise ValueError("accent_id is required for ParaformerFiLM")
        self._cur_accent_id = accent_id
        loss, stats, weight = super().forward(
            speech, speech_lengths, text, text_lengths, **kwargs
        )

        # ---- Cluster 路由正则（仅当 encoder 支持且开关打开）----
        if int(os.environ.get("ROUTING_LOSS", "0")) == 1 and hasattr(
            self.encoder, "get_routing_loss"
        ):
            lam_out = float(os.environ.get("ROUTING_LAMBDA_OUTSIDE", "0.01"))
            lam_util = float(os.environ.get("ROUTING_LAMBDA_UTIL", "0.01"))
            rl = self.encoder.get_routing_loss(lam_out, lam_util)
            r_loss = rl["loss"]  # 带梯度的标量 tensor
            # super 返回的 loss 已被 force_gatherable 变成 [1] 形状，标量相加会广播
            loss = loss + r_loss
            # 诊断项写入 stats（detach，供日志观察路由是否坍缩）
            dev = loss.device
            stats["loss_route"] = torch.tensor([rl["total"]], device=dev)
            # 计算所有层的平均 outside 和 util
            if rl["details"]:
                avg_outside = sum(d[list(d.keys())[0]]["outside"] for d in rl["details"]) / len(rl["details"])
                avg_util = sum(d[list(d.keys())[0]]["util"] for d in rl["details"]) / len(rl["details"])
                stats["route_outside"] = torch.tensor([avg_outside], device=dev)
                stats["route_util"] = torch.tensor([avg_util], device=dev)

            # rank 0 每 ROUTING_DUMP_EVERY 步 dump 一次 9×K 路由矩阵
            if int(os.environ.get("LOCAL_RANK", "0")) == 0 and self.training:
                self._route_step = getattr(self, "_route_step", 0) + 1
                every = int(os.environ.get("ROUTING_DUMP_EVERY", "500"))
                dump_path = os.environ.get("ROUTING_DUMP_PATH", "")
                if dump_path and hasattr(self.encoder, "dump_routing_matrix") and \
                        self._route_step % every == 1:
                    try:
                        self.encoder.dump_routing_matrix(dump_path, step=self._route_step)
                    except Exception as e:
                        print(f"[routing-dump] 跳过（{e}）", flush=True)

        # ---- 残差 Frobenius 正则 L_res（v2 文档 §9，仅当 encoder 支持且开关打开）----
        if int(os.environ.get("RES_LOSS", "0")) == 1 and hasattr(
            self.encoder, "get_res_loss"
        ):
            lam_res = float(os.environ.get("RES_LAMBDA", "0.01"))
            rl = self.encoder.get_res_loss(lam_res)
            loss = loss + rl["loss"]  # 带梯度标量，广播到 [1]
            dev = loss.device
            stats["loss_res"] = torch.tensor([rl["total"]], device=dev)
            stats["res_raw"] = torch.tensor([rl["res"]], device=dev)

        return loss, stats, weight

    def encode(self, speech, speech_lengths, **kwargs):
        """
        覆盖 encode 方法，传递 accent_id 给 encoder
        """
        accent_id = getattr(self, "_cur_accent_id", None)
        if accent_id is None:
            raise ValueError("accent_id not set; call forward() or set _cur_accent_id")

        # SpecAug + Normalize (保持原逻辑)
        with autocast(False):
            if self.specaug is not None and self.training:
                speech, speech_lengths = self.specaug(speech, speech_lengths)
            if self.normalize is not None:
                speech, speech_lengths = self.normalize(speech, speech_lengths)

        # Encoder forward (传递 accent_id)
        encoder_out, encoder_out_lens, _ = self.encoder(speech, speech_lengths, accent_id)

        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        return encoder_out, encoder_out_lens

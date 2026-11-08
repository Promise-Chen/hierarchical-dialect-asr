#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
SANMEncoderStage2V2 —— 严格按 global_cluster_single_v2.md §5-7 实现第三部分

与失败版 SANMEncoderStage2 的根本区别：
  ✗ 失败版：独立 bottleneck adapter 插 encoder 24-34 层 + 静态标量 gate
            → 双零初始化死锁（gate=0/up=0 互锁），CER 精确复现 6.65% no-op
  ✓ 本版（v2 文档第三部分）：per-dialect 低秩残差 ΔW=A·B（rank r_D=4）寄生在
            Cluster Adapter 投影权重上 + Dynamic Gate（语句 attentive pool + 方言 emb 生成）
            → 反死锁初始化（B=0 恒等启动，A Xavier 破对称），DDP 稠密安全

实现完全落在扩展后的 DialectClusterModule（enable_dialect_residual=True）里，
本类只负责用正确参数重建 self.cluster，其余（Global FiLM、逐层前向、路由损失）全继承。
"""
import os
from modules.sanm_encoder_global_cluster import SANMEncoderGlobalCluster
from modules.cluster_layer import DialectClusterModule
from funasr.register import tables


@tables.register("encoder_classes", "SANMEncoderStage2V2")
class SANMEncoderStage2V2(SANMEncoderGlobalCluster):
    def __init__(
        self,
        input_size: int,
        output_size: int = 512,
        num_accents: int = 9,
        global_film_rank: int = 16,
        global_film_layers: str = "all",
        global_film_alpha_init: float = 0.0,
        num_clusters: int = 4,
        cluster_bottleneck: int = 64,
        cluster_emb_dim: int = 32,
        cluster_temperature: float = 0.1,
        cluster_gate_init: float = 0.0,
        # 第三部分 —— Dynamic Dialect Residual
        residual_rank: int = 4,
        residual_dialect_emb_dim: int = 16,
        residual_pool_proj_dim: int = 32,
        residual_att_dim: int = 128,
        **kwargs,
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            num_accents=num_accents,
            global_film_rank=global_film_rank,
            global_film_layers=global_film_layers,
            global_film_alpha_init=global_film_alpha_init,
            num_clusters=num_clusters,
            cluster_bottleneck=cluster_bottleneck,
            cluster_emb_dim=cluster_emb_dim,
            cluster_temperature=cluster_temperature,
            cluster_gate_init=cluster_gate_init,
            **kwargs,
        )

        # 重建 cluster：开启第三部分动态残差。
        # 基础 cluster 权重（adapters/gate/router）名称不变 → Stage1 checkpoint 严格加载；
        # 新增 dialect_residual/residual_* 是新参数（init_param 时 ignore_init_mismatch 跳过）。
        self.cluster = DialectClusterModule(
            hidden_size=output_size,
            num_dialects=num_accents,
            num_clusters=num_clusters,
            bottleneck_dim=cluster_bottleneck,
            cluster_emb_dim=cluster_emb_dim,
            temperature=cluster_temperature,
            gate_init=cluster_gate_init,
            enable_dialect_residual=True,
            residual_rank=residual_rank,
            residual_dialect_emb_dim=residual_dialect_emb_dim,
            residual_pool_proj_dim=residual_pool_proj_dim,
            residual_att_dim=residual_att_dim,
        )

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            n_res = sum(p.numel() for n, p in self.cluster.named_parameters()
                        if ("dialect_residual" in n or "residual_" in n))
            print(
                f"[Stage2V2-Encoder] Dynamic Dialect Residual: rank={residual_rank}, "
                f"dialect_emb={residual_dialect_emb_dim}, pool_proj={residual_pool_proj_dim}, "
                f"新增参数={n_res/1e3:.1f}K",
                flush=True,
            )

    def get_res_loss(self, lambda_res: float = 0.01):
        """
        残差 Frobenius 正则（v2 文档 §9）：λ_res·L_res，供训练循环调用。
        Returns: {"res": L_res(float), "total": λ_res·L_res(float), "loss": Tensor(带梯度)}
        """
        lres = self.cluster.dialect_residual.res_loss()  # 标量 tensor，带梯度
        return {"res": lres.item(), "total": (lambda_res * lres).item(),
                "loss": lambda_res * lres}

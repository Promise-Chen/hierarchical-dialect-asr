#!/usr/bin/env python3
"""
Global + Cluster 训练入口 —— Stage 1 完整版

策略：
  - 冻结主干（默认），只训 global_film.* + cluster.*
  - 损失：L_ASR + λ_out * L_outside + λ_util * L_util
  - global_film / cluster 单独分组，可用不同 lr 倍数

环境变量：
  FREEZE_BACKBONE          默认 1（冻结）；0 = 全解冻联合训
  GLOBAL_FILM_LR_MULT      global_film lr 倍数，默认 15
  CLUSTER_LR_MULT          cluster lr 倍数，默认 15
  LAMBDA_OUTSIDE           Top-2 外惩罚权重，默认 0.01
  LAMBDA_UTIL              簇利用率 KL 权重，默认 0.01
"""
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from funasr.register import tables
import modules  # noqa: F401

if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print("[global-cluster] ✅ 自定义模块已注册", flush=True)

import torch
from funasr.train_utils import trainer_ds as _td


def freeze_and_scale_lr(self, model, **kwargs):
    """冻结主干，只训 global_film + cluster；两者可用不同 lr。支持额外冻结 predictor。"""
    from funasr.optimizers import optim_classes
    from funasr.schedulers import scheduler_classes

    real = model.module if hasattr(model, "module") else model
    optim_conf = dict(kwargs.get("optim_conf", {}))
    base_lr = optim_conf.get("lr", 2e-5)

    film_mult = float(os.environ.get("GLOBAL_FILM_LR_MULT", "15"))
    cluster_mult = float(os.environ.get("CLUSTER_LR_MULT", "15"))
    freeze_backbone = int(os.environ.get("FREEZE_BACKBONE", "1"))
    freeze_predictor = int(os.environ.get("FREEZE_PREDICTOR", "0"))  # 新增
    freeze_global_film = int(os.environ.get("FREEZE_GLOBAL_FILM", "0"))  # 新增
    freeze_cluster = int(os.environ.get("FREEZE_CLUSTER", "0"))  # 新增

    film_params, cluster_params, predictor_params, main_params = [], [], [], []
    n_film, n_cluster, n_pred, n_main = 0, 0, 0, 0

    for name, p in real.named_parameters():
        if "global_film" in name:
            p.requires_grad = True
            if freeze_global_film == 0:
                film_params.append(p)
            n_film += p.numel()
        elif "cluster" in name:
            p.requires_grad = True
            if freeze_cluster == 0:
                cluster_params.append(p)
            n_cluster += p.numel()
        elif "predictor" in name:
            p.requires_grad = True
            if freeze_predictor == 0:
                predictor_params.append(p)
            n_pred += p.numel()
        else:
            p.requires_grad = True  # 保持 True 让梯度流通
            if freeze_backbone == 0:
                main_params.append(p)
            n_main += p.numel()

    film_conf = dict(optim_conf)
    film_conf["lr"] = base_lr * film_mult

    cluster_conf = dict(optim_conf)
    cluster_conf["lr"] = base_lr * cluster_mult

    if freeze_backbone:
        groups = []
        if film_params:
            groups.append({"params": film_params, **film_conf})
        if cluster_params:
            groups.append({"params": cluster_params, **cluster_conf})
        if not freeze_predictor and predictor_params:
            groups.append({"params": predictor_params, "lr": base_lr})

        train_parts = []
        if film_params:
            train_parts.append(f"global_film")
        if cluster_params:
            train_parts.append(f"cluster")
        if not freeze_predictor and predictor_params:
            train_parts.append(f"predictor {n_pred/1e6:.1f}M")

        mode_desc = (f"冻结主干 {n_main/1e6:.1f}M，只训 " +
                     " + ".join(train_parts) if train_parts else "全部冻结")
    else:
        groups = [
            {"params": main_params, "lr": base_lr},
            {"params": film_params, **film_conf},
            {"params": cluster_params, **cluster_conf},
        ]
        if predictor_params:
            groups.append({"params": predictor_params, "lr": base_lr})
        mode_desc = f"主干+分支都训（主干 {n_main/1e6:.1f}M @ {base_lr:.2e}）"

    optim = optim_classes.get(kwargs.get("optim", "adam"))(groups, **optim_conf)
    scheduler = scheduler_classes.get(kwargs.get("scheduler", "warmuplr"))(
        optim, **kwargs.get("scheduler_conf", {}))

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[global-cluster] 🔧 {mode_desc}", flush=True)
        if film_params:
            print(f"[global-cluster] ✅ global_film: {n_film/1e3:.1f}K @ {base_lr*film_mult:.2e}", flush=True)
        if cluster_params:
            print(f"[global-cluster] ✅ cluster: {n_cluster/1e3:.1f}K @ {base_lr*cluster_mult:.2e}", flush=True)

    return model, optim, scheduler


_td.Trainer.warp_optim_scheduler = freeze_and_scale_lr

# ⚠️ TODO: 路由损失集成（需改 ParaformerFiLM wrapper 在 forward 里加）
# 当前版本只训 ASR loss，路由损失暂时跳过

try:
    import _ddp_save_patch  # noqa
except Exception as e:
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[global-cluster] ⚠️  _ddp_save_patch import failed: {e}", flush=True)

if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print("[global-cluster] ✅ patches applied", flush=True)
    print("[global-cluster] ⚠️  路由损失暂未集成（需改 model wrapper），先跑 ASR loss only", flush=True)

from funasr.bin.train_ds import main_hydra
main_hydra()


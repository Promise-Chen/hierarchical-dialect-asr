#!/usr/bin/env python3
"""
Stage 2 修正版训练入口 —— 冻结backbone + 解冻Cluster + 训Residual

策略：
  - 冻结 backbone（主干）
  - 解冻 global_film + cluster（极小lr，允许梯度流动）
  - 训练 dialect_residual（正常lr）

环境变量：
  FREEZE_BACKBONE   默认 1（冻结主干）
  GLOBAL_LR_MULT    global_film lr倍数，默认1（极小）
  CLUSTER_LR_MULT   cluster lr倍数，默认1（极小）
  DIALECT_LR_MULT   residual lr倍数，默认50（正常）
"""
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from funasr.register import tables
import modules  # noqa: F401

if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print("[stage2-fixed] ✅ 自定义模块已注册", flush=True)

import torch
from funasr.train_utils import trainer_ds as _td


def _is_residual_param(name: str) -> bool:
    """第三部分新参数：低秩残差 bank + gate 网络 + pool + 方言 emb"""
    return ("dialect_residual" in name) or ("residual_" in name)


def freeze_backbone_train_cluster_residual(self, model, **kwargs):
    """冻结backbone，解冻cluster（极小lr），训练residual（正常lr）"""
    from funasr.optimizers import optim_classes
    from funasr.schedulers import scheduler_classes

    real = model.module if hasattr(model, "module") else model
    optim_conf = dict(kwargs.get("optim_conf", {}))
    base_lr = optim_conf.get("lr", 2e-5)

    global_mult = float(os.environ.get("GLOBAL_LR_MULT", "1"))
    cluster_mult = float(os.environ.get("CLUSTER_LR_MULT", "1"))
    dialect_mult = float(os.environ.get("DIALECT_LR_MULT", "50"))
    freeze_backbone = int(os.environ.get("FREEZE_BACKBONE", "1"))

    residual_params, global_params, cluster_params, backbone_params = [], [], [], []
    n_res, n_global, n_cluster, n_backbone = 0, 0, 0, 0

    for name, p in real.named_parameters():
        p.requires_grad = True  # DDP 要求
        if _is_residual_param(name):
            residual_params.append(p)
            n_res += p.numel()
        elif "global_film" in name:
            global_params.append(p)
            n_global += p.numel()
        elif "cluster" in name:
            cluster_params.append(p)
            n_cluster += p.numel()
        else:
            backbone_params.append(p)
            n_backbone += p.numel()

    res_conf = dict(optim_conf); res_conf["lr"] = base_lr * dialect_mult
    global_conf = dict(optim_conf); global_conf["lr"] = base_lr * global_mult
    cluster_conf = dict(optim_conf); cluster_conf["lr"] = base_lr * cluster_mult

    if freeze_backbone:
        groups = [
            {"params": residual_params, **res_conf},
            {"params": global_params, **global_conf},
            {"params": cluster_params, **cluster_conf},
            {"params": backbone_params, "lr": 0.0},
        ]
        mode_desc = (f"冻结 backbone {n_backbone/1e6:.1f}M，"
                     f"解冻 global {n_global/1e3:.1f}K @ {base_lr*global_mult:.2e}，"
                     f"解冻 cluster {n_cluster/1e3:.1f}K @ {base_lr*cluster_mult:.2e}，"
                     f"训 residual {n_res/1e3:.1f}K @ {base_lr*dialect_mult:.2e}")
    else:
        groups = [
            {"params": backbone_params, "lr": base_lr},
            {"params": global_params, **global_conf},
            {"params": cluster_params, **cluster_conf},
            {"params": residual_params, **res_conf},
        ]
        mode_desc = f"全部解冻联合训"

    optim = optim_classes.get(kwargs.get("optim", "adam"))(groups, **optim_conf)
    scheduler = scheduler_classes.get(kwargs.get("scheduler", "warmuplr"))(
        optim, **kwargs.get("scheduler_conf", {}))

    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[stage2-fixed] 🔧 {mode_desc}", flush=True)
        print(f"[stage2-fixed] ✅ residual: {n_res/1e3:.1f}K @ {base_lr*dialect_mult:.2e}", flush=True)
        print(f"[stage2-fixed] ✅ global: {n_global/1e3:.1f}K @ {base_lr*global_mult:.2e}", flush=True)
        print(f"[stage2-fixed] ✅ cluster: {n_cluster/1e3:.1f}K @ {base_lr*cluster_mult:.2e}", flush=True)
        print(f"[stage2-fixed] 🔒 backbone: {n_backbone/1e6:.1f}M @ {'frozen' if freeze_backbone else base_lr}", flush=True)

    return model, optim, scheduler


# Monkey patch - 使用正确的方法名
_td.Trainer.warp_optim_scheduler = freeze_backbone_train_cluster_residual

# 修复 save_checkpoint 的 KeyError（mid-epoch保存时val_acc缺失）
try:
    import _ddp_save_patch  # noqa
except Exception as e:
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"[stage2-fixed] ⚠️  _ddp_save_patch import failed: {e}", flush=True)

if int(os.environ.get("LOCAL_RANK", 0)) == 0:
    print("[stage2-fixed] ✅ patches applied", flush=True)

# 使用与 train_stage2v2_entry.py 相同的调用方式
from funasr.bin.train_ds import main_hydra
main_hydra()

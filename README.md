# Hierarchical Dialect-Adaptive ASR on Paraformer

> Coarse-to-fine dialect adaptation for multi-dialect Mandarin speech recognition.


A parameter-efficient adaptation stack built on top of a pretrained **Paraformer**
(FunASR) encoder. It models dialect differences in three tiers — a global
calibration, a shared cross-dialect layer, and an utterance-level per-dialect
residual — trained as lightweight adapters while the backbone stays frozen.



Evaluated on the **KeSpeech** corpus (9 Mandarin dialects). The full stack lowers
average CER from **6.81% → 6.56%** over the fine-tuned Paraformer baseline, while
training only a small fraction of adapter parameters.
。

## Architecture 

```
          H  (frozen Paraformer encoder hidden states)
          │
          ▼
 ┌─────────────────────────────┐
 │ 1. Global Dialect Calibration│  Low-rank FiLM, per-dialect γ/β
 │                │  coarse, utterance-wide shift
 └─────────────────────────────┘
          │  H^G
          ▼
 ┌─────────────────────────────┐
 │ 2. Latent Dialect Cluster    │  learnable prototypes + Top-2 routing
 │                │  K bottleneck cluster adapters
 └─────────────────────────────┘
          │  H^C
          ▼
 ┌─────────────────────────────┐
 │ 3. Dynamic Dialect Residual  │  per-dialect low-rank ΔW (rank 4)
 │                │  gated by dialect emb + utterance
 └─────────────────────────────┘
          │  H^D  →  Predictor → Decoder → text
```

| Tier | Mechanism | Conditioned on | Role |
|---|---|---|---|
| Global | Low-rank FiLM | dialect label | coarse, utterance-wide distribution calibration |
| Cluster | Prototype router + Top-2 adapters | dialect label | latent cross-dialect sharing (auto-learned grouping) |
| Residual | Low-rank ΔW × dynamic gate | dialect label + current utterance | fine-grained per-utterance personalization |

Two routing regularizers keep the cluster layer healthy: a **Top-2 concentration**
term (each dialect focuses on a few clusters) and a **utilization KL** term
(clusters are not all collapsed onto a few experts). Full derivations and the loss
definitions are in [`docs/architecture.md`](docs/architecture.md).

> 路由采用两个互补正则项：**Top-2 集中约束**（单方言集中使用少数簇）
> 与**簇利用率 KL**（防止所有方言坍缩到同一批簇）。完整公式见
> [`docs/architecture.md`](docs/architecture.md)。

## Repository layout 

```
modules/
  paraformer_film.py              ParaformerFiLM — BiCifParaformer wrapper, threads dialect_ids
  sanm_encoder_global_cluster.py  Stage 1 encoder: Global FiLM + dialect-cluster routing
  sanm_encoder_stage2_v2.py       Stage 2 encoder: + dynamic per-dialect residual
  global_film_layer.py            low-rank FiLM modulation
  cluster_layer.py                cluster router, adapters, residual bank, attentive pool
  audio_dataset_with_accent.py    dataset class carrying a per-utterance dialect label
  index_ds_with_accent.py         jsonl index reader that parses the dialect field
configs/
  paraformer_global_cluster.yaml  Stage 1 config (Global + Cluster)
  paraformer_stage2_v2.yaml       Stage 2 config (+ Dynamic Residual)
scripts/
  train_global_cluster_entry.py   Stage 1 training entry (freeze backbone, train adapters)
  train_stage2_fixed_entry.py     Stage 2 training entry (train residual only)
  50_eval_cer.sh                  overall + per-dialect CER report (jiwer)
docs/
  architecture.md                 full method write-up with equations
  final_cer_report.txt            per-dialect CER on the KeSpeech test set
```

## Setup 

```bash
conda env create -f environment.yml
conda activate paraformer
```

The pretrained Paraformer-zh checkpoint (model, `am.mvn`, `tokens.json`, `seg_dict`)
is downloaded separately from ModelScope. Point the configs at it via an environment
variable — all paths in the YAML use `${PRETRAINED_DIR}` and `${STAGE1_CKPT}`
placeholders instead of hardcoded absolute paths.


> `${PRETRAINED_DIR}` / `${STAGE1_CKPT}` 占位符，请指向你本地的实际路径。

```bash
export PRETRAINED_DIR=/path/to/pretrained_models
export STAGE1_CKPT=exp/stage1_global_cluster/model.pt.avg3
```

### Data format 

Each training example is a jsonl record carrying a `dialect` label alongside the
audio path and transcript, consumed by `IndexDSJsonlWithAccent` /
`AudioDatasetWithAccent`. Experiments use the **KeSpeech** corpus (~1,428 h,
9 Mandarin dialects).

> 每条样本为带 `dialect` 字段的 jsonl 记录，由 `IndexDSJsonlWithAccent` 读取。
> 实验基于 **KeSpeech** 语料库（约 1,428 小时，9 种汉语方言）。

## Training

The model is trained in three stages (shared structure → per-dialect residual →
joint fine-tune). Backbone stays frozen; only the lightweight adapters are trained.

```bash
# Stage 1: Global FiLM + dialect-cluster adapters (backbone frozen)
torchrun --nproc_per_node=2 scripts/train_global_cluster_entry.py \
  --config-path configs --config-name paraformer_global_cluster

# Stage 2: dynamic per-dialect low-rank residual (Stage 1 frozen)
torchrun --nproc_per_node=2 scripts/train_stage2_fixed_entry.py \
  --config-path configs --config-name paraformer_stage2_v2
```

Layer freezing and per-group learning rates are controlled by environment variables
documented at the top of each entry script (e.g. `FREEZE_BACKBONE`,
`CLUSTER_LR_MULT`, `DIALECT_LR_MULT`).

## Evaluation 

```bash
bash scripts/50_eval_cer.sh <run_name>   # overall + per-dialect CER via jiwer
```

## Results

Character Error Rate (CER) on the KeSpeech test set (19,723 utterances, 9 dialects).
Adapting only lightweight low-rank modules on top of a frozen Paraformer backbone:

| | Overall CER |
|---|---|
| Paraformer baseline | 6.81% |
| + Hierarchical dialect adaptation | **6.56%** |

Per-dialect breakdown is in [`docs/final_cer_report.txt`](docs/final_cer_report.txt).


> [`docs/final_cer_report.txt`](docs/final_cer_report.txt)。

## Notes

- This repository contains the model code, configs, and training/eval entry points
  for the final architecture. Training data, pretrained checkpoints, and experiment
  outputs are not included (see `.gitignore`).
- Built on [FunASR](https://github.com/modelscope/FunASR); the custom classes
  register into FunASR's table registry at import time.

。

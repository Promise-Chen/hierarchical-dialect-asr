"""
Custom module registration for the hierarchical dialect-adaptive ASR stack.

Importing this package triggers the @tables.register decorators so FunASR can
resolve the custom model / encoder / dataset classes referenced in the YAML
configs under configs/.

Core architecture (see docs/architecture.md):
  - ParaformerFiLM              : BiCifParaformer wrapper that threads dialect_ids
  - SANMEncoderGlobalCluster    : Stage 1 encoder  (Global FiLM + dialect-cluster routing)
  - SANMEncoderStage2V2         : Stage 2 encoder  (+ dynamic per-dialect low-rank residual)
  - AudioDatasetWithAccent      : dataset class that carries a dialect label per utterance
  - IndexDSJsonlWithAccent      : jsonl index reader that parses the dialect field
"""

from modules.paraformer_film import ParaformerFiLM
from modules.sanm_encoder_global_cluster import SANMEncoderGlobalCluster
from modules.sanm_encoder_stage2_v2 import SANMEncoderStage2V2
from modules.audio_dataset_with_accent import AudioDatasetWithAccent
from modules.index_ds_with_accent import IndexDSJsonlWithAccent

__all__ = [
    "ParaformerFiLM",
    "SANMEncoderGlobalCluster",
    "SANMEncoderStage2V2",
    "AudioDatasetWithAccent",
    "IndexDSJsonlWithAccent",
]

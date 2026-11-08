"""
IndexDSJsonlWithAccent — 保留原始 manifest 中所有字段（包括 accent_id）的 IndexDS。

为什么需要：FunASR 自带的 IndexDSJsonl 在加载 jsonl 时只保留 5 个字段
(source / prompt / target / source_len / target_len)，把 accent_id、key、dialect
等字段全部丢弃了。

本类做最小改动：完整解析 jsonl，保留每条原 dict（同时补齐 source_len/target_len 字段，
避免下游 batch_sampler 找不到长度）。
"""

import json
import logging
import torch
from funasr.register import tables


@tables.register("index_ds_classes", "IndexDSJsonlWithAccent")
class IndexDSJsonlWithAccent(torch.utils.data.Dataset):

    def __init__(self, path: str, **kwargs):
        super().__init__()
        self.max_source_length = kwargs.get("max_source_length", 2048)
        self.min_source_length = kwargs.get("min_source_length", 0)
        self.max_target_length = kwargs.get("max_target_length", 2048)
        self.min_target_length = kwargs.get("min_target_length", 0)
        self.max_token_length = kwargs.get("max_token_length", 2200)
        is_training = kwargs.get("is_training", True)

        # 兼容 jsonl 列表文件 / 单 jsonl
        if not (path.endswith(".jsonl") or path.endswith(".json")):
            data_split_num = kwargs.get("data_split_num", 1)
            data_split_i = kwargs.get("data_split_i", 0)
            if not is_training:
                data_split_num, data_split_i = 1, 0
            with open(path, encoding="utf-8") as fin:
                file_list_all = fin.readlines()
            num_per_slice = (len(file_list_all) - 1) // data_split_num + 1
            file_list = file_list_all[data_split_i * num_per_slice : (data_split_i + 1) * num_per_slice]
        else:
            file_list = [path]

        contents = []
        n_skipped_no_accent = 0
        for file_json in file_list:
            with open(file_json.strip(), encoding="utf-8") as fin:
                for line in fin:
                    data = json.loads(line.strip())
                    # 只处理标准 ASR 样本（必须有 source 和 target）
                    if "source" not in data or "target" not in data:
                        continue

                    # 如果没有 accent_id 但有 dialect，将 dialect 映射到 accent_id
                    if "accent_id" not in data:
                        if "dialect" in data:
                            # KeSpeech 方言 -> accent_id 映射
                            dialect_to_id = {
                                "Beijing": 0, "Ji-Lu": 1, "Jiang-Huai": 2, "Jiao-Liao": 3,
                                "Lan-Yin": 4, "Mandarin": 5, "Northeastern": 6,
                                "Southwestern": 7, "Zhongyuan": 8
                            }
                            dialect_name = data["dialect"]
                            if dialect_name in dialect_to_id:
                                data["accent_id"] = dialect_to_id[dialect_name]
                            else:
                                n_skipped_no_accent += 1
                                continue
                        else:
                            n_skipped_no_accent += 1
                            continue

                    # 补齐长度字段（下游 BatchSampler 用 get_source_len 取这个）
                    data.setdefault("source_len", 1)
                    data.setdefault("target_len", len(data["target"]) if isinstance(data["target"], str) else 0)
                    # 保留完整 dict，accent_id 和其他字段全在
                    contents.append(data)

        self.contents = contents
        if n_skipped_no_accent:
            logging.warning(f"[IndexDSJsonlWithAccent] 跳过 {n_skipped_no_accent} 条缺 accent_id 的样本")
        logging.info(f"total_num of samplers: {len(self.contents)}, {path}")

        # 自检：确认第一条确实带 accent_id
        if self.contents:
            sample = self.contents[0]
            assert "accent_id" in sample, f"内部错误：contents 里居然有不带 accent_id 的样本：{sample}"

    def __len__(self):
        return len(self.contents)

    def __getitem__(self, index):
        return self.contents[index]

    def get_source_len(self, data_dict):
        return data_dict.get("source_len", 1)

    def get_target_len(self, data_dict):
        return data_dict.get("target_len", 0)

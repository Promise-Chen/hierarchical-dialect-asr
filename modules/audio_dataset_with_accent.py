"""
AudioDatasetWithAccent — 透传 accent_id 字段到 batch 的 Dataset
继承 FunASR 的 AudioDataset，最小侵入：只多返回一个 accent_id tensor。

核心修复：FunASR 原生 AudioDataset.__getitem__ 硬编码返回字段
{speech, speech_lengths, text, text_lengths}，永远丢掉 accent_id。
本类把它加回来，collator 是 generic 的（实测代码看过），会自动 batch。
"""

import torch
from funasr.datasets.audio_datasets.datasets import AudioDataset
from funasr.register import tables


@tables.register("dataset_classes", "AudioDatasetWithAccent")
class AudioDatasetWithAccent(AudioDataset):
    """
    在父类 __getitem__ 的返回字典里加上 accent_id。
    其它字段（speech/speech_lengths/text/text_lengths）原样保留。

    要求：manifest 每条都有 accent_id 字段（KeSpeech 100% 已加）。
    """

    def __getitem__(self, index):
        # 父类做完音频特征提取和文本 tokenize
        ret = super().__getitem__(index)

        # 从 manifest 重新拿这一条的元数据（IndexDSJsonl 直接返回 manifest dict）
        item = self.index_ds[index]

        # 防御：缺标签直接报错，不静默丢弃
        assert "accent_id" in item, (
            f"[AudioDatasetWithAccent] 样本缺 accent_id 字段: key={item.get('key', '?')}。"
            "请检查 manifest 是否用了 add_accent_id.py 处理过。"
        )
        accent_id = item["accent_id"]
        assert isinstance(accent_id, int) and 0 <= accent_id < 9, (
            f"[AudioDatasetWithAccent] accent_id 不合法: {accent_id} "
            f"（key={item.get('key', '?')}）。应为 [0, 9) 整数。"
        )

        # 注意：单样本里写成长度=1 的 1D tensor，collator 会 pad_sequence 成 [B, 1]
        # 后面在 ParaformerWithAccent.encode 里会 squeeze 成 [B]
        ret["accent_id"] = torch.tensor([accent_id], dtype=torch.int64)
        return ret

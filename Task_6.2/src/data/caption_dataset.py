"""Pairs a cached image feature with one caption. Each of the 5 captions per image is its own sample."""
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from src.text.tokenizer import Tokenizer


class CaptionDataset(Dataset):
    def __init__(
        self,
        image_ids: List[str],
        captions_map: Dict[str, List[str]],
        features: Dict[str, "np.ndarray"],
        tokenizer: Tokenizer,
    ):
        self.tokenizer = tokenizer
        self.samples: List[Tuple[str, str]] = [
            (img_id, cap)
            for img_id in image_ids
            for cap in captions_map.get(img_id, [])
            if img_id in features
        ]
        self.features = features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, caption = self.samples[idx]
        feat = torch.tensor(self.features[img_id], dtype=torch.float32)
        ids = torch.tensor(self.tokenizer.encode(caption), dtype=torch.long)
        return feat, ids

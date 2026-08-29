"""
Flickr8k loading, cleaning, splitting.

Expects the Kaggle layout (adityajn105/flickr8k):
    data/flickr8k/Images/*.jpg
    data/flickr8k/captions.txt   (columns: image,caption)
"""
import os
import re
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import pandas as pd

from src.config import CONFIG


def clean_caption(text: str) -> str:
    """Lowercase, strip punctuation/numbers, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_captions(captions_file: str = None) -> Dict[str, List[str]]:
    """Read captions.txt into {image_filename: [caption1, ..., caption5]}."""
    captions_file = captions_file or CONFIG.captions_file
    df = pd.read_csv(captions_file)
    df.columns = [c.strip().lower() for c in df.columns]  # image, caption

    mapping: Dict[str, List[str]] = defaultdict(list)
    for _, row in df.iterrows():
        img_id = row["image"].strip()
        caption = clean_caption(str(row["caption"]))
        if caption:
            mapping[img_id].append(caption)
    return dict(mapping)


def split_dataset(
    mapping: Dict[str, List[str]],
    val_ratio: float = None,
    test_ratio: float = None,
    seed: int = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Split by image id, not caption, so all 5 captions of an image land in the same split."""
    val_ratio = val_ratio if val_ratio is not None else CONFIG.val_ratio
    test_ratio = test_ratio if test_ratio is not None else CONFIG.test_ratio
    seed = seed if seed is not None else CONFIG.random_seed

    image_ids = sorted(mapping.keys())
    rng = random.Random(seed)
    rng.shuffle(image_ids)

    n = len(image_ids)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)

    val_ids = image_ids[:n_val]
    test_ids = image_ids[n_val:n_val + n_test]
    train_ids = image_ids[n_val + n_test:]
    return train_ids, val_ids, test_ids


def image_path(image_id: str) -> str:
    return os.path.join(CONFIG.images_dir, image_id)


if __name__ == "__main__":
    mapping = load_captions()
    train, val, test = split_dataset(mapping)
    print(f"Images: {len(mapping)} | train={len(train)} val={len(val)} test={len(test)}")

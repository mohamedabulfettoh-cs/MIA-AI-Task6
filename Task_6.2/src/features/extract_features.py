"""
Extracts image features using a pretrained, frozen CNN backbone.

Keeps the spatial feature map (before pooling) instead of collapsing it
to one vector, so attention actually has something to attend over.

Backbones supported: inception_v3, resnet50, efficientnet_b0
Features get cached to disk as a pickle so we're not re-running the CNN
every epoch. Heads up, the spatial maps take a lot more space than a
pooled vector would (InceptionV3 at 8x8x2048 floats is roughly 512KB
per image, so around 4GB for all of Flickr8k). Fine for local/Kaggle,
switch to efficientnet_b0 if disk is tight.
"""
import os
import pickle
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

from src.config import CONFIG

# which layer to pull spatial features from, per backbone
_RETURN_NODE = {
    "inception_v3": "Mixed_7c",     # (B, 2048, 8, 8) for 299x299 input
    "resnet50": "layer4",           # (B, 2048, 7, 7) for 224x224 input
    "efficientnet_b0": "features",  # (B, 1280, 7, 7) for 224x224 input
}


def build_backbone(name: str, device: torch.device) -> nn.Module:
    """Frozen pretrained CNN, returns the spatial feature map (before pooling)."""
    if name == "inception_v3":
        base = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1,
                                    aux_logits=True)
        base.aux_logits = False
    elif name == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif name == "efficientnet_b0":
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown backbone: {name}")

    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    return_node = _RETURN_NODE[name]
    extractor = create_feature_extractor(base, return_nodes={return_node: "feat"})
    extractor.eval()
    return extractor.to(device)


def spatial_forward(net: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) -> (B, num_positions, channels), flattening the spatial grid."""
    feat_map = net(batch)["feat"]              # (B, C, H, W)
    B, C, H, W = feat_map.shape
    return feat_map.reshape(B, C, H * W).permute(0, 2, 1)  # (B, H*W, C)


def get_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def extract_features(
    image_ids: List[str],
    backbone_name: str = None,
    batch_size: int = 32,
    device: str = None,
) -> Dict[str, "np.ndarray"]:
    """Run the frozen CNN over all images and return {image_id: feature_vector}."""
    backbone_name = backbone_name or CONFIG.vision_backbone
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    net = build_backbone(backbone_name, device)
    tfm = get_transform(CONFIG.image_size)

    features = {}
    for i in tqdm(range(0, len(image_ids), batch_size), desc=f"Extracting ({backbone_name})"):
        batch_ids = image_ids[i:i + batch_size]
        imgs = []
        for img_id in batch_ids:
            path = os.path.join(CONFIG.images_dir, img_id)
            img = Image.open(path).convert("RGB")
            imgs.append(tfm(img))
        batch = torch.stack(imgs).to(device)
        out = spatial_forward(net, batch).cpu().numpy()  # (B, num_positions, C)
        for img_id, vec in zip(batch_ids, out):
            features[img_id] = vec
    return features


def extract_and_cache(image_ids: List[str], cache_path: str = None) -> Dict[str, "np.ndarray"]:
    cache_path = cache_path or CONFIG.features_cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        missing = [i for i in image_ids if i not in cached]
        if not missing:
            return cached
        new_feats = extract_features(missing)
        cached.update(new_feats)
    else:
        cached = extract_features(image_ids)

    with open(cache_path, "wb") as f:
        pickle.dump(cached, f)
    return cached


if __name__ == "__main__":
    from src.data.dataset import load_captions

    mapping = load_captions()
    extract_and_cache(list(mapping.keys()))

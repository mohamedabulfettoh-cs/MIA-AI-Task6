"""
Inference layer, used by both the API and the Gradio UI.

Two modes: "custom" loads our own trained checkpoint from artifacts/,
"pretrained" falls back to a HF captioning model so the app still works
before you've trained anything. Set CAPTION_MODEL_MODE to control this,
defaults to "auto" which picks custom if a checkpoint exists.
"""
import io
import os

import torch
from PIL import Image

MODE = os.getenv("CAPTION_MODEL_MODE", "auto")  # auto | custom | pretrained


class CustomCaptioner:
    def __init__(self):
        from src.config import CONFIG
        from src.text.tokenizer import Tokenizer
        from src.models.caption_model import CaptioningModel
        from src.features.extract_features import build_backbone, get_transform, spatial_forward

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = Tokenizer.load(CONFIG.tokenizer_path)
        ckpt = torch.load(CONFIG.best_model_path, map_location=self.device)

        self.model = CaptioningModel(ckpt["feature_dim"], ckpt["vocab_size"]).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        self.backbone = build_backbone(CONFIG.vision_backbone, self.device)
        self.transform = get_transform(CONFIG.image_size)
        self._spatial_forward = spatial_forward

    @torch.no_grad()
    def caption(self, image: Image.Image) -> str:
        img = image.convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        feat = self._spatial_forward(self.backbone, tensor)  # (1, N, C) spatial features
        return self.model.caption_image(feat, self.tokenizer)


class PretrainedCaptioner:
    """Fallback for when there's no trained checkpoint yet."""

    MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"

    def __init__(self):
        from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = VisionEncoderDecoderModel.from_pretrained(self.MODEL_ID).to(self.device)
        self.feature_extractor = ViTImageProcessor.from_pretrained(self.MODEL_ID)
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_ID)
        self.model.eval()

    @torch.no_grad()
    def caption(self, image: Image.Image) -> str:
        img = image.convert("RGB")
        pixel_values = self.feature_extractor(images=[img], return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        output_ids = self.model.generate(pixel_values, max_length=34, num_beams=4)
        text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return text.strip()


_captioner = None


def _has_custom_checkpoint() -> bool:
    from src.config import CONFIG
    return os.path.exists(CONFIG.best_model_path) and os.path.exists(CONFIG.tokenizer_path)


def get_captioner():
    """Lazily builds (and caches) whichever captioner MODE points to."""
    global _captioner
    if _captioner is not None:
        return _captioner

    use_custom = MODE == "custom" or (MODE == "auto" and _has_custom_checkpoint())
    if use_custom:
        print("[inference] Loading custom trained model...")
        _captioner = CustomCaptioner()
    else:
        print("[inference] No custom checkpoint found, loading pretrained fallback.")
        _captioner = PretrainedCaptioner()
    return _captioner


def caption_image_bytes(image_bytes: bytes) -> str:
    image = Image.open(io.BytesIO(image_bytes))
    return get_captioner().caption(image)


def caption_pil(image: Image.Image) -> str:
    return get_captioner().caption(image)

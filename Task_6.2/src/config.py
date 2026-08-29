"""Config for the whole project, so hyperparameters aren't scattered everywhere."""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # paths
    data_dir: str = os.getenv("DATA_DIR", "data/flickr8k")
    images_dir: str = os.getenv("IMAGES_DIR", "data/flickr8k/Images")
    captions_file: str = os.getenv("CAPTIONS_FILE", "data/flickr8k/captions.txt")
    features_cache: str = os.getenv("FEATURES_CACHE", "data/cache/image_features.pkl")
    tokenizer_path: str = os.getenv("TOKENIZER_PATH", "artifacts/tokenizer.json")
    checkpoint_dir: str = os.getenv("CHECKPOINT_DIR", "artifacts/checkpoints")
    best_model_path: str = os.getenv("BEST_MODEL_PATH", "artifacts/checkpoints/best_model.pt")

    # split
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42

    # vision backbone: inception_v3 / resnet50 / efficientnet_b0
    # keep the spatial feature map (not pooled) so attention has something to work with
    vision_backbone: str = os.getenv("VISION_BACKBONE", "inception_v3")
    # image_size has to match the backbone: 299 for inception_v3, 224 for the other two
    image_size: int = 299

    # text
    max_caption_len: int = 34
    min_word_freq: int = 2  # words rarer than this become <unk>
    start_token: str = "<start>"
    end_token: str = "<end>"
    pad_token: str = "<pad>"
    unk_token: str = "<unk>"

    # model
    embedding_dim: int = 256
    attention_dim: int = 256
    decoder_units: int = 512
    decoder_type: str = "gru"  # or "lstm"
    dropout: float = 0.5

    # training
    batch_size: int = 64
    epochs: int = 30
    learning_rate: float = 1e-3
    lr_patience: int = 3
    early_stopping_patience: int = 5
    grad_clip: float = 5.0
    num_workers: int = 2

    # eval
    beam_size: int = 3
    metrics: list = field(default_factory=lambda: ["bleu", "rouge", "meteor"])


CONFIG = Config()

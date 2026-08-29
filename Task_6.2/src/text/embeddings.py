"""
Optional: init the decoder's embedding layer from pretrained Word2Vec,
GloVe, or FastText vectors instead of training from scratch. Words that
aren't found just stay randomly initialized.

    matrix = build_embedding_matrix(tokenizer, "glove", "artifacts/glove.6B.100d.txt")
    model.decoder.embedding.weight.data.copy_(matrix)
"""
import numpy as np
import torch

from src.config import CONFIG
from src.text.tokenizer import Tokenizer


def _load_glove(path: str) -> dict:
    vectors = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            vectors[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    return vectors


def _load_gensim_format(path: str, kind: str) -> dict:
    """kind is 'word2vec' or 'fasttext', both load through gensim."""
    from gensim.models import KeyedVectors

    if kind == "fasttext":
        from gensim.models.fasttext import load_facebook_vectors
        kv = load_facebook_vectors(path)
    else:
        binary = path.endswith(".bin")
        kv = KeyedVectors.load_word2vec_format(path, binary=binary)
    return {w: kv[w] for w in kv.key_to_index}


def build_embedding_matrix(
    tokenizer: Tokenizer,
    method: str,
    vectors_path: str,
    embed_dim: int = None,
) -> torch.Tensor:
    embed_dim = embed_dim or CONFIG.embedding_dim
    method = method.lower()

    if method == "glove":
        vectors = _load_glove(vectors_path)
    elif method in ("word2vec", "fasttext"):
        vectors = _load_gensim_format(vectors_path, method)
    else:
        raise ValueError(f"Unknown embedding method: {method}")

    matrix = np.random.normal(scale=0.1, size=(tokenizer.vocab_size, embed_dim)).astype(np.float32)
    hits = 0
    for word, idx in tokenizer.word2idx.items():
        vec = vectors.get(word)
        if vec is not None and len(vec) == embed_dim:
            matrix[idx] = vec
            hits += 1

    print(f"[embeddings] {method}: initialised {hits}/{tokenizer.vocab_size} words "
          f"({hits / tokenizer.vocab_size:.1%} coverage)")
    return torch.tensor(matrix)

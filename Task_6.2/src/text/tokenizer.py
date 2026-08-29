"""Caption text pipeline: tokenizing, building the vocab, padding, encoding."""
import json
from collections import Counter
from typing import Dict, List

from src.config import CONFIG


class Tokenizer:
    def __init__(self, min_freq: int = None):
        self.min_freq = min_freq or CONFIG.min_word_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self._specials = [CONFIG.pad_token, CONFIG.start_token,
                           CONFIG.end_token, CONFIG.unk_token]

    # ---------- building ----------
    def fit(self, captions: List[str]) -> "Tokenizer":
        counter = Counter()
        for cap in captions:
            counter.update(cap.split())

        vocab = list(self._specials) + [
            w for w, c in counter.items() if c >= self.min_freq
        ]
        self.word2idx = {w: i for i, w in enumerate(vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.word2idx)

    # ---------- encode / decode ----------
    def encode(self, caption: str, max_len: int = None) -> List[int]:
        max_len = max_len or CONFIG.max_caption_len
        unk = self.word2idx[CONFIG.unk_token]
        tokens = [self.word2idx[CONFIG.start_token]]
        tokens += [self.word2idx.get(w, unk) for w in caption.split()]
        tokens.append(self.word2idx[CONFIG.end_token])

        tokens = tokens[:max_len]
        pad_id = self.word2idx[CONFIG.pad_token]
        tokens += [pad_id] * (max_len - len(tokens))
        return tokens

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        words = []
        for i in ids:
            w = self.idx2word.get(int(i), CONFIG.unk_token)
            if strip_special and w in self._specials:
                if w == CONFIG.end_token:
                    break
                continue
            words.append(w)
        return " ".join(words)

    # ---------- persistence ----------
    def save(self, path: str = None):
        path = path or CONFIG.tokenizer_path
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"word2idx": self.word2idx, "min_freq": self.min_freq}, f)

    @classmethod
    def load(cls, path: str = None) -> "Tokenizer":
        path = path or CONFIG.tokenizer_path
        with open(path) as f:
            data = json.load(f)
        tok = cls(min_freq=data.get("min_freq", CONFIG.min_word_freq))
        tok.word2idx = data["word2idx"]
        tok.idx2word = {int(i): w for w, i in tok.word2idx.items()}
        return tok


if __name__ == "__main__":
    from src.data.dataset import load_captions

    mapping = load_captions()
    all_caps = [c for caps in mapping.values() for c in caps]
    tok = Tokenizer().fit(all_caps)
    tok.save()
    print(f"Vocab size: {tok.vocab_size}")

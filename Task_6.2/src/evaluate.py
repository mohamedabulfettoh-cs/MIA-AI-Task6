"""
Evaluate a trained model on the test split: BLEU, ROUGE-L, METEOR, plus
a few qualitative examples. Run with: python -m src.evaluate
"""
import json
import os
from typing import Dict, List

import nltk
import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from src.config import CONFIG


def _ensure_nltk_data():
    """METEOR needs wordnet locally, download once if missing."""
    for pkg, path in [("wordnet", "corpora/wordnet"), ("omw-1.4", "corpora/omw-1.4")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)
from src.data.dataset import load_captions, split_dataset
from src.features.extract_features import extract_and_cache
from src.text.tokenizer import Tokenizer
from src.models.caption_model import CaptioningModel
from src.utils import get_device


def compute_bleu(references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
    refs_tok = [[r.split() for r in refs] for refs in references]
    hyps_tok = [h.split() for h in hypotheses]
    smoothie = SmoothingFunction().method4
    return {
        f"bleu-{n}": corpus_bleu(refs_tok, hyps_tok, weights=tuple([1 / n] * n), smoothing_function=smoothie)
        for n in (1, 2, 3, 4)
    }


def compute_rouge(references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for refs, hyp in zip(references, hypotheses):
        best = max(scorer.score(ref, hyp)["rougeL"].fmeasure for ref in refs)
        scores.append(best)
    return {"rougeL": sum(scores) / len(scores)}


def compute_meteor(references: List[List[str]], hypotheses: List[str]) -> Dict[str, float]:
    scores = [
        meteor_score([r.split() for r in refs], hyp.split())
        for refs, hyp in zip(references, hypotheses)
    ]
    return {"meteor": sum(scores) / len(scores)}


def evaluate(num_qualitative: int = 10):
    _ensure_nltk_data()
    device = get_device()
    tokenizer = Tokenizer.load()
    ckpt = torch.load(CONFIG.best_model_path, map_location=device)

    model = CaptioningModel(ckpt["feature_dim"], ckpt["vocab_size"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mapping = load_captions()
    _, _, test_ids = split_dataset(mapping)
    features = extract_and_cache(test_ids)

    references, hypotheses, qualitative = [], [], []
    with torch.no_grad():
        for img_id in test_ids:
            if img_id not in features:
                continue
            refs = mapping[img_id]
            feat = torch.tensor(features[img_id], dtype=torch.float32).unsqueeze(0).to(device)
            hyp = model.caption_image(feat, tokenizer)

            references.append(refs)
            hypotheses.append(hyp)
            if len(qualitative) < num_qualitative:
                qualitative.append({"image": img_id, "generated": hyp, "references": refs})

    results = {}
    results.update(compute_bleu(references, hypotheses))
    results.update(compute_rouge(references, hypotheses))
    results.update(compute_meteor(references, hypotheses))

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/eval_results.json", "w") as f:
        json.dump({"metrics": results, "examples": qualitative}, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved qualitative examples to artifacts/eval_results.json")
    return results


if __name__ == "__main__":
    evaluate()

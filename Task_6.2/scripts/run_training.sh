#!/usr/bin/env bash
# Runs the full pipeline: feature extraction -> tokenizer -> training -> evaluation
set -e

echo "== 1. Extracting & caching CNN image features =="
python -m src.features.extract_features

echo "== 2. Training the captioning model =="
python -m src.train

echo "== 3. Evaluating on the held-out test split (BLEU / ROUGE / METEOR) =="
python -m src.evaluate

echo "All done. See artifacts/checkpoints/best_model.pt and artifacts/eval_results.json"

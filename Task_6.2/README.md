---
title: Image Caption Generator
emoji: image
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# Image Caption Generator

End to end image caption generator trained on Flickr8k. Pretrained CNN encoder (transfer learning), Bahdanau attention, GRU/LSTM decoder, served through FastAPI with a Gradio UI, packaged in Docker.

## 1. Overview

| | |
|---|---|
| Task | given an image, generate a caption describing it |
| Dataset | [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k), 8,000 images x 5 human captions each |
| Approach | CNN feature extractor (frozen) -> linear projection -> Bahdanau attention -> GRU/LSTM decoder |
| Serving | FastAPI (/caption, /health) + Gradio UI, one container |
| Evaluation | BLEU-1 to BLEU-4, ROUGE-L, METEOR, plus qualitative examples |

## 2. Architecture

```
Image -> Pretrained CNN (InceptionV3 / ResNet50 / EfficientNet-B0, frozen)
              |  (spatial feature grid, e.g. 8x8x2048)
              v
        Linear projection + LayerNorm
              |
              v
   For each decoding step:
     Bahdanau attention over the spatial grid -> context vector
     concat(word embedding, context) -> GRUCell / LSTMCell
     Linear -> softmax over vocab -> next word
```

The CNN backbone only does feature extraction, it's frozen, not trained from scratch. The important bit here is keeping the spatial feature map instead of pooling it down to one vector, since a single pooled vector gives attention nothing to actually attend over. Word embeddings can be trained from scratch or initialized from Word2Vec/GloVe/FastText (`src/text/embeddings.py`). Decoder is GRU by default, switchable to LSTM via config.

## 3. Preprocessing

- Images resized to whatever the backbone expects, normalized with ImageNet stats.
- Captions lowercased, punctuation/digits stripped, wrapped with `<start>`/`<end>`, padded/truncated, rare words mapped to `<unk>`.
- Split by image id, not by caption, so all 5 captions of one image land in the same split. Keeps train/val/test clean.

## 4. Training

```bash
bash scripts/download_dataset.sh   # pulls Flickr8k from Kaggle
bash scripts/run_training.sh       # features -> tokenizer -> train -> evaluate
```

Runs in order: `src/features/extract_features.py` caches CNN features, `src/train.py` trains with teacher forcing, ReduceLROnPlateau, gradient clipping, and early stopping (checkpoints the best val loss to `artifacts/checkpoints/best_model.pt`), then `src/evaluate.py` scores the test split.

Hyperparameters live in `src/config.py`, override with env vars or just edit the defaults.

## 5. Evaluation metrics and results

Computed on the held out test split via `src/evaluate.py`:

- BLEU-1 to BLEU-4, n-gram precision against the 5 references.
- ROUGE-L, longest common subsequence F-measure.
- METEOR, accounts for synonyms/stemming, correlates better with human judgement than BLEU alone.

Run `python -m src.evaluate` after training. Results and qualitative examples get written to `artifacts/eval_results.json`.

| Metric | Score |
|---|---|
| BLEU-1 | 0.593 |
| BLEU-2 | 0.408 |
| BLEU-3 | 0.275 |
| BLEU-4 | 0.184 |
| ROUGE-L | 0.442 |
| METEOR | 0.386 |

Trained for 11 epochs (early stopping, patience 5, best checkpoint from epoch 6, val_loss 2.84). These are in line with typical results for this kind of attention-based encoder-decoder on Flickr8k, BLEU drops off from BLEU-1 to BLEU-4 as expected since exact 4-gram matches are a lot harder than single-word overlap.

## 6. Project structure

```
image-caption-generator/
├── app/                  FastAPI + Gradio serving layer
│   ├── main.py           REST API, mounts Gradio at "/"
│   ├── gradio_app.py      Gradio UI
│   ├── inference.py       loads model (custom or pretrained fallback)
│   └── schemas.py
├── src/
│   ├── config.py          all hyperparameters/paths
│   ├── data/               dataset loading, splitting, torch Dataset
│   ├── features/           CNN feature extraction + caching
│   ├── text/                tokenizer, vocab, pretrained embeddings
│   ├── models/               encoder projection + attention + decoder
│   ├── train.py              training loop
│   ├── evaluate.py            BLEU / ROUGE / METEOR
│   └── utils.py
├── tests/                  unit tests
├── scripts/                 download_dataset.sh, run_training.sh
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 7. Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# train your own model first if you want (see section 4)
# without a checkpoint the app falls back to a pretrained model automatically

uvicorn app.main:app --host 0.0.0.0 --port 7860
```

Open http://localhost:7860 for the Gradio UI, or http://localhost:7860/docs for the API docs.

With Docker:

```bash
docker build -t caption-app .
docker run -p 7860:7860 caption-app
# or:
docker compose up --build
```

## 8. Using the app/API

Gradio UI: open the root URL, upload an image, click Generate caption.

REST API:
```bash
curl -X POST "http://localhost:7860/caption" \
  -F "file=@/path/to/image.jpg"
# -> {"caption": "a dog running through a grassy field", "model_mode": "auto"}

curl http://localhost:7860/health
# -> {"status": "ok", "mode": "auto"}
```

`CAPTION_MODEL_MODE` controls which model gets served: `auto` (default) uses your trained checkpoint if it's there, otherwise falls back to a pretrained HF model. `custom` forces your own model (errors if there's no checkpoint). `pretrained` forces the fallback.

## 9. Model storage and sharing

Trained checkpoint and tokenizer are hosted on Hugging Face:

**Model: [Botooo/image-caption-flickr8k](https://huggingface.co/Botooo/image-caption-flickr8k/tree/main)**

Uploaded via the Hugging Face web UI (Files and versions > Add file > Upload files):
- `best_model.pt`, trained weights (best checkpoint, epoch 6, val_loss 2.84)
- `tokenizer.json`, the vocabulary used to train it

To reproduce or automate the upload:

```bash
pip install huggingface_hub
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("Botooo/image-caption-flickr8k", repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj="artifacts/checkpoints/best_model.pt",
                 path_in_repo="best_model.pt",
                 repo_id="Botooo/image-caption-flickr8k")
api.upload_file(path_or_fileobj="artifacts/tokenizer.json",
                 path_in_repo="tokenizer.json",
                 repo_id="Botooo/image-caption-flickr8k")
PY
```

## 10. Example inputs and outputs
[<video controls src="20260829-1231-31.7429474-1.mp4" title="Title"></video>](https://github.com/user-attachments/assets/d483690a-4c2b-4536-8177-fe6d09fff0de)

There are obviously inaccuracies in the generated captions, with the biggest inacccuracy being with the flower image, that makes sense since the model was trained on Flickr8k which is mostly outdoor scenes with people and animals, not flowers. The other two images are more in line with the training data, so the model does somewhat better with them, the second image the model gets the gender of the person wrong as it says it is a female when it is a guy, the third image the model makes a mistake because of the dogs' silhouttes and shadows causing it to think it is a group of people riding horses, the fourth and final image the model gets right because it perfectly aligns with the Flickr8k dataset (not part of the dataset, just similar to what it includes).

## 11. Running tests

```bash
pip install pytest
pytest tests/
```

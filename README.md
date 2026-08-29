# Task 6, AI Team Training '27 (Phase II)

Two subtasks, each in its own folder with its own full README.

## Task_6.1, English to French Translation

Seq2Seq machine translation model, BiLSTM encoder with attention, modernized with FastText word embeddings trained on the corpus itself, evaluated with BLEU and ROUGE.

See `Task_6.1/machine-translation.ipynb` directly for the full implementation.

## Task_6.2, Image Caption Generator

End to end image captioning system trained on Flickr8k, pretrained CNN encoder, Bahdanau attention over spatial features, GRU decoder, served through FastAPI + Gradio, packaged in Docker. Evaluated with BLEU, ROUGE-L, and METEOR.

See [Task_6.2/README.md](Task_6.2/README.md) for the full writeup, including the trained model link on Hugging Face and how to run it.
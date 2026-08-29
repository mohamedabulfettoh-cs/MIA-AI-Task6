#!/usr/bin/env bash
# Downloads the Flickr8k dataset from Kaggle into data/flickr8k/
# Requires: pip install kaggle, and ~/.kaggle/kaggle.json with your API token
# (Account -> Create New API Token on kaggle.com).
set -e

mkdir -p data/flickr8k
kaggle datasets download -d adityajn105/flickr8k -p data/flickr8k --unzip

echo "Done. Expect:"
echo "  data/flickr8k/Images/*.jpg"
echo "  data/flickr8k/captions.txt"

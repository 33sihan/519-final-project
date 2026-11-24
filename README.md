# CIS 5190 Final Project B — News Source Classification

Classify a news article link as **Fox News (0)** or **NBC News (1)** using only the headline part of the URL.

**Leaderboard result:** 0.978 accuracy on the hidden test set (`url_val16k`).

## Model

A wide-and-deep, fastText-style neural network (`model.py`):

- The headline is split into hashed features (crc32): words, word pairs and triples, character 2–5-grams, the first and last word, and the length.
- **Deep part:** an EmbeddingBag (mean, 64-dim) feeds an MLP (256 hidden units) that outputs 2 logits.
- **Wide part:** one learned weight per feature per class, summed. This is logistic regression on the same features.
- Prediction = deep + wide, then argmax.

It needs no vocabulary file. Everything is in `model.pt`, a plain `state_dict`.

## Data

- `url_train.csv`: the 4,800 provided links.
- Links we scraped from Fox News and NBC News's public sitemaps (`harvest.py`, `cdx_links.py`, `scrape_links.py`). About 2.28M links were collected; `harvest.csv` is not in this repo because of its size.
- Training used links from mid-2020 onward, about 580K unique headlines. Headlines that appear on both outlets were dropped.

## Training

```powershell
py train_big.py --since 2020-07-01 --epochs 8 --dropout 0.1 --smoothing 0 --val 0 --out model.pt
```

The optimizer is SparseAdam for the embeddings and AdamW for the MLP. The loss is class-weighted cross-entropy, and the provided rows are repeated 3×.

## Results

| Model | Leaderboard | Headlines unseen in training |
|---|---|---|
| First submission (Nov 2) | 0.755 | – |
| 2022+ data, 2 epochs (`model_balanced.pt`) | 0.913 | ~0.88 |
| **2020-07+ data, 8 epochs (`model.pt`)** | **0.978** | ~0.85 |

The scraped data comes from the same two websites as the test links, so some test headlines may also appear in the training data. We never accessed the test set. The "unseen" column measures accuracy on url_train headlines that were removed from all training data.

## Files

| File | Purpose |
|---|---|
| `model.py`, `preprocess.py`, `model.pt` | Leaderboard submission |
| `train_big.py` | Main training script |
| `sweep.py` | Runs a setting on 3 validation splits and averages the results |
| `check_submission.py` | Local check that mimics the leaderboard backend |
| `harvest.py`, `cdx_links.py`, `scrape_links.py` | Link scrapers |

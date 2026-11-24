"""
train.py  —  trains NewsClassifier (v2, train.py v3) and writes model.pt (a state_dict).

Usage (Windows: use `py`, Mac/Linux: `python3`):
    py train.py url_train.csv                          # provided data only
    py train.py url_train.csv scraped_urls.csv         # + scraped links (recommended)
    py train.py url_train.csv scraped_matched.csv        # after match_window.py (best)
    py train.py url_train.csv scraped_matched.csv --provided-weight 5

Steps:
  1. Reduce every link to headline words (same function the test set uses).
  2. Hold out a validation set; ALWAYS validate on the provided url_train.csv rows,
     so the number you see is comparable to the leaderboard.
  3. Train, pick the best epoch by validation accuracy.
  4. (default) Retrain from scratch on ALL data for that many epochs -> model.pt.
"""
import argparse
import random

import torch
import torch.nn as nn

from model import NewsClassifier, text_to_features
from preprocess import LABELS, prepare_data


def load(csv_path):
    X, y = prepare_data(csv_path)
    if not y:
        raise ValueError(f"{csv_path}: could not read labels (need full foxnews/nbcnews links)")
    return [(t, LABELS.index(l)) for t, l in zip(X, y) if t]


def dedupe(rows):
    seen, out = set(), []
    for t, l in rows:
        if (t, l) not in seen:
            seen.add((t, l))
            out.append((t, l))
    return out


def encode_all(rows):
    return [(text_to_features(t), l) for t, l in rows]


def batches(enc, bs, shuffle):
    idx = list(range(len(enc)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [enc[j] for j in idx[i:i + bs]]
        hashes, offsets, labels = [], [], []
        for feats, l in chunk:
            offsets.append(len(hashes))
            hashes.extend(feats)
            labels.append(l)
        yield (torch.tensor(hashes, dtype=torch.long),
               torch.tensor(offsets, dtype=torch.long),
               torch.tensor(labels, dtype=torch.long))


def accuracy(model, enc):
    model.eval()
    correct = 0
    with torch.no_grad():
        for h, o, y in batches(enc, 2048, False):
            correct += (model.forward_encoded(h, o).argmax(1) == y).sum().item()
    return correct / max(1, len(enc))


def train(train_enc, epochs, args, val_enc=None):
    torch.manual_seed(args.seed)
    model = NewsClassifier()
    n_fox = sum(1 for _, l in train_enc if l == 0)
    n_nbc = len(train_enc) - n_fox
    # class weights so an unbalanced scrape doesn't bias the model toward one outlet
    w = torch.tensor([len(train_enc) / (2 * max(1, n_fox)), len(train_enc) / (2 * max(1, n_nbc))])
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = epochs * ((len(train_enc) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(1, steps),
                                                pct_start=0.1)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for h, o, y in batches(train_enc, args.batch, True):
            loss = loss_fn(model.forward_encoded(h, o), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y)
            n += len(y)
        msg = f"epoch {epoch:2d}  loss {total / n:.4f}"
        if val_enc:
            acc = accuracy(model, val_enc)
            history.append(acc)
            msg += f"  val_acc {acc:.4f}"
        print(msg, flush=True)
    return model, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="first CSV = provided url_train.csv")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--val", type=float, default=0.2, help="fraction of url_train.csv held out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--provided-weight", type=int, default=3,
                    help="repeat url_train.csv rows this many times (they match the test data best)")
    ap.add_argument("--no-final", action="store_true", help="skip retraining on all data")
    ap.add_argument("--out", default="model.pt")
    args = ap.parse_args()
    random.seed(args.seed)

    provided = dedupe(load(args.csvs[0]))
    extra = dedupe([r for p in args.csvs[1:] for r in load(p)])
    random.shuffle(provided)
    n_val = int(len(provided) * args.val)
    val = provided[:n_val]
    prov_train = provided[n_val:]
    prov_texts = {t for t, _ in provided}
    extra = [r for r in extra if r[0] not in prov_texts]       # no overlap with course data
    w = max(1, args.provided_weight)
    train_rows = prov_train * w + extra
    print(f"provided={len(provided)}  extra={len(extra)}  ->  train={len(prov_train)}x{w}"
          f" + {len(extra)} = {len(train_rows)}  val={len(val)}")

    print("encoding...", flush=True)
    train_enc, val_enc = encode_all(train_rows), encode_all(val)

    model, hist = train(train_enc, args.epochs, args, val_enc)
    best_epoch = max(range(len(hist)), key=lambda i: hist[i]) + 1 if hist else args.epochs
    print(f"best val_acc {max(hist) if hist else float('nan'):.4f} at epoch {best_epoch}")

    if not args.no_final and val:
        print(f"retraining on ALL data for {best_epoch} epochs...", flush=True)
        model, _ = train(encode_all(provided * w + extra), best_epoch, args)
    elif hist and best_epoch != args.epochs:
        print("(saving last-epoch model; rerun with --epochs", best_epoch, "to match best)")

    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
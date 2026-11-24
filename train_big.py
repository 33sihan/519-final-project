"""
train_big.py  —  train NewsClassifier (model.py v4) on url_train.csv + all scraped links.

    py train_big.py --epochs 4 --out model.pt

Data:
  * every link -> headline (preprocess.headline_from_url, same as the test set)
  * one row per unique headline; headlines seen on BOTH outlets are dropped
  * 20% of url_train.csv is held out as validation, and those headlines are removed
    from ALL training data (incl. scraped), so val accuracy = accuracy on headlines
    the model has never seen
  * url_train training rows are repeated --provided-weight times
Training: features are hashed on the fly in DataLoader workers; sparse embeddings
with SparseAdam, Adam for the MLP; class-weighted cross-entropy (scrape has more Fox).
Saves the epoch with the best validation accuracy.
"""
import argparse
import random
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import model as M
from model import NewsClassifier, pad_wordids, text_to_features, text_to_wordids
from preprocess import headline_from_url, label_from_url

SCRAPED = ["harvest.csv", "cdx.csv", "scraped_dated.csv", "scraped_urls.csv"]


class Rows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        t, l = self.rows[i]
        return text_to_features(t), text_to_wordids(t), l


def collate(batch):
    hashes, offsets, wids, labels = [], [], [], []
    for feats, w, l in batch:
        offsets.append(len(hashes))
        hashes.extend(feats)
        wids.append(w)
        labels.append(l)
    return (torch.tensor(hashes, dtype=torch.long), torch.tensor(offsets, dtype=torch.long),
            pad_wordids(wids), torch.tensor(labels, dtype=torch.long))


def load_rows(train_csv, scraped, val_frac, seed, since="", old_frac=0.0, keep_scraped=False):
    tr = pd.read_csv(train_csv)
    prov = [(headline_from_url(u), label_from_url(u)) for u in tr["url"]]
    prov = [(h, l) for h, l in dict.fromkeys(prov) if h]
    random.Random(seed).shuffle(prov)          # seed = which validation split
    n_val = int(len(prov) * val_frac)
    val, prov_tr = prov[:n_val], prov[n_val:]
    val_heads = {h for h, _ in val}

    labels = {}
    for p in scraped:
        df = pd.read_csv(p, dtype=str).fillna("")
        if since and "date" in df.columns:          # keep links dated on/after --since,
            old = df[(df["date"] < since) & (df["date"] != "")]  # plus a sample of older ones
            df = pd.concat([df[df["date"] >= since],
                            old.sample(frac=old_frac, random_state=0) if old_frac else old.iloc[:0]])
        for u in df["url"].drop_duplicates():
            h, l = headline_from_url(u), label_from_url(u)
            if l is None or len(h.split()) < 2:
                continue
            labels.setdefault(h, set()).add(l)
        print(f"  {p}: {len(labels)} unique headlines so far", flush=True)
    for h, l in prov_tr:
        labels.setdefault(h, set()).add(l)
    # keep_scraped: a held-out url_train headline stays in training ONLY if it was also
    # scraped from the website (= the leaderboard situation: test links come from the
    # same sites we scraped). Otherwise held-out headlines are removed everywhere.
    extra = [(h, next(iter(s))) for h, s in labels.items()
             if len(s) == 1 and (keep_scraped or h not in val_heads)]
    n_amb = sum(1 for s in labels.values() if len(s) > 1)
    return prov_tr, extra, val, n_amb


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ok = n = 0
    for h, o, w, y in loader:
        ok += (model.forward_encoded(h, o, w).argmax(1) == y).sum().item()
        n += len(y)
    return ok / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="url_train.csv")
    ap.add_argument("--scraped", nargs="*", default=SCRAPED)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--provided-weight", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--since", default="", help="drop dated scraped links before this "
                    "date (YYYY-MM-DD); files without a date column are kept")
    ap.add_argument("--old-frac", type=float, default=0.0,
                    help="with --since: also keep this random fraction of older dated links")
    ap.add_argument("--val-seed", type=int, default=0, help="which validation split to use")
    ap.add_argument("--val-keep-scraped", action="store_true",
                    help="leaderboard-like validation: held-out url_train headlines that were "
                         "also scraped stay in the scraped training data")
    ap.add_argument("--wide-bits", type=int, default=0, help="WIDE_BUCKETS = 2**bits")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--smoothing", type=float, default=0.02)
    ap.add_argument("--cnn", action="store_true", help="add the word-CNN branch "
                    "(set USE_CNN = True in model.py before submitting such a model)")
    ap.add_argument("--out", default="model.pt")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    prov_tr, extra, val, n_amb = load_rows(args.train, args.scraped, args.val, args.val_seed,
                                           args.since, args.old_frac, args.val_keep_scraped)
    rows = prov_tr * max(1, args.provided_weight) + extra
    n_fox = sum(1 for _, l in rows if l == 0)
    print(f"train rows {len(rows)} (provided {len(prov_tr)}x{args.provided_weight}, "
          f"extra {len(extra)}; fox {n_fox}, nbc {len(rows) - n_fox}); "
          f"ambiguous dropped {n_amb}; val {len(val)} unseen url_train headlines", flush=True)

    train_dl = DataLoader(Rows(rows), batch_size=args.batch, shuffle=True, collate_fn=collate,
                          num_workers=args.workers, persistent_workers=True, prefetch_factor=4)
    val_dl = DataLoader(Rows(val), batch_size=2048, collate_fn=collate)

    M.USE_CNN = args.cnn
    if args.wide_bits:                      # must match WIDE_BUCKETS in model.py when submitting
        M.WIDE_BUCKETS = 2 ** args.wide_bits
    model = NewsClassifier(sparse=True, dropout=args.dropout)
    sparse_mods = [model.emb, model.wide] + ([model.wemb] if args.cnn else [])
    sparse_ids = {id(p) for m in sparse_mods for p in m.parameters()}
    opt_s = torch.optim.SparseAdam([p for m in sparse_mods for p in m.parameters()], lr=args.lr)
    opt_d = torch.optim.AdamW([p for p in model.parameters() if id(p) not in sparse_ids],
                              lr=args.lr, weight_decay=1e-4)
    total = args.epochs * len(train_dl)
    lr_at = lambda step: args.lr * max(0.05, 1 - step / total)
    w = torch.tensor([len(rows) / (2 * n_fox), len(rows) / (2 * (len(rows) - n_fox))])
    loss_fn = nn.CrossEntropyLoss(weight=w, label_smoothing=args.smoothing)

    best, step = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for i, (h, o, wd, y) in enumerate(train_dl):
            for opt in (opt_s, opt_d):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
            loss = loss_fn(model.forward_encoded(h, o, wd), y)
            opt_s.zero_grad(); opt_d.zero_grad()
            loss.backward()
            opt_s.step(); opt_d.step()
            step += 1
            tot += loss.item() * len(y); n += len(y)
            if i % 1000 == 0:
                print(f"  epoch {epoch} batch {i}/{len(train_dl)} loss {tot / n:.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        acc = evaluate(model, val_dl) if val else float("nan")
        print(f"epoch {epoch}: loss {tot / n:.4f}  val_acc(unseen) {acc:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if not val or acc > best:          # no validation set: keep the last epoch
            best = acc
            sd = {k: v for k, v in model.state_dict().items()}
            torch.save(sd, args.out)
            print(f"  saved {args.out}", flush=True)
    print(f"best val_acc(unseen) {best:.4f}")


if __name__ == "__main__":
    main()

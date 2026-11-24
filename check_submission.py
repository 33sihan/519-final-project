"""
check_submission.py  —  load preprocess.py / model.py / model.pt the way the leaderboard
backend does, run on a CSV, report accuracy (if labels are readable) and runtime.

    py check_submission.py url_train.csv
"""
import sys
import time

import torch

import model as M
import preprocess as P

csv_path = sys.argv[1] if len(sys.argv) > 1 else "url_train.csv"
t0 = time.time()
X, y = P.prepare_data(csv_path)
m = M.Model()
m.load_state_dict(torch.load("model.pt", map_location="cpu"))
m.eval()
preds = m.predict(X)
dt = time.time() - t0
assert len(preds) == len(X), "one prediction per row required"
print(f"{len(X)} rows, {dt:.1f}s, preds sample {preds[:10]}")
if y:
    gold = [P.LABELS.index(l) for l in y]
    print(f"accuracy {sum(int(p == g) for p, g in zip(preds, gold)) / len(gold):.4f}")

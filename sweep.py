"""
sweep.py  —  run one train_big.py setting on several validation splits and average.

    py sweep.py NAME [train_big.py args ...]
    e.g.  py sweep.py base --since 2022-01-01 --epochs 3

Runs --val-seed 0,1,2 (each holds out a different 20% of url_train.csv, removed from
all training data), and appends the mean unseen-headline accuracy per epoch to
runs/sweep.csv. Models go to runs/tmp_<NAME>.pt (not used for submission).
"""
import os
import re
import statistics
import subprocess
import sys

SEEDS = (0, 1, 2)
SCRAPED = ["--scraped", "harvest.csv", "cdx.csv", "scraped_dated.csv"]


def main():
    name, extra = sys.argv[1], sys.argv[2:]
    if "--scraped" not in extra:
        extra = SCRAPED + extra
    os.makedirs("runs", exist_ok=True)
    per_epoch = {}
    for s in SEEDS:
        cmd = [sys.executable, "train_big.py", "--val-seed", str(s), "--workers", "4",
               "--out", f"runs/tmp_{name}.pt"] + extra
        log = f"runs/sweep_{name}_v{s}.log"
        with open(log, "w", encoding="utf-8") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        for line in open(log, encoding="utf-8"):
            m = re.match(r"epoch (\d+): .*val_acc\(unseen\) ([\d.]+)", line)
            if m:
                per_epoch.setdefault(int(m.group(1)), []).append(float(m.group(2)))
        print(f"[{name}] split {s} done", flush=True)
    means = {e: statistics.mean(v) for e, v in per_epoch.items()}
    best_e = max(means, key=means.get)
    row = f"{name},{' '.join(extra)},{best_e},{means[best_e]:.4f}," + \
          " ".join(f"e{e}={means[e]:.4f}" for e in sorted(means))
    with open("runs/sweep.csv", "a", encoding="utf-8") as f:
        f.write(row + "\n")
    print(row)


if __name__ == "__main__":
    main()

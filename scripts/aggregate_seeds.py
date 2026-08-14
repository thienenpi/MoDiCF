#!/usr/bin/env python3
"""Aggregate best-Recall@20-epoch metrics across multiple seed logs into mean +/- std.

Follows the MoDiCF paper protocol (Sec 5.2.3): 10 repeated runs, report mean and
standard deviation of Recall/Precision/NDCG/F/F_fuse at K=10 and K=20. Per-run
epoch selection is the best test Recall@20, matching the released training code.

Usage:
    python scripts/aggregate_seeds.py "logs/deconf-group-baby-l1.0-k10-seed*.out" [--ddof 0|1]

The argument is a shell glob (keep it quoted so Python expands it, not the shell)
matching one log file per seed run.
"""
import glob
import re
import sys

import numpy as np

PAT = re.compile(
    r"Test_@\[.*?\]:\s*Recall=\[([^\]]+)\],\s*precision=\[([^\]]+)\],\s*hit=\[([^\]]+)\],\s*"
    r"ndcg=\[([^\]]+)\],.*?F=\[([^\]]+)\],\s*F_fuse=\[([^\]]+)\]"
)


def parse_floats(s):
    return [float(x) for x in s.split(",")]


def best_recall20(path):
    """Best epoch by Recall@20 within a single log file (Ks=[5,10,20,50], index 2)."""
    best = None
    best_r20 = -1.0
    with open(path) as f:
        for line in f:
            m = PAT.search(line)
            if not m:
                continue
            recall = parse_floats(m.group(1))
            precision = parse_floats(m.group(2))
            ndcg = parse_floats(m.group(4))
            fair_p = parse_floats(m.group(5))
            f_fuse = parse_floats(m.group(6))
            r20 = recall[2]
            if r20 > best_r20:
                best_r20 = r20
                best = dict(recall=recall, precision=precision, ndcg=ndcg,
                            fair_p=fair_p, f_fuse=f_fuse)
    return best


def main():
    argv = sys.argv[1:]
    ddof = 1
    if "--ddof" in argv:
        i = argv.index("--ddof")
        ddof = int(argv[i + 1])
        del argv[i:i + 2]
    if len(argv) != 1:
        print(__doc__)
        sys.exit(1)
    paths = sorted(glob.glob(argv[0]))
    if not paths:
        print(f"No files matched: {argv[0]}")
        sys.exit(1)

    per_seed = []
    for p in paths:
        b = best_recall20(p)
        if b is None:
            print(f"WARNING: no Test_@ lines found in {p}, skipping")
            continue
        per_seed.append(b)
        print(f"{p}: Recall@20={b['recall'][2]*100:.2f}")

    n = len(per_seed)
    if n == 0:
        print("No usable runs found.")
        sys.exit(1)

    def stat(key, idx):
        vals = np.array([b[key][idx] for b in per_seed]) * 100
        # ddof=1 (sample std) matches "average results and standard deviations"
        # over the paper's 10 repeated runs; pass --ddof 0 for population std.
        return vals.mean(), vals.std(ddof=ddof if n > ddof else 0)

    print(f"\nAggregated over {n} seed run(s)  [std: ddof={ddof}]:")
    for label, idx in [("@5", 0), ("@10", 1), ("@20", 2)]:
        lb = label.ljust(3)
        cells = []
        for name, key in [("Recall", "recall"), ("Prec", "precision"), ("NDCG", "ndcg"),
                          ("F", "fair_p"), ("F_fuse", "f_fuse")]:
            m, s = stat(key, idx)
            cells.append(f"{name}{lb}={m:.2f}±{s:.2f}")
        print("  " + "  ".join(cells))

    if n < 10:
        print(f"\nNote: only {n}/10 seed runs found — not yet the full 10-run paper protocol.")


if __name__ == "__main__":
    main()

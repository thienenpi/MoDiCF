#!/usr/bin/env python3
"""Aggregate best-Recall@20-epoch metrics across multiple seed logs into mean +/- std.

Usage:
    python scripts/aggregate_seeds.py "logs/gmddc-baby-k10-b0.3-seed*.out"
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
            ndcg = parse_floats(m.group(4))
            fair_p = parse_floats(m.group(5))
            f_fuse = parse_floats(m.group(6))
            r20 = recall[2]
            if r20 > best_r20:
                best_r20 = r20
                best = dict(recall=recall, ndcg=ndcg, fair_p=fair_p, f_fuse=f_fuse)
    return best


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    paths = sorted(glob.glob(sys.argv[1]))
    if not paths:
        print(f"No files matched: {sys.argv[1]}")
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
        return vals.mean(), vals.std()

    print(f"\nAggregated over {n} seed run(s):")
    for label, idx in [("@10", 0), ("@20", 2)]:
        r_m, r_s = stat("recall", idx)
        n_m, n_s = stat("ndcg", idx)
        f_m, f_s = stat("fair_p", idx)
        ff_m, ff_s = stat("f_fuse", idx)
        print(f"  Recall{label}={r_m:.2f}±{r_s:.2f}  NDCG{label}={n_m:.2f}±{n_s:.2f}  "
              f"F{label}={f_m:.2f}±{f_s:.2f}  F_fuse{label}={ff_m:.2f}±{ff_s:.2f}")

    if n < 10:
        print(f"\nNote: only {n}/10 seed runs found — not yet the full 10-run paper protocol.")


if __name__ == "__main__":
    main()

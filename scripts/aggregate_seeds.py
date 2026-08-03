#!/usr/bin/env python3
"""Aggregate best-Recall@20-epoch metrics across multiple seed logs into mean +/- std.

The per-run epoch is picked by max test Recall@20, matching the rule in the author's
main.py (self.best_recall20). Pass a second glob to also run the paired t-test the
paper reports (Sec 5.2.2, p < 0.05); runs are paired by the seed in the filename.

Usage:
    python scripts/aggregate_seeds.py "logs/gmddc-baby-k10-b0.3-seed*.out"
    python scripts/aggregate_seeds.py "logs/gmddc-baby-*-seed*.out" "logs/repro-baby-*-seed*.out"
"""
import glob
import os
import re
import sys

import numpy as np
from scipy import stats

PAT = re.compile(
    r"Test_@\[.*?\]:\s*Recall=\[([^\]]+)\],\s*precision=\[([^\]]+)\],\s*hit=\[([^\]]+)\],\s*"
    r"ndcg=\[([^\]]+)\],.*?F=\[([^\]]+)\],\s*F_fuse=\[([^\]]+)\]"
)


SEED_PAT = re.compile(r"-seed(\d+)-")


def parse_floats(s):
    return [float(x) for x in s.split(",")]


def seed_of(path):
    """Seed encoded in a log filename, e.g. gmddc-baby-k10-b0.3-seed3-74393.out -> 3.

    Returns None for logs that predate the -seed<N>- naming, which therefore cannot
    be paired against another arm.
    """
    m = SEED_PAT.search(os.path.basename(path))
    return int(m.group(1)) if m else None


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


def paired_ttest(treat, base):
    """Paired t-test of treat vs base (paper Sec 5.2.2 protocol: p < 0.05).

    treat/base map seed -> parsed metrics. Only seeds present in BOTH arms are used,
    since pairing is what removes the shared mask/init variance; unmatched seeds are
    reported so a silently thin comparison cannot pass for a full one.
    """
    seeds = sorted(set(treat) & set(base))
    if len(seeds) < 2:
        print(f"\nPaired t-test skipped: only {len(seeds)} seed(s) present in both arms.")
        return

    dropped = sorted((set(treat) | set(base)) - set(seeds))
    print(f"\nPaired t-test over {len(seeds)} matched seed(s): {seeds}")
    if dropped:
        print(f"  (unmatched seeds excluded: {dropped})")

    for label, idx in [("@10", 1), ("@20", 2)]:
        cells = []
        for key, name in [("recall", "Recall"), ("ndcg", "NDCG"),
                          ("fair_p", "F"), ("f_fuse", "F_fuse")]:
            t = np.array([treat[s][key][idx] for s in seeds]) * 100
            b = np.array([base[s][key][idx] for s in seeds]) * 100
            delta = t.mean() - b.mean()
            # zero variance in the differences makes the t statistic undefined
            if np.allclose(t, b):
                cells.append(f"{name}{label} Δ=0.00 (identical)")
                continue
            _, p = stats.ttest_rel(t, b)
            mark = "*" if p < 0.05 else " "
            cells.append(f"{name}{label} Δ={delta:+.2f}{mark} p={p:.4f}")
        print("  " + "  ".join(cells))
    print("  * = significant at p < 0.05")


def load_arm(pattern, label):
    """Parse every log matching pattern into (list of metrics, {seed: metrics})."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"No files matched: {pattern}")
        sys.exit(1)

    print(f"=== {label}: {pattern}")
    per_run, by_seed = [], {}
    for p in paths:
        b = best_recall20(p)
        if b is None:
            print(f"WARNING: no Test_@ lines found in {p}, skipping")
            continue
        per_run.append(b)
        s = seed_of(p)
        if s is None:
            print(f"WARNING: no -seed<N>- in {os.path.basename(p)}; excluded from t-test")
        elif s in by_seed:
            print(f"WARNING: seed {s} seen twice ({os.path.basename(p)}); keeping the first")
        else:
            by_seed[s] = b
        print(f"{p}: Recall@20={b['recall'][2]*100:.2f}")

    if not per_run:
        print("No usable runs found.")
        sys.exit(1)

    print(f"Aggregated over {len(per_run)} seed run(s):")
    # Ks = [5, 10, 20, 50] -> index 1 is @10, index 2 is @20
    for lab, idx in [("@10", 1), ("@20", 2)]:
        out = []
        for key, name in [("recall", "Recall"), ("ndcg", "NDCG"),
                          ("fair_p", "F"), ("f_fuse", "F_fuse")]:
            v = np.array([b[key][idx] for b in per_run]) * 100
            out.append(f"{name}{lab}={v.mean():.2f}±{v.std():.2f}")
        print("  " + "  ".join(out))

    if len(per_run) < 10:
        print(f"  Note: only {len(per_run)}/10 runs - not yet the full 10-run paper protocol.")
    return per_run, by_seed


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)

    _, treat = load_arm(sys.argv[1], "treatment")
    if len(sys.argv) == 3:
        print()
        _, base = load_arm(sys.argv[2], "baseline")
        paired_ttest(treat, base)


if __name__ == "__main__":
    main()

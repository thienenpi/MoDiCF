#!/usr/bin/env python3
"""Aggregate best-Recall@20-epoch metrics across multiple seed logs into mean +/- std.

Follows the MoDiCF paper protocol (Sec 5.2.3): 10 repeated runs, report mean and
standard deviation of Recall/Precision/NDCG/F/F_fuse at K=5, 10 and 20. The per-run
epoch is picked by max test Recall@20, matching the rule in the author's main.py
(self.best_recall20). Pass a second glob to also run the paired t-test the paper
reports (Sec 5.2.2, p < 0.05); runs are paired by the seed in the filename.

Usage:
    python scripts/aggregate_seeds.py "logs/gmddc-baby-k10-b0.3-seed*.out" [--ddof 0|1]
    python scripts/aggregate_seeds.py "logs/gmddc-baby-*-seed*.out" "logs/repro-baby-*-seed*.out"

Globs must stay quoted so Python expands them, not the shell. --ddof selects the std
denominator (default 1 = sample std).
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


# Reported metrics, in table order. Ks = [5, 10, 20, 50] indexes every value list.
METRICS = [("Recall", "recall"), ("Prec", "precision"), ("NDCG", "ndcg"),
           ("F", "fair_p"), ("F_fuse", "f_fuse")]


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
        for name, key in METRICS:
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


def load_arm(pattern, label, ddof=1):
    """Parse every log matching pattern into (list of metrics, {seed: metrics}).

    ddof is passed straight to np.std: 1 (sample std) matches "average results and
    standard deviations" over the paper's 10 repeated runs (Sec 5.2.3); 0 gives the
    population std. It is clamped to 0 when there are too few runs for ddof to be
    defined, so a single-run glob still prints instead of yielding nan.
    """
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

    n = len(per_run)
    eff_ddof = ddof if n > ddof else 0
    print(f"Aggregated over {n} seed run(s)  [std: ddof={eff_ddof}]:")
    # Ks = [5, 10, 20, 50] -> index 0 is @5, index 1 is @10, index 2 is @20
    for lab, idx in [("@5", 0), ("@10", 1), ("@20", 2)]:
        out = []
        for name, key in METRICS:
            v = np.array([b[key][idx] for b in per_run]) * 100
            out.append(f"{name}{lab.ljust(3)}={v.mean():.2f}±{v.std(ddof=eff_ddof):.2f}")
        print("  " + "  ".join(out))

    if n < 10:
        print(f"  Note: only {n}/10 runs - not yet the full 10-run paper protocol.")
    return per_run, by_seed


def main():
    argv = sys.argv[1:]
    ddof = 1
    if "--ddof" in argv:
        i = argv.index("--ddof")
        if i + 1 >= len(argv):
            print("--ddof needs a value (0 or 1)")
            sys.exit(1)
        ddof = int(argv[i + 1])
        del argv[i:i + 2]
    if len(argv) not in (1, 2):
        print(__doc__)
        sys.exit(1)

    _, treat = load_arm(argv[0], "treatment", ddof)
    if len(argv) == 2:
        print()
        _, base = load_arm(argv[1], "baseline", ddof)
        paired_ttest(treat, base)


if __name__ == "__main__":
    main()

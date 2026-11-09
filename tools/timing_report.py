"""Where are our label changes early or late? Public window only, counts per offset bucket.

Each true change in the public 5 minutes is put in one of three kinds -- stoppage start (to D),
restart (D to a team), possession change (team to team) -- and paired with our nearest change into
the same label. The signed offset (ours minus truth) is counted into buckets. Nothing but counts
is printed.

This is a correctness check of what the edges mean, on ~20 changes. It must not be used to tune a
number: EVALUATION.md shows that sample picking the wrong setting on a one-parameter baseline.

    python tools/timing_report.py [submission.csv ...]
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
FPS = 25
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
SEARCH_S = 10
BUCKETS = ((-SEARCH_S, -3), (-3, -1), (-1, -0.4), (-0.4, 0.4), (0.4, 1), (1, 3), (3, SEARCH_S))
DEFAULT = [os.path.join(ROOT, "evaluation", "results", "sonar.csv")]


def kind(transition):
    _, src, dst = transition
    if dst == pm.D:
        return "stoppage start"
    if src == pm.D:
        return "restart"
    return "possession change"


def offsets(gt_codes, pred_codes):
    runs = pm.valid_runs(gt_codes)
    gt_t = pm.transitions(gt_codes, runs)
    pred_t = pm.transitions(pm.fill_abstentions(pred_codes, runs), runs)
    out = {}
    for t in gt_t:
        same = [p[0] for p in pred_t if p[2] == t[2] and abs(p[0] - t[0]) <= SEARCH_S * FPS]
        delta = None if not same else min(same, key=lambda f: abs(f - t[0])) - t[0]
        out.setdefault(kind(t), []).append(delta)
    return out


PUBLIC_CHECK_MESSAGE = (
    "This tool reads the public 5-minute ground truth. RULES.md: those minutes are for checking "
    "that the pipeline is wired correctly, never for training, tuning or choosing between "
    "settings. Run it only to confirm the pipeline works, with --public-check, and do not act "
    "on its numbers.")


def require_public_check_flag():
    import sys
    if "--public-check" not in sys.argv:
        sys.exit(PUBLIC_CHECK_MESSAGE)
    sys.argv.remove("--public-check")


if __name__ == "__main__":
    require_public_check_flag()
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    header = ["none"] + ["{:+g}..{:+g}s".format(a, b) for a, b in BUCKETS]
    for path in sys.argv[1:] or DEFAULT:
        pred = pm.label_codes(pm.read_label_csv(path))
        print("\n{}  (ours minus truth; counts of true changes)".format(os.path.basename(path)))
        print("  {:<18}{:>4}".format("kind", "n") + "".join("{:>11}".format(h) for h in header))
        for k, deltas in sorted(offsets(gt, pred).items()):
            counts = [sum(d is None for d in deltas)]
            for a, b in BUCKETS:
                counts.append(sum(d is not None and a * FPS <= d < b * FPS for d in deltas))
            print("  {:<18}{:>4}".format(k, len(deltas)) +
                  "".join("{:>11}".format(c) for c in counts))

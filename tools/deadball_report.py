"""Fit the dead-ball detector and report on it without looking at any data row.

Three kinds of evidence, in the order they should be trusted for choosing anything:

  1. Cross-half check on the weak labels: fit on one half's Smart Tagging labels, score the other
     half. Label-free with respect to the ground truth, and out of sample.
  2. Plausibility of the decoded stoppages against what a football match looks like: dead share
     of graded time, number of stoppages, how many tagged set pieces get a restart near them.
  3. The public 5 minutes, as a correctness check on the chosen settings only. Never for choosing.

    python tools/deadball_report.py            # defaults
    python tools/deadball_report.py --grid     # also sweep the Viterbi settings, label-free only
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import GAME, half_video_blocks  # noqa: E402
import possession_metrics as pm  # noqa: E402
from sonar import atd, deadball  # noqa: E402
from starter import load_smart_tagging  # noqa: E402

# Params
FPS = 25
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
RESTART_SLACK_S = 5
SHORT_SPELL_S = 3


def auc(scores, labels):
    """Area under the ROC curve via ranks (ties averaged)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    pos = labels.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def cross_half(details, n):
    x, dead, alive = details["x"], details["weak_dead"], details["weak_alive"]
    halves = half_video_blocks(GAME)
    print("Cross-half check on weak labels (fit one half, score the other)")
    for train_half, test_half in ((0, 1), (1, 0)):
        tr = np.zeros(n, dtype=bool)
        te = np.zeros(n, dtype=bool)
        tr[halves[train_half][1]:halves[train_half][2] + 1] = True
        te[halves[test_half][1]:halves[test_half][2] + 1] = True
        tr &= dead | alive
        te &= dead | alive
        w = deadball.fit_logistic(x[tr], dead[tr].astype(float))
        llr = deadball.log_likelihood_ratio(x[te], w)
        y = dead[te]
        balanced = 0.5 * (np.mean(llr[y] > 0) + np.mean(llr[~y] <= 0))
        print("  train half {} -> test half {}: AUC {:.3f}, balanced accuracy {:.3f} "
              "({} dead / {} alive weak frames)".format(
                  train_half + 1, test_half + 1, auc(llr, y), balanced, int(y.sum()),
                  int((~y).sum())))


def spells(mask, first, last):
    """[(start, stop)] runs of True inside [first, last]."""
    window = mask[first:last + 1].astype(np.int8)
    edges = np.diff(np.concatenate(([0], window, [0])))
    return [(first + a, first + b) for a, b in zip(np.flatnonzero(edges == 1),
                                                   np.flatnonzero(edges == -1))]


def plausibility(dead, events, n):
    graded = deadball.graded_mask(n)
    all_spells = []
    for _, first, last in half_video_blocks(GAME):
        all_spells += spells(dead, first, last)
    durations = np.array([b - a for a, b in all_spells]) / FPS
    restarts = np.array([b for _, b in all_spells])
    setpieces = [e for e in events if deadball.code_of(e) == "set pieces"
                 and graded[min(max(e["start_frame"], 0), n - 1)]]
    slack = RESTART_SLACK_S * FPS
    hit = sum(np.any((restarts >= e["start_frame"] - slack) & (restarts < e["end_frame"] + slack))
              for e in setpieces)
    return {
        "dead_share": float(dead[graded].mean()),
        "n_stoppages": len(all_spells),
        "median_s": float(np.median(durations)) if len(durations) else 0.0,
        "short": int(np.sum(durations < SHORT_SPELL_S)),
        "setpiece_restart_recall": hit / max(len(setpieces), 1),
        "n_setpieces": len(setpieces),
    }


def print_plausibility(label, p):
    print("  {:<28} dead {:>5.1f}% of graded | {:>4} stoppages, median {:>5.1f} s, {:>3} under {} s"
          " | restart near {:>2}/{} set pieces ({:.0%})".format(
              label, 100 * p["dead_share"], p["n_stoppages"], p["median_s"], p["short"],
              SHORT_SPELL_S, int(round(p["setpiece_restart_recall"] * p["n_setpieces"])),
              p["n_setpieces"], p["setpiece_restart_recall"]))


def boundary_errors(gt_spells, pred_spells):
    """Median signed start and end error in seconds over GT spells with an overlapping
    prediction (positive = predicted later than the truth), and how many had one."""
    starts, ends = [], []
    for a, b in gt_spells:
        best, overlap = None, 0
        for c, d in pred_spells:
            inter = max(0, min(b, d) - max(a, c))
            if inter > overlap:
                best, overlap = (c, d), inter
        if best:
            starts.append(best[0] - a)
            ends.append(best[1] - b)
    if not starts:
        return None, None, 0
    return float(np.median(starts)) / FPS, float(np.median(ends)) / FPS, len(starts)


def public_check(dead, label="refined"):
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    window = gt != pm.X
    gt_dead = (gt == pm.D) & window
    pred_dead = dead & window
    tp = int(np.sum(gt_dead & pred_dead))
    precision = tp / max(int(pred_dead.sum()), 1)
    recall = tp / max(int(gt_dead.sum()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    first, last = np.flatnonzero(window)[[0, -1]]
    gt_spells = spells(gt_dead, first, last)
    pred_spells = spells(pred_dead, first, last)
    matched = 0
    for a, b in gt_spells:
        for c, d in pred_spells:
            inter = max(0, min(b, d) - max(a, c))
            if inter and inter / float(max(b, d) - min(a, c)) >= 0.5:
                matched += 1
                break
    start_err, end_err, n_overlap = boundary_errors(gt_spells, pred_spells)
    print("  {:<8} D precision {:.3f}  recall {:.3f}  F1 {:.3f} | predicted stoppages {} | "
          "GT stoppages matched at IoU>=0.5: {} of {}".format(
              label, precision, recall, f1, len(pred_spells), matched, len(gt_spells)))
    if n_overlap:
        print("  {:<8} median edge error over {} overlapping stoppages: start {:+.1f} s, "
              "end {:+.1f} s".format(label, n_overlap, start_err, end_err))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()

    data = atd.load()
    events = load_smart_tagging.load()
    dead, details = deadball.detect(data, events)
    n = len(dead)

    print("Fitted weights (log-likelihood ratio, dead over alive):")
    for name, value in details["weights"].items():
        print("  {:<16} {:>+8.4f}".format(name, value))
    print()
    cross_half(details, n)

    print("\nPlausibility of the decoded stoppages (a match: ~40% dead, ~80-130 stoppages):")
    print_plausibility("defaults", plausibility(dead, events, n))
    if args.grid:
        # Only switch_cost / temperature matters to the Viterbi path, so sweep that ratio: the
        # number of frames of unit evidence one switch costs.
        for frames in (12.5, 25, 50, 100, 200):
            grid_dead = deadball.decode(details["llr"], n, temperature=1.0 / FPS,
                                        switch_cost=frames / FPS)
            print_plausibility("switch = {:>5} frames".format(frames),
                               plausibility(grid_dead, events, n))
    weak_alive = details["weak_alive"]
    print("  predicted-dead frames on weak-alive (open-play) frames: {:.1%} of weak-alive".format(
        np.sum(dead & weak_alive) / max(weak_alive.sum(), 1)))
    print_plausibility("viterbi (before refine)", plausibility(details["viterbi_dead"], events, n))
    print("\nEdge refinement: {}".format(
        {k: (round(v, 2) if isinstance(v, float) else v) for k, v in details["refine"].items()}))

    gt_n = int(np.sum(pm.label_codes(pm.read_label_csv(PUBLIC_GT)) == pm.D))
    print("\nPublic 5 minutes (correctness check only; {} GT dead frames):".format(gt_n))
    public_check(details["viterbi_dead"], "viterbi")
    public_check(dead, "refined")

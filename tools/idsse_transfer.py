"""Label-free check of a dead-ball model on BAR-ATH: release weights vs weights fitted on IDSSE truth.

    python tools/idsse_transfer.py [--degrade 0.41]

Nothing here reads evaluation/gt. For each candidate it prints, over BAR-ATH's graded frames:
the dead share, the stoppage count and how many are under 3 s, and the ROC-AUC of the per-frame
log-likelihood ratio against the Smart Tagging weak labels (the release model is fitted on those
labels, so its AUC there is in-sample and flattered; the IDSSE model never saw them).

Prints aggregates only.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import idsse_data  # noqa: E402
import idsse_eval as ie  # noqa: E402
from sonar import atd, deadball, deadball_features as dbf  # noqa: E402
from starter import load_smart_tagging  # noqa: E402
from tools.deadball_report import auc  # noqa: E402


def stoppage_counts(dead, graded):
    n, short = 0, 0
    for _, first, last in deadball.half_video_blocks(deadball.GAME):
        seg = dead[first:last + 1].astype(np.int8)
        edges = np.diff(np.concatenate(([0], seg, [0])))
        lengths = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
        n += len(lengths)
        short += int((lengths < 3 * deadball.FPS).sum())
    return n, short


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--degrade", type=float, default=0.41)
    args = parser.parse_args()

    w_release = ie.bar_ath_weights()
    prepared = {m: ie.prepare(m, args.degrade, k) for k, m in enumerate(idsse_data.available())}
    w_dfl = ie.dfl_weights(prepared, exclude=None)
    ie.use_blocks(None)

    data = atd.load()
    events = load_smart_tagging.load()
    features = dbf.compute(data)
    x, names = deadball.design(features)
    n = len(x)
    graded = deadball.graded_mask(n)
    weak_dead, weak_alive = deadball.weak_labels(n, events)
    train = weak_dead | weak_alive
    print("BAR-ATH graded frames {}, weak-labelled {}".format(int(graded.sum()), int(train.sum())))
    for name, w in (("release", w_release), ("idsse-trained (all 7)", w_dfl)):
        llr = deadball.log_likelihood_ratio(x, w)
        dead = deadball.decode(llr, n)
        refined, _, _ = deadball.refine(dead, data, features)
        stops, short = stoppage_counts(refined, graded)
        print("  {:22s} dead share {:.1%}  stoppages {} (under 3 s {})  weak-label AUC {:.3f}"
              .format(name, refined[graded].mean(), stops, short,
                      auc(llr[train], weak_dead[train])))
    print("\nweights (bias first):")
    for k, col in enumerate(["bias"] + names):
        print("  {:16s} release {:+8.3f}   idsse {:+8.3f}".format(col, w_release[k], w_dfl[k]))


if __name__ == "__main__":
    main()

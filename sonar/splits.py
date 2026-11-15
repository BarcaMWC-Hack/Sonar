"""The team's shared train/test folds: graded frames only.

Everyone trains and validates on these folds, so results can be compared fold by fold.

Ungraded frames (before kick-off, halftime, after the final whistle) never take part: sonar/atd.py
blanks them when it loads the tracking, and every mask here is limited to graded frames.

Neighbouring frames are near copies, so splitting frames at random leaks: a test frame's neighbours
land in training and the score looks better than it is. So each half is cut into consecutive
blocks. Fold k tests block k of both halves, every training set contains both halves, and GAP_S
seconds either side of each test block are left out of training.

    n = source_frames("tactical")
    for fold, train, test in splits.folds(n):        # the standard check
        fit on (train & labelled), score on (test & labelled)

    for fold, train, test in splits.half_folds(n):   # stress test: split the halves and swap

    for outer, pool, test in splits.folds(n):        # tuning a setting: nested
        for inner, fit, val in splits.folds(n, within=pool, skip=[outer]):
            choose the setting on val, then score it once on test

Report the mean and the worst fold. Call one model better only if it wins at least 4 of 5 folds.

    python -m sonar.splits                           # print the folds and check them
    python -m sonar.splits --json folds.json         # the same frame ranges, for other tools

Only frame numbers are involved, so the output holds no match data.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks, source_frames  # noqa: E402

# Params
FPS = 25
N_FOLDS = 5     # consecutive blocks per half, and so the number of folds
GAP_S = 10      # covers the longest feature window (the 10 s ball-out memory)


def _frames(n, game):
    return source_frames("tactical", game) if n is None else n


def graded_mask(n=None, game=GAME):
    """True on the frames the competition grades."""
    mask = np.zeros(_frames(n, game), dtype=bool)
    for _, first, last in half_video_blocks(game):
        mask[first:last + 1] = True
    return mask


def block_ranges(game=GAME):
    """[(fold, half, first frame, last frame)], last inclusive."""
    out = []
    for half, first, last in half_video_blocks(game):
        edges = np.linspace(first, last + 1, N_FOLDS + 1).astype(np.int64)
        out += [(k, half, int(edges[k]), int(edges[k + 1]) - 1) for k in range(N_FOLDS)]
    return out


def block_ids(n=None, game=GAME):
    """The fold each frame is tested in; -1 on ungraded frames."""
    ids = np.full(_frames(n, game), -1, dtype=np.int8)
    for fold, _, first, last in block_ranges(game):
        ids[first:last + 1] = fold
    return ids


def grow(mask, frames):
    """mask widened by `frames` on both sides."""
    count = np.concatenate(([0], np.cumsum(mask)))
    index = np.arange(len(mask))
    return count[np.minimum(len(mask), index + frames + 1)] > count[np.maximum(0, index - frames)]


def folds(n=None, game=GAME, within=None, skip=()):
    """Yield (fold, train mask, test mask), graded frames only, GAP_S clear of each test block."""
    ids = block_ids(n, game)
    pool = ids >= 0 if within is None else (ids >= 0) & within
    for fold in range(N_FOLDS):
        if fold not in skip:
            block = ids == fold
            yield fold, pool & ~grow(block, GAP_S * FPS), pool & block


def half_folds(n=None, game=GAME):
    """Yield (fold, train mask, test mask) for fitting on one half and testing on the other."""
    masks = []
    for _, first, last in half_video_blocks(game):
        mask = np.zeros(_frames(n, game), dtype=bool)
        mask[first:last + 1] = True
        masks.append(mask)
    yield 0, masks[0], masks[1]
    yield 1, masks[1], masks[0]


def check(n=None, game=GAME):
    """Raise if a fold breaks a rule. Returns (graded frames, smallest training set)."""
    n = _frames(n, game)
    graded = graded_mask(n, game)
    halves = [mask for _, mask, _ in half_folds(n, game)]
    tested = np.zeros(n, dtype=np.int64)
    smallest = n
    for fold, train, test in folds(n, game):
        tested[test] += 1
        smallest = min(smallest, int(train.sum()))
        if np.any((train | test) & ~graded):
            raise ValueError("fold {} uses an ungraded frame".format(fold))
        if np.any(train & grow(test, GAP_S * FPS)):
            raise ValueError("fold {} trains inside the gap".format(fold))
        if not all(np.any(train & half) and np.any(test & half) for half in halves):
            raise ValueError("fold {} does not contain both halves".format(fold))
        for inner, fit, val in folds(n, game, within=train, skip=[fold]):
            if np.any((fit | val) & ~train) or np.any(fit & grow(val, GAP_S * FPS)):
                raise ValueError("fold {} inner {} leaves its pool".format(fold, inner))
    if np.any(tested[graded] != 1) or np.any(tested[~graded] != 0):
        raise ValueError("every graded frame must be tested exactly once")
    return int(graded.sum()), smallest


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", help="also write the frame ranges to this file")
    args = parser.parse_args()

    ranges = block_ranges()
    print("{} folds, {} s left out of training either side of each test block".format(
        N_FOLDS, GAP_S))
    print("fold  half       first frame  last frame  frames")
    for fold, half, first, last in sorted(ranges):
        print("{:>4}  {:<9}  {:>11}  {:>10}  {:>6}".format(fold, half, first, last,
                                                           last - first + 1))
    n_graded, smallest = check()
    print("checks passed: {} graded frames, each tested once; no ungraded frame in any fold; "
          "smallest training set {} frames".format(n_graded, smallest))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"game": GAME, "gap_seconds": GAP_S, "folds": [
                {"fold": f, "half": h, "first": a, "last": b} for f, h, a, b in sorted(ranges)]},
                handle, indent=2)
        print("wrote {}".format(args.json))


if __name__ == "__main__":
    main()

"""Pass/fail checks on a submission, beyond the format check in evaluation/validate_submission.py.

    python tools/check_submission.py                      # evaluation/results/sonar.csv
    python tools/check_submission.py path/to/other.csv

Each check uses only what the organisers state (FAQ.md, the graded halves) or a label-free football
norm, so a failure means the file is not what it claims to be. Prints pass/fail and counts only,
and exits 1 if any check fails.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402

FPS = 25
N_FRAMES = 199319
PRE_KICKOFF = (110808, 113742)      # FAQ.md: these graded frames are all D
MAX_ABSTAIN_SHARE = 0.01            # graded frames left X
DEAD_SHARE = (0.30, 0.50)           # a match is about 40% dead
STOPPAGES = (80, 130)               # football: roughly 80-130 stoppages a match
POSSESSIONS = (150, 400)            # possessions per match, from 10-15 s typical spells
MEAN_POSSESSION_S = (8.0, 20.0)


def check(name, ok, detail):
    print("{}  {:<30} {}".format("PASS" if ok else "FAIL", name, detail))
    return ok


def runs_of(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return int(np.sum(edges == 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission", nargs="?",
                        default=os.path.join(ROOT, "evaluation", "results", "sonar.csv"))
    args = parser.parse_args()

    frame = pd.read_csv(args.submission, dtype={"label": str}, keep_default_na=False)
    labels = frame["label"].to_numpy()
    graded = np.zeros(len(labels), dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True

    results = []
    results.append(check("rows", len(labels) == N_FRAMES,
                         "{} (expected {})".format(len(labels), N_FRAMES)))
    values, counts = np.unique(labels, return_counts=True)
    results.append(check("labels", set(values) <= set("HADX"),
                         "{}".format({v: int(c) for v, c in zip(values, counts)})))
    results.append(check("ungraded frames are X", np.all(labels[~graded] == "X"),
                         "{} ungraded frames".format(int(np.sum(~graded)))))
    lo, hi = PRE_KICKOFF
    results.append(check("pre-kickoff frames are D", np.all(labels[lo:hi] == "D"),
                         "frames {}..{}: {} of {} D".format(lo, hi - 1,
                                                          int(np.sum(labels[lo:hi] == "D")),
                                                          hi - lo)))

    abstain = np.mean(labels[graded] == "X")
    results.append(check("graded X share", abstain <= MAX_ABSTAIN_SHARE,
                         "{:.2%} (limit {:.0%})".format(abstain, MAX_ABSTAIN_SHARE)))

    dead = labels == "D"
    share = np.mean(dead[graded])
    results.append(check("dead share of graded", DEAD_SHARE[0] <= share <= DEAD_SHARE[1],
                         "{:.1%} (band {:.0%}-{:.0%})".format(share, *DEAD_SHARE)))

    stoppages = sum(runs_of(dead[first:last + 1]) for _, first, last in half_video_blocks(GAME))
    results.append(check("stoppages", STOPPAGES[0] <= stoppages <= STOPPAGES[1],
                         "{} (band {}-{})".format(stoppages, *STOPPAGES)))

    spells, spell_frames = 0, 0
    for _, first, last in half_video_blocks(GAME):
        seg = labels[first:last + 1].copy()
        for i in range(1, len(seg)):          # the scorer carries X forward within a half
            if seg[i] == "X":
                seg[i] = seg[i - 1]
        starts = [i for i in range(len(seg)) if seg[i] in "HA" and (i == 0 or seg[i - 1] != seg[i])]
        spells += len(starts)
        spell_frames += int(np.sum(np.isin(seg, ["H", "A"])))
    mean_s = spell_frames / max(spells, 1) / FPS
    results.append(check("possessions", POSSESSIONS[0] <= spells <= POSSESSIONS[1],
                         "{} (band {}-{})".format(spells, *POSSESSIONS)))
    results.append(check("mean possession", MEAN_POSSESSION_S[0] <= mean_s <= MEAN_POSSESSION_S[1],
                         "{:.1f} s (band {:.0f}-{:.0f} s)".format(mean_s, *MEAN_POSSESSION_S)))

    failed = len(results) - sum(results)
    print("\n{} of {} checks passed".format(len(results) - failed, len(results)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

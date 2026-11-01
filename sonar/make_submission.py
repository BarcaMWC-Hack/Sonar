"""Write Sonar's submission: dead-ball detector + a team call inside every alive stretch.

    python -m sonar.make_submission                  # evaluation/results/sonar.csv
    python -m sonar.make_submission --name try2      # evaluation/results/try2.csv
    python -m sonar.make_submission --team nearest   # baseline team call, for comparison

How a frame gets its label:
  - outside the two graded halves: X (never scored)
  - the dead-ball detector says dead: D
  - otherwise the frame is inside an alive stretch, and gets the team call. By default that is
    sonar/possession.py (control-weighted evidence, switching cost). With --team nearest it is a
    centred majority vote of the nearest-player codes, kept inside the stretch so a vote never
    reaches across a stoppage; frames with no vote take the nearest voted frame in the same
    stretch, so a possession is never filled with the label from before a stoppage.
  - A stretch with no team evidence at all is X, which the scorer carries forward.

Prints summary counts only.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from sonar import atd, deadball, possession, possession_baseline  # noqa: E402
from starter import load_smart_tagging  # noqa: E402
from starter.baseline import write_submission  # noqa: E402

# Params
RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
TEAM_WINDOW = 50        # frames either side for the team majority vote, as baseline_smooth50


def runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def team_labels_in(codes, window):
    """Labels for one alive stretch: 'H' / 'A' by local majority, nearest-filled; or all 'X'."""
    n = len(codes)
    home = np.concatenate(([0], np.cumsum(codes == possession_baseline.HOME)))
    away = np.concatenate(([0], np.cumsum(codes == possession_baseline.AWAY)))
    index = np.arange(n)
    low = np.maximum(0, index - window)
    high = np.minimum(n, index + window + 1)
    h, a = home[high] - home[low], away[high] - away[low]
    voted = (h + a) > 0
    if not voted.any():
        return np.full(n, "X", dtype="<U1")
    labels = np.where(h >= a, "H", "A")
    # Nearest voted frame for the rest: forward fill, then back-fill the head.
    carry = np.maximum.accumulate(np.where(voted, index, -1))
    carry[carry < 0] = np.flatnonzero(voted)[0]
    return labels[carry]


def build_labels(dead, codes, game=GAME, window=TEAM_WINDOW):
    labels = np.full(len(dead), "X", dtype="<U1")
    for _, first, last in half_video_blocks(game):
        stop = last + 1
        labels[first:stop][dead[first:stop]] = "D"
        for a, b in runs(~dead[first:stop]):
            labels[first + a:first + b] = team_labels_in(codes[first + a:first + b], window)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default="sonar")
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument("--team", choices=("control", "nearest"), default="control",
                        help="control: sonar/possession.py (default); nearest: the baseline's "
                             "nearest player with a +/-2 s majority vote")
    args = parser.parse_args()

    data = atd.load()
    events = load_smart_tagging.load()
    dead, _ = deadball.detect(data, events)
    if args.team == "control":
        evidence, _ = possession.evidence(data, dead)
        labels = possession.labels(dead, evidence, possession.tag_evidence(events, len(dead)))
    else:
        labels = build_labels(dead, possession_baseline.nearest_team(data))

    path = write_submission(os.path.join(args.out, args.name + ".csv"), labels)
    graded = deadball.graded_mask(len(labels))
    counts = {k: int(np.sum(labels[graded] == k)) for k in "HADX"}
    n_poss = 0
    for _, first, last in half_video_blocks(GAME):
        seg = labels[first:last + 1]
        filled = seg.copy()
        # possessions as the scorer sees them: X carried forward
        for i in range(1, len(filled)):
            if filled[i] == "X":
                filled[i] = filled[i - 1]
        n_poss += sum(1 for i in range(len(filled))
                      if filled[i] in "HA" and (i == 0 or filled[i - 1] != filled[i]))
    print("graded frames: H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("possessions (H/A spells after X carry-forward): {}".format(n_poss))
    print("wrote {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()

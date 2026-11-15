"""Does the broadcast show replays and close-ups when the ball is dead? Label-free measurement only.

    python tools/broadcast_cue.py

Mapping (found label-free by matching players and ball, tools/broadcast_align.py and the second-half
search): tactical frame = broadcast frame + 160.5 s in the first half and + 1076 s in the second half
(halftime is cut from the broadcast). Every mapped frame is checked again with players below.

Cue (per broadcast frame, every 5th frame):
  close-up : camera calibration not valid, or fewer than 4 players visible
  replay   : calibration valid, 4+ players visible, but those players sit more than 3 m (median)
             from the nearest tactical player of the same team at the mapped time
Cue stretches shorter than 3 s are ignored.

Gate (fixed before any number was seen): the cue is a candidate dead-ball signal only if
  (a) at least 80% of cue time falls in our predicted stoppages or within 5 s of one, and
  (b) at least 60% of the cue stretches that fall in our predicted live time show stoppage signs
      (median player speed below the 25th percentile of predicted-live frames).
Nothing is adopted here. Prints statistics only; nothing leaves this machine.
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import broadcast_align as ba  # noqa: E402
from games import GAME, half_video_blocks, require  # noqa: E402
from sonar import deadball  # noqa: E402
from starter import load_smart_tagging  # noqa: E402

FPS = 25
SHIFT_1, SHIFT_2 = int(160.5 * FPS), 1076 * FPS
STEP = 5


def runs_of(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def main():
    tac, brd = ba.load_stream("tactical"), ba.load_stream("broadcast")
    n_t, n_b = len(tac["ball"]), len(brd["ball"])
    (hom_path,) = require(["homography"], GAME, None, "broadcast")
    hom = pd.read_csv(hom_path, usecols=["frame", "valid"])
    valid = np.zeros(n_b, dtype=bool)
    frames = hom["frame"].to_numpy()
    keep = (frames >= 0) & (frames < n_b)
    valid[frames[keep]] = hom["valid"].to_numpy()[keep].astype(bool)

    blocks = half_video_blocks(GAME)
    first_end = blocks[0][2]
    # Broadcast frames before this map with the first-half shift, after it with the second.
    boundary = first_end - SHIFT_1 + 1

    TP, BP = tac["players"][:, :, :2].astype(float), brd["players"][:, :, :2].astype(float)
    cue = np.zeros(n_t, dtype=bool)
    kinds = {"close-up": 0, "replay": 0, "matching": 0}
    check = []
    for b in range(0, n_b, STEP):
        t = b + (SHIFT_1 if b < boundary else SHIFT_2)
        if not 0 <= t < n_t:
            continue
        nvis = int(np.isfinite(BP[b, :, 0]).sum())
        if not valid[b] or nvis < 4:
            kinds["close-up"] += 1
            cue[t:t + STEP] = True
            continue
        dists = []
        for code in (0, 1):
            bp = BP[b, brd["team"] == code]
            tp = TP[t, tac["team"] == code]
            bp, tp = bp[np.isfinite(bp[:, 0])], tp[np.isfinite(tp[:, 0])]
            if len(bp) and len(tp):
                dists.extend(np.sqrt(((bp[:, None] - tp[None]) ** 2).sum(2)).min(1).tolist())
        med = np.median(dists) if dists else np.inf
        if med > 3.0:
            kinds["replay"] += 1
            cue[t:t + STEP] = True
        else:
            kinds["matching"] += 1
            check.append(med)
    print("broadcast frames sampled every {}: {}".format(STEP, kinds))
    print("mapping check on matching frames: median player residual {:.2f} m".format(
        float(np.median(check))))

    graded = np.zeros(n_t, dtype=bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    covered = np.zeros(n_t, dtype=bool)
    for a, b in ((max(0, -SHIFT_1), boundary + SHIFT_1), (boundary + SHIFT_2, n_b + SHIFT_2)):
        covered[max(0, a):min(n_t, b)] = True
    cue &= graded & covered
    segs = [(a, b) for a, b in runs_of(cue) if b - a >= 3 * FPS]
    cue_long = np.zeros(n_t, dtype=bool)
    for a, b in segs:
        cue_long[a:b] = True

    data = ba.load_stream("tactical")
    dead, _ = deadball.detect(data, load_smart_tagging.load())
    near_dead = dead.copy()
    for a, b in runs_of(dead):
        near_dead[max(0, a - 5 * FPS):min(n_t, b + 5 * FPS)] = True
    speed = np.nanmedian(data["players"][:, :, 2].astype(float), axis=1)
    live = graded & covered & ~dead
    p25 = np.nanpercentile(speed[live], 25)

    share_dead = float((cue_long & near_dead).sum()) / max(int(cue_long.sum()), 1)
    live_segs = [(a, b) for a, b in segs if not near_dead[a:b].any() or (~near_dead[a:b]).mean() > 0.5]
    slow = sum(1 for a, b in live_segs if np.nanmedian(speed[a:b]) < p25)
    share_slow = slow / max(len(live_segs), 1)
    print("graded tactical time covered by the broadcast: {:.1%}".format((graded & covered).mean() / graded.mean()))
    print("cue stretches (3 s+): {}, total {:.0f} s".format(len(segs), cue_long.sum() / FPS))
    print("(a) cue time in or within 5 s of our predicted stoppages: {:.1%}".format(share_dead))
    print("    our predicted-dead time (covered) that the cue marks: {:.1%}".format(
        float((cue_long & dead & covered).sum()) / max(int((dead & graded & covered).sum()), 1)))
    print("(b) cue stretches mainly in our predicted live time: {}; with stoppage signs: {} ({:.0%})".format(
        len(live_segs), slow, share_slow))
    print("    live-time cue that would turn into D if adopted: {:.0f} s".format(
        sum(b - a for a, b in live_segs) / FPS))
    gate = share_dead >= 0.80 and share_slow >= 0.60
    print("GATE (a >= 80% and b >= 60%):", "PASS: candidate dead-ball cue" if gate else "FAIL")


if __name__ == "__main__":
    main()

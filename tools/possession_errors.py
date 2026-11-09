"""Where could possession switches go wrong? Label-free counts on the full match.

  1. Interpolated ball: the tracker fills short gaps itself. A filled stretch is a straight line
     at constant speed (zero second difference), and it can pass a player who never touched the
     ball. How much of the live, controlled evidence sits on such frames?
  2. Switches at gap ends: with no evidence during a gap, a change can only land where evidence
     resumes. How many of our switches sit right at the end of a long ball gap?
  3. Switch support: how much control evidence does the new team build in the 2 s after each of
     our switches, and how many switches are reversed within 3 s?

    python tools/possession_errors.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from sonar import atd, deadball, deadball_features as dbf, possession  # noqa: E402
from starter import load_smart_tagging  # noqa: E402

# Params
FPS = 25
LINEAR_TOL = 1e-3       # m; second difference below this is a straight-line (filled) frame


def linear_frames(ball_xy):
    """True where the ball's position is exactly on a straight line through its neighbours."""
    second = ball_xy[2:] - 2 * ball_xy[1:-1] + ball_xy[:-2]
    out = np.zeros(len(ball_xy), dtype=bool)
    with np.errstate(invalid="ignore"):
        out[1:-1] = np.hypot(*second.T) < LINEAR_TOL
    return out


def switches(labels):
    out = []
    for _, first, last in half_video_blocks(GAME):
        seg = labels[first:last + 1]
        for i in np.flatnonzero((seg[1:] != seg[:-1]) & np.isin(seg[1:], ("H", "A")) &
                                np.isin(seg[:-1], ("H", "A"))) + 1:
            out.append((first + int(i), seg[i]))
    return out


if __name__ == "__main__":
    data = atd.load()
    events = load_smart_tagging.load()
    dead, details = deadball.detect(data, events)
    n = len(dead)
    alive = deadball.graded_mask(n) & ~dead
    ball_xy = data["ball"][:, :2].astype(np.float64)
    known = ~np.isnan(ball_xy[:, 0])
    full, det = possession.evidence(data, dead)
    ev = det["ball"]        # support is measured on ball control alone
    labels = possession.labels(dead, full, possession.tag_evidence(events, n))

    lin = linear_frames(ball_xy) & known
    edges = np.diff(np.concatenate(([0], lin.astype(np.int8), [0])))
    lens = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    print("1. Straight-line (filled) ball frames")
    print("   {:.1%} of tracked live frames; runs: {} total, {} of 5+ frames, {} of 25+ frames".format(
        np.sum(lin & alive) / max(np.sum(known & alive), 1), len(lens), int(np.sum(lens >= 5)),
        int(np.sum(lens >= 25))))
    strong = alive & (np.abs(ev) > 0.5)
    print("   strong control evidence (|e| > 0.5) on live frames: {} frames, {:.1%} of them on "
          "straight-line runs of 5+".format(
              int(strong.sum()),
              np.sum(strong & (dbf.run_lengths(lin) >= 5)) / max(strong.sum(), 1)))

    sw = switches(labels)
    _, gap_ends = deadball.long_gaps(known)
    at_gap_end = sum(np.any((gap_ends <= f) & (gap_ends >= f - FPS // 2)) for f, _ in sw)
    print("\n2. Our possession switches inside live play: {}".format(len(sw)))
    print("   within 0.5 s after a long ball gap ends: {}".format(at_gap_end))

    support, reversed_ = [], 0
    for f, team in sw:
        sign = 1.0 if team == "H" else -1.0
        support.append(np.sum(np.clip(sign * ev[f:f + 2 * FPS], 0, None)))
        if np.any(labels[f + 1:f + 3 * FPS] != team) and \
                np.any(labels[f + 1:f + 3 * FPS] == ("A" if team == "H" else "H")):
            reversed_ += 1
    support = np.array(support)
    print("\n3. Evidence for the new team in the 2 s after each switch (frames of full control):")
    print("   < 3: {} | 3-10: {} | 10-25: {} | 25+: {}".format(
        int(np.sum(support < 3)), int(np.sum((support >= 3) & (support < 10))),
        int(np.sum((support >= 10) & (support < 25))), int(np.sum(support >= 25))))
    print("   switches reversed within 3 s: {}".format(reversed_))

    t_frames, t_sides = possession.touches(data)
    live_t = alive[t_frames]
    live_minutes = alive.sum() / FPS / 60.0
    print("\n4. Touches (ball velocity jumps > {} m/s within {} m of a player)".format(
        possession.TOUCH_DV, possession.TOUCH_RADIUS))
    print("   {} in live play, {:.1f} per live minute, {:.0%} by Barca".format(
        int(live_t.sum()), live_t.sum() / live_minutes, np.mean(t_sides[live_t] > 0)))

    side = possession.control_side(ev, t_frames, t_sides, n)
    localized, counts = possession.localize(labels, dead, ev, data)

    def on_ball(lab):
        hit = total = 0
        for f, team in switches(lab):
            sign = 1.0 if team == "H" else -1.0
            hit += bool(np.any(side[max(0, f - 5):f + 6] == sign))
            total += 1
        return hit, total
    shifts = np.array(counts["shifts"]) / FPS
    print("\n5. Re-placing switches at the new team's first touch or control")
    print("   moved {}, kept {} | earlier by 1-3 s {}, 0.4-1 s {} | within 0.4 s {} | "
          "later by 0.4-1 s {}, 1-3 s {}".format(
              counts["moved"], counts["kept"], int(np.sum(shifts <= -1)),
              int(np.sum((shifts > -1) & (shifts <= -0.4))), int(np.sum(np.abs(shifts) < 0.4)),
              int(np.sum((shifts >= 0.4) & (shifts < 1))), int(np.sum(shifts >= 1))))
    for name, lab in (("before", labels), ("after", localized)):
        hit, total = on_ball(lab)
        print("   {:<6} new team touching or controlling the ball within 0.2 s of the switch: "
              "{} of {}".format(name, hit, total))

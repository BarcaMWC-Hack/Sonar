"""What does the team signal look like during live play? Shares and counts only.

Uses the dead-ball detector to find live play, then reports how often there is any ball evidence,
how close the nearest player is, how fast the ball moves, how much the raw nearest-player call
flickers, and what each Smart Tagging code's Team tag means relative to that call.

    python tools/possession_diagnostics.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from sonar import atd, deadball, possession_baseline as pb  # noqa: E402
from sonar.possession import nearest_by_team  # noqa: E402
from starter import load_smart_tagging  # noqa: E402

# Params
FPS = 25
TEAM_TAGS = {"barcelona": pb.HOME, "atletic club": pb.AWAY}


if __name__ == "__main__":
    data = atd.load()
    events = load_smart_tagging.load()
    dead, details = deadball.detect(data, events)
    n = len(dead)
    alive = deadball.graded_mask(n) & ~dead
    known = details["features"]["ball_known"]
    speed = deadball.instant_ball_speed(data["ball"][:, :2].astype(np.float64))
    d_home, d_away = nearest_by_team(data)
    nearest = np.minimum(d_home, d_away)
    codes = pb.nearest_team(data)

    ak = alive & known
    print("live play (detector): {} frames; ball tracked on {:.1%}".format(
        int(alive.sum()), ak.sum() / alive.sum()))
    with np.errstate(invalid="ignore"):
        for r in (1.0, 2.0, 3.0, 5.0):
            print("  nearest player within {:.0f} m of the ball: {:.1%} of tracked live frames".format(
                r, np.mean(nearest[ak] < r)))
        sp = speed[ak]
        sp = sp[~np.isnan(sp)]
        print("  ball speed < 3 m/s {:.1%} | 3-8 m/s {:.1%} | > 8 m/s {:.1%}".format(
            np.mean(sp < 3), np.mean((sp >= 3) & (sp <= 8)), np.mean(sp > 8)))
        margin = np.abs(d_home - d_away)[ak]
        print("  |home - away nearest distance| < 1 m (contested): {:.1%}".format(np.mean(margin < 1)))

    # Flicker of the raw call, and of the call restricted to controlled frames.
    def changes_per_min(mask):
        idx = np.flatnonzero(mask)
        c = codes[idx]
        flips = np.sum((c[1:] != c[:-1]) & (np.diff(idx) < FPS))
        return flips / (alive.sum() / FPS / 60.0)
    with np.errstate(invalid="ignore"):
        controlled = ak & (nearest < 2.0) & (speed < 6.0)
    print("  raw nearest-team changes per live minute: {:.1f}; on controlled frames only "
          "(nearest < 2 m, ball < 6 m/s): {:.1f}".format(changes_per_min(ak),
                                                         changes_per_min(controlled)))
    print("  controlled frames: {:.1%} of live frames".format(controlled.sum() / alive.sum()))

    # Live stretches with no tracked-ball frame at all.
    edges = np.diff(np.concatenate(([0], alive.astype(np.int8), [0])))
    stretches = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))
    empty = sum(1 for a, b in stretches if not known[a:b].any())
    print("  live stretches: {}, with no tracked ball at all: {}".format(len(stretches), empty))

    # Smart Tagging: does the Team tag name the side with the ball? Per code, share of its
    # controlled live frames whose nearest-player team equals the tagged team.
    print("\nSmart Tagging Team tag vs controlled nearest-player team (share agreeing):")
    rows = {}
    for e in events:
        team = TEAM_TAGS.get((e["tags"].get("Team") or "").strip().lower())
        if team is None:
            continue
        sl = slice(max(0, e["start_frame"]), min(n, e["end_frame"]))
        m = controlled[sl]
        if not m.any():
            continue
        agree = np.sum(codes[sl][m] == team)
        tot = rows.setdefault(deadball.code_of(e), [0, 0, 0])
        tot[0] += agree
        tot[1] += int(m.sum())
        tot[2] += 1
    for code, (agree, total, count) in sorted(rows.items(), key=lambda kv: -kv[1][1]):
        print("  {:<30} {:>4} tags  {:>7} frames  agree {:.0%}".format(
            code, count, total, agree / max(total, 1)))

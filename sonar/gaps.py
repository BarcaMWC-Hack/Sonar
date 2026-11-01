"""Place a possession switch inside a ball-tracking gap.

The decoder (sonar/possession.py) pays for a switch when the first evidence of the new team
arrives, which after a gap is when the ball reappears. The switch itself happened somewhere in
the gap. Whether possession changed is still decided only from observed tracking; this only
moves the switch earlier, inside the gap that ends just before it, and never past the previous
switch or stoppage.

Rules (place_switches_in_gaps):
  end        leave the switch where the decoder put it (the status quo)
  mid        halfway through the gap
  proximity  the first frame the new team's nearest player is closer than the old team's to the
             ball's straight-line path across the gap (the path is used for timing only, never
             as control evidence); if that never happens, leave the switch
"""

import numpy as np

FPS = 25
MIN_GAP_S = 0.4         # shorter gaps are ordinary jitter
SLACK_S = 2.0           # a switch counts as "after the gap" up to this long after the gap ends
RULES = ("end", "mid", "proximity")


def nan_runs(missing):
    edges = np.diff(np.concatenate(([0], missing.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def _nearest(players, team, point, side):
    sel = team == side
    d = np.sqrt(((players[sel] - point) ** 2).sum(axis=1))
    d = d[np.isfinite(d)]
    return d.min() if len(d) else np.inf


def place_switches_in_gaps(labels, dead, ball, data, rule="proximity"):
    """(labels with switches moved into the gap before them, number moved)."""
    if rule not in RULES:
        raise ValueError(rule)
    out = labels.copy()
    if rule == "end":
        return out, 0
    xy = ball[:, :2].astype(np.float64)
    gaps = [(a, b) for a, b in nan_runs(np.isnan(xy[:, 0])) if b - a >= MIN_GAP_S * FPS]
    ends = np.array([b for _, b in gaps]) if gaps else np.array([], dtype=int)
    players = data["players"][:, :, :2].astype(np.float64)
    team = data["team"]
    dead_idx = np.flatnonzero(dead)
    moved, barrier = 0, 0
    for s in (np.flatnonzero(labels[1:] != labels[:-1]) + 1).tolist():
        old, new = labels[s - 1], labels[s]
        if old not in "HA" or new not in "HA":
            barrier = s
            continue
        k = int(np.searchsorted(ends, s, side="right")) - 1
        if k < 0 or s - ends[k] > SLACK_S * FPS:
            barrier = s
            continue
        a, b = gaps[k]
        before = dead_idx[dead_idx < s]
        lo = max(a, barrier + 1, (int(before[-1]) + 1) if len(before) else 0)
        if lo >= s:
            barrier = s
            continue
        if rule == "mid":
            target = (lo + s) // 2
        else:
            start = xy[a - 1] if a > 0 else xy[b]
            stop = xy[b] if b < len(xy) else start
            if np.isnan(start).any() or np.isnan(stop).any():
                barrier = s
                continue
            new_side = 0 if new == "H" else 1
            target = s
            for t in range(lo, min(b, s)):
                frac = (t - a + 1) / float(b - a + 1)
                point = start + frac * (stop - start)
                if _nearest(players[t], team, point, new_side) < \
                        _nearest(players[t], team, point, 1 - new_side):
                    target = t
                    break
        if target < s:
            out[target:s] = new
            moved += 1
        barrier = min(target, s)
    return out, moved

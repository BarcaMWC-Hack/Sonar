"""Label-free edge-case census of the current submission on the provided match.

    python tools/edge_census.py

Reads only the tracking, the Smart Tagging and our own output; no ground truth of any kind.
Prints counts only (no frame-level data). Each count says how often a known failure mode can occur,
so we know which fixes are worth building.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from sonar import atd, deadball, possession  # noqa: E402
from sonar import make_submission as ms  # noqa: E402
from starter import load_smart_tagging  # noqa: E402

FPS = 25


def runs_of(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def main():
    data = atd.load()
    events = load_smart_tagging.load()
    dead, _ = deadball.detect(data, events)
    evidence, details = possession.evidence(data, dead)
    tags = possession.tag_evidence(events, len(dead))
    labels = possession.labels(dead, evidence, tags)
    labels, _ = possession.localize(labels, dead, details["ball"], data)
    labels = ms.fill_graded_abstentions(labels)
    graded = deadball.graded_mask(len(labels))
    ball = data["ball"][:, :2].astype(np.float64)
    known = ~np.isnan(ball[:, 0])
    speed = deadball.instant_ball_speed(ball)
    ball_ev = details["ball"]
    out = {}

    live_runs, dead_runs, spells, switches = [], [], [], []
    for _, first, last in half_video_blocks(GAME):
        seg = labels[first:last + 1]
        for a, b in runs_of(seg != "D"):
            live_runs.append((first + a, first + b))
        for a, b in runs_of(seg == "D"):
            dead_runs.append((first + a, first + b))
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], change, [len(seg)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if seg[a] in "HA":
                spells.append((first + a, first + b))
        for c in change:
            if seg[c - 1] in "HA" and seg[c] in "HA":
                switches.append(first + c)

    lengths = np.array([b - a for a, b in live_runs]) / FPS
    out["live stretches"] = len(live_runs)
    out["live stretches under 2 s (likely false: each is a 1-spell false positive)"] = int((lengths < 2).sum())
    out["live stretches under 5 s"] = int((lengths < 5).sum())
    dl = np.array([b - a for a, b in dead_runs]) / FPS
    out["stoppages"] = len(dead_runs)
    out["stoppages under 3 s"] = int((dl < 3).sum())
    moving = 0
    for a, b in dead_runs:
        s = speed[a:b]
        s = s[np.isfinite(s)]
        if len(s) > FPS and np.median(s) > 2.0:
            moving += 1
    out["stoppages where the seen ball keeps moving > 2 m/s (median)"] = moving
    out["dead share of graded time"] = round(float((labels[graded] == "D").mean()), 3)

    sl = np.array([b - a for a, b in spells]) / FPS
    out["possession spells"] = len(spells)
    out["spells under 2 s"] = int((sl < 2).sum())
    out["mean spell length s"] = round(float(sl.mean()), 1)
    out["H<->A switches"] = len(switches)
    no_ball = 0
    in_gap = 0
    for c in switches:
        w = slice(max(0, c - FPS), c + FPS)
        if not np.any(np.abs(ball_ev[w]) > possession.CONTROL_THRESHOLD):
            no_ball += 1
        if not known[c]:
            in_gap += 1
    out["switches with no ball-control evidence within 1 s"] = no_ball
    out["switches placed while the ball is unseen"] = in_gap

    live = graded & (labels != "D")
    out["ball unseen on predicted-live frames (share)"] = round(float((~known[live]).mean()), 3)
    out["ball unseen on predicted-dead frames (share)"] = round(float((~known[graded & (labels == "D")]).mean()), 3)

    players = data["players"][:, :, :2].astype(np.float64)
    team = data["team"]
    vis = np.isfinite(players[:, :, 0])
    for code, name in ((0, "Barcelona"), (1, "Athletic")):
        cnt = vis[:, team == code].sum(axis=1)
        out["{} tracked players (mean on graded)".format(name)] = round(float(cnt[graded].mean()), 2)
        out["{} graded frames with fewer than 9 tracked".format(name)] = int((cnt[graded] < 9).sum())

    home, away = np.flatnonzero(team == 0), np.flatnonzero(team == 1)
    close = np.zeros(len(players), dtype=bool)
    for i in home:
        d = np.sqrt(((players[:, away, :] - players[:, i:i + 1, :]) ** 2).sum(axis=2))
        close |= np.nanmin(np.where(np.isfinite(d), d, np.inf), axis=1) < 0.3
    long_close = sum(1 for a, b in runs_of(close & graded) if b - a > FPS)
    out["opposite-team players within 0.3 m for over 1 s (episodes)"] = long_close

    no_evidence = sum(1 for a, b in live_runs if not np.any(evidence[a:b] != 0))
    out["live stretches where Smart Tagging decides (no tracking evidence)"] = no_evidence
    out["graded X frames"] = int((labels[graded] == "X").sum())

    for k, v in out.items():
        print("{:<72} {}".format(k, v))


if __name__ == "__main__":
    main()

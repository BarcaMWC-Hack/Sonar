"""Label-free checks on the possession call, and a sweep of its switching cost. Counts only.

For each setting: how many possessions (maximal H or A runs, X carried forward as the scorer
does), their mean and median length, how many are under 2 s, Barça's share of live time, and how
often our call agrees with the side Smart Tagging names in possession phases (64-87% for the raw
controlled call, so this is a consistency check, not an answer key).

    python tools/possession_report.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from sonar import atd, deadball, possession, possession_baseline  # noqa: E402
from sonar.make_submission import build_labels  # noqa: E402
from starter import load_smart_tagging  # noqa: E402

# Params
FPS = 25
SWEEP = (1.5, 3.0, 6.0, 12.0)


def carried(labels):
    out = labels.copy()
    for _, first, last in half_video_blocks(GAME):
        for i in range(first + 1, last + 1):
            if out[i] == "X":
                out[i] = out[i - 1]
    return out


def possession_runs(labels):
    runs = []
    for _, first, last in half_video_blocks(GAME):
        seg = labels[first:last + 1]
        start = None
        for i in range(len(seg) + 1):
            here = seg[i] if i < len(seg) else "#"
            if start is not None and here != seg[start]:
                if seg[start] in "HA":
                    runs.append(i - start)
                start = i
            elif start is None:
                start = i
    return np.array(runs) / FPS


def tag_agreement(labels, events):
    agree = total = 0
    for e in events:
        side = possession.TEAM_TAGS.get((e["tags"].get("Team") or "").strip().lower())
        if side is None or deadball.code_of(e) not in possession.POSSESSION_CODES:
            continue
        seg = labels[max(0, e["start_frame"]):e["end_frame"]]
        live = np.isin(seg, ("H", "A"))
        agree += np.sum(seg[live] == ("H" if side > 0 else "A"))
        total += int(live.sum())
    return agree / max(total, 1)


def report(name, labels, events):
    labels = carried(labels)
    runs = possession_runs(labels)
    graded = deadball.graded_mask(len(labels))
    live = np.isin(labels[graded], ("H", "A"))
    print("  {:<24} {:>4} possessions | mean {:>5.1f} s, median {:>5.1f} s, {:>3} under 2 s | "
          "Barca {:>4.1%} of live | tag agreement {:.1%}".format(
              name, len(runs), runs.mean(), np.median(runs), int(np.sum(runs < 2)),
              np.mean(labels[graded][live] == "H"), tag_agreement(labels, events)))


if __name__ == "__main__":
    data = atd.load()
    events = load_smart_tagging.load()
    dead, _ = deadball.detect(data, events)
    print("Possession call on the full match (football reference: possessions of ~10-15 s)")
    report("nearest, +/-2 s vote", build_labels(dead, possession_baseline.nearest_team(data)),
           events)
    print("Ball evidence only (no tag prior), so tag agreement is an independent check:")
    for radius, falloff, slow, fast in ((1.0, 1.0, 4.0, 8.0), (1.5, 1.5, 5.0, 10.0)):
        ball_only = possession.ball_evidence(data, radius, falloff, slow, fast)
        for cost in SWEEP:
            report("r{} v{}-{} sw {}".format(radius, slow, fast, cost),
                   possession.labels(dead, ball_only, switch_cost=cost), events)
    print("Default (tags name the team only for live stretches with no ball evidence):")
    report("default", possession.labels(dead, possession.ball_evidence(data),
                                        possession.tag_evidence(events, len(dead))), events)

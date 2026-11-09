"""Label-free checks on the possession call, and a sweep of its settings. Counts only.

For each setting: how many possessions (maximal H or A runs, X carried forward as the scorer
does) and their length; how many switches inside live play are reversed within 3 s, or get
almost no control by the new team in the 2 s after (both are touches and deflections, which
CHALLENGE.md says are not changes); Barca's share of live time; and how often the call agrees with
the side Smart Tagging names in possession phases. The tags feed neither the ball nor the shape
evidence, so that agreement is an independent check -- but it rises with longer possessions,
because the phases are long, so compare settings at a similar possession count.

    python tools/possession_report.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from sonar import atd, deadball, possession, possession_baseline, team_shape  # noqa: E402
from sonar.make_submission import build_labels  # noqa: E402
from starter import load_smart_tagging  # noqa: E402
from tools.deadball_report import auc  # noqa: E402

# Params
FPS = 25
SWITCH_COSTS = (3.0, 6.0, 12.0, 25.0)
SHAPE_WEIGHTS = (0.0, 0.05, 0.1, 0.2)
WEAK_SUPPORT = 3.0      # frames of full control in the 2 s after a switch


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
        breaks = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(seg)]))
        runs += [b - a for a, b in zip(bounds[:-1], bounds[1:]) if seg[a] in "HA"]
    return np.array(runs) / FPS


def switch_quality(labels, ball_ev):
    """(reversed within 3 s, weakly supported) over switches between H and A inside live play."""
    reversed_ = weak = 0
    for _, first, last in half_video_blocks(GAME):
        seg = labels[first:last + 1]
        for i in np.flatnonzero((seg[1:] != seg[:-1]) & np.isin(seg[1:], ("H", "A")) &
                                np.isin(seg[:-1], ("H", "A"))) + 1:
            f, team = first + int(i), seg[i]
            other = "A" if team == "H" else "H"
            reversed_ += bool(np.any(labels[f + 1:f + 3 * FPS] == other))
            sign = 1.0 if team == "H" else -1.0
            weak += np.sum(np.clip(sign * ball_ev[f:f + 2 * FPS], 0, None)) < WEAK_SUPPORT
    return reversed_, weak


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


def report(name, labels, events, ball_ev):
    labels = carried(labels)
    runs = possession_runs(labels)
    graded = deadball.graded_mask(len(labels))
    live = np.isin(labels[graded], ("H", "A"))
    reversed_, weak = switch_quality(labels, ball_ev)
    print("  {:<22} {:>4} poss | mean {:>5.1f} s, median {:>5.1f} s | reversed<3s {:>3}, "
          "weak {:>3} | Barca {:>4.1%} | tags {:.1%}".format(
              name, len(runs), runs.mean(), np.median(runs), reversed_, weak,
              np.mean(labels[graded][live] == "H"), tag_agreement(labels, events)))


def shape_cross_half(x, ball_ev, alive):
    blocks = half_video_blocks(GAME)
    first = np.zeros(len(x), dtype=bool)
    first[blocks[0][1]:blocks[0][2] + 1] = True
    usable = team_shape.training_mask(ball_ev, alive, x)
    y = ball_ev > 0
    for train, test, label in ((first, ~first, "1->2"), (~first, first, "2->1")):
        w = deadball.fit_logistic(x[usable & train], y[usable & train].astype(float),
                                  l2=team_shape.L2)
        s = team_shape.log_odds(x[usable & test], w)
        print("  team-shape model, fit half {} and score the other: AUC {:.3f} on {} frames".format(
            label, auc(s, y[usable & test]), int((usable & test).sum())))


if __name__ == "__main__":
    data = atd.load()
    events = load_smart_tagging.load()
    dead, _ = deadball.detect(data, events)
    n = len(dead)
    tags = possession.tag_evidence(events, n)
    _, det = possession.evidence(data, dead)
    ball, shape = det["ball"], det["shape"]

    print("Team-shape evidence (labels: live frames with clear ball control)")
    print("  weights: {}".format(dict(zip(("bias",) + team_shape.NAMES,
                                          det["shape_weights"].tolist()))))
    shape_cross_half(det["shape_x"], ball, det["alive"])

    print("\nPossession call on the full match (football reference: ~10-15 s possessions)")
    report("nearest, +/-2 s vote", build_labels(dead, possession_baseline.nearest_team(data)),
           events, ball)
    for weight in SHAPE_WEIGHTS:
        for cost in SWITCH_COSTS:
            tag = "  <- default" if (cost, weight) == (possession.SWITCH_COST,
                                                       possession.SHAPE_WEIGHT) else ""
            report("sw {:>4} shape {:<4}".format(cost, weight),
                   possession.labels(dead, ball + weight * shape, tags, switch_cost=cost),
                   events, ball)
            if tag:
                print(tag)
    final = possession.labels(dead, ball + possession.SHAPE_WEIGHT * shape, tags)
    final, _ = possession.localize(final, dead, ball, data)
    report("default + localize", final, events, ball)

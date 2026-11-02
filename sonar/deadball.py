"""Dead-ball detection: which graded frames have the ball out of play.

Three steps, all inspectable:

  1. Features (sonar/deadball_features.py): player motion, ball gaps, ball out of bounds, ball
     speed, where a still ball sits.
  2. A logistic model turns them into a per-frame log-likelihood ratio, dead vs alive. It is
     trained on WEAK labels taken from Metrica's Smart Tagging, never on any ground truth:
        dead  = the 8 s before each SET PIECES interval starts (Smart Tagging only; the 2nd-half
                pre-kickoff 110808..113741 that FAQ.md states is all D is applied when deciding,
                not used to train)
        alive = the inside of open-play phases (Progression, Build Up, ...), trimmed by 1 s and
                kept 10 s clear of any set piece
     Classes are reweighted to equal mass, so the model's logit is a likelihood ratio and the
     match-level prior is added separately.
  3. A two-state Viterbi pass over each half with a cost per switch, so a stoppage has to be
     supported by a few seconds of evidence before it is declared and possessions are not cut
     in half by a single slow frame.

Nothing here reads evaluation/gt. The public ground truth is used afterwards, by the scorer, as a
correctness check only.
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from sonar import deadball_features as dbf  # noqa: E402

# Params
FPS = 25
SECOND_HALF_KICKOFF = 113742        # FAQ.md: 110808..113741 are graded and all D
PRE_SETPIECE_S = 8
OPEN_PLAY_CODES = {"progression", "build up", "creating chances", "attacking transition",
                   "defensive transition", "ball in final third", "long ball"}
OPEN_PLAY_TRIM_S = 1
SETPIECE_GUARD_S = 10
OOB_MARGIN = 0.5                    # metres beyond a line before "out" counts
OOB_MEMORY_S = 10                   # a ball seen out stays evidence for this long
L2 = 1.0
WEIGHT_DECIMALS = 4                 # rounded so the fit is identical across BLAS builds

# Viterbi. TEMPERATURE scales per-frame evidence down because the features are smoothed over
# about a second, so neighbouring frames are not independent observations. SWITCH_COST is paid
# on every alive<->dead change. Only their ratio changes the path; at these values one switch
# costs one second of unit evidence. Chosen without any ground truth, from the label-free sweep
# in tools/deadball_report.py --grid: it is the setting whose stoppage count (80) sits inside the
# 80-130 a football match has, and lower costs start producing sub-3 s stoppages.
# PRIOR_DEAD is the share of graded time a match spends with the ball out of play (roughly 40%:
# about 60 of ~106 graded minutes are effective play).
TEMPERATURE = 1.0 / FPS
SWITCH_COST = 1.0
PRIOR_DEAD = 0.40


# -- masks ----------------------------------------------------------------------------------------

def graded_mask(n, game=GAME):
    mask = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(game):
        mask[first:last + 1] = True
    return mask


def interval_mask(n, intervals):
    mask = np.zeros(n, dtype=bool)
    for start, stop in intervals:
        mask[max(0, start):max(0, min(n, stop))] = True
    return mask


def known_dead_mask(n, game=GAME):
    """Frames the documentation says are dead: the 2nd half before its kickoff."""
    mask = np.zeros(n, dtype=bool)
    mask[half_video_blocks(game)[1][1]:SECOND_HALF_KICKOFF] = True
    return mask


def code_of(event):
    return (event["code"] or "").strip().lower()


def weak_labels(n, events, game=GAME):
    """(dead, alive) boolean masks from Smart Tagging only.

    The documented 2nd-half pre-kickoff frames (FAQ.md) are applied when deciding (known_dead_mask),
    but are NOT training labels: the model learns from Smart Tagging alone.
    """
    graded = graded_mask(n, game)
    setpieces = [e for e in events if code_of(e) == "set pieces"]
    pre = interval_mask(n, [(e["start_frame"] - PRE_SETPIECE_S * FPS, e["start_frame"])
                            for e in setpieces])
    guard = interval_mask(n, [(e["start_frame"] - SETPIECE_GUARD_S * FPS,
                               e["end_frame"] + SETPIECE_GUARD_S * FPS) for e in setpieces])
    trim = OPEN_PLAY_TRIM_S * FPS
    play = interval_mask(n, [(e["start_frame"] + trim, e["end_frame"] - trim)
                             for e in events if code_of(e) in OPEN_PLAY_CODES])
    dead = pre & graded
    alive = play & ~guard & graded & ~dead
    return dead, alive


# -- model ----------------------------------------------------------------------------------------

def recent(mask, frames):
    """True where mask was True at any point in the previous `frames` frames (inclusive)."""
    count = np.concatenate(([0], np.cumsum(mask)))
    index = np.arange(len(mask))
    return (count[index + 1] - count[np.maximum(0, index + 1 - frames)]) > 0


def design(f):
    """(matrix, column names). Continuous columns are clipped, never standardised by the data,
    so a column means the same thing on any match."""
    with np.errstate(invalid="ignore"):
        known = f["ball_known"]
        bs = f["ball_speed"]
        oob = known & (f["ball_oob"] > OOB_MARGIN)
        columns = [
            ("mean_speed", np.clip(np.nan_to_num(f["mean_speed"]), 0, 6)),
            ("mean_speed_long", np.clip(np.nan_to_num(f["mean_speed_long"]), 0, 6)),
            ("moving_share", np.nan_to_num(f["moving_share"])),
            ("top_speed", np.clip(np.nan_to_num(f["top_speed"]), 0, 9)),
            ("sprint_share", np.nan_to_num(f["sprint_share"])),
            ("n_outside", np.clip(np.nan_to_num(f["n_outside"]), 0, 3)),
            ("kickoff_shape", (np.nan_to_num(f["own_half_share"]) > 0.97).astype(float)),
            ("ball_gap", (f["missing_run"] > FPS).astype(float)),
            ("known_share", np.nan_to_num(f["ball_known_share"])),
            ("oob_now", oob.astype(float)),
            ("oob_recent", recent(oob, OOB_MEMORY_S * FPS).astype(float)),
            ("ball_still", (bs < 0.5).astype(float)),
            ("ball_slow", ((bs >= 0.5) & (bs < 2.0)).astype(float)),
            ("ball_fast", (bs > 8.0).astype(float)),
            ("spot_centre", (f["spot"] == dbf.SPOT_CENTRE).astype(float)),
            ("spot_corner", (f["spot"] == dbf.SPOT_CORNER).astype(float)),
            ("spot_goal_area", (f["spot"] == dbf.SPOT_GOAL_AREA).astype(float)),
            ("spot_touchline", (f["spot"] == dbf.SPOT_TOUCHLINE).astype(float)),
            ("few_visible", (f["n_visible"] < 18).astype(float)),
        ]
    names = [name for name, _ in columns]
    return np.column_stack([values for _, values in columns]), names


def fit_logistic(x, y, l2=L2, iterations=100):
    """Class-balanced L2 logistic regression by Newton's method. Returns [bias, *weights]."""
    xb = np.column_stack([np.ones(len(x)), x])
    sample_weight = np.where(y == 1, 0.5 / max(y.sum(), 1), 0.5 / max((1 - y).sum(), 1)) * len(y)
    reg = l2 * np.eye(xb.shape[1])
    reg[0, 0] = 0.0
    w = np.zeros(xb.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(xb @ w)))
        gradient = xb.T @ (sample_weight * (p - y)) + reg @ w
        hessian = (xb * (sample_weight * p * (1 - p))[:, None]).T @ xb + reg
        step = np.linalg.solve(hessian, gradient)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return np.round(w, WEIGHT_DECIMALS)


def log_likelihood_ratio(x, w):
    return w[0] + x @ w[1:]


# -- decoding -------------------------------------------------------------------------------------

def viterbi_two_state(evidence, switch_cost):
    """Best alive(0)/dead(1) path maximising sum(evidence on dead frames) - switch_cost * switches.

    evidence is the per-frame score for 'dead' over 'alive', already scaled and with the prior
    folded in. Plain dynamic programming; ~0.3 s per half in pure Python.
    """
    n = len(evidence)
    back = np.zeros((n, 2), dtype=np.int8)
    score_alive, score_dead = 0.0, float(evidence[0])
    for t in range(1, n):
        stay_alive, from_dead = score_alive, score_dead - switch_cost
        stay_dead, from_alive = score_dead, score_alive - switch_cost
        back[t, 0] = 0 if stay_alive >= from_dead else 1
        back[t, 1] = 1 if stay_dead >= from_alive else 0
        score_alive = max(stay_alive, from_dead)
        score_dead = max(stay_dead, from_alive) + float(evidence[t])
    path = np.zeros(n, dtype=np.int8)
    path[-1] = 1 if score_dead > score_alive else 0
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path.astype(bool)


def decode(llr, n, game=GAME, temperature=TEMPERATURE, switch_cost=SWITCH_COST,
           prior_dead=PRIOR_DEAD):
    """Dead mask over every frame; only graded frames are decoded, the rest stay False."""
    prior = math.log(prior_dead / (1.0 - prior_dead))
    evidence = temperature * (llr + prior)
    dead = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(game):
        dead[first:last + 1] = viterbi_two_state(evidence[first:last + 1], switch_cost)
    return dead | known_dead_mask(n, game)


# -- boundary refinement --------------------------------------------------------------------------
#
# The Viterbi path is built from features smoothed over seconds -- up to a 10 s window -- so its
# edges can sit seconds away from the instants the ground truth's edges mark: the ball crossing
# a line, and the restart kick. EVALUATION.md puts the cost of being 1 s late everywhere at ~35%
# of spellF1, so each edge is re-placed from unsmoothed signals near it:
#
#   start: the first frame the ball is seen over a line. Otherwise the moment the players
#          decelerate -- the strongest drop in mean player speed, 2 s after vs 2 s before -- or,
#          if the ball vanishes into a long gap just before that, the frame it vanishes.
#   end:   found from the moment the players accelerate (the strongest rise, same measure). That
#          peak trails the restart, because players speed up over a second or two, so the
#          restart is the EARLIEST ball event from CHANGE_LEAD_S before the peak to
#          CHANGE_LAG_S after it: a held or still ball struck, or the ball reappearing from a
#          long gap; failing those, any ball becoming fast. Every restart -- throw-in, goal kick,
#          corner, free kick, kick-off -- is the first strike of a held, placed or unseen ball;
#          later strikes in the window are open play. (A first version took the latest event
#          before the peak; that picks the first pass after the restart, and was dropped.)

START_BEFORE_S, START_AFTER_S = 6, 2
END_BEFORE_S, END_AFTER_S = 6, 3
CHANGE_WINDOW = 2 * FPS     # frames either side of a candidate edge for the speed contrast
CHANGE_LEAD_S = 3           # restart ball events up to this long before the players' peak
CHANGE_LAG_S = 1            # ...or this long after it (attackers start runs before a corner)
GAP_LAG_S = 0.5             # a ball vanishing up to this long after the deceleration still counts
MIN_STOPPAGE = FPS          # a refined stoppage keeps at least one second
CROSS_MARGIN = 0.2          # m; the ball's centre this far over a line means it is wholly over
CROSS_MIN_FRAMES = 3        # ...for at least this many frames, so one noisy frame is not a crossing
LONG_GAP = 5 * FPS          # the tracker fills shorter gaps itself (see deadball_features)
KICK_SPEED = 5.0            # m/s, ball speed just after a restart strike or throw
HELD_SPEED = 2.5            # m/s, ball speed while placed or carried before the restart
HELD_FRAMES = 25            # look back this far for the held/still ball
HELD_MIN_KNOWN = 5
# Restart candidates, most specific group first; within a group the earliest frame wins.
END_EVENT_GROUPS = (("kick", "reacquire"), ("strike",))


def motion_contrast(raw_speed, half=CHANGE_WINDOW):
    """Mean player speed over the next `half` frames minus the previous `half`, NaN-aware.

    Peaks where the players accelerate (a restart) and dips where they decelerate (a stoppage).
    """
    valid = ~np.isnan(raw_speed)
    total = np.concatenate(([0.0], np.cumsum(np.where(valid, raw_speed, 0.0))))
    count = np.concatenate(([0], np.cumsum(valid)))
    n = len(raw_speed)
    t = np.arange(n)
    high, low = np.minimum(n, t + half), np.maximum(0, t - half)
    after = (total[high] - total[t]) / np.maximum(count[high] - count[t], 1)
    before = (total[t] - total[low]) / np.maximum(count[t] - count[low], 1)
    return after - before


def instant_ball_speed(ball_xy, half=2):
    n = len(ball_xy)
    out = np.full(n, np.nan)
    out[half:n - half] = np.hypot(*(ball_xy[2 * half:] - ball_xy[:n - 2 * half]).T) / (
        2.0 * half / FPS)
    return out


def crossing_starts(ball_oob):
    """Frames where a run of >= CROSS_MIN_FRAMES frames with the ball over a line begins."""
    with np.errstate(invalid="ignore"):
        over = ball_oob > CROSS_MARGIN
    edges = np.diff(np.concatenate(([0], over.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return starts[(stops - starts) >= CROSS_MIN_FRAMES]


def long_gaps(known):
    """(starts, stops) of missing-ball runs of at least LONG_GAP frames; stop = first seen frame."""
    edges = np.diff(np.concatenate(([0], (~known).astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    keep = (stops - starts) >= LONG_GAP
    return starts[keep], stops[keep]


def strike_onsets(ball_xy):
    """Frames where the ball becomes fast, from slow or from not being tracked."""
    speed = instant_ball_speed(ball_xy)
    with np.errstate(invalid="ignore"):
        fast = speed > KICK_SPEED
    return np.flatnonzero(fast & ~np.roll(fast, 3))


def kick_frames(ball_xy):
    """Frames where a held or still ball is struck: fast now, slow over the second before."""
    speed = instant_ball_speed(ball_xy)
    n = len(speed)
    with np.errstate(invalid="ignore"):
        fast = speed > KICK_SPEED
        rising = fast & ~(np.roll(speed, 3) > KICK_SPEED)
    out = []
    for t in np.flatnonzero(rising):
        lo, hi = max(0, t - HELD_FRAMES), max(0, t - 3)
        before = speed[lo:hi]
        before = before[~np.isnan(before)]
        if len(before) >= HELD_MIN_KNOWN and before.max() < HELD_SPEED and t < n:
            out.append(t)
    return np.array(out, dtype=np.int64)


def refine(dead, data, features, game=GAME):
    """(refined dead mask, counts of which rule moved each edge, final stoppages with rules)."""
    ball_xy = data["ball"][:, :2].astype(np.float64)
    crossings = crossing_starts(features["ball_oob"])
    gaps, gap_ends = long_gaps(features["ball_known"])
    end_events = {"kick": kick_frames(ball_xy), "strike": strike_onsets(ball_xy),
                  "reacquire": gap_ends}
    contrast = motion_contrast(features["mean_speed_raw"])
    fixed = known_dead_mask(len(dead), game)
    counts = {}
    shifts = {"start": [], "end": []}
    kept = []

    def window(lo, hi, first, last):
        return max(lo, first), min(hi, last)

    out = np.zeros_like(dead)
    for _, first, last in half_video_blocks(game):
        spells = []
        mask = dead[first:last + 1].astype(np.int8)
        edges = np.diff(np.concatenate(([0], mask, [0])))
        for a, b in zip(np.flatnonzero(edges == 1) + first, np.flatnonzero(edges == -1) + first):
            new_a, new_b = a, b
            start_rule = "documented"
            if not fixed[a]:
                lo, hi = window(a - START_BEFORE_S * FPS, a + START_AFTER_S * FPS, first, last)
                c = crossings[(crossings >= lo) & (crossings <= hi)]
                if len(c):
                    new_a, start_rule = int(c[np.argmin(np.abs(c - a))]), "crossing"
                else:
                    drop = lo + int(np.argmin(contrast[lo:hi + 1]))
                    g = gaps[(gaps >= drop - CHANGE_LEAD_S * FPS) &
                             (gaps <= drop + GAP_LAG_S * FPS)]
                    if len(g):
                        new_a, start_rule = int(g[np.argmin(np.abs(g - drop))]), "gap"
                    else:
                        new_a, start_rule = drop, "motion"
            counts["start_" + start_rule] = counts.get("start_" + start_rule, 0) + 1

            if fixed[a] and abs(b - SECOND_HALF_KICKOFF) <= END_BEFORE_S * FPS:
                new_b, end_rule = SECOND_HALF_KICKOFF, "documented"
            else:
                lo, hi = window(max(b - END_BEFORE_S * FPS, new_a + MIN_STOPPAGE),
                                b + END_AFTER_S * FPS, first, last)
                if hi <= lo:
                    new_b, end_rule = b, "kept"
                else:
                    rise = lo + int(np.argmax(contrast[lo:hi + 1]))
                    new_b, end_rule = rise, "motion"
                    for group in END_EVENT_GROUPS:
                        found = []
                        for name in group:
                            frames = end_events[name]
                            e = frames[(frames >= rise - CHANGE_LEAD_S * FPS) &
                                       (frames <= rise + CHANGE_LAG_S * FPS) &
                                       (frames >= new_a + MIN_STOPPAGE)]
                            if len(e):
                                found.append((int(e.min()), name))
                        if found:
                            new_b, end_rule = min(found)
                            break
            counts["end_" + end_rule] = counts.get("end_" + end_rule, 0) + 1
            shifts["start"].append(new_a - a)
            shifts["end"].append(new_b - b)
            spells.append((new_a, new_b, start_rule, end_rule))
        # Keep spells ordered and non-overlapping; one that collapses is dropped.
        previous_end = first
        for a, b, start_rule, end_rule in spells:
            a = max(a, previous_end, first)
            b = min(b, last + 1)
            if b > a:
                out[a:b] = True
                previous_end = b
                kept.append({"start": int(a), "stop": int(b), "start_rule": start_rule,
                             "end_rule": end_rule})
    out |= fixed
    counts["median_abs_shift_start_s"] = float(np.median(np.abs(shifts["start"]))) / FPS
    counts["median_abs_shift_end_s"] = float(np.median(np.abs(shifts["end"]))) / FPS
    return out, counts, kept


def detect(data, events, game=GAME, refine_edges=True, **decode_kwargs):
    """(dead mask, details). details carries the fitted weights and the per-frame LLR."""
    features = dbf.compute(data)
    x, names = design(features)
    n = len(x)
    weak_dead, weak_alive = weak_labels(n, events, game)
    train = weak_dead | weak_alive
    w = fit_logistic(x[train], weak_dead[train].astype(float))
    llr = log_likelihood_ratio(x, w)
    viterbi_dead = decode(llr, n, game, **decode_kwargs)
    dead, refine_counts, stoppages = (refine(viterbi_dead, data, features, game) if refine_edges
                                      else (viterbi_dead, {}, []))
    return dead, {"weights": dict(zip(["bias"] + names, w.tolist())), "llr": llr,
                  "features": features, "x": x, "names": names,
                  "weak_dead": weak_dead, "weak_alive": weak_alive,
                  "viterbi_dead": viterbi_dead, "refine": refine_counts,
                  "stoppages": stoppages}

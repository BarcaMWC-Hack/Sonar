"""Who has the ball while it is alive: control-weighted evidence decoded with a switching cost.

The rules in CHALLENGE.md say possession changes only when the other team gains *controlled*
possession, and stays with the last team through loose or contested phases. The nearest-player
call ignores that: on the delivered file it changes team 25 times per live minute, against 2.6
when counted only on frames where a player is close to a slow ball (tools/possession_diagnostics.py).
So:

  1. Evidence per frame, as log-odds Barça over Athletic:
        control  1 when the nearest player is within 1 m of the ball, fading with distance
        speed    1 for a ball slower than SLOW_BALL, fading to 0 at FAST_BALL (in flight: nobody
                 controls it, so it says nothing about a change)
        side     +1 / -1 by which team's nearest player is closer, saturating within a metre or two
     evidence = BALL_WEIGHT * control * speed * side
  2. A small prior from Smart Tagging phases whose Team tag names the side in possession
     (Build Up, Progression, ...: 64-87% agreement with the controlled call on this file), used
     only on frames where the ball gives no evidence, so it fills tracking gaps without
     overriding the tracking. The "Defending in ..." codes are mirror copies naming the other
     side and are not used.
  3. A two-state Viterbi per live stretch with SWITCH_COST: a change of possession has to be
     paid for with that much accumulated control evidence by the new team.

A live stretch with no evidence at all is left X, which the scorer carries forward.
"""

import numpy as np

from sonar import deadball, possession_baseline as pb

# Params
FPS = 25
CONTROL_RADIUS = 1.0        # m; within this the nearest player has the ball at their feet
CONTROL_FALLOFF = 1.0       # m; evidence falls by e every metre beyond it
SLOW_BALL, FAST_BALL = 4.0, 8.0     # m/s
UNKNOWN_SPEED_GATE = 0.5    # tracked ball whose speed cannot be computed (edge of a gap)
SIDE_SCALE = 1.5            # m; difference in nearest distance that counts as clearly one side
BALL_WEIGHT = 1.0
TAG_WEIGHT = 0.05           # per frame, and only where the ball gives no evidence
# Chosen without ground truth, from tools/possession_report.py: at 3.0 the match has 266
# possessions of mean 14.9 s, inside the ~10-15 s typical of football; 25 gave 132 of 30 s and
# 1.5 gave 312 with 47 under 2 s. At a matched count, a 1 m control radius agrees with the Smart
# Tagging side more often than 1.5 m (72.0% vs 70.6%), so the tighter definition is kept.
SWITCH_COST = 3.0
POSSESSION_CODES = {"build up", "progression", "creating chances", "attacking transition",
                    "ball in final third", "ball in the box", "long ball"}
TEAM_TAGS = {"barcelona": +1.0, "atletic club": -1.0}


def nearest_by_team(data):
    """(distance to nearest home player, distance to nearest away player) per frame, metres."""
    ball = data["ball"][:, :2].astype(np.float64)
    players = data["players"][:, :, :2].astype(np.float64)
    d = np.sqrt(((players - ball[:, None, :]) ** 2).sum(axis=2))
    d = np.where(np.isnan(d), np.inf, d)
    home = data["team"] == pb.HOME
    return d[:, home].min(axis=1), d[:, ~home].min(axis=1)


def ball_evidence(data, radius=None, falloff=None, slow=None, fast=None):
    """Per-frame log-odds H over A from who controls the ball; 0 where there is none."""
    radius = CONTROL_RADIUS if radius is None else radius
    falloff = CONTROL_FALLOFF if falloff is None else falloff
    slow = SLOW_BALL if slow is None else slow
    fast = FAST_BALL if fast is None else fast
    d_home, d_away = nearest_by_team(data)
    nearest = np.minimum(d_home, d_away)
    speed = deadball.instant_ball_speed(data["ball"][:, :2].astype(np.float64))
    with np.errstate(invalid="ignore"):
        control = np.exp(-np.maximum(0.0, nearest - radius) / falloff)
        gate = np.clip((fast - speed) / (fast - slow), 0.0, 1.0)
        gate = np.where(np.isnan(speed), UNKNOWN_SPEED_GATE, gate)
        side = np.tanh((d_away - d_home) / SIDE_SCALE)
    evidence = BALL_WEIGHT * control * gate * side
    return np.where(np.isfinite(nearest), evidence, 0.0)


def tag_evidence(events, n):
    """Per-frame +TAG_WEIGHT inside Barça possession phases, -TAG_WEIGHT inside Athletic's."""
    out = np.zeros(n)
    for e in events:
        sign = TEAM_TAGS.get((e["tags"].get("Team") or "").strip().lower())
        if sign is None or deadball.code_of(e) not in POSSESSION_CODES:
            continue
        out[max(0, e["start_frame"]):max(0, min(n, e["end_frame"]))] += TAG_WEIGHT * sign
    # Overlapping phases of the same side should not stack into strong evidence.
    return np.clip(out, -TAG_WEIGHT, TAG_WEIGHT)


def evidence(data, events):
    """Ball evidence, with the Smart Tagging prior filling only the frames it leaves at 0."""
    ball = ball_evidence(data)
    return np.where(ball != 0, ball, tag_evidence(events, len(ball)))


def labels(dead, ev, switch_cost=SWITCH_COST):
    """Per-frame 'H' / 'A' / 'D' / 'X' over the graded halves; X elsewhere."""
    out = np.full(len(dead), "X", dtype="<U1")
    for _, first, last in deadball.half_video_blocks(deadball.GAME):
        stop = last + 1
        half_dead = dead[first:stop]
        out[first:stop][half_dead] = "D"
        edges = np.diff(np.concatenate(([0], (~half_dead).astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1) + first, np.flatnonzero(edges == -1) + first):
            stretch = ev[a:b]
            if not np.any(stretch != 0):
                continue    # no evidence at all: abstain, the scorer carries the last label
            home = deadball.viterbi_two_state(stretch, switch_cost)
            out[a:b] = np.where(home, "H", "A")
    return out

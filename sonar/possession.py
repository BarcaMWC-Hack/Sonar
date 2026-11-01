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
  2. Team shape (sonar/team_shape.py): the side with the ball spreads out, so a small
     SHAPE_WEIGHT times the shape log-odds is added on every frame. It is slow, so it mostly
     decides whether a run of touches really was a change, while the ball evidence still sets
     where the change lands.
  3. A two-state Viterbi per live stretch with SWITCH_COST: a change of possession has to be
     paid for with that much accumulated evidence by the new team.
  4. Smart Tagging phases whose Team tag names the side in possession (Build Up, Progression,
     ...: 64-87% agreement with the controlled call on this file) name the team for a live
     stretch the ball says nothing about. They never place a change: their edges are seconds
     off. The "Defending in ..." codes are mirror copies naming the other side and are not used.

A live stretch with no evidence of either kind is left X, which the scorer carries forward.
"""

import numpy as np

from sonar import deadball, possession_baseline as pb, team_shape

# Params
FPS = 25
CONTROL_RADIUS = 1.0        # m; within this the nearest player has the ball at their feet
CONTROL_FALLOFF = 1.0       # m; evidence falls by e every metre beyond it
SLOW_BALL, FAST_BALL = 4.0, 8.0     # m/s
UNKNOWN_SPEED_GATE = 0.5    # tracked ball whose speed cannot be computed (edge of a gap)
SIDE_SCALE = 1.5            # m; difference in nearest distance that counts as clearly one side
BALL_WEIGHT = 1.0
TAG_WEIGHT = 1.0            # only its sign over a whole evidence-free stretch is used
# Chosen without ground truth, from tools/possession_report.py: at 3.0 the match has 265
# possessions of mean 14.6 s, inside the ~10-15 s typical of football; 12 gave 173 of 22 s and
# 1.5 gave 312 with 51 under 2 s. At a matched count, a 1 m control radius agrees with the Smart
# Tagging side more often than 1.5 m (72.2% vs 71.0%), so the tighter definition is kept.
SWITCH_COST = 3.0
# Per frame, times the team-shape log-odds (sonar/team_shape.py). From the sweep in
# tools/possession_report.py: at switch cost 3, weight 0.05 raises agreement with the Smart
# Tagging side from 72.3% to 77.4% at a similar possession count (277 vs 267), and cuts switches
# the new team barely controls from 59 to 43. Larger weights are no clearer (77.1-78.4%), and the
# signal lags turnovers by seconds, so the smallest weight that captures the gain is kept.
SHAPE_WEIGHT = 0.05
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


def evidence(data, dead, shape_weight=None):
    """(evidence for the decoder, details): ball control plus the team-shape term."""
    shape_weight = SHAPE_WEIGHT if shape_weight is None else shape_weight
    ball = ball_evidence(data)
    alive = deadball.graded_mask(len(dead)) & ~dead
    x = team_shape.features(data)
    w = team_shape.fit(x, ball, alive)
    shape = team_shape.log_odds(x, w)
    return ball + shape_weight * shape, {"ball": ball, "shape": shape, "shape_x": x,
                                         "shape_weights": w, "alive": alive}


def labels(dead, ball_ev, tag_ev=None, switch_cost=SWITCH_COST):
    """Per-frame 'H' / 'A' / 'D' / 'X' over the graded halves; X elsewhere.

    Inside each live stretch the tracking evidence (ball_ev: ball control, plus team shape when
    it comes from evidence()) is decoded, so only the tracking ever places a change of
    possession. Smart Tagging (tag_ev) speaks only for a stretch where that evidence is zero
    throughout, and then names one team for the whole stretch: its phase edges are seconds off,
    which is fine for "who" and wrong for "when".
    """
    out = np.full(len(dead), "X", dtype="<U1")
    for _, first, last in deadball.half_video_blocks(deadball.GAME):
        stop = last + 1
        half_dead = dead[first:stop]
        out[first:stop][half_dead] = "D"
        edges = np.diff(np.concatenate(([0], (~half_dead).astype(np.int8), [0])))
        for a, b in zip(np.flatnonzero(edges == 1) + first, np.flatnonzero(edges == -1) + first):
            stretch = ball_ev[a:b]
            if np.any(stretch != 0):
                home = deadball.viterbi_two_state(stretch, switch_cost)
                out[a:b] = np.where(home, "H", "A")
            elif tag_ev is not None and tag_ev[a:b].sum() != 0:
                out[a:b] = "H" if tag_ev[a:b].sum() > 0 else "A"
            # else: no evidence at all; X, which the scorer carries from the last label
    return out

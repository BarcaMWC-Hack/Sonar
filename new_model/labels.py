"""Weak per-frame labels from Smart Tagging, for training and feature selection only.

Smart Tagging is not ground truth.  It is a phase annotation, and its team tag names the side
performing the phase, not the tracker's per-frame owner.  These labels are therefore a proxy:
they are good enough to rank features and train a first classifier, and they are not a
submission.  The public five minutes is never used here.

What becomes a label
--------------------
Alive (0) + team (+1/-1)
    frames inside a possession phase whose Team tag names Barcelona or Atletic Club:
        BUILD UP, PROGRESSION, CREATING CHANCES, ATTACKING TRANSITION,
        BALL IN FINAL THIRD, BALL IN THE BOX, LONG BALL
    The phase is trimmed by TRIM_S at both ends, because its edges are seconds off and a phase
    can span a turnover.

    PLAYERS IN THE BOX is deliberately *not* here.  It is a state, not a possession: the tag
    names the attacking side, which may not have the ball (a counter-attack leaves the box
    populated by the side that has just lost it).  It is also the largest code, so including it
    would let one noisy state dominate the alive labels.

Dead (1)
    frames in the PRE_S seconds before a SET PIECES tag starts, plus the published second-half
    pre-kickoff block.  The SET PIECES interval itself is *not* labelled dead: the restart is
    inside it, and on the public window goal kicks are ~34% alive while free kicks are ~76% dead.

Dropped
    all DEFENDING IN ... and DEFENSIVE TRANSITION codes: exact mirror copies of the possession
    codes with the defending team tagged.  Using them would label the same frames to both sides.
    GOALS: four intervals with a mean duration of 165 s, not a possession.
    Team = N/A: every throw-in, and 32 of 59 set pieces.

Unlabelled (-1 / 0 / weight 0)
    everything else.  Do not treat it as alive; the classifier should predict there, not train
    there.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES  # noqa: E402
from starter.load_smart_tagging import load as load_events  # noqa: E402

FPS = 25

POSSESSION_CODES = {
    "build up", "progression", "creating chances", "attacking transition",
    "ball in final third", "ball in the box", "long ball",
}
DROP_CODES = {
    "defending in attacking third", "defending in middle third",
    "defending in defensive third", "defensive transition", "goals",
}
TEAM_TAGS = {"barcelona": +1, "atletic club": -1}

PRE_S = 8.0             # seconds before a set piece that are dead-ball evidence
TRIM_S = 1.5            # seconds trimmed from each end of a possession phase
GUARD_S = 10.0          # seconds around a set piece where alive labels are dropped

# Published in FAQ.md: the second half starts at 113742, and the preceding block is all dead.
SECOND_HALF_DEAD = (110808, 113742)


def build(events=None, n_frames=None, game=GAME):
    """(dead, team, weight) arrays over video frames.

    dead    1 = dead, 0 = alive, -1 = unlabelled
    team    +1 = home (Barcelona), -1 = away (Atletic Club), 0 = unlabelled/dead
    weight  1 where the frame is a training example, 0 elsewhere
    """
    if events is None:
        events = load_events(game)
    if n_frames is None:
        n_frames = GAMES[game]["n_video_frames"]

    dead = np.full(n_frames, -1, dtype=np.int8)
    team = np.zeros(n_frames, dtype=np.int8)
    weight = np.zeros(n_frames, dtype=np.float32)

    trim = int(round(TRIM_S * FPS))
    pre = int(round(PRE_S * FPS))
    guard = int(round(GUARD_S * FPS))

    # 1. Alive frames and their team.
    for event in events:
        code = (event["code"] or "").strip().lower()
        if code not in POSSESSION_CODES:
            continue
        sign = TEAM_TAGS.get((event["tags"].get("Team") or "").strip().lower())
        if sign is None:
            continue
        a = max(0, event["start_frame"] + trim)
        b = min(n_frames, event["end_frame"] - trim)
        if b <= a:
            continue
        dead[a:b] = 0
        team[a:b] = sign
        weight[a:b] = 1.0

    # 2. Dead frames: the stoppage before each set piece.
    set_starts, set_ends = [], []
    for event in events:
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        start = event["start_frame"]
        set_starts.append(start)
        set_ends.append(event["end_frame"])
        a = max(0, start - pre)
        b = min(n_frames, start)
        if b <= a:
            continue
        dead[a:b] = 1
        team[a:b] = 0
        weight[a:b] = 1.0

    # 3. Published second-half pre-kickoff block.
    a, b = SECOND_HALF_DEAD
    dead[a:b] = 1
    team[a:b] = 0
    weight[a:b] = 1.0

    # 4. Guard alive labels around set pieces: their phase edges are seconds off.
    guard_mask = np.zeros(n_frames, dtype=bool)
    for start in set_starts:
        guard_mask[max(0, start - guard):min(n_frames, start + guard)] = True
    for end in set_ends:
        guard_mask[max(0, end - guard):min(n_frames, end + guard)] = True
    drop = (dead == 0) & guard_mask
    dead[drop] = -1
    team[drop] = 0
    weight[drop] = 0.0

    return dead, team, weight


def summary(dead, team, weight):
    return {
        "labelled": int(np.sum(weight > 0)),
        "dead": int(np.sum(dead == 1)),
        "alive": int(np.sum(dead == 0)),
        "home": int(np.sum(team == 1)),
        "away": int(np.sum(team == -1)),
        "unknown": int(np.sum(dead == -1)),
        "dead_share_of_labelled": float(np.sum(dead == 1) / max(np.sum(weight > 0), 1)),
    }


if __name__ == "__main__":
    events = load_events()
    dead, team, weight = build(events)
    print("Smart Tagging instances: {}".format(len(events)))
    for key, value in summary(dead, team, weight).items():
        print("  {:<24} {}".format(key, value))

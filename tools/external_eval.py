"""Other-game yardstick: score possession variants on games that are not the provided match.

    python tools/external_eval.py                 # clean tracking
    python tools/external_eval.py --degrade 0.41  # mask the ball on 41% of frames, seeded

Why: the rules bar tuning on the provided game and its public minutes, and bar hand labels on it.
Games 1 and 2 of Metrica's open sample data (data/external/metrica) are other games, so tuning
and comparing variants on them is allowed. Their tracking is clean, so --degrade masks the ball
to mimic our tracker's gaps (simulated; the numbers are sim_ numbers, not match results).

The truth here is a PROXY derived from the event data, not true possession:
  - a PASS, SHOT, SET PIECE or RECOVERY puts the ball with that event's team from its start frame
  - a BALL OUT or FAULT RECEIVED starts a dead period, which the next restart event ends
Dead frames are taken from that proxy for every variant ("oracle dead"), so the numbers compare
possession calls only, not the dead-ball detector.

Scored with evaluation/possession_metrics.py, the same spellF1 as the competition.
"""

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from sonar import deadball, possession, possession_baseline as pb  # noqa: E402
from sonar.make_submission import team_labels_in  # noqa: E402

DATA = os.path.join(ROOT, "data", "external", "metrica", "data")
LENGTH, WIDTH = 105.0, 68.0
GAMES = (1, 2)
RESTARTS = {"PASS", "SHOT", "SET PIECE", "RECOVERY"}
STOPS = {"BALL OUT", "FAULT RECEIVED"}


def _read_tracking(path):
    with open(path, newline="") as f:
        lines = f.read().splitlines()
    names = lines[2].split(",")
    arr = np.genfromtxt(lines[3:], delimiter=",")
    return names, arr


def load_game(game):
    """data dict as sonar expects (metres, players (n, P, 2), team (P,) 0 home / 1 away)."""
    base = os.path.join(DATA, "Sample_Game_{}".format(game),
                        "Sample_Game_{}_RawTrackingData_{{}}_Team.csv".format(game))
    xs, ys, team = [], [], []
    ball = None
    for side, code in (("Home", pb.HOME), ("Away", pb.AWAY)):
        names, arr = _read_tracking(base.format(side))
        for i, name in enumerate(names):
            if name.startswith("Player"):
                xs.append(arr[:, i] * LENGTH)
                ys.append(arr[:, i + 1] * WIDTH)
                team.append(code)
            elif name == "Ball" and side == "Home":
                ball = np.stack([arr[:, i] * LENGTH, arr[:, i + 1] * WIDTH], axis=1)
    players = np.stack([np.stack(xs, axis=1), np.stack(ys, axis=1)], axis=2)
    return {"ball": ball, "players": players, "team": np.array(team, dtype=np.int8),
            "pitch": (LENGTH, WIDTH)}


def proxy_truth(game, n):
    """Per-frame 'H' / 'A' / 'D' / 'X' from the event data (see module docstring)."""
    path = os.path.join(DATA, "Sample_Game_{}".format(game),
                        "Sample_Game_{}_RawEventsData.csv".format(game))
    with open(path, newline="") as f:
        events = list(csv.DictReader(f))
    events.sort(key=lambda e: (int(e["Start Frame"]), ))
    labels = np.full(n, "X", dtype="<U1")
    state, dead, start, period = None, False, None, None
    for e in events:
        frame = max(0, min(n, int(e["Start Frame"]) - 1))
        if e["Period"] != period:
            # A new period: nothing carries over, and the gap since the last event stays X.
            state, dead, start, period = None, False, None, e["Period"]
        if start is not None and state is not None:
            labels[start:frame] = "D" if dead else state
        if e["Type"] in RESTARTS:
            state, dead = ("H" if e["Team"] == "Home" else "A"), False
        elif e["Type"] in STOPS:
            dead = True
        start = frame
    if start is not None and state is not None:
        labels[start:min(n, start + 1)] = "D" if dead else state
    return labels


def mask_ball(ball, fraction, mean_gap_s=2.0, seed=0):
    """Copy of ball with seeded gaps (simulated) so that about `fraction` of frames are NaN."""
    rng = np.random.default_rng(seed)
    out = ball.copy()
    n = len(out)
    gap = mean_gap_s * 25.0
    seen = gap * (1.0 - fraction) / fraction
    t, hidden = 0, False
    while t < n:
        length = max(1, int(rng.exponential(gap if hidden else seen)))
        if hidden:
            out[t:t + length] = np.nan
        t += length
        hidden = not hidden
    return out


def stretches(alive):
    edges = np.diff(np.concatenate(([0], alive.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def nearest_variant(data, dead, window=50):
    codes = pb.nearest_team(data)
    out = np.where(dead, "D", "X").astype("<U1")
    for a, b in stretches(~dead):
        out[a:b] = team_labels_in(codes[a:b], window)
    return out


def control_variant(data, dead, radius, cost):
    ev = possession.ball_evidence(data, radius=radius)
    out = np.where(dead, "D", "X").astype("<U1")
    for a, b in stretches(~dead):
        stretch = ev[a:b]
        if np.any(stretch != 0):
            out[a:b] = np.where(deadball.viterbi_two_state(stretch, cost), "H", "A")
    return out


def score(gt_labels, pred_labels):
    gt = pm.label_codes(gt_labels)
    pred = pm.label_codes(pred_labels)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    result = pm.spell_scores(gt, filled, runs)
    return result["mean_f1"], result["n_gt"], len(pm.possession_spells(filled, runs))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--degrade", type=float, default=0.0,
                        help="fraction of frames with the ball masked (simulated), e.g. 0.41")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    variants = [("nearest, +/-2 s vote", lambda d, dead: nearest_variant(d, dead))]
    for radius in (1.0, 1.5):
        for cost in (0.5, 1.0, 1.5, 2.0, 3.0, 6.0):
            variants.append(("control r{} cost {}".format(radius, cost),
                             lambda d, dead, r=radius, c=cost: control_variant(d, dead, r, c)))

    rows = {name: [] for name, _ in variants}
    for game in GAMES:
        data = load_game(game)
        if args.degrade:
            data["ball"] = mask_ball(data["ball"], args.degrade, seed=args.seed + game)
        truth = proxy_truth(game, len(data["ball"]))
        dead = truth == "D"
        print("game {}: {} frames, ball missing on {:.0f}%, proxy truth {} possessions".format(
            game, len(truth), 100.0 * np.isnan(data["ball"][:, 0]).mean(),
            score(truth, truth)[1]))
        for name, fn in variants:
            rows[name].append(score(truth, fn(data, dead)))

    print("\nsim_ spellF1 on proxy truth, oracle dead (per game, then mean) | predicted/true spells")
    for name, scores in rows.items():
        f1s = [s[0] for s in scores]
        spells = " ".join("{}/{}".format(s[2], s[1]) for s in scores)
        print("  {:26s} {:6.3f} {:6.3f}  mean {:6.3f} | {}".format(name, f1s[0], f1s[1],
                                                                  float(np.mean(f1s)), spells))


if __name__ == "__main__":
    main()

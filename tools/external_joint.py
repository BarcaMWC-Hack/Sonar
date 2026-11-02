"""Pre-registered joint retune on three other games, two proxy truths, leave-one-game-out.

    python tools/external_joint.py            # writes a table; run once, do not iterate on it

Protocol (fixed before any number was seen):
  games      Metrica sample games 1, 2, 3
  proxy A    possession follows the latest PASS/SHOT/SET PIECE/RECOVERY (+CARRY in game 3)
  proxy B    like A, but a change of team needs two consecutive such events by the new team
  grid       switch cost {2,3,4} x shape weight {0.03,0.05,0.08} x control falloff {1,2}
             x speed thresholds {(4,8),(5,10)} m/s; radius 1.0 and side scale 1.5 fixed
  default    cost 3, shape 0.05, falloff 1, speed (4,8)
  selection  held-out game g: pick the setting with the best mean over the other games of
             mean(proxy A, proxy B), first-touch placement OFF (it is circular with event starts)
  adopt iff  mean held-out gain >= +0.02, no game below 0, each proxy's mean >= +0.01,
             and first-touch ON does not reverse the sign; masked seeds are checked separately
All numbers are sim_: proxy truth, oracle dead periods.
"""

import csv
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import external_eval as ee  # noqa: E402
import external_full as ef  # noqa: E402
import external_game3 as g3  # noqa: E402
from sonar import deadball, possession, team_shape  # noqa: E402

GAMES = (1, 2, 3)
COST = (2.0, 3.0, 4.0)
SHAPE = (0.03, 0.05, 0.08)
FALLOFF = (1.0, 2.0)
SPEED = ((4.0, 8.0), (5.0, 10.0))
DEFAULT = (3.0, 0.05, 1.0, (4.0, 8.0))
RESTART12 = {"PASS", "SHOT", "SET PIECE", "RECOVERY"}
STOP12 = {"BALL OUT", "FAULT RECEIVED"}


def load(game):
    return g3.load()[0] if game == 3 else ee.load_game(game)


def events(game):
    """[(frame0, 'H'/'A', kind 'r'/'s', period)] sorted by frame."""
    out = []
    if game == 3:
        for e in g3._events():
            name = e["type"]["name"]
            kind = "r" if name in g3.RESTARTS else ("s" if name in g3.STOPS else None)
            if kind:
                out.append((e["start"]["frame"] - 1, "H" if e["team"]["id"] == "FIFATMA" else "A",
                            kind, str(e["period"])))
    else:
        path = os.path.join(ee.DATA, "Sample_Game_{}".format(game),
                            "Sample_Game_{}_RawEventsData.csv".format(game))
        for e in csv.DictReader(open(path, newline="")):
            kind = "r" if e["Type"] in RESTART12 else ("s" if e["Type"] in STOP12 else None)
            if kind:
                out.append((int(e["Start Frame"]) - 1, "H" if e["Team"] == "Home" else "A",
                            kind, e["Period"]))
    out.sort(key=lambda t: t[0])
    return out


def truth(game, n, strict):
    labels = np.full(n, "X", dtype="<U1")
    evs = events(game)
    state, dead, start, period = None, False, None, None
    for i, (frame, team, kind, per) in enumerate(evs):
        frame = max(0, min(n, frame))
        if per != period:
            state, dead, start, period = None, False, None, per
        if start is not None and state is not None:
            labels[start:frame] = "D" if dead else state
        if kind == "r":
            change = state is not None and team != state
            if strict and change:
                nxt = next((e for e in evs[i + 1:] if e[2] == "r" and e[3] == per), None)
                if nxt is None or nxt[1] != team:
                    start = frame      # a single touch by the other team: not a change
                    dead = False
                    continue
            state, dead = team, False
        else:
            dead = True
        start = frame
    return labels


def blocks(game, n, tr):
    if game == 3:
        period = g3.periods(n)
    else:
        path = os.path.join(ee.DATA, "Sample_Game_{}".format(game),
                            "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(game))
        rows = open(path).read().splitlines()[3:]
        period = np.array([int(r.split(",", 1)[0]) for r in rows[:n]], dtype=np.int8)
    out = []
    for p in (1, 2):
        idx = np.flatnonzero((period == p) & (tr != "X"))
        out.append((p, int(idx[0]), int(idx[-1])))
    ef._BLOCKS["current"] = out


def pieces(data, dead, falloff, speed):
    ball = possession.ball_evidence(data, falloff=falloff, slow=speed[0], fast=speed[1])
    x = team_shape.features(data)
    alive = deadball.graded_mask(len(dead)) & ~dead
    return ball, team_shape.log_odds(x, team_shape.fit(x, ball, alive))


def score_setting(data, trs, dead, setting, cache):
    cost, sw, falloff, speed = setting
    key = (falloff, speed)
    if key not in cache:
        cache[key] = pieces(data, dead, falloff, speed)
    ball, shape = cache[key]
    labels = possession.labels(dead, ball + sw * shape, None, switch_cost=cost)
    placed, _ = possession.localize(labels, dead, ball, data)
    return {(p, loc): ee.score(trs[p], lab)[0]
            for p, trs_ in (("A", 0), ("B", 0)) for loc, lab in ((False, labels), (True, placed))}


def main():
    grid = list(itertools.product(COST, SHAPE, FALLOFF, SPEED))
    table = {}
    for game in GAMES:
        data = load(game)
        n = len(data["ball"])
        trs = {"A": truth(game, n, False), "B": truth(game, n, True)}
        dead = trs["A"] == "D"
        blocks(game, n, trs["A"])
        cache = {}
        for s in grid:
            table[(game, s)] = score_setting(data, trs, dead, s, cache)
        print("done game", game, flush=True)

    def sel(s, games):
        return np.mean([np.mean([table[(g, s)][("A", False)], table[(g, s)][("B", False)]])
                        for g in games])

    print("\nHeld-out gains of the picked setting over the default (spellF1, sim_)")
    print("fold  picked(cost,shape,falloff,speed)            A/off   B/off   A/on    B/on")
    gains = []
    for held in GAMES:
        others = [g for g in GAMES if g != held]
        pick = max(grid, key=lambda s: (sel(s, others), s == DEFAULT))
        d = [table[(held, pick)][k] - table[(held, DEFAULT)][k]
             for k in (("A", False), ("B", False), ("A", True), ("B", True))]
        gains.append(d)
        print("{:4d}  {:42s} {:+.4f} {:+.4f} {:+.4f} {:+.4f}".format(held, str(pick), *d))
    g = np.array(gains)
    mean_off = float(g[:, :2].mean())
    print("\nmean gain (off, A+B) {:+.4f} | min fold {:+.4f} | A mean {:+.4f} B mean {:+.4f} | "
          "on-mean {:+.4f}".format(mean_off, float(g[:, :2].mean(axis=1).min()),
                                   float(g[:, 0].mean()), float(g[:, 1].mean()),
                                   float(g[:, 2:].mean())))
    ok = (mean_off >= 0.02 and g[:, :2].mean(axis=1).min() >= 0 and g[:, 0].mean() >= 0.01
          and g[:, 1].mean() >= 0.01 and g[:, 2:].mean() >= 0)
    print("ADOPT under the pre-registered rule:", "YES" if ok else "NO")
    best_all = max(grid, key=lambda s: (sel(s, GAMES), s == DEFAULT))
    print("best on all three games (not held out, descriptive only):", best_all,
          "mean {:+.4f} vs default".format(sel(best_all, GAMES) - sel(DEFAULT, GAMES)))
    np.save(os.path.join(os.environ.get("TEMP", "."), "joint_table.npy"),
            np.array([(g_, s, v) for (g_, s), v in table.items()], dtype=object),
            allow_pickle=True)


if __name__ == "__main__":
    main()

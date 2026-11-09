"""Full possession pipeline (ball + team shape + first-touch placement) on the other-game yardstick.

    python tools/external_full.py [--degrade 0.41]

Same proxy truth and oracle dead as tools/external_eval.py, so every number is sim_ and compares
possession calls only. The BAR-ATH half blocks that sonar hard-codes are replaced by the two
periods of each sample game; nothing else in the pipeline is changed.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import external_eval as ee  # noqa: E402
from sonar import deadball, possession, team_shape  # noqa: E402

_BLOCKS = {}


def _blocks(game, adjust=0):
    return _BLOCKS["current"]


deadball.half_video_blocks = _blocks
team_shape.half_video_blocks = _blocks


def set_blocks(game, n, truth):
    """Two (half, first, last) blocks from the tracking's periods, trimmed to labelled frames."""
    import csv
    path = os.path.join(ee.DATA, "Sample_Game_{}".format(game),
                        "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(game))
    period = np.empty(n, dtype=np.int8)
    with open(path, newline="") as f:
        rows = f.read().splitlines()[3:]
    for i, line in enumerate(rows[:n]):
        period[i] = int(line.split(",", 1)[0])
    labelled = truth != "X"
    blocks = []
    for p in (1, 2):
        idx = np.flatnonzero((period == p) & labelled)
        blocks.append((p, int(idx[0]), int(idx[-1])))
    _BLOCKS["current"] = blocks
    return blocks


def full_variant(data, dead, shape_weight, cost, localize, parts):
    ball, x, w, shape, alive = parts
    ev = ball + shape_weight * shape
    labels = possession.labels(dead, ev, None, switch_cost=cost)
    if localize:
        labels, _ = possession.localize(labels, dead, ball, data)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--degrade", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    grid = [(sw, cost, loc) for sw in (0.0, 0.05, 0.1, 0.2) for cost in (1.0, 1.5, 2.0, 3.0)
            for loc in (False, True)]
    table = {g: [] for g in grid}
    for game in ee.GAMES:
        data = ee.load_game(game)
        if args.degrade:
            data["ball"] = ee.mask_ball(data["ball"], args.degrade, seed=args.seed + game)
        truth = ee.proxy_truth(game, len(data["ball"]))
        dead = truth == "D"
        set_blocks(game, len(truth), truth)
        ball = possession.ball_evidence(data)
        x = team_shape.features(data)
        alive = deadball.graded_mask(len(dead)) & ~dead
        w = team_shape.fit(x, ball, alive)
        parts = (ball, x, w, team_shape.log_odds(x, w), alive)
        for g in grid:
            pred = full_variant(data, dead, g[0], g[1], g[2], parts)
            table[g].append(ee.score(truth, pred))

    print("sim_ spellF1, proxy truth, oracle dead | game1 game2 mean | spells pred/true g1 g2")
    print("shape  cost  localize")
    for g, scores in table.items():
        f1 = [s[0] for s in scores]
        spells = " ".join("{}/{}".format(s[2], s[1]) for s in scores)
        print("{:5.2f} {:5.1f}  {:5s}   {:.3f} {:.3f}  mean {:.3f} | {}".format(
            g[0], g[1], "yes" if g[2] else "no", f1[0], f1[1], float(np.mean(f1)), spells))

    print("\nHeld out: pick the best setting on one game, score it on the other")
    print("(localize 'no' only: the proxy truth starts a change at the new team's event, which")
    print(" first-touch placement may favour by construction)")
    default = (0.05, 3.0, False)
    for held in range(len(ee.GAMES)):
        other = 1 - held
        pool = [g for g in grid if not g[2]]
        best = max(pool, key=lambda g: (table[g][other][0], -g[1], -g[0]))
        print("  game {}: picked {} on game {} -> {:.3f}; team default {} -> {:.3f}".format(
            ee.GAMES[held], best[:2], ee.GAMES[other], table[best][held][0], default[:2],
            table[default][held][0]))


if __name__ == "__main__":
    main()

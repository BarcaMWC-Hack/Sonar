"""Held-out sweep of the ball-evidence thresholds and a minimum-spell filter (other-game yardstick).

    python tools/external_params.py [--degrade 0.41 --seed 0]

Grid is fixed in advance. Pipeline: ball evidence (thresholds below) + team shape (refit per
setting) at the default shape weight and switch cost, decoded per live stretch; then optionally
every spell shorter than MIN_SPELL_S is flipped to its neighbours' team. Selection is held out:
the best setting on one game is scored on the other. sim_ numbers: proxy truth, oracle dead.
"""

import argparse
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import external_eval as ee  # noqa: E402
import external_full as ef  # noqa: E402
from sonar import deadball, possession, team_shape  # noqa: E402

RADIUS = (0.75, 1.0, 1.5)
FALLOFF = (0.5, 1.0, 2.0)
SPEED = ((3.0, 6.0), (4.0, 8.0), (5.0, 10.0))
SIDE = (1.0, 1.5, 3.0)
MIN_SPELL_S = (0.0, 1.0, 2.0, 3.0)
DEFAULT = (1.0, 1.0, (4.0, 8.0), 1.5)


def merge_short(labels, min_frames):
    """Flip each H/A spell shorter than min_frames to the team on both sides, if they agree."""
    out = labels.copy()
    if min_frames <= 0:
        return out
    for _ in range(3):
        change = np.flatnonzero(out[1:] != out[:-1]) + 1
        bounds = np.concatenate(([0], change, [len(out)]))
        flipped = False
        for k in range(1, len(bounds) - 2):
            a, b = bounds[k], bounds[k + 1]
            if out[a] in "HA" and b - a < min_frames and out[a - 1] == out[b] and out[b] in "HA":
                out[a:b] = out[b]
                flipped = True
        if not flipped:
            break
    return out


def evaluate(data, truth, dead, setting):
    radius, falloff, (slow, fast), side = setting
    possession.SIDE_SCALE = side
    ball = possession.ball_evidence(data, radius=radius, falloff=falloff, slow=slow, fast=fast)
    x = team_shape.features(data)
    alive = deadball.graded_mask(len(dead)) & ~dead
    shape = team_shape.log_odds(x, team_shape.fit(x, ball, alive))
    labels = possession.labels(dead, ball + possession.SHAPE_WEIGHT * shape, None)
    return {m: ee.score(truth, merge_short(labels, int(m * 25)))[0] for m in MIN_SPELL_S}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--degrade", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="time a few settings only")
    args = parser.parse_args()

    settings = list(itertools.product(RADIUS, FALLOFF, SPEED, SIDE))
    if args.limit:
        settings = settings[:args.limit]
    scores = {}
    for game in ee.GAMES:
        data = ee.load_game(game)
        if args.degrade:
            data["ball"] = ee.mask_ball(data["ball"], args.degrade, seed=args.seed * 10 + game)
        truth = ee.proxy_truth(game, len(data["ball"]))
        dead = truth == "D"
        ef.set_blocks(game, len(truth), truth)
        for s in settings:
            for m, v in evaluate(data, truth, dead, s).items():
                scores[(s, m, game)] = v
    keys = [(s, m) for s in settings for m in MIN_SPELL_S]
    g1, g2 = ee.GAMES
    print("grid {} settings x {} filters; degrade {}".format(len(settings), len(MIN_SPELL_S),
                                                              args.degrade))
    for held, other in ((g1, g2), (g2, g1)):
        best = max(keys, key=lambda k: (scores[k + (other,)], k[1] == 0))
        dflt = (DEFAULT, 0.0)
        if dflt + (held,) in scores:
            print("  held out game {}: picked {} on game {} -> {:.3f}; default -> {:.3f}".format(
                held, best, other, scores[best + (held,)], scores[dflt + (held,)]))
    np.save(os.path.join(ROOT_TMP, "params_{}_{}.npy".format(args.degrade, args.seed)),
            np.array([(k, scores[k + (g,)]) for k in keys for g in (g1, g2)], dtype=object),
            allow_pickle=True)
    print("top 5 by mean over both games:")
    for k in sorted(keys, key=lambda k: -(scores[k + (g1,)] + scores[k + (g2,)]))[:5]:
        print("  {} -> g1 {:.3f} g2 {:.3f}".format(k, scores[k + (g1,)], scores[k + (g2,)]))
    d = (DEFAULT, 0.0)
    if d + (g1,) in scores:
        print("default {} -> g1 {:.3f} g2 {:.3f}".format(d, scores[d + (g1,)], scores[d + (g2,)]))


ROOT_TMP = os.environ.get("TEMP", ".")

if __name__ == "__main__":
    main()

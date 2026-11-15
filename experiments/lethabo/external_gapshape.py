"""Does team shape deserve more weight where the ball gives no evidence? (other-game yardstick)

    python tools/external_gapshape.py

Evidence = ball + w * shape, with w = SHAPE_WEIGHT where the ball speaks and w_gap where ball
evidence is exactly 0 (ball missing, or in flight). Grid fixed in advance; seeds 0-2 development,
3-5 unseen; ball masked to ~41% (simulated). sim_ numbers: proxy truth, oracle dead.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import external_eval as ee  # noqa: E402
import external_full as ef  # noqa: E402
from sonar import deadball, possession, team_shape  # noqa: E402

W_GAP = (0.05, 0.10, 0.20, 0.40)
SEEDS = {"dev": (0, 1, 2), "unseen": (3, 4, 5)}


def run(game, seed):
    data = ee.load_game(game)
    data["ball"] = ee.mask_ball(data["ball"], 0.41, seed=seed * 10 + game)
    truth = ee.proxy_truth(game, len(data["ball"]))
    dead = truth == "D"
    ef.set_blocks(game, len(truth), truth)
    ball = possession.ball_evidence(data)
    x = team_shape.features(data)
    alive = deadball.graded_mask(len(dead)) & ~dead
    shape = team_shape.log_odds(x, team_shape.fit(x, ball, alive))
    out = {}
    for w in W_GAP:
        weight = np.where(ball == 0, w, possession.SHAPE_WEIGHT)
        out[w] = ee.score(truth, possession.labels(dead, ball + weight * shape, None))[0]
    return out


def main():
    for name, seeds in SEEDS.items():
        rows = {w: [] for w in W_GAP}
        for seed in seeds:
            for game in ee.GAMES:
                r = run(game, seed)
                for w in W_GAP:
                    rows[w].append(r[w])
        base = np.array(rows[W_GAP[0]])
        print("\n{} seeds {}: spellF1 (g1+g2 mean over seeds) | vs w_gap=0.05".format(name, seeds))
        for w in W_GAP:
            v = np.array(rows[w])
            d = v - base
            print("  w_gap {:.2f}  mean {:.3f} | {:+.4f} (min {:+.4f}, wins {}/{})".format(
                w, v.mean(), d.mean(), d.min(), int((d > 0).sum()), d.size))


if __name__ == "__main__":
    main()

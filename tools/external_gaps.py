"""Does placing switches inside ball gaps help? Other-game yardstick, seeded masks (simulated).

    python tools/external_gaps.py [--localize]

Full pipeline (ball + team shape, default settings) on Metrica games 1-2 with the ball masked to
about 41% missing. Rules are fixed in advance (sonar/gaps.py); seeds 0-2 are the development
masks and seeds 3-5 are unseen. sim_ numbers: proxy truth, oracle dead.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import external_eval as ee  # noqa: E402
import external_full as ef  # noqa: E402
from sonar import deadball, gaps, possession, team_shape  # noqa: E402

SEEDS = {"dev": (0, 1, 2), "unseen": (3, 4, 5)}


def run(game, seed, localize, fraction=0.41):
    data = ee.load_game(game)
    data["ball"] = ee.mask_ball(data["ball"], fraction, seed=seed * 10 + game)
    truth = ee.proxy_truth(game, len(data["ball"]))
    dead = truth == "D"
    ef.set_blocks(game, len(truth), truth)
    ball = possession.ball_evidence(data)
    x = team_shape.features(data)
    alive = deadball.graded_mask(len(dead)) & ~dead
    shape = team_shape.log_odds(x, team_shape.fit(x, ball, alive))
    base = possession.labels(dead, ball + possession.SHAPE_WEIGHT * shape, None)
    scores = {}
    for rule in gaps.RULES:
        labels, moved = gaps.place_switches_in_gaps(base, dead, data["ball"], data, rule)
        if localize:
            labels, _ = possession.localize(labels, dead, ball, data)
        scores[rule] = (ee.score(truth, labels)[0], moved)
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--localize", action="store_true")
    args = parser.parse_args()
    for name, seeds in SEEDS.items():
        print("\n{} seeds {}  (localize {}):  spellF1 game1 game2 | switches moved".format(
            name, seeds, "yes" if args.localize else "no"))
        table = {r: [] for r in gaps.RULES}
        for seed in seeds:
            for game in ee.GAMES:
                s = run(game, seed, args.localize)
                for r in gaps.RULES:
                    table[r].append(s[r])
        for r in gaps.RULES:
            f1 = np.array([v[0] for v in table[r]]).reshape(len(seeds), len(ee.GAMES))
            mv = np.array([v[1] for v in table[r]]).reshape(len(seeds), len(ee.GAMES))
            diff = f1 - np.array([v[0] for v in table["end"]]).reshape(len(seeds), len(ee.GAMES))
            print("  {:9s} mean {:.3f}  g1 {:.3f} g2 {:.3f} | vs end: {:+.4f} (min {:+.4f}, "
                  "wins {}/{}) | moved ~{:.0f}".format(
                      r, f1.mean(), f1[:, 0].mean(), f1[:, 1].mean(), diff.mean(), diff.min(),
                      int((diff > 0).sum()), diff.size, mv.mean()))


if __name__ == "__main__":
    main()

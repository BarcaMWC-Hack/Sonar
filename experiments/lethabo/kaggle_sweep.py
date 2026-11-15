"""Big parallel sweep on the other-game yardstick (meant for a Kaggle CPU session).

    python tools/kaggle_sweep.py --out sweep.csv [--limit N] [--workers W]

Reads only the open Metrica sample games (data/external/metrica). Never touches the provided
match. Each worker handles one (game, control radius, falloff, speed, side) group: ball evidence
and the shape fit are computed once, then every (switch cost, shape weight) is decoded, with and
without first-touch placement, scored against proxy truth A and B (tools/external_joint.py).
Writes one CSV row per (game, setting). Analyse with tools/joint_report.py.
"""

import argparse
import csv
import itertools
import multiprocessing as mp
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

RADIUS = (0.75, 1.0, 1.5)
FALLOFF = (0.5, 1.0, 2.0)
SPEED = ((3.0, 6.0), (4.0, 8.0), (5.0, 10.0))
SIDE = (1.0, 1.5, 3.0)
COST = (1.5, 2.0, 3.0, 4.0, 6.0)
SHAPE = (0.0, 0.03, 0.05, 0.08, 0.12)
GAMES = (1, 2, 3)
_CACHE = {}


def _game(game):
    if game not in _CACHE:
        import external_eval as ee
        import external_joint as ej
        data = ej.load(game)
        n = len(data["ball"])
        trs = {"A": ej.truth(game, n, False), "B": ej.truth(game, n, True)}
        ej.blocks(game, n, trs["A"])
        _CACHE.clear()
        _CACHE[game] = (data, trs, trs["A"] == "D")
    return _CACHE[game]


def work(task):
    game, radius, falloff, speed, side = task
    import external_eval as ee
    import external_joint as ej
    from sonar import deadball, possession, team_shape
    data, trs, dead = _game(game)
    ej.blocks(game, len(dead), trs["A"])
    possession.SIDE_SCALE = side
    ball = possession.ball_evidence(data, radius=radius, falloff=falloff, slow=speed[0],
                                    fast=speed[1])
    x = team_shape.features(data)
    alive = deadball.graded_mask(len(dead)) & ~dead
    shape = team_shape.log_odds(x, team_shape.fit(x, ball, alive))
    rows = []
    for cost, sw in itertools.product(COST, SHAPE):
        labels = possession.labels(dead, ball + sw * shape, None, switch_cost=cost)
        placed, _ = possession.localize(labels, dead, ball, data)
        rows.append([game, radius, falloff, speed[0], speed[1], side, cost, sw] +
                    [round(ee.score(trs[p], lab)[0], 5) for lab in (labels, placed)
                     for p in ("A", "B")])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="sweep.csv")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    tasks = [(g,) + t for g in GAMES for t in itertools.product(RADIUS, FALLOFF, SPEED, SIDE)]
    if args.limit:
        tasks = tasks[:args.limit]
    # Group by game so each worker keeps one game's arrays in memory.
    tasks.sort(key=lambda t: t[0])
    with open(args.out, "w", newline="") as f, mp.Pool(args.workers) as pool:
        w = csv.writer(f)
        w.writerow(["game", "radius", "falloff", "slow", "fast", "side", "cost", "shape",
                    "A_off", "B_off", "A_on", "B_on"])
        done = 0
        for rows in pool.imap(work, tasks, chunksize=1):
            w.writerows(rows)
            f.flush()
            done += 1
            print("{}/{} groups".format(done, len(tasks)), flush=True)


if __name__ == "__main__":
    main()

"""Label-free tracking statistics, to see which open games the competition tracking resembles.

    python tools/domain_stats.py --local               # Metrica games + the provided match (local only)
    python tools/domain_stats.py --sportec             # Sportec matches (needs internet + kloppy)

Uses no labels of any kind: only the tracking arrays. Writes a small JSON of summary numbers
(no frame-level data). The provided match is never uploaded anywhere: --local runs on this machine.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

FPS = 25.0


def runs_of(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def stats(data):
    ball = data["ball"][:, :2].astype(np.float64)
    players = data["players"][:, :, :2].astype(np.float64)
    n = len(ball)
    present = ~np.isnan(ball[:, 0])
    a, b = runs_of(~present)
    lengths = (b - a) if len(a) else np.array([0])
    missing = int((~present).sum())
    step = np.sqrt((np.diff(ball, axis=0) ** 2).sum(axis=1)) * FPS
    both = present[1:] & present[:-1]
    ball_speed = step[both]
    second = np.abs(np.diff(ball, n=2, axis=0))
    second = second[np.isfinite(second).all(axis=1)]
    pstep = np.sqrt((np.diff(players, axis=0) ** 2).sum(axis=2)) * FPS
    pstep = pstep[np.isfinite(pstep)]
    tracked = np.isfinite(players[:, :, 0]).sum(axis=1)
    return {
        "frames": n,
        "ball_missing_frac": round(missing / n, 4),
        "missing_runs": int(len(a)),
        "median_missing_run_s": round(float(np.median(lengths)) / FPS, 2),
        "share_missing_in_runs_5s_plus": round(float(lengths[lengths >= 5 * FPS].sum()) / max(missing, 1), 3),
        "ball_speed_p50": round(float(np.percentile(ball_speed, 50)), 2),
        "ball_speed_p95": round(float(np.percentile(ball_speed, 95)), 2),
        "ball_jitter_p50_m": round(float(np.percentile(second.sum(axis=1), 50)), 4),
        "player_speed_p50": round(float(np.percentile(pstep, 50)), 3),
        "player_speed_p95": round(float(np.percentile(pstep, 95)), 3),
        "tracked_players_mean": round(float(tracked.mean()), 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--sportec", action="store_true")
    parser.add_argument("--out", default="domain_stats.json")
    args = parser.parse_args()
    out = {}
    if args.local:
        import external_eval as ee
        import external_game3 as g3
        from sonar import atd
        for g in (1, 2):
            out["metrica{}".format(g)] = stats(ee.load_game(g))
        out["metrica3"] = stats(g3.load()[0])
        data = atd.load()
        graded = np.zeros(len(data["ball"]), dtype=bool)
        from games import half_video_blocks, GAME
        for _, first, last in half_video_blocks(GAME):
            graded[first:last + 1] = True
        d = dict(data)
        d["ball"] = data["ball"][graded]
        d["players"] = data["players"][graded]
        out["PROVIDED_MATCH_graded_frames"] = stats(d)
    if args.sportec:
        import confirm_sportec as cs
        for m in cs.MATCHES:
            try:
                out[m] = stats(cs.load_match(m)[0])
            except Exception as e:
                out[m] = {"error": type(e).__name__}
            print(m, out[m], flush=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()

"""Load SkillCorner's open A-League broadcast tracking (MIT licence) in the shape sonar expects.

    python tools/skillcorner_data.py         # parse and cache every downloaded match, print counts

Source: https://github.com/SkillCorner/opendata (20 matches of 2024/25 A-League, 10 fps
broadcast tracking). Put each match folder under data/external/skillcorner/<id>/ (git-ignored),
or set SONAR_EXTERNAL_DIR to the folder holding skillcorner/.

Why it matters: this is BROADCAST tracking, like ours, so the ball and players drop out of view
for real. No simulation is needed. The ball is NaN wherever SkillCorner did not detect it;
players keep SkillCorner's extrapolated positions.

Truth (graded = frames inside the two periods):
  D  = not inside any phase of play (SkillCorner defines phases only while the ball is in play)
  H/A = the frame's possession group (home / away team); where that is empty inside a phase,
        the phase's team in possession
This is SkillCorner's own definition, close to but not the same as the competition's.

Frames are resampled from 10 to 25 fps (linear for positions, nearest for truth) so every sonar
constant keeps its meaning. Speeds are recomputed from positions, as in tools/idsse_data.py.

Prints counts only.
"""

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from idsse_data import _speed  # noqa: E402

EXTERNAL = os.environ.get("SONAR_EXTERNAL_DIR") or os.path.join(ROOT, "data", "external")
DATA = os.path.join(EXTERNAL, "skillcorner")
CACHE = os.path.join(EXTERNAL, "skillcorner_cache")
SRC_FPS, FPS = 10, 25
GAP = 1500
CACHE_VERSION = 1


def _paths(match):
    base = os.path.join(DATA, match, match + "_")
    return base + "match.json", base + "tracking_extrapolated.jsonl", base + "phases_of_play.csv"


def build(match):
    info_path, track_path, phase_path = _paths(match)
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    home_id = info["home_team"]["id"]
    length, width = float(info["pitch_length"]), float(info["pitch_width"])
    side = {p["id"]: (0 if p["team_id"] == home_id else 1) for p in info["players"]}
    people = sorted(side)
    slot = {p: i for i, p in enumerate(people)}

    frames, periods, ball, players, group = [], [], [], [], []
    with open(track_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("period") not in (1, 2):
                continue
            frames.append(r["frame"])
            periods.append(r["period"])
            b = r.get("ball_data") or {}
            seen = b.get("is_detected") and b.get("x") is not None
            ball.append((b["x"], b["y"]) if seen else (np.nan, np.nan))
            row = np.full((len(people), 2), np.nan)
            for p in r.get("player_data") or []:
                j = slot.get(p["player_id"])
                if j is not None:
                    row[j] = (p["x"], p["y"])
            players.append(row)
            g = (r.get("possession") or {}).get("group")
            group.append(1 if g == "home team" else (-1 if g == "away team" else 0))
    frames, periods = np.asarray(frames), np.asarray(periods)
    ball, players, group = np.asarray(ball), np.asarray(players), np.asarray(group)

    alive_team = {}
    with open(phase_path, newline="", encoding="utf-8") as f:
        phases = list(csv.DictReader(f))
    in_phase = np.zeros(len(frames), dtype=bool)
    phase_team = np.zeros(len(frames), dtype=np.int8)
    order = np.argsort(frames)
    for ph in phases:
        lo, hi = int(ph["frame_start"]), int(ph["frame_end"])
        a, b = np.searchsorted(frames[order], [lo, hi + 1])
        idx = order[a:b]
        in_phase[idx] = True
        phase_team[idx] = 1 if int(ph["team_in_possession_id"]) == home_id else -1
    del alive_team

    # Resample each period to 25 fps.
    out_ball, out_players, out_truth, blocks = [], [], [], []
    total = 0
    for k, per in enumerate((1, 2)):
        sel = np.flatnonzero(periods == per)
        sel = sel[np.argsort(frames[sel])]
        src = frames[sel] - frames[sel][0]
        n = int(src[-1] * FPS / SRC_FPS) + 1
        t = np.arange(n) * SRC_FPS / FPS
        lo = np.clip(np.searchsorted(src, t, side="right") - 1, 0, len(src) - 1)
        hi = np.clip(lo + 1, 0, len(src) - 1)
        span = np.maximum(src[hi] - src[lo], 1e-9)
        frac = np.where(hi > lo, (t - src[lo]) / span, 0.0)
        near = np.where(frac < 0.5, lo, hi)
        # source frames more than one step apart are a hole in the feed: no interpolation
        hole = (src[hi] - src[lo]) > 1

        def interp(values):
            v0, v1 = values[sel][lo], values[sel][hi]
            f = frac.reshape((-1,) + (1,) * (v0.ndim - 1))
            out = v0 + (v1 - v0) * f
            out[hole] = np.nan
            return out

        b = interp(ball)
        p = interp(players)
        g = group[sel][near]
        alive = in_phase[sel][near]
        team = np.where(g != 0, g, phase_team[sel][near])
        truth = np.where(~alive, "D", np.where(team > 0, "H", np.where(team < 0, "A", "D")))
        out_ball.append(b)
        out_players.append(p)
        out_truth.append(truth)
        blocks.append((per, total, total + n - 1))
        total += n
        if k == 0:
            out_ball.append(np.full((GAP, 2), np.nan))
            out_players.append(np.full((GAP, len(people), 2), np.nan))
            out_truth.append(np.full(GAP, "X"))
            total += GAP
    b = np.concatenate(out_ball)
    p = np.concatenate(out_players)
    # SkillCorner uses a centre origin; sonar uses a corner origin.
    b = b + np.array([length / 2, width / 2])
    p = p + np.array([length / 2, width / 2])
    speed = np.stack([_speed(p[:, j, 0], p[:, j, 1]) for j in range(p.shape[1])], axis=1)
    players3 = np.concatenate([p, speed[:, :, None]], axis=2).astype(np.float32)
    ball3 = np.concatenate([b, np.full((len(b), 1), np.nan)], axis=1).astype(np.float32)
    return {"ball": ball3, "players": players3,
            "team": np.array([side[q] for q in people], dtype=np.int8),
            "pitch": (length, width), "truth": np.concatenate(out_truth).astype("<U1"),
            "blocks": blocks}


def load(match, rebuild=False):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, "{}_v{}.npz".format(match, CACHE_VERSION))
    if os.path.exists(path) and not rebuild:
        z = np.load(path)
        return {"ball": z["ball"], "players": z["players"], "team": z["team"],
                "pitch": tuple(z["pitch"].tolist()), "truth": z["truth"],
                "blocks": [tuple(int(v) for v in b) for b in z["blocks"]]}
    d = build(match)
    np.savez_compressed(path, ball=d["ball"], players=d["players"], team=d["team"],
                        pitch=np.array(d["pitch"]), truth=d["truth"], blocks=np.array(d["blocks"]))
    return d


def available():
    if not os.path.isdir(DATA):
        return []
    out = []
    for m in sorted(os.listdir(DATA)):
        paths = _paths(m)
        if all(os.path.exists(p) for p in paths) and os.path.getsize(paths[1]) > 1000:
            out.append(m)
    return out


if __name__ == "__main__":
    for match in (sys.argv[1:] or available()):
        d = load(match, rebuild="--rebuild" in sys.argv)
        t = d["truth"]
        g = t != "X"
        poss = 0
        for _, a, b in d["blocks"]:
            s = t[a:b + 1]
            poss += int(np.sum((s[1:] != s[:-1]) & np.isin(s[1:], ["H", "A"]))) + int(s[0] in "HA")
        print("{}: graded {}, players {}, H {:.1%} A {:.1%} D {:.1%}, ball missing {:.1%}, "
              "possessions {}".format(match, int(g.sum()), d["players"].shape[1],
                                      np.mean(t[g] == "H"), np.mean(t[g] == "A"),
                                      np.mean(t[g] == "D"), np.mean(np.isnan(d["ball"][g, 0])),
                                      poss))

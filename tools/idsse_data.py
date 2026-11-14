"""Load the open DFL / IDSSE matches (Bassek et al. 2025, CC-BY 4.0) in the shape sonar expects.

    python tools/idsse_data.py            # parse and cache every downloaded match, print counts

Source: https://huggingface.co/datasets/pysport/idsse-data (mirror of figshare
10.6084/m9.figshare.28196177): 7 Bundesliga matches, TRACAB tracking at 25 Hz. Put the files in
data/external/idsse (git-ignored), or set SONAR_EXTERNAL_DIR to the folder holding idsse/.

Why it is a real yardstick: the ball's FrameSet carries BallStatus (0 dead / 1 alive) and
BallPossession (1 home / 2 away) on EVERY frame -- the same mechanical kind of truth the
competition derives from corrected tracking. So unlike the Metrica proxy (tools/external_eval.py)
this scores the dead-ball detector and the possession calls together, on other games, which the
rules allow us to tune on.

Layout of a loaded match: first half, a 1,500-frame ungraded gap, second half. Coordinates are
moved from DFL's centre origin to sonar's corner origin. Speeds are recomputed from positions
(m/s over 0.2 s), so they mean the same thing as in sonar's own tracking. Truth is 'D' where
BallStatus is 0, else 'H' (home) or 'A' (away), and 'X' in the gap.

Prints counts only.
"""

import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXTERNAL = os.environ.get("SONAR_EXTERNAL_DIR") or os.path.join(ROOT, "data", "external")
DATA = os.path.join(EXTERNAL, "idsse")
CACHE = os.path.join(EXTERNAL, "idsse_cache")
MATCHES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
GAP = 1500
FPS = 25
CACHE_VERSION = 1

_FRAMESET = re.compile(r'<FrameSet ([^>]*)>')
_ATTR = re.compile(r'(\w+)="([^"]*)"')
_FRAME = re.compile(r'<Frame N="(\d+)"[^>]*? X="([-\d.]+)" Y="([-\d.]+)"([^>]*)/>')


def _file(kind, match):
    for name in os.listdir(DATA):
        if name.startswith("DFL_" + kind) and name.endswith(match + ".xml"):
            return os.path.join(DATA, name)
    raise FileNotFoundError("{} for {} not in {}".format(kind, match, DATA))


def _info(match):
    with open(_file("02_01_matchinformation", match), encoding="utf-8") as f:
        text = f.read()
    home = re.search(r'HomeTeamId="([^"]+)"', text).group(1)
    away = re.search(r'GuestTeamId="([^"]+)"', text).group(1)
    length = float(re.search(r'PitchX="([\d.]+)"', text).group(1))
    width = float(re.search(r'PitchY="([\d.]+)"', text).group(1))
    return home, away, length, width


def _parse(match):
    home, away, length, width = _info(match)
    sets = []           # (section, team, person, frames, x, y, status, possession)
    current = None
    with open(_file("04_03_positions_raw_observed", match), encoding="utf-8") as f:
        for line in f:
            if "<Frame " in line:
                m = _FRAME.search(line)
                if m is None:
                    continue
                current[3].append(int(m.group(1)))
                current[4].append(float(m.group(2)))
                current[5].append(float(m.group(3)))
                if current[1] == "BALL":
                    rest = dict(_ATTR.findall(m.group(4)))
                    current[6].append(int(rest.get("BallStatus", "0")))
                    current[7].append(int(rest.get("BallPossession", "0")))
            elif "<FrameSet " in line:
                attrs = dict(_ATTR.findall(_FRAMESET.search(line).group(1)))
                current = (attrs["GameSection"], attrs["TeamId"], attrs["PersonId"],
                           [], [], [], [], [])
                sets.append(current)
    return home, away, length, width, sets


def _speed(x, y, half=2):
    """m/s from the displacement over +/-half frames; NaN where a neighbour is missing."""
    out = np.full(len(x), np.nan)
    if len(x) > 2 * half:
        dx = x[2 * half:] - x[:-2 * half]
        dy = y[2 * half:] - y[:-2 * half]
        out[half:-half] = np.hypot(dx, dy) * FPS / (2 * half)
    return out


def build(match):
    home, away, length, width, sets = _parse(match)
    sections = sorted({s[0] for s in sets}, key=lambda s: ("first" not in s, s))
    ranges = {}
    for sec in sections:
        frames = [f for s in sets if s[0] == sec and s[1] == "BALL" for f in s[3]]
        ranges[sec] = (min(frames), max(frames))
    offsets, total = {}, 0
    blocks = []
    for k, sec in enumerate(sections):
        lo, hi = ranges[sec]
        offsets[sec] = total - lo
        blocks.append((k + 1, total, total + hi - lo))
        total += hi - lo + 1 + (GAP if k < len(sections) - 1 else 0)
    people = sorted({(s[1], s[2]) for s in sets if s[1] in (home, away)})
    slot = {p: i for i, p in enumerate(people)}
    players = np.full((total, len(people), 3), np.nan, dtype=np.float32)
    ball = np.full((total, 3), np.nan, dtype=np.float32)
    truth = np.full(total, "X", dtype="<U1")
    for sec, team, person, frames, xs, ys, status, poss in sets:
        if sec not in offsets:
            continue
        idx = np.asarray(frames) + offsets[sec]
        x = np.asarray(xs) + length / 2.0
        y = np.asarray(ys) + width / 2.0
        if team == "BALL":
            ball[idx, 0], ball[idx, 1] = x, y
            st, po = np.asarray(status), np.asarray(poss)
            truth[idx] = np.where(st == 0, "D", np.where(po == 1, "H", np.where(po == 2, "A", "D")))
        elif (team, person) in slot:
            j = slot[(team, person)]
            players[idx, j, 0], players[idx, j, 1] = x, y
            # frames of one person are contiguous runs; speed over each run
            full_x = np.full(total, np.nan)
            full_y = np.full(total, np.nan)
            full_x[idx], full_y[idx] = x, y
            players[idx, j, 2] = _speed(full_x, full_y)[idx]
    team_codes = np.array([0 if t == home else 1 for t, _ in people], dtype=np.int8)
    return {"ball": ball, "players": players, "team": team_codes, "pitch": (length, width),
            "truth": truth, "blocks": blocks}


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
    for m in MATCHES:
        if os.path.exists(os.path.join(CACHE, "{}_v{}.npz".format(m, CACHE_VERSION))):
            out.append(m)
            continue
        try:
            path, _ = _file("04_03_positions_raw_observed", m), _file("02_01_matchinformation", m)
        except FileNotFoundError:
            continue
        with open(path, "rb") as f:          # a download still in progress has no closing tag
            f.seek(max(0, os.path.getsize(path) - 200))
            if b"</PutDataRequest>" in f.read():
                out.append(m)
    return out


if __name__ == "__main__":
    for match in (sys.argv[1:] or available()):
        d = load(match, rebuild="--rebuild" in sys.argv)
        t = d["truth"]
        graded = t != "X"
        print("{}: frames {}, graded {}, players {}, H {:.1%} A {:.1%} D {:.1%}, ball missing {:.1%}"
              .format(match, len(t), int(graded.sum()), d["players"].shape[1],
                      np.mean(t[graded] == "H"), np.mean(t[graded] == "A"),
                      np.mean(t[graded] == "D"), np.mean(np.isnan(d["ball"][graded, 0]))))

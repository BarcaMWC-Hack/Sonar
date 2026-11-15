"""Load the tactical ATD tracking file into dense numpy arrays, once, then cache it.

The raw file is one line per video frame:

    frame_id:player1;player2;...;player46:ball

Each group is ``x,y,speed`` with ``NaN`` for a missing observation.  Coordinates are normalised
0-1, so they are multiplied by the real pitch size (105.3 x 68.0 m) before use.

    ball     (n_frames, 3)      x, y in metres, speed as provided (mostly NaN)
    players  (n_frames, 46, 3)  x, y in metres, speed in m/s, per track slot
    team     (46,)              0 = home (Barcelona), 1 = away (Atletic Club)

Row ``i`` is video frame ``i`` (the file's frame_id counts from 1), so every array lines up with a
submission row.

Why a cache: parsing 199,319 lines takes tens of seconds.  The cache is keyed by a SHA-256 of the
raw file, so a different input can never silently reuse an old array.
"""

import hashlib
import io
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import DATA_DIR, GAME, GAMES, require, source_frames  # noqa: E402
from starter.baseline import team_names, team_of_each_track  # noqa: E402

N_SLOTS = 46
VALUES_PER_GROUP = 3
CHUNK_LINES = 20000
CACHE_VERSION = 1
FLATTEN = str.maketrans({":": ",", ";": ","})


def cache_path(game=GAME, data_dir=None):
    return os.path.join(data_dir or DATA_DIR, "cache", "{}_atd.npz".format(game))


def _content_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return np.frombuffer(digest.digest(), dtype=np.int64)


def _parse(raw_path, n_frames):
    n_cols = 1 + VALUES_PER_GROUP * (N_SLOTS + 1)
    flat = np.full((n_frames, n_cols - 1), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        while True:
            lines = [line for _, line in zip(range(CHUNK_LINES), handle)]
            if not lines:
                break
            text = "".join(lines).translate(FLATTEN)
            chunk = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64,
                                na_values=["NaN"], engine="c").to_numpy()
            if chunk.shape[1] != n_cols:
                raise ValueError("expected {} values per line, got {}".format(
                    n_cols, chunk.shape[1]))
            frames = chunk[:, 0].astype(np.int64) - 1
            flat[frames] = chunk[:, 1:].astype(np.float32)
    return flat


def _assemble(flat, team, length, width):
    scale = np.array([length, width, 1.0], dtype=np.float32)
    players = flat[:, :N_SLOTS * VALUES_PER_GROUP].reshape(-1, N_SLOTS, VALUES_PER_GROUP) * scale
    ball = flat[:, N_SLOTS * VALUES_PER_GROUP:] * scale
    return {"ball": ball, "players": players, "team": team, "pitch": (length, width)}


def load(game=GAME, data_dir=None, rebuild=False):
    """dict with ``ball``, ``players``, ``team``, ``pitch``, ``home``, ``away``."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    stamp = np.concatenate(([CACHE_VERSION, n_frames], _content_hash(raw_path))).astype(np.int64)

    path = cache_path(game, data_dir)
    if not rebuild and os.path.isfile(path):
        cached = np.load(path)
        if np.array_equal(cached["stamp"], stamp):
            data = _assemble(cached["flat"], team_of_each_track(meta_path), length, width)
            data["home"], data["away"] = team_names(meta_path)
            return data

    flat = _parse(raw_path, n_frames)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, flat=flat, stamp=stamp)
    data = _assemble(flat, team_of_each_track(meta_path), length, width)
    data["home"], data["away"] = team_names(meta_path)
    return data


if __name__ == "__main__":
    import time

    start = time.time()
    data = load(rebuild="--rebuild" in sys.argv)
    print("loaded in {:.1f} s".format(time.time() - start))
    print("ball {}  players {}  home {}  away {}".format(
        data["ball"].shape, data["players"].shape, data["home"], data["away"]))
    print("ball position known on {} of {} frames".format(
        int(np.sum(~np.isnan(data["ball"][:, 0]))), len(data["ball"])))

"""The tactical ATD as numpy arrays, parsed once and cached.

starter/baseline.py streams the raw file line by line, which is fine for one pass but slow for
anything that needs every slot of every frame. This parses the same file into dense arrays:

    ball     (n_frames, 3)      x, y in metres, ball speed as provided (mostly NaN, see below)
    players  (n_frames, 46, 3)  x, y in metres, speed in m/s, per track slot
    team     (46,)              0 = home (Barcelona), 1 = away, per track slot

Row i is video frame i (= ATD frame_id - 1), so every array lines up with a submission row.

Schema facts, from tools/probe_schema.py on the delivered file: every line has 46 player groups
and one ball group, three values each. The ball's third value is NaN on 91% of the frames where
its position is known, so ball speed should be derived from positions rather than read.

The cache lives in data/cache/ (git-ignored with the rest of data/) and is rebuilt whenever the
raw file's content changes, so it can never go stale silently.

Ungraded frames are blanked to NaN on load, so no model or smoothing window can reach them.
"""

import io
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, DATA_DIR, require, source_frames  # noqa: E402
from sonar.splits import graded_mask  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402

# Params
N_SLOTS = 46
VALUES_PER_GROUP = 3
CHUNK_LINES = 20000
CACHE_VERSION = 1
# ':' separates frame / players / ball and ';' separates groups; mapping both to ',' turns a line
# into one flat CSV row the C parser can read: frame, 46 x (x, y, s), ball (x, y, s).
FLATTEN = str.maketrans({":": ",", ";": ","})


def cache_path(game=GAME, data_dir=None):
    return os.path.join(data_dir or DATA_DIR, "cache", "{}_tactical_atd.npz".format(game))


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
            frames = chunk[:, 0].astype(np.int64) - 1   # frame_id counts from 1
            flat[frames] = chunk[:, 1:].astype(np.float32)
    return flat


def _content_hash(path):
    """SHA-256 of a file as four int64 values."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return np.frombuffer(digest.digest(), dtype=np.int64)


def load(game=GAME, data_dir=None, rebuild=False, graded_only=True):
    """dict with ball, players, team and pitch (length, width) in metres."""
    meta_path, raw_path = require(["atd_meta", "atd_raw"], game, data_dir)
    n_frames = source_frames("tactical", game)
    length, width = GAMES[game]["pitch"]
    keep = graded_mask(n_frames, game) if graded_only else None
    # Keyed by the file CONTENT (SHA-256), not its size and time, so a different input can never
    # silently reuse an old cache.
    stamp = np.concatenate(([CACHE_VERSION, n_frames], _content_hash(raw_path))).astype(np.int64)

    path = cache_path(game, data_dir)
    if not rebuild and os.path.isfile(path):
        cached = np.load(path)
        if np.array_equal(cached["stamp"], stamp):
            return _assemble(cached["flat"], team_of_each_track(meta_path), length, width, keep)

    flat = _parse(raw_path, n_frames)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, flat=flat, stamp=stamp)
    return _assemble(flat, team_of_each_track(meta_path), length, width, keep)


def _assemble(flat, team, length, width, keep=None):
    scale = np.array([length, width, 1.0], dtype=np.float32)
    players = flat[:, :N_SLOTS * VALUES_PER_GROUP].reshape(-1, N_SLOTS, VALUES_PER_GROUP) * scale
    ball = flat[:, N_SLOTS * VALUES_PER_GROUP:] * scale
    if keep is not None:
        players[~keep] = np.nan
        ball[~keep] = np.nan
    return {"ball": ball, "players": players, "team": team, "pitch": (length, width)}


if __name__ == "__main__":
    import time
    start = time.time()
    data = load(rebuild="--rebuild" in sys.argv)
    print("loaded in {:.1f} s".format(time.time() - start))
    print("ball {}  players {}  team slots home {} away {}".format(
        data["ball"].shape, data["players"].shape,
        int(np.sum(data["team"] == 0)), int(np.sum(data["team"] == 1))))
    print("ball position known on {} frames".format(int(np.sum(~np.isnan(data["ball"][:, 0])))))

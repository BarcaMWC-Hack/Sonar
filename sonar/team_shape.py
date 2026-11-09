"""Team shape as possession evidence: the side with the ball spreads out and moves forward.

Available on every frame, including the 18% of live frames where the ball is not tracked. On the
delivered file the ball gives no evidence on 46% of live frames: in flight, untracked, or away
from every player. It is a slow signal -- a team takes seconds to spread out after winning the
ball -- so it is used to decide WHETHER possession changed, never WHEN: the switch point stays
with the ball evidence.

Features, each Barca minus Athletic, smoothed over 2 s:
  spread_y   standard deviation of the players' y (width). On its own this separates who
             controls the ball with AUC 0.86 on the delivered file.
  spread_x   standard deviation of x (length)
  forward    speed of the team's centroid towards the goal it attacks
  advance    how far both teams' centroids sit into Athletic's half (where the block is)

A logistic model maps them to log-odds Barca over Athletic. It is trained on the live frames where
the ball evidence is strong (a player within ~1.5 m of a slow ball), so its labels come from the
tracking, not from any ground truth; tools/possession_report.py reports its cross-half AUC.
"""

import warnings

import numpy as np

from games import GAME, half_video_blocks
from sonar import deadball
from sonar.deadball_features import rolling_mean

# Params
FPS = 25
SMOOTH_HALF = 25
LABEL_EVIDENCE = 0.5        # |ball evidence| above this makes a frame a training example
CLIP = 4.0                  # cap on the shape log-odds, so a freak shape cannot dominate
L2 = 1.0
NAMES = ("spread_y", "spread_x", "forward", "advance")


def features(data, game=GAME):
    """(n_frames, 4) shape features, NaN where a team has no tracked player."""
    players = data["players"].astype(np.float64)
    home = data["team"] == 0
    length = data["pitch"][0]
    n = len(players)
    out = np.full((n, len(NAMES)), np.nan)
    blocks = half_video_blocks(game)
    bounds = [(0, blocks[1][1]), (blocks[1][1], n)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for (start, stop), (_, first, last) in zip(bounds, blocks):
            px, py = players[start:stop, :, 0], players[start:stop, :, 1]
            hx, ax = np.nanmean(px[:, home], axis=1), np.nanmean(px[:, ~home], axis=1)
            # +1 when Barca attack towards larger x in this half, read from where they stand.
            sign = 1.0 if (np.nanmean(players[first:last + 1][:, home, 0]) <
                           np.nanmean(players[first:last + 1][:, ~home, 0])) else -1.0
            out[start:stop, 0] = np.nanstd(py[:, home], axis=1) - np.nanstd(py[:, ~home], axis=1)
            out[start:stop, 1] = np.nanstd(px[:, home], axis=1) - np.nanstd(px[:, ~home], axis=1)
            # Barca's forward speed minus Athletic's; Athletic attack the other way.
            out[start:stop, 2] = sign * (np.gradient(hx) + np.gradient(ax)) * FPS
            out[start:stop, 3] = sign * ((hx - length / 2) + (ax - length / 2))
    for j in range(len(NAMES)):
        out[:, j] = rolling_mean(out[:, j], SMOOTH_HALF)
    return out


def centred(x, alive, game=GAME):
    """x with each half's mean over its live frames removed.

    The gap between the teams' shapes has its own baseline in each half (tactics, substitutions).
    Left in, a model fitted on one half is biased on the other by up to 0.6 log-odds. No labels
    are used.
    """
    out = x.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for _, first, last in half_video_blocks(game):
            half = np.zeros(len(x), dtype=bool)
            half[first:last + 1] = True
            out[half] -= np.nanmean(x[half & alive], axis=0)
    return out


def training_mask(ball_ev, alive, x):
    return alive & (np.abs(ball_ev) > LABEL_EVIDENCE) & ~np.isnan(x).any(axis=1)


def fit(x, ball_ev, alive):
    """[bias, *weights] of log-odds Barca over Athletic, from the frames the ball is clear on."""
    train = training_mask(ball_ev, alive, x)
    return deadball.fit_logistic(x[train], (ball_ev[train] > 0).astype(float), l2=L2)


def log_odds(x, w):
    """Per-frame shape log-odds, 0 where a feature is missing, clipped to +/- CLIP."""
    out = w[0] + np.nan_to_num(x) @ w[1:]
    out[np.isnan(x).any(axis=1)] = 0.0
    return np.clip(out, -CLIP, CLIP)

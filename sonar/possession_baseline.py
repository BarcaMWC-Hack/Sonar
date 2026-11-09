"""Stand-in team call: the nearest tracked player to the ball, as in starter/baseline.py.

This is the placeholder the possession work (role 2) replaces. It exists so the dead-ball
detector can be scored end to end today. Same rule as the published baseline, vectorised over
the cached arrays instead of streaming the raw file.
"""

import numpy as np

HOME, AWAY, NOBODY = 0, 1, -1


def nearest_team(data):
    """Per-frame team code of the player nearest the ball; NOBODY where the ball is not tracked."""
    ball = data["ball"][:, :2].astype(np.float64)
    players = data["players"][:, :, :2].astype(np.float64)
    d2 = ((players - ball[:, None, :]) ** 2).sum(axis=2)
    d2 = np.where(np.isnan(d2), np.inf, d2)
    nearest = np.argmin(d2, axis=1)
    found = np.isfinite(d2[np.arange(len(d2)), nearest])
    return np.where(found, data["team"][nearest], NOBODY).astype(np.int8)

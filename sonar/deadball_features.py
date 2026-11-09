"""Per-frame evidence for "the ball is dead", computed from the ATD alone.

Every feature is an array over video frames, NaN where it cannot be computed. None of them is a
decision; sonar/deadball.py turns them into one.

What each one is after
----------------------
ball_oob          metres the ball sits outside the touchlines / goal lines (<= 0 inside). A ball
                  clearly over a line is the most direct evidence of a stoppage there is.
ball_speed        ball displacement over one second, in m/s. A placed ball (free kick, goal kick,
                  corner, kick-off) sits still; a ball in play rarely does for long.
ball_known_share  share of the surrounding two seconds on which the ball is tracked at all.
missing_run       length in frames of the run of missing-ball frames this frame belongs to. The
                  tracker already fills short gaps: on the delivered file there are no missing
                  runs between 50 and 124 frames, and 142 runs of 125+. A missing ball therefore
                  means a long gap, which mostly happens when the ball is out of play.
mean_speed        mean speed of the tracked players, smoothed. Players walk during stoppages.
mean_speed_long   the same over 10 s, for context either side of a short burst.
top_speed         mean speed of the TOP_N fastest players, smoothed. Someone is nearly always
                  running while the ball is alive.
sprint_share      share of tracked players faster than SPRINT_SPEED, smoothed.
n_outside         players more than OUTSIDE_MARGIN off the pitch, smoothed: throw-in and corner
                  takers, substitutions.
own_half_share    share of players in their own half; 1.0 is a kick-off line-up.
moving_share      share of tracked players faster than MOVING_SPEED, smoothed.
spot              where a still ball is: kick-off spot, corner, goal area, touchline (or none).
"""

import numpy as np

from games import GAME, half_video_blocks

# Params
FPS = 25
BALL_SPEED_HALF = 12          # frames either side for the 1 s ball displacement
SMOOTH_HALF = 25              # frames either side for the player-motion averages (2 s window)
LONG_HALF = 125               # frames either side for the slow player-motion average (10 s window)
KNOWN_HALF = 25
MOVING_SPEED = 2.0            # m/s; above a walk
SPRINT_SPEED = 4.0            # m/s; a run, which nobody needs while the ball is dead
TOP_N = 3                     # the fastest few players; one sprint is enough to mean "in play"
OUTSIDE_MARGIN = 1.0          # m beyond a line before a player counts as off the pitch
SPOT_RADIUS = 2.5             # m, around the centre spot
CORNER_RADIUS = 3.0           # m, around a corner flag
GOAL_AREA_DEPTH, GOAL_AREA_WIDTH = 5.5 + 1.0, 18.32 + 2.0   # six-yard box plus tracking slack
TOUCHLINE_BAND = 2.0          # m either side of a touchline

SPOT_NONE, SPOT_CENTRE, SPOT_CORNER, SPOT_GOAL_AREA, SPOT_TOUCHLINE = 0, 1, 2, 3, 4
SPOT_NAMES = {SPOT_NONE: "none", SPOT_CENTRE: "centre", SPOT_CORNER: "corner",
              SPOT_GOAL_AREA: "goal_area", SPOT_TOUCHLINE: "touchline"}


def rolling_mean(values, half):
    """Centred mean over +/- half frames, ignoring NaN. NaN where the window has no value."""
    valid = ~np.isnan(values)
    total = np.concatenate(([0.0], np.cumsum(np.where(valid, values, 0.0))))
    count = np.concatenate(([0], np.cumsum(valid)))
    index = np.arange(len(values))
    low = np.maximum(0, index - half)
    high = np.minimum(len(values), index + half + 1)
    n = count[high] - count[low]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (total[high] - total[low]) / np.maximum(n, 1), np.nan)


def run_lengths(mask):
    """For each frame, the length of the run of True it belongs to (0 where mask is False)."""
    out = np.zeros(len(mask), dtype=np.int64)
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    for start, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[start:stop] = stop - start
    return out


def ball_outside(ball_xy, pitch):
    """Metres outside the pitch rectangle; negative inside (distance to the nearest line)."""
    length, width = pitch
    x, y = ball_xy[:, 0], ball_xy[:, 1]
    beyond = np.maximum.reduce([-x, x - length, -y, y - width])
    return beyond


def ball_speed(ball_xy, half=BALL_SPEED_HALF):
    """|p(t+half) - p(t-half)| / time, where both ends are tracked."""
    n = len(ball_xy)
    out = np.full(n, np.nan)
    ahead = ball_xy[2 * half:]
    behind = ball_xy[:n - 2 * half]
    out[half:n - half] = np.hypot(*(ahead - behind).T) / (2.0 * half / FPS)
    return out


def still_spot(ball_xy, pitch):
    """Which restart location the ball is at, whether or not it is still."""
    length, width = pitch
    x, y = ball_xy[:, 0], ball_xy[:, 1]
    spot = np.full(len(x), SPOT_NONE, dtype=np.int8)
    known = ~np.isnan(x)

    touch = known & ((np.abs(y) < TOUCHLINE_BAND) | (np.abs(y - width) < TOUCHLINE_BAND) |
                     (y < 0) | (y > width))
    spot[touch] = SPOT_TOUCHLINE

    goal_x = np.minimum(np.abs(x), np.abs(length - x))
    goal_area = known & (goal_x < GOAL_AREA_DEPTH) & (np.abs(y - width / 2) < GOAL_AREA_WIDTH / 2)
    spot[goal_area] = SPOT_GOAL_AREA

    corner_d = np.hypot(goal_x, np.minimum(np.abs(y), np.abs(width - y)))
    spot[known & (corner_d < CORNER_RADIUS)] = SPOT_CORNER

    centre_d = np.hypot(x - length / 2, y - width / 2)
    spot[known & (centre_d < SPOT_RADIUS)] = SPOT_CENTRE
    return spot


def compute(data):
    """dict of per-frame feature arrays, all of length n_frames."""
    ball = data["ball"].astype(np.float64)
    players = data["players"].astype(np.float64)
    ball_xy = ball[:, :2]
    known = ~np.isnan(ball_xy[:, 0])
    # A frame with no players at all is outside graded play, so a ball gap stops at its edge.
    present = ~np.isnan(players[:, :, 0]).all(axis=1)

    speeds = players[:, :, 2]
    visible = ~np.isnan(players[:, :, 0])
    with np.errstate(invalid="ignore"):
        n_visible = visible.sum(axis=1)
        speed_known = ~np.isnan(speeds)
        mean_speed = np.where(speed_known.sum(axis=1) > 0,
                              np.nansum(speeds, axis=1) / np.maximum(speed_known.sum(axis=1), 1),
                              np.nan)
        moving = np.where(speed_known.sum(axis=1) > 0,
                          (speeds > MOVING_SPEED).sum(axis=1) / np.maximum(speed_known.sum(axis=1), 1),
                          np.nan)

    length, width = data["pitch"]
    px, py = players[:, :, 0], players[:, :, 1]
    with np.errstate(invalid="ignore"):
        ranked = np.sort(np.where(speed_known, speeds, -1.0), axis=1)[:, -TOP_N:]
        top_speed = np.where((ranked >= 0).all(axis=1), ranked.mean(axis=1), np.nan)
        sprint = np.where(speed_known.sum(axis=1) > 0,
                          (speeds > SPRINT_SPEED).sum(axis=1) / np.maximum(speed_known.sum(axis=1), 1),
                          np.nan)
        outside = ((px < -OUTSIDE_MARGIN) | (px > length + OUTSIDE_MARGIN) |
                   (py < -OUTSIDE_MARGIN) | (py > width + OUTSIDE_MARGIN)).sum(axis=1)

    return {
        "ball_known": known,
        "ball_oob": ball_outside(ball_xy, data["pitch"]),
        "ball_speed": ball_speed(ball_xy),
        "ball_known_share": rolling_mean(known.astype(np.float64), KNOWN_HALF),
        "missing_run": run_lengths(~known & present),
        "frame_present": present,
        "n_visible": n_visible,
        "mean_speed_raw": mean_speed,
        "mean_speed": rolling_mean(mean_speed, SMOOTH_HALF),
        "mean_speed_long": rolling_mean(mean_speed, LONG_HALF),
        "moving_share": rolling_mean(moving, SMOOTH_HALF),
        "top_speed": rolling_mean(top_speed, SMOOTH_HALF),
        "sprint_share": rolling_mean(sprint, SMOOTH_HALF),
        "n_outside": rolling_mean(outside.astype(np.float64), SMOOTH_HALF),
        "own_half_share": own_half_share(px, visible, data["team"], length),
        "spot": still_spot(ball_xy, data["pitch"]),
    }


def own_half_share(px, visible, team, length):
    """Share of tracked players standing in their own team's half. 1.0 is a kick-off line-up.

    Which end each team defends is read per half from where its players stand on average, so
    nothing about the direction of play is hard-coded. Halftime and the gaps outside the halves
    are given the first half's orientation; they are never graded.
    """
    out = np.full(len(px), np.nan)
    home = team == 0
    blocks = half_video_blocks(GAME)
    bounds = [(0, blocks[1][1])] + [(blocks[1][1], len(px))]
    for (start, stop), (_, first, last) in zip(bounds, blocks):
        sl = slice(first, last + 1)
        with np.errstate(invalid="ignore"):
            home_left = np.nanmean(px[sl][:, home]) < np.nanmean(px[sl][:, ~home])
        own_left = home if home_left else ~home      # slots whose own half is x < length / 2
        seg = px[start:stop]
        on_left = seg < length / 2
        own = np.where(own_left[None, :], on_left, ~on_left) & visible[start:stop]
        n_vis = visible[start:stop].sum(axis=1)
        out[start:stop] = np.where(n_vis > 0, own.sum(axis=1) / np.maximum(n_vis, 1), np.nan)
    return out

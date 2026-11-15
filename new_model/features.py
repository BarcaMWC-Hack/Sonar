"""Build the per-frame feature table from the tracking data.

This module is deliberately label-free.  It reads the ATD tracking file and the Smart Tagging
context (set-piece *timing and type* only, never the possession team), and returns

    X       (199319, n_features) float32, NaN where a feature cannot be computed
    names   feature names, in column order
    groups  family per column: ball, geo, shape, motion, time, sp, homo

The families
------------
ball     what the ball itself is doing: visible? out of play? how fast? where?
geo      geometry between the ball and the nearest players: who is closer, how close, how many
shape    team shape: width, depth, compactness, forward speed, players ahead of the ball
motion   how hard the players are running, and how many are off the pitch
time     rolling windows of the above, plus time since the ball/touch was last seen
sp       set-piece context: active, pre-restart, time to/from it, type
homo     homography quality flags (camera calibration, not tracking)

Everything is computed for all 199,319 video frames, including halftime and the ungraded edges.
Rows are video frames, matching the submission contract.

Usage
-----
    python new_model/features.py            # build and cache data/cache/BAR-ATH_features.npz
    python new_model/features.py --rebuild  # rebuild the ATD cache too
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import DATA_DIR, GAME, GAMES, half_video_blocks, require  # noqa: E402
from new_model import atd  # noqa: E402
from starter.load_smart_tagging import load as load_events  # noqa: E402

FPS = 25

# Ball
SHORT_HALF = 2                 # +/- frames for the instantaneous velocity
SPEED_HALF = 12                # +/- frames for the 1 s speed
TOUCH_LAG = 6                  # frames between the velocity before and after a touch
TOUCH_DV = 5.0                 # m/s change that counts as a touch
TOUCH_RADIUS = 1.5             # m; the touching player must be this close
KNOWN_HALF = 25                # +/- frames for the ball-known share
STILL_SPEED = 1.0              # m/s; below this the ball is effectively placed

# Geometry
CONTROL_RADIUS = 1.0           # m; at the feet
CONTROL_FALLOFF = 1.0          # m; evidence falls by e per metre beyond the radius
SIDE_SCALE = 1.5               # m; distance difference that counts as clearly one side
PLAYER_MAX_SPEED = 7.0         # m/s; for the time-to-ball proxy
RADII = (2.0, 5.0, 10.0)

# Shape / motion
SMOOTH_HALF = 25               # 2 s
LONG_HALF = 125                # 10 s
MOVING_SPEED = 2.0             # m/s
SPRINT_SPEED = 4.0             # m/s
OUTSIDE_MARGIN = 1.0           # m beyond a line before a player is off the pitch
PENALTY_DEPTH = 16.5           # m
PENALTY_HALF_WIDTH = 20.16     # m

# Set pieces
PRE_S = 8                      # seconds before a set-piece tag that are dead-ball evidence

# Spot codes
SPOT_NONE, SPOT_TOUCHLINE, SPOT_GOAL_AREA, SPOT_CORNER, SPOT_CENTRE = range(5)


# ── small numeric helpers ────────────────────────────────────────────────────
def rolling_mean(values, half):
    """Centred mean over +/- ``half`` frames, ignoring NaN. NaN where the window is empty."""
    values = np.asarray(values, dtype=np.float64)
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


def ball_velocity(ball_xy, half=SHORT_HALF):
    """(n, 2) ball velocity in m/s from positions ``half`` frames either side; NaN if unknown."""
    n = len(ball_xy)
    out = np.full((n, 2), np.nan)
    out[half:n - half] = (ball_xy[2 * half:] - ball_xy[:n - 2 * half]) / (2.0 * half / FPS)
    return out


def ball_speed(ball_xy, half=SPEED_HALF):
    """|p(t+half) - p(t-half)| / time, in m/s."""
    n = len(ball_xy)
    out = np.full(n, np.nan)
    out[half:n - half] = np.hypot(*(ball_xy[2 * half:] - ball_xy[:n - 2 * half]).T) / (
        2.0 * half / FPS)
    return out


def speed_change(ball_xy, lag=TOUCH_LAG):
    """Change in ball velocity over +/- ``lag`` frames, in m/s. A touch shows as a jump."""
    v = ball_velocity(ball_xy)
    n = len(v)
    out = np.full(n, np.nan)
    out[lag:n - lag] = np.hypot(*(v[2 * lag:] - v[:n - 2 * lag]).T)
    return out


def oriented_x(x, dir_sign):
    """x with the home side always attacking towards larger values.

    ``dir_sign`` is +1 when home attacks +x in this half, -1 when it attacks -x.  Pass a column
    vector when ``x`` is 2-D so the sign broadcasts over players.
    """
    length = GAMES[GAME]["pitch"][0]
    return np.where(np.asarray(dir_sign) > 0, x, length - x)


def _nanmean(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanstd(a, axis=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanstd(a, axis=axis)


def _finite(a):
    """Copy with infinities turned into NaN (distances to a team with no visible player)."""
    out = np.asarray(a, dtype=np.float64).copy()
    out[~np.isfinite(out)] = np.nan
    return out


# ── orientation and half index ───────────────────────────────────────────────
def half_orientation(data, game=GAME):
    """(dir_sign, half, frame_in_half), all length n.

    The tracking is pitch-fixed, so teams swap ends at halftime.  The direction is read from
    where each team actually stands in each half, not hard-coded from the metadata.
    """
    n = len(data["ball"])
    blocks = half_video_blocks(game)
    dir_sign = np.ones(n, dtype=np.float32)
    half = np.zeros(n, dtype=np.int8)
    frame_in_half = np.full(n, np.nan, dtype=np.float32)
    home = data["team"] == 0
    px = data["players"][:, :, 0]

    for h, first, last in blocks:
        sl = slice(first, last + 1)
        half[sl] = h
        frame_in_half[sl] = np.linspace(0.0, 1.0, last - first + 1)
        hx = _nanmean(px[sl][:, home])
        ax = _nanmean(px[sl][:, ~home])
        dir_sign[sl] = 1.0 if hx < ax else -1.0

    first_end = blocks[0][2]
    second_start = blocks[1][1]
    dir_sign[:second_start] = dir_sign[first_end]
    dir_sign[second_start:] = dir_sign[second_start]
    return dir_sign, half, frame_in_half


# ── ball ─────────────────────────────────────────────────────────────────────
def spot_codes(ball_xy, pitch):
    length, width = pitch
    x, y = ball_xy[:, 0], ball_xy[:, 1]
    known = ~np.isnan(x)
    code = np.full(len(x), SPOT_NONE, dtype=np.int8)

    touch = known & ((np.abs(y) < 2.0) | (np.abs(y - width) < 2.0) | (y < 0) | (y > width))
    code[touch] = SPOT_TOUCHLINE

    goal_x = np.minimum(np.abs(x), np.abs(length - x))
    goal_area = known & (goal_x < PENALTY_DEPTH - 10.0) & (
        np.abs(y - width / 2) < 10.16)
    code[goal_area] = SPOT_GOAL_AREA

    corner_d = np.hypot(goal_x, np.minimum(np.abs(y), np.abs(width - y)))
    code[known & (corner_d < 3.0)] = SPOT_CORNER

    centre_d = np.hypot(x - length / 2, y - width / 2)
    code[known & (centre_d < 2.5)] = SPOT_CENTRE
    return code


def ball_features(data, dir_sign):
    ball = data["ball"].astype(np.float64)
    bx, by = ball[:, 0], ball[:, 1]
    known = ~np.isnan(bx)
    length, width = data["pitch"]
    xy = ball[:, :2]

    oob = np.maximum.reduce([-bx, bx - length, -by, by - width])
    speed_1s = ball_speed(xy, SPEED_HALF)
    speed_short = ball_speed(xy, SHORT_HALF)
    dv = speed_change(xy)
    v = ball_velocity(xy)
    spot = spot_codes(xy, data["pitch"])
    progress = oriented_x(bx, dir_sign)

    f = {
        "ball_known": known.astype(np.float32),
        "ball_x": bx,
        "ball_y": by,
        "ball_progress": progress,
        "ball_oob_m": oob,
        "ball_speed_1s": speed_1s,
        "ball_speed_short": speed_short,
        "ball_speed_change": dv,
        "ball_known_share_2s": rolling_mean(known.astype(np.float64), KNOWN_HALF),
        "ball_missing_run": run_lengths(~known).astype(np.float64),
        "ball_still": np.where(np.isnan(speed_1s), np.nan,
                               (speed_1s < STILL_SPEED).astype(np.float64)),
        "ball_vx": v[:, 0],
        "ball_vy": v[:, 1],
        "ball_vx_progress": v[:, 0] * dir_sign,
        "ball_spot_touchline": np.where(known, (spot == SPOT_TOUCHLINE).astype(np.float64), np.nan),
        "ball_spot_goal_area": np.where(known, (spot == SPOT_GOAL_AREA).astype(np.float64), np.nan),
        "ball_spot_corner": np.where(known, (spot == SPOT_CORNER).astype(np.float64), np.nan),
        "ball_spot_centre": np.where(known, (spot == SPOT_CENTRE).astype(np.float64), np.nan),
        # Zones are oriented to the home side's attacking direction, so they are comparable
        # across halves.  They say where the ball is, not who has it.
        "ball_defensive_third": np.where(known, (progress < length / 3).astype(np.float64), np.nan),
        "ball_middle_third": np.where(
            known, ((progress >= length / 3) & (progress <= 2 * length / 3)).astype(np.float64),
            np.nan),
        "ball_final_third": np.where(
            known, (progress > 2 * length / 3).astype(np.float64), np.nan),
        "ball_penalty_box": np.where(
            known, ((progress > length - PENALTY_DEPTH)
                    & (np.abs(by - width / 2) < PENALTY_HALF_WIDTH)).astype(np.float64), np.nan),
        "ball_six_yard": np.where(
            known, ((progress > length - 5.5)
                    & (np.abs(by - width / 2) < 9.16)).astype(np.float64), np.nan),
        "ball_dist_to_goal": length - progress,
    }
    return f


# ── ball-player geometry ─────────────────────────────────────────────────────
def geometry_features(data, chunk=20000):
    n = len(data["ball"])
    home = data["team"] == 0
    ball_xy = data["ball"][:, :2].astype(np.float64)
    players_xy = data["players"][:, :, :2].astype(np.float64)
    ball_known = ~np.isnan(ball_xy[:, 0])

    keys = (
        "geo_d_home", "geo_d_away", "geo_d_nearest", "geo_d_diff",
        "geo_control_home", "geo_control_away", "geo_control_diff",
        "geo_n_home_2m", "geo_n_away_2m", "geo_n_home_5m", "geo_n_away_5m",
        "geo_n_home_10m", "geo_n_away_10m", "geo_home_2nd", "geo_away_2nd",
        "geo_home_mean5", "geo_away_mean5", "geo_time_home", "geo_time_away",
        "geo_time_diff", "geo_soft_home_10", "geo_soft_away_10",
        "geo_soft_home_20", "geo_soft_away_20", "geo_pressure_diff_10",
        "geo_pressure_diff_20",
    )
    out = {k: np.full(n, np.nan, dtype=np.float64) for k in keys}

    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        b = ball_xy[start:stop]
        p = players_xy[start:stop]
        diff = p - b[:, None, :]
        d = np.sqrt((diff * diff).sum(axis=2))
        d = np.where(np.isnan(d), np.inf, d)
        dh = d[:, home]
        da = d[:, ~home]

        out["geo_n_home_2m"][start:stop] = (dh < 2.0).sum(axis=1)
        out["geo_n_away_2m"][start:stop] = (da < 2.0).sum(axis=1)
        out["geo_n_home_5m"][start:stop] = (dh < 5.0).sum(axis=1)
        out["geo_n_away_5m"][start:stop] = (da < 5.0).sum(axis=1)
        out["geo_n_home_10m"][start:stop] = (dh < 10.0).sum(axis=1)
        out["geo_n_away_10m"][start:stop] = (da < 10.0).sum(axis=1)
        out["geo_soft_home_10"][start:stop] = np.exp(-dh / 10.0).sum(axis=1)
        out["geo_soft_away_10"][start:stop] = np.exp(-da / 10.0).sum(axis=1)
        out["geo_soft_home_20"][start:stop] = np.exp(-dh / 20.0).sum(axis=1)
        out["geo_soft_away_20"][start:stop] = np.exp(-da / 20.0).sum(axis=1)

        dh.sort(axis=1)
        da.sort(axis=1)
        out["geo_d_home"][start:stop] = dh[:, 0]
        out["geo_d_away"][start:stop] = da[:, 0]
        out["geo_home_2nd"][start:stop] = dh[:, 1]
        out["geo_away_2nd"][start:stop] = da[:, 1]
        out["geo_home_mean5"][start:stop] = _nanmean(_finite(dh[:, :5]), axis=1)
        out["geo_away_mean5"][start:stop] = _nanmean(_finite(da[:, :5]), axis=1)

    # Turn "no visible player" (inf) into NaN before doing arithmetic on distances.
    for k in ("geo_d_home", "geo_d_away", "geo_home_2nd", "geo_away_2nd",
              "geo_home_mean5", "geo_away_mean5"):
        out[k] = _finite(out[k])

    out["geo_d_nearest"] = np.fmin(out["geo_d_home"], out["geo_d_away"])
    out["geo_d_diff"] = out["geo_d_away"] - out["geo_d_home"]
    out["geo_control_home"] = np.where(
        np.isfinite(out["geo_d_home"]),
        np.exp(-np.maximum(0.0, out["geo_d_home"] - CONTROL_RADIUS) / CONTROL_FALLOFF), 0.0)
    out["geo_control_away"] = np.where(
        np.isfinite(out["geo_d_away"]),
        np.exp(-np.maximum(0.0, out["geo_d_away"] - CONTROL_RADIUS) / CONTROL_FALLOFF), 0.0)
    out["geo_control_diff"] = out["geo_control_home"] - out["geo_control_away"]
    out["geo_time_home"] = out["geo_d_home"] / PLAYER_MAX_SPEED
    out["geo_time_away"] = out["geo_d_away"] / PLAYER_MAX_SPEED
    out["geo_time_diff"] = out["geo_time_away"] - out["geo_time_home"]
    out["geo_pressure_diff_10"] = out["geo_soft_home_10"] - out["geo_soft_away_10"]
    out["geo_pressure_diff_20"] = out["geo_soft_home_20"] - out["geo_soft_away_20"]

    # Without a ball every geometry feature is undefined.
    for k in keys:
        out[k][~ball_known] = np.nan
    return out


# ── team shape ───────────────────────────────────────────────────────────────
def shape_features(data, dir_sign):
    players = data["players"].astype(np.float64)
    px, py = players[:, :, 0], players[:, :, 1]
    home = data["team"] == 0
    length, width = data["pitch"]
    ball_x = data["ball"][:, 0].astype(np.float64)
    ball_known = ~np.isnan(ball_x)

    home_prog = oriented_x(px, dir_sign[:, None])
    away_prog = oriented_x(px, -dir_sign[:, None])
    ball_prog_home = oriented_x(ball_x, dir_sign)
    ball_prog_away = oriented_x(ball_x, -dir_sign)

    vis_home = ~np.isnan(px[:, home])
    vis_away = ~np.isnan(px[:, ~home])
    n_home = vis_home.sum(axis=1)
    n_away = vis_away.sum(axis=1)

    hx, hy = _nanmean(px[:, home], axis=1), _nanmean(py[:, home], axis=1)
    ax, ay = _nanmean(px[:, ~home], axis=1), _nanmean(py[:, ~home], axis=1)
    hprog = _nanmean(home_prog[:, home], axis=1)
    aprog = _nanmean(away_prog[:, ~home], axis=1)

    h_sx, h_sy = _nanstd(px[:, home], axis=1), _nanstd(py[:, home], axis=1)
    a_sx, a_sy = _nanstd(px[:, ~home], axis=1), _nanstd(py[:, ~home], axis=1)
    h_compact = _nanmean(np.hypot(px[:, home] - hx[:, None], py[:, home] - hy[:, None]), axis=1)
    a_compact = _nanmean(np.hypot(px[:, ~home] - ax[:, None], py[:, ~home] - ay[:, None]), axis=1)

    h_own = np.where(n_home > 0,
                     ((home_prog[:, home] < length / 2) & vis_home).sum(axis=1)
                     / np.maximum(n_home, 1), np.nan)
    a_own = np.where(n_away > 0,
                     ((away_prog[:, ~home] < length / 2) & vis_away).sum(axis=1)
                     / np.maximum(n_away, 1), np.nan)

    h_ahead = ((home_prog[:, home] > ball_prog_home[:, None]) & vis_home).sum(axis=1)
    a_ahead = ((away_prog[:, ~home] > ball_prog_away[:, None]) & vis_away).sum(axis=1)
    h_box = ((home_prog[:, home] > length - PENALTY_DEPTH)
             & (np.abs(py[:, home] - width / 2) < PENALTY_HALF_WIDTH) & vis_home).sum(axis=1)
    a_box = ((away_prog[:, ~home] > length - PENALTY_DEPTH)
             & (np.abs(py[:, ~home] - width / 2) < PENALTY_HALF_WIDTH) & vis_away).sum(axis=1)

    f = {
        "shape_home_cx": hprog,
        "shape_home_cy": hy,
        "shape_away_cx": aprog,
        "shape_away_cy": ay,
        "shape_cx_diff": hprog - aprog,
        "shape_cy_diff": hy - ay,
        "shape_home_sx": h_sx,
        "shape_home_sy": h_sy,
        "shape_away_sx": a_sx,
        "shape_away_sy": a_sy,
        "shape_sx_diff": h_sx - a_sx,
        "shape_sy_diff": h_sy - a_sy,
        "shape_home_fwd": np.gradient(hprog) * FPS,
        "shape_away_fwd": np.gradient(aprog) * FPS,
        "shape_home_compact": h_compact,
        "shape_away_compact": a_compact,
        "shape_home_own_half": h_own,
        "shape_away_own_half": a_own,
        "shape_n_home": n_home.astype(np.float64),
        "shape_n_away": n_away.astype(np.float64),
        "shape_home_ahead": h_ahead.astype(np.float64),
        "shape_away_ahead": a_ahead.astype(np.float64),
        "shape_home_in_box": h_box.astype(np.float64),
        "shape_away_in_box": a_box.astype(np.float64),
    }
    for k in ("shape_home_ahead", "shape_away_ahead", "shape_home_in_box", "shape_away_in_box"):
        f[k][~ball_known] = np.nan
    return f


# ── player motion ────────────────────────────────────────────────────────────
def motion_features(data):
    players = data["players"].astype(np.float64)
    speeds = players[:, :, 2]
    visible = ~np.isnan(players[:, :, 0])
    home = data["team"] == 0
    length, width = data["pitch"]

    known = ~np.isnan(speeds)
    n_known = known.sum(axis=1)
    mean_speed = np.where(n_known > 0, np.nansum(speeds, axis=1) / np.maximum(n_known, 1), np.nan)
    home_mean = np.where(known[:, home].sum(axis=1) > 0,
                         np.nansum(speeds[:, home], axis=1)
                         / np.maximum(known[:, home].sum(axis=1), 1), np.nan)
    away_mean = np.where(known[:, ~home].sum(axis=1) > 0,
                         np.nansum(speeds[:, ~home], axis=1)
                         / np.maximum(known[:, ~home].sum(axis=1), 1), np.nan)

    top3 = np.sort(np.where(known, speeds, -1.0), axis=1)[:, -3:]
    top3 = np.where((top3 >= 0).all(axis=1), top3.mean(axis=1), np.nan)

    moving = np.where(n_known > 0, (speeds > MOVING_SPEED).sum(axis=1)
                      / np.maximum(n_known, 1), np.nan)
    sprint = np.where(n_known > 0, (speeds > SPRINT_SPEED).sum(axis=1)
                      / np.maximum(n_known, 1), np.nan)
    home_sprint = np.where(known[:, home].sum(axis=1) > 0,
                           (speeds[:, home] > SPRINT_SPEED).sum(axis=1)
                           / np.maximum(known[:, home].sum(axis=1), 1), np.nan)
    away_sprint = np.where(known[:, ~home].sum(axis=1) > 0,
                           (speeds[:, ~home] > SPRINT_SPEED).sum(axis=1)
                           / np.maximum(known[:, ~home].sum(axis=1), 1), np.nan)

    px, py = players[:, :, 0], players[:, :, 1]
    outside = ((px < -OUTSIDE_MARGIN) | (px > length + OUTSIDE_MARGIN)
               | (py < -OUTSIDE_MARGIN) | (py > width + OUTSIDE_MARGIN)).sum(axis=1)

    f = {
        "motion_mean_speed": mean_speed,
        "motion_mean_speed_2s": rolling_mean(mean_speed, SMOOTH_HALF),
        "motion_mean_speed_10s": rolling_mean(mean_speed, LONG_HALF),
        "motion_home_speed": home_mean,
        "motion_away_speed": away_mean,
        "motion_top3_2s": rolling_mean(top3, SMOOTH_HALF),
        "motion_moving_2s": rolling_mean(moving, SMOOTH_HALF),
        "motion_sprint_2s": rolling_mean(sprint, SMOOTH_HALF),
        "motion_home_sprint": home_sprint,
        "motion_away_sprint": away_sprint,
        "motion_n_outside_2s": rolling_mean(outside.astype(np.float64), SMOOTH_HALF),
    }
    return f


# ── temporal context ─────────────────────────────────────────────────────────
def temporal_features(f, half, frame_in_half):
    n = len(f["ball_known"])
    ball_known = f["ball_known"] > 0.5
    index = np.arange(n)

    last_known = np.maximum.accumulate(np.where(ball_known, index, -1))
    since_known = np.where(last_known >= 0, index - last_known, np.nan)
    next_known = np.minimum.accumulate(np.where(ball_known, index, n)[::-1])[::-1]
    until_known = np.where(next_known < n, next_known - index, np.nan)

    touch = (f["ball_speed_change"] > TOUCH_DV) & (f["geo_d_nearest"] < TOUCH_RADIUS)
    touch = np.nan_to_num(touch, nan=0.0).astype(bool)
    last_touch = np.maximum.accumulate(np.where(touch, index, -1))
    since_touch = np.where(last_touch >= 0, index - last_touch, np.nan)

    return {
        "time_ball_speed_3s": rolling_mean(f["ball_speed_1s"], 75),
        "time_ball_speed_10s": rolling_mean(f["ball_speed_1s"], 250),
        "time_d_nearest_2s": rolling_mean(f["geo_d_nearest"], 50),
        "time_control_2s": rolling_mean(f["geo_control_diff"], 50),
        "time_control_10s": rolling_mean(f["geo_control_diff"], 250),
        "time_sy_diff_2s": rolling_mean(f["shape_sy_diff"], 50),
        "time_since_ball_known": np.minimum(since_known, 250),
        "time_until_ball_known": np.minimum(until_known, 250),
        "time_since_touch": np.minimum(since_touch, 250),
        "time_touch_rate_5s": rolling_mean(touch.astype(np.float64), 62),
        "time_half": half.astype(np.float64),
        "time_frame_in_half": frame_in_half,
        "time_ball_known_rate_10s": rolling_mean(ball_known.astype(np.float64), 250),
    }


# ── set-piece context ────────────────────────────────────────────────────────
SP_TYPE_KEYS = ("throw_in", "corner", "goal_kick", "free_kick", "kickoff", "other")


def _set_type(tags):
    value = (tags.get("Type") or "").strip().lower().replace("-", " ")
    if "throw" in value:
        return "throw_in"
    if "corner" in value:
        return "corner"
    if "goal kick" in value:
        return "goal_kick"
    if "free kick" in value:
        return "free_kick"
    if "kick off" in value:
        return "kickoff"
    return "other"


def setpiece_features(events, n):
    active = np.zeros(n, dtype=np.float32)
    pre = np.zeros(n, dtype=np.float32)
    onehot = {k: np.zeros(n, dtype=np.float32) for k in SP_TYPE_KEYS}
    starts, ends = [], []

    for event in events:
        if (event["code"] or "").strip().lower() != "set pieces":
            continue
        a = max(0, event["start_frame"])
        b = max(0, min(n, event["end_frame"]))
        if b <= a:
            continue
        active[a:b] = 1.0
        pre[max(0, a - PRE_S * FPS):a] = 1.0
        onehot[_set_type(event["tags"])][a:b] = 1.0
        starts.append(a)
        ends.append(b)

    index = np.arange(n)
    start_at = np.full(n, -1, dtype=np.int64)
    end_at = np.full(n, n, dtype=np.int64)
    for a in starts:
        start_at[a] = a
    for b in ends:
        end_at[b] = b
    last_start = np.maximum.accumulate(start_at)
    next_end = np.minimum.accumulate(end_at[::-1])[::-1]
    since_start = np.where(last_start >= 0, index - last_start, np.nan)
    until_end = np.where(next_end < n, next_end - index, np.nan)

    start_indicator = np.zeros(n, dtype=np.float64)
    start_indicator[starts] = 1.0

    f = {
        "sp_active": active,
        "sp_pre": pre,
        "sp_since_start": np.minimum(since_start, 250),
        "sp_until_end": np.minimum(until_end, 250),
        "sp_count_10s": rolling_mean(start_indicator, 125) * (2 * 125 + 1),
    }
    for key in SP_TYPE_KEYS:
        f["sp_" + key] = onehot[key]
    return f


# ── homography quality ───────────────────────────────────────────────────────
def homography_features(n, game=GAME):
    path = require(["homography"], game)[0]
    table = pd.read_csv(path)
    frame = table["frame"].to_numpy()
    error = np.full(n, np.nan)
    valid = np.zeros(n)
    error[frame] = table["reprojection_error"].to_numpy()
    valid[frame] = table["valid"].to_numpy()
    return {
        "homo_valid": valid,
        "homo_error": error,
        "homo_error_log": np.log1p(error),
        "homo_error_2s": rolling_mean(error, 50),
    }


# ── assembly ─────────────────────────────────────────────────────────────────
def build(data=None, events=None, game=GAME):
    """(X, names, groups) for every video frame."""
    if data is None:
        data = atd.load(game)
    if events is None:
        events = load_events(game)

    n = len(data["ball"])
    dir_sign, half, frame_in_half = half_orientation(data, game)

    f = {}
    f.update(ball_features(data, dir_sign))
    f.update(geometry_features(data))
    f.update(shape_features(data, dir_sign))
    f.update(motion_features(data))
    f.update(temporal_features(f, half, frame_in_half))
    f.update(setpiece_features(events, n))
    f.update(homography_features(n, game))

    names = list(f.keys())
    groups = [name.split("_", 1)[0] for name in names]
    X = np.column_stack([f[name] for name in names]).astype(np.float32)
    return X, names, groups


def save(X, names, groups, game=GAME, data_dir=None):
    path = os.path.join(data_dir or DATA_DIR, "cache", "{}_features.npz".format(game))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=X.astype(np.float32),
                        names=np.array(names), groups=np.array(groups))
    return path


def load_saved(game=GAME, data_dir=None):
    path = os.path.join(data_dir or DATA_DIR, "cache", "{}_features.npz".format(game))
    cached = np.load(path)
    return cached["X"], list(cached["names"]), list(cached["groups"])


if __name__ == "__main__":
    import time

    start = time.time()
    data = atd.load(rebuild="--rebuild" in sys.argv)
    X, names, groups = build(data)
    path = save(X, names, groups)
    print("built {} x {} in {:.1f} s".format(*X.shape, time.time() - start))
    print("cached -> {}".format(os.path.relpath(path, ROOT)))
    print("\n{:<8} {:>5}  {:>9}".format("group", "n", "mean NaN%"))
    for group in dict.fromkeys(groups):
        cols = [j for j, g in enumerate(groups) if g == group]
        nan_share = np.isnan(X[:, cols]).mean() * 100.0
        print("{:<8} {:>5}  {:>8.1f}%".format(group, len(cols), nan_share))

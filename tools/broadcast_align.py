"""Feasibility test: can the broadcast tracking be aligned to the tactical timeline, label-free?

    python tools/broadcast_align.py

Both streams are automatic tracking in pitch coordinates. For each 30 s broadcast window we search
for the tactical time offset where the two BALL tracks agree best (coarse at 1 Hz over the whole
match, then refined at 25 Hz). A window is accepted only if its best match is tight (median ball
distance under 1.5 m), clearly better than any other offset more than 10 s away, and confirmed by
the PLAYERS (median distance from each visible broadcast player to the nearest tactical player of
the same team under 2 m), which were not used to find the offset.

Gate (fixed in advance): at least 70% of the broadcast windows that have a ball are accepted.
Then we report how many tactical frames with no ball could get one from an accepted window.
Uses no labels; prints statistics only. Nothing leaves this machine.
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from sonar import atd  # noqa: E402
from starter.baseline import team_of_each_track  # noqa: E402

FPS = 25
WINDOW_S = 30


def load_stream(source):
    meta, raw = require(["atd_meta", "atd_raw"], GAME, None, source)
    with open(raw, encoding="utf8") as handle:
        slots = len(handle.readline().split(":")[1].split(";"))
    saved = atd.N_SLOTS
    try:
        atd.N_SLOTS = slots
        flat = atd._parse(raw, source_frames(source, GAME))
        length, width = GAMES[GAME]["pitch"]
        data = atd._assemble(flat, team_of_each_track(meta), length, width)
    finally:
        atd.N_SLOTS = saved
    return data


def cost_curve(b, t, offsets, min_overlap):
    """Median ball distance between broadcast samples b and tactical samples shifted by each offset."""
    out = np.full(len(offsets), np.inf)
    idx = np.arange(len(b))
    ok_b = np.isfinite(b[:, 0])
    for k, o in enumerate(offsets):
        j = idx + o
        valid = ok_b & (j >= 0) & (j < len(t))
        jj = j[valid]
        both = np.isfinite(t[jj, 0])
        if both.sum() < min_overlap:
            continue
        d = np.sqrt(((b[valid][both] - t[jj][both]) ** 2).sum(axis=1))
        out[k] = np.median(d)
    return out


def main():
    tac = load_stream("tactical")
    brd = load_stream("broadcast")
    tb, bb = tac["ball"][:, :2].astype(np.float64), brd["ball"][:, :2].astype(np.float64)
    print("frames: tactical {}, broadcast {}; ball seen: tactical {:.1%}, broadcast {:.1%}".format(
        len(tb), len(bb), np.isfinite(tb[:, 0]).mean(), np.isfinite(bb[:, 0]).mean()), flush=True)

    t1, b1 = tb[::FPS], bb[::FPS]                       # 1 Hz
    w = WINDOW_S
    windows = [(s, s + w) for s in range(0, len(b1) - w, w)]
    accepted, with_ball = [], 0
    for s, e in windows:
        seg = b1[s:e]
        if np.isfinite(seg[:, 0]).sum() < w // 2:
            continue
        with_ball += 1
        offsets = np.arange(-s, len(t1) - s)
        curve = cost_curve(seg, t1, offsets, min_overlap=w // 3)
        best = int(np.argmin(curve))
        if not np.isfinite(curve[best]):
            continue
        far = np.abs(offsets - offsets[best]) > 10
        second = np.min(curve[far]) if np.any(np.isfinite(curve[far])) else np.inf
        # cost_curve offsets map a window-local index to an absolute tactical index, so the
        # broadcast-to-tactical shift is that offset minus the window start.
        coarse = (offsets[best] - s) * FPS
        # Refine at 25 Hz within +/- 2 s of the coarse offset.
        fs, fe = s * FPS, e * FPS
        fine_offsets = np.arange(coarse - 2 * FPS, coarse + 2 * FPS + 1)
        fine = np.full(len(fine_offsets), np.inf)
        bseg = bb[fs:fe]
        for k, o in enumerate(fine_offsets):
            j = np.arange(fs, fe) + o
            valid = (j >= 0) & (j < len(tb)) & np.isfinite(bseg[:, 0])
            jj = j[valid]
            both = np.isfinite(tb[jj, 0])
            if both.sum() < w * FPS // 3:
                continue
            fine[k] = np.median(np.sqrt(((bseg[valid][both] - tb[jj][both]) ** 2).sum(axis=1)))
        kbest = int(np.argmin(fine))
        offset, resid = int(fine_offsets[kbest]), float(fine[kbest])
        # Independent check with players (not used to choose the offset), every 5th frame.
        pd = []
        for f in range(fs, fe, 5):
            g = f + offset
            if not (0 <= g < len(tb)):
                continue
            for code in (0, 1):
                bp = brd["players"][f, brd["team"] == code, :2]
                tp = tac["players"][g, tac["team"] == code, :2]
                bp, tp = bp[np.isfinite(bp[:, 0])], tp[np.isfinite(tp[:, 0])]
                if len(bp) and len(tp):
                    d = np.sqrt(((bp[:, None, :] - tp[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
                    pd.extend(d.tolist())
        player_med = float(np.median(pd)) if pd else np.inf
        ok = resid < 1.5 and second > 2 * max(resid, 0.5) and player_med < 2.0
        if ok:
            accepted.append((fs, fe, offset, resid, player_med))
    print("broadcast windows with a ball: {}; accepted: {} ({:.0%})".format(
        with_ball, len(accepted), len(accepted) / max(with_ball, 1)), flush=True)
    if accepted:
        offs = np.array([a[2] for a in accepted])
        print("accepted offsets (s): min {:.1f}, median {:.1f}, max {:.1f}; distinct to the second: {}".format(
            offs.min() / FPS, np.median(offs) / FPS, offs.max() / FPS, len(np.unique(np.round(offs / FPS)))))
        print("median ball residual {:.2f} m; median player residual {:.2f} m".format(
            float(np.median([a[3] for a in accepted])), float(np.median([a[4] for a in accepted]))))
    graded = np.zeros(len(tb), dtype=bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    missing = graded & ~np.isfinite(tb[:, 0])
    fill = np.zeros(len(tb), dtype=bool)
    for fs, fe, o, _, _ in accepted:
        j = np.arange(fs, fe) + o
        valid = (j >= 0) & (j < len(tb)) & np.isfinite(bb[fs:fe, 0])
        fill[j[valid]] = True
    print("graded tactical frames with no ball: {}; of these, an aligned broadcast ball exists for {} ({:.1%})".format(
        int(missing.sum()), int((missing & fill).sum()), (missing & fill).sum() / max(missing.sum(), 1)))
    gate = len(accepted) / max(with_ball, 1) >= 0.70
    print("GATE (>= 70% of ball windows accepted):", "PASS" if gate else "FAIL")


if __name__ == "__main__":
    main()

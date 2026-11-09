"""How well does each candidate dead-ball feature separate dead from alive? Shares only.

Prints, for several groups of frames, the percentage of frames on which each candidate condition
holds. No coordinate, speed or row is ever printed, so the output is safe to share anywhere.

Groups:
  public_alive / public_dead  the 5-minute public ground truth (a correctness check, small sample)
  prekick_2h                  110808..113741, graded and all D per FAQ.md
  pre_setpiece                8 s before each Smart Tagging SET PIECES start (weak: likely dead)
  in_setpiece                 inside SET PIECES intervals (the restart happens in here)
  open_play                   inside open-play Smart Tagging phases, away from set pieces
                              (weak: likely alive)
  graded                      every graded frame

    python tools/deadball_diagnostics.py
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from sonar import atd, deadball, deadball_features as dbf  # noqa: E402
from sonar.deadball import FPS, PRE_SETPIECE_S, graded_mask, interval_mask  # noqa: E402
from starter import load_smart_tagging  # noqa: E402

# Params
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def groups(n, graded, events):
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    setpieces = [e for e in events if deadball.code_of(e) == "set pieces"]
    pre = interval_mask(n, [(e["start_frame"] - PRE_SETPIECE_S * FPS, e["start_frame"])
                            for e in setpieces])
    inside = interval_mask(n, [(e["start_frame"], e["end_frame"]) for e in setpieces])
    _, play = deadball.weak_labels(n, events)
    return [
        ("public_alive", (gt == pm.H) | (gt == pm.A)),
        ("public_dead", gt == pm.D),
        ("prekick_2h", deadball.known_dead_mask(n)),
        ("pre_setpiece", pre & graded),
        ("in_setpiece", inside & graded),
        ("open_play", play),
        ("graded", graded),
    ], gt, setpieces


def conditions(f):
    with np.errstate(invalid="ignore"):
        return [
            ("known", f["ball_known"]),
            ("oob>0.5", f["ball_oob"] > 0.5),
            ("oob>2", f["ball_oob"] > 2.0),
            ("bs<0.5", f["ball_speed"] < 0.5),
            ("bs<2", f["ball_speed"] < 2.0),
            ("bs>8", f["ball_speed"] > 8.0),
            ("kn<30%", f["ball_known_share"] < 0.3),
            ("miss>2s", f["missing_run"] > 2 * FPS),
            ("miss>5s", f["missing_run"] > 5 * FPS),
            ("ms<1.5", f["mean_speed"] < 1.5),
            ("ms<2.0", f["mean_speed"] < 2.0),
            ("ms<2.5", f["mean_speed"] < 2.5),
            ("mv<20%", f["moving_share"] < 0.2),
            ("mv<35%", f["moving_share"] < 0.35),
            ("nv<18", f["n_visible"] < 18),
            ("centre", f["spot"] == dbf.SPOT_CENTRE),
            ("corner", f["spot"] == dbf.SPOT_CORNER),
            ("g_area", f["spot"] == dbf.SPOT_GOAL_AREA),
            ("touch", f["spot"] == dbf.SPOT_TOUCHLINE),
        ]


def tag_alignment(gt, setpieces):
    """Counts only: where, relative to each tagged set piece, does the public GT go dead->alive?"""
    window = np.flatnonzero(gt != pm.X)
    first, last = window[0], window[-1]
    restarts = [i for i in range(first + 1, last + 1)
                if gt[i - 1] == pm.D and gt[i] in (pm.H, pm.A)]
    tags = [e for e in setpieces if first <= e["start_frame"] <= last]
    bins = {"before start-5s": 0, "start-5s..start": 0, "inside tag": 0, "end..end+5s": 0,
            "none within 5s": 0}
    for e in tags:
        s, t = e["start_frame"], e["end_frame"]
        near = [r for r in restarts if s - 5 * FPS <= r < t + 5 * FPS]
        if not near:
            bins["none within 5s"] += 1
        elif any(s <= r < t for r in near):
            bins["inside tag"] += 1
        elif any(r < s for r in near):
            bins["start-5s..start"] += 1
        else:
            bins["end..end+5s"] += 1
    print("\nPublic window: {} dead->alive restarts in GT, {} SET PIECES tags start inside it".format(
        len(restarts), len(tags)))
    print("  nearest restart relative to each tag: {}".format(
        {k: v for k, v in bins.items() if k != "before start-5s"}))
    tagged_restarts = sum(any(e["start_frame"] - 5 * FPS <= r < e["end_frame"] + 5 * FPS
                              for e in tags) for r in restarts)
    print("  GT restarts with a SET PIECES tag within 5 s: {} of {}".format(
        tagged_restarts, len(restarts)))


PUBLIC_CHECK_MESSAGE = (
    "This tool reads the public 5-minute ground truth. RULES.md: those minutes are for checking "
    "that the pipeline is wired correctly, never for training, tuning or choosing between "
    "settings. Run it only to confirm the pipeline works, with --public-check, and do not act "
    "on its numbers.")


def require_public_check_flag():
    import sys
    if "--public-check" not in sys.argv:
        sys.exit(PUBLIC_CHECK_MESSAGE)
    sys.argv.remove("--public-check")


if __name__ == "__main__":
    require_public_check_flag()
    data = atd.load()
    n = len(data["ball"])
    features = dbf.compute(data)
    events = load_smart_tagging.load()
    graded = graded_mask(n)
    group_list, gt, setpieces = groups(n, graded, events)
    conds = conditions(features)

    names = [name for name, _ in conds]
    print("% of frames in each group where the condition holds\n")
    print("{:<13}{:>8}  ".format("group", "frames") + "".join("{:>8}".format(c) for c in names))
    for gname, gmask in group_list:
        size = int(gmask.sum())
        row = "".join("{:>8.1f}".format(100.0 * np.sum(cmask & gmask) / max(size, 1))
                      for _, cmask in conds)
        print("{:<13}{:>8}  ".format(gname, size) + row)

    tag_alignment(gt, setpieces)
    types = {}
    for e in setpieces:
        types[e["tags"].get("Type", "?")] = types.get(e["tags"].get("Type", "?"), 0) + 1
    print("\nSET PIECES by Type: {}".format(types))

"""Honest yardstick: the full production pipeline vs the official baselines on Sportec (DFL) games.

    python tools/dfl_yardstick.py [--smoke] [--out dfl_yardstick.csv]      # needs internet + kloppy

Why: every earlier other-game number took the dead-ball frames from the truth and used clean
tracking. Here the truth is DFL's own frame-level ball status and ball possession (derived from
corrected tracking, the same kind of truth as the competition's), the pipeline predicts its own dead
periods, and the tracking is degraded to look like the competition match. All numbers are sim_.

Fixed before any number was seen:
  noise   (calibrated to label-free statistics of the competition match: 40.7% ball missing, almost
          all in gaps of 5 s or more, ball jitter ~0.0116 m median second difference, 20.5 of 22
          players tracked)
          - ball dropped for each true dead stretch of 2 s+ with probability 0.9, from up to 2 s
            after its start to up to 1 s before its end
          - live gaps of 5-20 s (lognormal, median 7 s) until 8% of live frames are missing
          - short live gaps of 0.2-2 s, one per minute, filled by straight-line interpolation (the
            competition tracker fills gaps under 5 s)
          - Gaussian ball jitter sigma 0.0037 m; each player dropped in 10 s windows with p 0.068
          - not simulated: identity swaps, duplicates, homography error
  tagger  (stands in for Metrica Smart Tagging, derived from the truth with noise, so sim_)
          - a SET PIECES interval at 55% of true restarts, start jittered +/-3 s, 5 s long
          - open-play intervals over 70% of true live stretches of 6 s+, trimmed 1 s, with the
            possessing team (wrong team 15% of the time)
  methods the production pipeline (dead-ball detector, possession, first-touch placement, no graded
          X) and the official nearest-player baselines at +/-25, 50 and 125 frames; plus two
          diagnostics: our possession with TRUE dead periods, and our dead periods with TRUE team.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import external_eval as ee  # noqa: E402
import external_full as ef  # noqa: E402  (patches the BAR-ATH half blocks)
import possession_metrics as pm  # noqa: E402
from sonar import deadball, possession, possession_baseline as pb, team_shape  # noqa: E402,F401
from sonar import make_submission as ms  # noqa: E402
from starter import baseline as official  # noqa: E402

FPS = 25
MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
team_shape.half_video_blocks = ef._blocks
ms.half_video_blocks = ef._blocks


def runs_of(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def load(match_id):
    """(data with x, y, speed per slot, truth labels, period per frame)."""
    from kloppy import sportec
    ds = sportec.load_open_tracking_data(match_id=match_id, only_alive=False,
                                         coordinates="metrica")
    meta = ds.metadata
    df = ds.to_df() if hasattr(ds, "to_df") else ds.to_pandas()
    n = len(df)
    length = getattr(meta.pitch_dimensions, "pitch_length", None) or 105.0
    width = getattr(meta.pitch_dimensions, "pitch_width", None) or 68.0
    home_id = meta.teams[0].team_id
    team_of = {}
    for t, code in ((meta.teams[0], 0), (meta.teams[1], 1)):
        for p in t.players:
            team_of[p.player_id] = code
    cols = [c[:-2] for c in df.columns if c.endswith("_x") and c[:-2] in team_of]
    xy = np.full((n, len(cols), 2), np.nan)
    for j, pid in enumerate(cols):
        xy[:, j, 0] = df[pid + "_x"].to_numpy(dtype=float) * length
        xy[:, j, 1] = df[pid + "_y"].to_numpy(dtype=float) * width
    ball = np.stack([df["ball_x"].to_numpy(dtype=float) * length,
                     df["ball_y"].to_numpy(dtype=float) * width], axis=1)
    state_col = next(c for c in df.columns if c.startswith("ball_state"))
    owner_col = next(c for c in df.columns if c.startswith("ball_owning_team"))
    alive = df[state_col].astype(str).str.lower().str.contains("alive").to_numpy()
    owner = df[owner_col].astype(str).to_numpy()
    truth = np.where(~alive, "D", np.where(owner == str(home_id), "H", "A")).astype("<U1")
    known_owner = np.array([o not in ("None", "nan", "") for o in owner])
    # An alive frame with no owner keeps the last owner (possession persists through loose play).
    last = "H"
    for i in np.flatnonzero(alive):
        if known_owner[i]:
            last = truth[i]
        else:
            truth[i] = last
    period = df["period_id"].to_numpy()
    print("columns used:", state_col, owner_col, "| alive", round(float(alive.mean()), 3),
          "| alive without owner", round(float((alive & ~known_owner).mean()), 4), flush=True)
    data = {"players": xy, "ball": ball, "team": np.array([team_of[c] for c in cols], dtype=np.int8),
            "pitch": (length, width)}
    return data, truth, period


def with_speed(players_xy):
    """Append a speed channel (m/s, smoothed over 5 frames), as the competition ATD carries one."""
    step = np.sqrt((np.diff(players_xy, axis=0) ** 2).sum(axis=2)) * FPS
    step = np.vstack([step[:1], step])
    kernel = np.ones(5) / 5.0
    speed = np.apply_along_axis(lambda s: np.convolve(np.nan_to_num(s), kernel, mode="same"), 0, step)
    speed[np.isnan(players_xy[:, :, 0])] = np.nan
    return np.concatenate([players_xy, speed[:, :, None]], axis=2)


def degrade(data, truth, seed):
    rng = np.random.default_rng(seed)
    ball = data["ball"].copy()
    n = len(ball)
    dead = truth == "D"
    for a, b in runs_of(dead):
        if b - a >= 2 * FPS and rng.random() < 0.9:
            s = a + int(rng.uniform(0, 2) * FPS)
            e = b - int(rng.uniform(0, 1) * FPS)
            if e > s:
                ball[s:e] = np.nan
    live = np.flatnonzero(~dead)
    target = 0.08 * len(live)
    missing_live = int(np.isnan(ball[live, 0]).sum())
    while missing_live < target:
        start = int(rng.choice(live))
        length = int(np.clip(rng.lognormal(np.log(7.0), 0.4), 5, 20) * FPS)
        seg = slice(start, min(n, start + length))
        before = int(np.isnan(ball[seg][:, 0]).sum())
        ball[seg] = np.nan
        missing_live += (min(n, start + length) - start) - before
    for _ in range(int(len(live) / (60 * FPS))):
        start = int(rng.choice(live))
        length = max(2, int(rng.uniform(0.2, 2.0) * FPS))
        a, b = start - 1, min(n - 1, start + length)
        if a < 0 or np.isnan(ball[a, 0]) or np.isnan(ball[b, 0]):
            continue
        t = np.linspace(0, 1, b - a + 1)[:, None]
        ball[a:b + 1] = ball[a] + t * (ball[b] - ball[a])
    ball = ball + rng.normal(0, 0.0037, ball.shape)
    players = data["players"].copy()
    for j in range(players.shape[1]):
        for w0 in range(0, n, 10 * FPS):
            if rng.random() < 0.068:
                players[w0:w0 + 10 * FPS, j] = np.nan
    ball3 = np.concatenate([ball, np.full((n, 1), np.nan)], axis=1)
    return {"ball": ball3, "players": with_speed(players), "team": data["team"],
            "pitch": data["pitch"]}


def tagger(truth, seed):
    rng = np.random.default_rng(seed + 1000)
    events = []
    dead = truth == "D"
    for a, b in runs_of(dead):
        if b - a >= 2 * FPS and b < len(truth) and rng.random() < 0.55:
            s = b + int(rng.uniform(-3, 3) * FPS)
            events.append({"code": "SET PIECES", "start_frame": s, "end_frame": s + 5 * FPS,
                           "tags": {"Team": "N/A"}})
    for a, b in runs_of(~dead):
        if b - a >= 6 * FPS and rng.random() < 0.7:
            seg = truth[a:b]
            team = "H" if (seg == "H").sum() >= (seg == "A").sum() else "A"
            if rng.random() < 0.15:
                team = "A" if team == "H" else "H"
            events.append({"code": "Progression", "start_frame": a + FPS, "end_frame": b - FPS,
                           "tags": {"Team": "Barcelona" if team == "H" else "Atletic Club"}})
    events.sort(key=lambda e: e["start_frame"])
    return events


def set_blocks(period, truth):
    blocks = []
    for p in (1, 2):
        idx = np.flatnonzero(period == p)
        blocks.append((p, int(idx[0]), int(idx[-1])))
    ef._BLOCKS["current"] = blocks
    # No documented pre-kickoff on these games: make the BAR-ATH constant a no-op.
    deadball.SECOND_HALF_KICKOFF = blocks[1][1]
    graded = np.zeros(len(truth), dtype=bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    return np.where(graded, truth, "X").astype("<U1")


def production(data, events, n, dead=None):
    if dead is None:
        dead, _ = deadball.detect(data, events)
    evidence, details = possession.evidence(data, dead)
    labels = possession.labels(dead, evidence, possession.tag_evidence(events, n))
    labels, _ = possession.localize(labels, dead, details["ball"], data)
    return ms.fill_graded_abstentions(labels), dead


def scores(gt, pred):
    g, p = pm.label_codes(gt), pm.label_codes(pred)
    runs = pm.valid_runs(g)
    spell = pm.spell_scores(g, pm.fill_abstentions(p, runs), runs)["mean_f1"]
    graded = g != pm.X
    gd, pd = (g == pm.D) & graded, (p == pm.D) & graded
    tp = int((gd & pd).sum())
    prec, rec = tp / max(int(pd.sum()), 1), tp / max(int(gd.sum()), 1)
    return spell, 2 * prec * rec / max(prec + rec, 1e-9), rec


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="dfl_yardstick.csv")
    args = parser.parse_args()
    matches = MATCHES[:1] if args.smoke else MATCHES
    rows = []
    for k, m in enumerate(matches):
        raw, truth_full, period = load(m)
        truth = set_blocks(period, truth_full)
        n = len(truth)
        graded = truth != "X"
        print("{}: graded dead share {:.3f}, true possessions {}".format(
            m, float((truth[graded] == "D").mean()),
            len(pm.possession_spells(pm.label_codes(truth), pm.valid_runs(pm.label_codes(truth))))),
            flush=True)
        events = tagger(truth, k)
        for cond in ("noisy", "clean"):
            if cond == "noisy":
                data = degrade(raw, truth, k)
            else:
                data = {"ball": np.concatenate([raw["ball"], np.full((n, 1), np.nan)], axis=1),
                        "players": with_speed(raw["players"]), "team": raw["team"],
                        "pitch": raw["pitch"]}
            ours, dead = production(data, events, n)
            out = {"ours": ours}
            out["ours, true dead"] = production(data, events, n, dead=truth == "D")[0]
            out["our dead, true team"] = np.where(dead, "D", np.where(truth == "D", "H", truth))
            codes = pb.nearest_team(data)
            for w in (25, 50, 125):
                out["baseline_smooth{}".format(w)] = official.smooth(codes, w)
            for name, pred in out.items():
                sp, df1, drec = scores(truth, pred)
                rows.append((m, cond, name, sp, df1, drec))
                print("  {:6s} {:22s} spellF1 {:.3f}  D-F1 {:.3f}  D-recall {:.3f}".format(
                    cond, name, sp, df1, drec), flush=True)
    with open(args.out, "w") as f:
        f.write("match,condition,method,spellF1,D_F1,D_recall\n")
        for r in rows:
            f.write("{},{},{},{:.5f},{:.5f},{:.5f}\n".format(*r))
    print("\nMean over matches (sim_):")
    for cond in ("noisy", "clean"):
        for name in sorted({r[2] for r in rows}):
            v = [r[3] for r in rows if r[1] == cond and r[2] == name]
            print("  {:6s} {:22s} spellF1 {:.3f}  (n={})".format(cond, name, float(np.mean(v)), len(v)))


if __name__ == "__main__":
    main()

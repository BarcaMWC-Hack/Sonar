"""Frozen confirmation of the learned decoder on Sportec's open Bundesliga matches.

    python tools/confirm_sportec.py [--matches J03WMX,...] [--smoke]      # needs internet + kloppy

Protocol (fixed before any Sportec number was seen): train once on the three Metrica games exactly as
in tools/learned_possession.py (masked 0/20/41%, 2 GRU seeds, 14 epochs x 60 steps, logistic
regression baseline), freeze, then score each Sportec match with no tuning: the rule model at its
team defaults, gru at switch cost 3.0 and 5.0 (5.0 is what both earlier folds chose), lr at 8.0.
Truth: the same event-derived proxies A and B as the Metrica games, built from the Sportec events;
dead periods from the proxy (oracle). Clean tracking and 41% ball masking (seeds 3-5).
Adopt-bar: mean gain over the rule >= +0.03, each proxy >= +0.02, no match below 0, in the clean
and the masked tests. All numbers are sim_.
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
import external_joint as ej  # noqa: E402
import learned_possession as lp  # noqa: E402

MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")
RESTARTS = {"PASS", "SHOT", "RECOVERY", "TAKE_ON", "CARRY", "INTERCEPTION", "CLEARANCE"}
STOPS = {"BALL_OUT", "FOUL_COMMITTED"}
FPS = 25.0
_STORE = {}


def load_match(match_id):
    """(data dict, events list [(frame0, 'H'/'A', kind, period)]) for one Sportec match."""
    from kloppy import sportec
    ds = sportec.load_open_tracking_data(match_id=match_id, only_alive=False,
                                         coordinates="metrica")
    ev = sportec.load_open_event_data(match_id=match_id)
    meta = ds.metadata
    home_id, away_id = meta.teams[0].team_id, meta.teams[1].team_id
    df = ds.to_pandas() if hasattr(ds, "to_pandas") else ds.to_df()
    n = len(df)
    length = getattr(meta.pitch_dimensions, "pitch_length", None) or 105.0
    width = getattr(meta.pitch_dimensions, "pitch_width", None) or 68.0
    team_of = {}
    for t, code in ((meta.teams[0], 0), (meta.teams[1], 1)):
        for p in t.players:
            team_of[p.player_id] = code
    cols = [c[:-2] for c in df.columns if c.endswith("_x") and c[:-2] in team_of]
    players = np.full((n, len(cols), 2), np.nan)
    for j, pid in enumerate(cols):
        players[:, j, 0] = df[pid + "_x"].to_numpy(dtype=float) * length
        players[:, j, 1] = df[pid + "_y"].to_numpy(dtype=float) * width
    ball = np.stack([df["ball_x"].to_numpy(dtype=float) * length,
                     df["ball_y"].to_numpy(dtype=float) * width], axis=1)
    team = np.array([team_of[c] for c in cols], dtype=np.int8)
    period = df["period_id"].to_numpy()
    seconds = df["timestamp"].dt.total_seconds().to_numpy()
    rows_of = {int(p): np.flatnonzero(period == p) for p in np.unique(period)}
    events = []
    for e in ev.events:
        name = e.event_type.name
        kind = "r" if name in RESTARTS else ("s" if name in STOPS else None)
        if kind is None:
            continue
        p = int(e.period.id)
        if p not in rows_of:
            continue
        if kind == "r":
            if e.team is None or e.team.team_id not in (home_id, away_id):
                continue            # a restart needs a known team
            side = "H" if e.team.team_id == home_id else "A"
        else:
            side = "H"              # a stop's side is never used
        rows = rows_of[p]
        pos = int(np.searchsorted(seconds[rows], e.timestamp.total_seconds()))
        frame = int(rows[min(pos, len(rows) - 1)])
        events.append((frame, side, kind, str(p)))
    events.sort(key=lambda t: t[0])
    return {"ball": ball, "players": players, "team": team, "pitch": (length, width)}, events, period


def install(match_ids):
    """Make external_joint treat game ids 100+i as Sportec matches (load, events, blocks)."""
    orig_load, orig_events, orig_blocks = ej.load, ej.events, ej.blocks

    def load(game):
        # A fresh dict per call: masking replaces data["ball"], and must never leak into the cache.
        return dict(_STORE[game][0]) if game >= 100 else orig_load(game)

    def events(game):
        return _STORE[game][1] if game >= 100 else orig_events(game)

    def blocks(game, n, tr):
        if game < 100:
            return orig_blocks(game, n, tr)
        period = _STORE[game][2]
        out = []
        for p in (1, 2):
            idx = np.flatnonzero((period == p) & (tr != "X"))
            out.append((p, int(idx[0]), int(idx[-1])))
        ej.ef._BLOCKS["current"] = out

    ej.load, ej.events, ej.blocks = load, events, blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", default=",".join(MATCHES))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="sportec_results.csv")
    args = parser.parse_args()
    epochs, steps, seeds = (1, 5, 1) if args.smoke else (14, 60, 2)

    ids = []
    for k, m in enumerate(args.matches.split(",")):
        try:
            _STORE[100 + k] = load_match(m)
            ids.append((100 + k, m))
            print("loaded", m, _STORE[100 + k][0]["players"].shape, flush=True)
        except Exception as e:      # a match that fails to load is reported, never silently dropped
            print("FAILED", m, type(e).__name__, str(e)[:200], flush=True)
    if not ids:
        sys.exit("no Sportec match could be loaded")
    install(ids)

    train = [lp.build(g, fr, 100 + so) for g in lp.GAMES for fr, so in lp.TRAIN_MASKS]
    lr = lp.fit_lr(train)
    grus = [lp.fit_gru(train, s, epochs, steps) for s in range(seeds)]
    print("trained", flush=True)

    rows = []
    for gid, name in ids:
        for frac, seed in lp.TEST_MASKS:
            v = lp.build(gid, frac, seed)
            z = np.mean([g(v["X"]) for g in grus], axis=0)
            variants = {"rule": (v["rule"], 3.0), "gru@3": (z, 3.0), "gru@5": (z, 5.0),
                        "lr@8": (lr(v["X"]), 8.0)}
            for vn, (logit, cost) in variants.items():
                a, b = lp.f1(v, logit, cost)
                rows.append((name, frac, seed, vn, a, b))
                print("{} mask {:.2f} seed {} {:7s} A {:.4f} B {:.4f}".format(
                    name, frac, seed, vn, a, b), flush=True)
    with open(args.out, "w") as f:
        f.write("match,mask,seed,variant,A,B\n")
        for r in rows:
            f.write("{},{},{},{},{:.5f},{:.5f}\n".format(*r))

    names = [n for _, n in ids]
    print("\nHeld-out gain over the rule, per match (mean of A and B) and overall")
    verdicts = []
    for variant in ("gru@3", "gru@5", "lr@8"):
        for cond, sel in (("clean", lambda m: m == 0.0), ("masked41", lambda m: m > 0)):
            gA, gB, per = [], [], []
            for nm in names:
                def mean(vn, col):
                    return np.mean([r[col] for r in rows if r[0] == nm and sel(r[1]) and r[3] == vn])
                da, db = mean(variant, 4) - mean("rule", 4), mean(variant, 5) - mean("rule", 5)
                gA.append(da)
                gB.append(db)
                per.append((da + db) / 2)
            ok = (np.mean(per) >= 0.03 and np.mean(gA) >= 0.02 and np.mean(gB) >= 0.02
                  and min(per) >= 0)
            if variant.startswith("gru"):
                verdicts.append(ok)
            print("  {:8s} {:6s} A {:+.4f} B {:+.4f} mean {:+.4f} worst match {:+.4f} wins {}/{}"
                  .format(cond, variant, np.mean(gA), np.mean(gB), np.mean(per), min(per),
                          int(sum(p > 0 for p in per)), len(per)))
    print("matches used:", len(names), names)
    print("CONFIRMED under the pre-registered bar (gru@3 and gru@5, clean and masked):",
          "YES" if all(verdicts) else "NO")


if __name__ == "__main__":
    main()

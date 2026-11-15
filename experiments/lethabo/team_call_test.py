"""Pre-registered test of team-call variants on the DFL yardstick (sim_).

    python tools/team_call_test.py [--smoke]      # needs internet + kloppy (Kaggle)

Variants (fixed before any number was seen), each against the current pipeline:
  P5, P10  first-touch placement needs the new team to keep control (5 or 10 frames within 2 s,
           no old-team touch), so a deflection or block does not move the switch
  G        team shape counts only where the ball is seen, so it cannot place a switch in a gap
  M1, M2   possession spells shorter than 1 or 2 s are merged into their neighbours when both
           neighbours are the same team
Data and noise exactly as tools/dfl_yardstick.py (7 Sportec matches, truth = DFL ball status and
possession, our own predicted dead periods, noise calibrated to the competition match).
Adopt-gate per variant: mean spellF1 gain on noisy tracking >= +0.02, at least 5 of 7 matches better,
worst match >= -0.03, and mean gain on clean tracking > 0. Among passing variants the largest noisy
gain is the candidate (at most one). A label-free sanity check on the real match follows separately.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dfl_yardstick as ys  # noqa: E402
from sonar import deadball, possession  # noqa: E402

VARIANTS = {"base": {}, "P5": {"persist": 5}, "P10": {"persist": 10},
            "G": {"gate": True}, "M1": {"min_spell": 1.0}, "M2": {"min_spell": 2.0}}


def merge_short(labels, seconds):
    """Flip H/A spells shorter than `seconds` to the team on both sides, until none is left."""
    out = labels.copy()
    limit = int(seconds * ys.FPS)
    while True:
        change = np.flatnonzero(out[1:] != out[:-1]) + 1
        bounds = np.concatenate(([0], change, [len(out)]))
        flipped = False
        for k in range(1, len(bounds) - 2):
            a, b = bounds[k], bounds[k + 1]
            if out[a] in "HA" and b - a < limit and out[a - 1] == out[b] and out[b] in "HA" \
                    and out[a - 1] != out[a]:
                out[a:b] = out[b]
                flipped = True
                break           # recompute the boundaries after every merge
        if not flipped:
            return out


def run(data, events, dead, n, persist=0, gate=False, min_spell=0.0):
    ev, details = possession.evidence(data, dead, shape_only_with_ball=gate)
    labels = possession.labels(dead, ev, possession.tag_evidence(events, n))
    labels, _ = possession.localize(labels, dead, details["ball"], data, persist=persist)
    if min_spell:
        labels = merge_short(labels, min_spell)
    return ys.ms.fill_graded_abstentions(labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    matches = ys.MATCHES[:1] if args.smoke else ys.MATCHES
    rows = []
    for k, m in enumerate(matches):
        raw, truth_full, period = ys.load(m)
        truth = ys.set_blocks(period, truth_full)
        n = len(truth)
        events = ys.tagger(truth, k)
        for cond in ("noisy", "clean"):
            if cond == "noisy":
                data = ys.degrade(raw, truth, k)
            else:
                data = {"ball": np.concatenate([raw["ball"], np.full((n, 1), np.nan)], axis=1),
                        "players": ys.with_speed(raw["players"]), "team": raw["team"],
                        "pitch": raw["pitch"]}
            dead, _ = deadball.detect(data, events)
            for name, opts in VARIANTS.items():
                sp = ys.scores(truth, run(data, events, dead, n, **opts))[0]
                rows.append((m, cond, name, sp))
            print(m, cond, {r[2]: round(r[3], 4) for r in rows if r[0] == m and r[1] == cond},
                  flush=True)
    names = sorted({r[0] for r in rows})
    print("\nGain over base (spellF1, sim_)")
    passing = []
    for v in VARIANTS:
        if v == "base":
            continue
        g = {c: np.array([[r[3] for r in rows if r[0] == m and r[1] == c and r[2] == v][0] -
                          [r[3] for r in rows if r[0] == m and r[1] == c and r[2] == "base"][0]
                          for m in names]) for c in ("noisy", "clean")}
        ok = (g["noisy"].mean() >= 0.02 and (g["noisy"] > 0).sum() >= 5 and g["noisy"].min() >= -0.03
              and g["clean"].mean() > 0)
        if ok:
            passing.append((g["noisy"].mean(), v))
        print("  {:4s} noisy mean {:+.4f} wins {}/{} worst {:+.4f} | clean mean {:+.4f} -> {}".format(
            v, g["noisy"].mean(), int((g["noisy"] > 0).sum()), len(names), g["noisy"].min(),
            g["clean"].mean(), "PASS" if ok else "fail"))
    if args.smoke or len(names) < 7:
        print("DIAGNOSTIC ONLY (smoke or incomplete roster); no verdict")
    elif passing:
        print("CANDIDATE:", max(passing)[1], "(pending the label-free check on the real match)")
    else:
        print("NO VARIANT PASSES; keep the current pipeline")


if __name__ == "__main__":
    main()

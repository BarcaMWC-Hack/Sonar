"""Score Sonar builds on SkillCorner's 20 open broadcast-tracking matches (real ball loss).

    python tools/skillcorner_eval.py

Truth and loading: tools/skillcorner_data.py. No ball masking: the broadcast feed loses the
ball by itself (about 40% of graded frames, like our tracker). Dead-ball weight sets compared:

  release      fitted on BAR-ATH Smart Tagging weak labels (the current build)
  idsse        fitted on the 7 IDSSE matches only (never sees SkillCorner)
  sc-lomo      fitted on the true dead flags of the other 19 SkillCorner matches
  both-lomo    fitted on the 7 IDSSE matches plus the other 19 SkillCorner matches

Each is run with possession switching costs 3.0 (current), 2.0 and 1.5. Prints aggregates only.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import idsse_data  # noqa: E402
import idsse_eval as ie  # noqa: E402
import skillcorner_data  # noqa: E402
from sonar import deadball, deadball_features as dbf  # noqa: E402


def prepare(match):
    d = skillcorner_data.load(match)
    ie.use_blocks(d["blocks"])
    d["features"] = dbf.compute(d)
    d["x"], d["names"] = deadball.design(d["features"])
    return d


def gap_stats(prepared):
    lengths, dead_miss, alive_miss, miss = [], [], [], []
    for d in prepared.values():
        t = d["truth"]
        m = np.isnan(d["ball"][:, 0]) & (t != "X")
        edges = np.diff(np.concatenate(([0], m.astype(np.int8), [0])))
        lengths.extend((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).tolist())
        miss.append(m[t != "X"].mean())
        dead_miss.append(m[t == "D"].mean())
        alive_miss.append(m[(t == "H") | (t == "A")].mean())
    lengths = np.array(lengths) / 25.0
    print("ball missing {:.1%} (dead {:.1%}, alive {:.1%}); gaps per match {:.0f}, length s "
          "p50 {:.1f} p90 {:.1f}; share of missing time in gaps > 5 s {:.2f}".format(
              np.mean(miss), np.mean(dead_miss), np.mean(alive_miss),
              len(lengths) / len(prepared), np.median(lengths), np.percentile(lengths, 90),
              lengths[lengths > 5].sum() / lengths.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--matches", nargs="*", default=None)
    parser.add_argument("--costs", type=float, nargs="+", default=[3.0, 2.0, 1.5])
    args = parser.parse_args()
    start = time.time()

    w_release = ie.bar_ath_weights()
    matches = args.matches or skillcorner_data.available()
    prepared = {m: prepare(m) for m in matches}
    print("{} SkillCorner matches prepared in {:.0f} s".format(len(prepared), time.time() - start))
    gap_stats(prepared)

    idsse = {m: ie.prepare(m, 0.41, k, "bar-ath") for k, m in enumerate(idsse_data.available())}
    w_idsse = ie.dfl_weights(idsse, exclude=None)
    weights = {"release": lambda m: w_release, "idsse": lambda m: w_idsse}
    if len(prepared) > 1:
        sc_w = {m: ie.dfl_weights(prepared, m) for m in prepared}
        both = dict(idsse)
        both.update(prepared)
        both_w = {m: ie.dfl_weights(both, m) for m in prepared}
        weights["sc-lomo"] = lambda m: sc_w[m]
        weights["both-lomo"] = lambda m: both_w[m]

    print("\nper-match spellF1 in the order: " + " ".join(prepared))
    table = {}
    for wname, wf in weights.items():
        for cost in args.costs:
            rows = [ie.score(d["truth"], ie.pipeline(d, wf(m), poss_cost=cost))
                    for m, d in prepared.items()]
            table[(wname, cost)] = rows
            ie.summarise("{} cost {}".format(wname, cost), rows)
    for cost in args.costs[:2]:
        rows = [ie.score(d["truth"], ie.pipeline(d, None, dead=d["truth"] == "D", poss_cost=cost))
                for d in prepared.values()]
        ie.summarise("oracle dead cost {}".format(cost), rows)

    base = np.array([r["spell"] for r in table[("release", args.costs[0])]])
    print("\nagainst the current build (release, cost {}):".format(args.costs[0]))
    for key, rows in table.items():
        if key == ("release", args.costs[0]):
            continue
        s = np.array([r["spell"] for r in rows])
        diff = s - base
        se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
        print("  {:12s} cost {:3.1f}: wins {}/{}, mean gain {:+.3f} (se {:.3f})".format(
            key[0], key[1], int((diff > 0).sum()), len(diff), diff.mean(), se))
    print("done in {:.0f} s".format(time.time() - start))


if __name__ == "__main__":
    main()

"""Leave-one-game-out learned decoder over all ten open games (3 Metrica + 7 Sportec Bundesliga).

    python tools/ten_games.py [--smoke]          # needs internet + kloppy (Kaggle GPU)

Pre-registered (fixed before any number was seen). For each of the 10 games: train on the other 9
(masked 0/20/41%, 2 GRU seeds, 14 epochs x 60 steps; logistic regression baseline), score the held-out
game with no tuning: rule at team defaults, gru at switch cost 3.0 and 5.0, lr at 8.0; truth =
event-derived proxies A and B, oracle dead; clean tracking and 41% masking (seeds 3-5).
Bar (same as before): mean gain over the rule >= +0.03, each proxy mean >= +0.02, no game below 0,
for gru@3 and gru@5, in the clean and the masked test. One run; if it fails the learned decoder stops.
Also reported, not used for the decision: wins, and the means by provider (Metrica vs Sportec).
All numbers are sim_.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import confirm_sportec as cs  # noqa: E402
import learned_possession as lp  # noqa: E402

METRICA = (1, 2, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="ten_games_results.csv")
    args = parser.parse_args()
    epochs, steps, seeds = (1, 5, 1) if args.smoke else (14, 60, 2)
    matches = cs.MATCHES[:1] if args.smoke else cs.MATCHES

    names = {1: "metrica1", 2: "metrica2", 3: "metrica3"}
    for k, m in enumerate(matches):
        try:
            cs._STORE[100 + k] = cs.load_match(m)
            names[100 + k] = m
            print("loaded", m, flush=True)
        except Exception as e:
            print("FAILED", m, type(e).__name__, str(e)[:200], flush=True)
    cs.install(None)
    games = [g for g in (1, 2, 3) + tuple(range(100, 100 + len(matches))) if g in names]
    if args.smoke:
        games = [1, 100]

    train, test = {}, {}
    for g in games:
        train[g] = [lp.build(g, fr, 100 + so) for fr, so in lp.TRAIN_MASKS]
        test[g] = [(0.0, train[g][0])] + [(0.41, lp.build(g, 0.41, sd)) for sd in (3, 4, 5)]
        print("built", names[g], flush=True)

    rows = []
    for held in games:
        tr = [v for g in games if g != held for v in train[g]]
        lr = lp.fit_lr(tr)
        grus = [lp.fit_gru(tr, s, epochs, steps) for s in range(seeds)]
        for frac, v in test[held]:
            z = np.mean([m(v["X"]) for m in grus], axis=0)
            variants = {"rule": (v["rule"], 3.0), "gru@3": (z, 3.0), "gru@5": (z, 5.0),
                        "lr@8": (lr(v["X"]), 8.0)}
            for vn, (logit, cost) in variants.items():
                a, b = lp.f1(v, logit, cost)
                rows.append((names[held], frac, vn, a, b))
        print("held out", names[held], "done", flush=True)
    with open(args.out, "w") as f:
        f.write("game,mask,variant,A,B\n")
        for r in rows:
            f.write("{},{},{},{:.5f},{:.5f}\n".format(*r))

    gnames = [names[g] for g in games]

    def gains(variant, sel, subset):
        gA, gB, per = [], [], []
        for nm in subset:
            def mean(vn, col):
                return np.mean([r[col] for r in rows if r[0] == nm and sel(r[1]) and r[2] == vn])
            da, db = mean(variant, 3) - mean("rule", 3), mean(variant, 4) - mean("rule", 4)
            gA.append(da)
            gB.append(db)
            per.append((da + db) / 2)
        return np.array(gA), np.array(gB), np.array(per)

    print("\nGain over the rule, leave-one-game-out (mean of A and B per game)")
    verdicts = []
    for variant in ("gru@3", "gru@5", "lr@8"):
        for cond, sel in (("clean", lambda m: m == 0.0), ("masked41", lambda m: m > 0)):
            gA, gB, per = gains(variant, sel, gnames)
            ok = per.mean() >= 0.03 and gA.mean() >= 0.02 and gB.mean() >= 0.02 and per.min() >= 0
            if variant.startswith("gru"):
                verdicts.append(ok)
            print("  {:8s} {:6s} A {:+.4f} B {:+.4f} mean {:+.4f} worst {:+.4f} wins {}/{}".format(
                cond, variant, gA.mean(), gB.mean(), per.mean(), per.min(),
                int((per > 0).sum()), len(per)))
    for label, subset in (("Metrica", [n for n in gnames if n.startswith("metrica")]),
                          ("Sportec", [n for n in gnames if not n.startswith("metrica")])):
        if subset:
            for cond, sel in (("clean", lambda m: m == 0.0), ("masked41", lambda m: m > 0)):
                _, _, per = gains("gru@5", sel, subset)
                print("  by provider {:8s} {:8s} gru@5 mean {:+.4f} wins {}/{}".format(
                    label, cond, per.mean(), int((per > 0).sum()), len(per)))
    print("games:", gnames)
    complete = (len(gnames) == 10) and not args.smoke
    if not complete:
        print("DIAGNOSTIC ONLY: smoke run or incomplete roster ({} of 10 games); no verdict".format(
            len(gnames)))
    else:
        print("PASSES the pre-registered bar (gru@3 and gru@5, clean and masked):",
              "YES" if all(verdicts) else "NO")


if __name__ == "__main__":
    main()

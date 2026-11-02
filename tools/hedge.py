"""Hedged decoder: rule evidence blended with a learned decoder, tested on other games only.

    python tools/hedge.py [--smoke]          # needs internet + kloppy (Kaggle GPU)

Pre-registered (fixed before any number was seen):
  blend      z = rule evidence + 0.5 x learned log-odds, decoded with the rule's Viterbi at switch
             cost 3.0 (primary). Also reported, not used to decide: weights 0.25 and 1.0, cost 5.0,
             the pure learned decoder (cost 5.0) and the rule alone.
  training   Metrica open sample games only (masked 0/20/41%, 2 GRU seeds, 14 epochs x 60 steps).
  test A     leave-one-Metrica-game-out (train on two, score the third): our own provider.
  test B     train on all three Metrica games, score the 7 Sportec Bundesliga matches: another provider.
  truth      event-derived proxies A and B, oracle dead periods; clean tracking and 41% masking
             (seeds 3-5). All numbers are sim_.
  caveat     the team-shape model is fitted on each evaluated game's own tracking (label-free, from
             ball evidence), for the rule and the learned decoder alike. That is test-time adaptation
             of an unsupervised kind, declared here, not a leak of truth.
  decision   A: mean gain over the rule >= +0.03, each proxy mean >= +0.02, no fold below 0, in the
             clean and the masked test.  B: mean gain >= -0.01 in the clean and the masked test.
             Both must hold. If not, the blend is dropped.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import confirm_sportec as cs  # noqa: E402
import learned_possession as lp  # noqa: E402

PRIMARY = ("blend0.5@3", 0.5, 3.0)
EXTRA = (("blend0.25@3", 0.25, 3.0), ("blend1.0@3", 1.0, 3.0), ("blend0.5@5", 0.5, 5.0))


def evaluate(test_versions, name, models, rows):
    for frac, v in test_versions:
        z = np.mean([m(v["X"]) for m in models], axis=0)
        variants = {"rule": (v["rule"], 3.0), "gru@5": (z, 5.0)}
        for vn, lam, cost in (PRIMARY,) + EXTRA:
            variants[vn] = (v["rule"] + lam * z, cost)
        for vn, (logit, cost) in variants.items():
            a, b = lp.f1(v, logit, cost)
            rows.append((name, frac, vn, a, b))


def gains(rows, variant, sel, names):
    gA, gB, per = [], [], []
    for nm in names:
        def mean(vn, col):
            return np.mean([r[col] for r in rows if r[0] == nm and sel(r[1]) and r[2] == vn])
        da, db = mean(variant, 3) - mean("rule", 3), mean(variant, 4) - mean("rule", 4)
        gA.append(da)
        gB.append(db)
        per.append((da + db) / 2)
    return np.array(gA), np.array(gB), np.array(per)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="hedge_results.csv")
    args = parser.parse_args()
    epochs, steps, seeds = (1, 5, 1) if args.smoke else (14, 60, 2)
    matches = cs.MATCHES[:1] if args.smoke else cs.MATCHES

    names = {}
    for k, m in enumerate(matches):
        try:
            cs._STORE[100 + k] = cs.load_match(m)
            names[100 + k] = m
            print("loaded", m, flush=True)
        except Exception as e:
            print("FAILED", m, type(e).__name__, str(e)[:200], flush=True)
    cs.install(None)
    sportec = sorted(names)

    metrica = (1, 2) if args.smoke else (1, 2, 3)
    train = {g: [lp.build(g, fr, 100 + so) for fr, so in lp.TRAIN_MASKS] for g in metrica}
    test = {g: [(0.0, train[g][0])] + [(0.41, lp.build(g, 0.41, sd)) for sd in (3, 4, 5)]
            for g in metrica}
    print("metrica built", flush=True)

    rows_a, rows_b = [], []
    for held in metrica:                                       # test A: our own provider
        tr = [v for g in metrica if g != held for v in train[g]]
        models = [lp.fit_gru(tr, s, epochs, steps) for s in range(seeds)]
        evaluate(test[held], "metrica{}".format(held), models, rows_a)
        print("A held out metrica", held, flush=True)

    tr = [v for g in metrica for v in train[g]]                # test B: another provider
    models = [lp.fit_gru(tr, s, epochs, steps) for s in range(seeds)]
    for gid in sportec:
        vs = [(0.0, lp.build(gid, 0.0, 0))] + [(0.41, lp.build(gid, 0.41, sd)) for sd in (3, 4, 5)]
        evaluate(vs, names[gid], models, rows_b)
        print("B scored", names[gid], flush=True)

    with open(args.out, "w") as f:
        f.write("test,game,mask,variant,A,B\n")
        for tag, rows in (("A", rows_a), ("B", rows_b)):
            for r in rows:
                f.write("{},{},{},{},{:.5f},{:.5f}\n".format(tag, *r))

    names_a = ["metrica{}".format(g) for g in metrica]
    names_b = [names[g] for g in sportec]
    print("\nGain over the rule (mean of A and B per game)")
    ok = []
    for label, rows, nm, tag in (("A in-domain LOGO", rows_a, names_a, "A"),
                                 ("B Bundesliga", rows_b, names_b, "B")):
        for variant in ("blend0.5@3", "blend0.25@3", "blend1.0@3", "blend0.5@5", "gru@5"):
            for cond, sel in (("clean", lambda m: m == 0.0), ("masked41", lambda m: m > 0)):
                gA, gB, per = gains(rows, variant, sel, nm)
                print("  {:18s} {:11s} {:8s} A {:+.4f} B {:+.4f} mean {:+.4f} worst {:+.4f} "
                      "wins {}/{}".format(label, variant, cond, gA.mean(), gB.mean(), per.mean(),
                                          per.min(), int((per > 0).sum()), len(per)))
                if variant == PRIMARY[0]:
                    if tag == "A":
                        ok.append(per.mean() >= 0.03 and gA.mean() >= 0.02 and gB.mean() >= 0.02
                                  and per.min() >= 0)
                    else:
                        ok.append(per.mean() >= -0.01)
    complete = (not args.smoke) and len(names_b) == 7 and len(names_a) == 3
    if not complete:
        print("DIAGNOSTIC ONLY: smoke run or incomplete roster; no verdict")
    else:
        print("BLEND PASSES the pre-registered decision (A in-domain and B downside limit):",
              "YES" if all(ok) else "NO")


if __name__ == "__main__":
    main()

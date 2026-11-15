"""Score the whole Sonar pipeline (dead ball + possession) on the 7 open DFL / IDSSE matches.

    python tools/idsse_eval.py                      # release pipeline vs variants, ball masked 41%
    python tools/idsse_eval.py --degrade 0          # clean tracking
    python tools/idsse_eval.py --grid               # decoder settings, leave-one-match-out

Truth is each match's own per-frame BallStatus / BallPossession (tools/idsse_data.py), so this
scores the dead-ball detector and the possession calls together, with the competition's spellF1
(evaluation/possession_metrics.py). These are other games, which the rules let us tune on.

The release pipeline's dead-ball weights are fitted exactly as the submission fits them, on
BAR-ATH's Smart Tagging weak labels, then carried to each IDSSE match unchanged: a straight test
of whether the detector generalises. The "dfl-trained" variant instead fits the same logistic
model on the true dead flags of the OTHER six IDSSE matches (never the match being scored).

--degrade masks the ball on that share of frames with seeded gaps (simulated), to mimic our
tracker, which loses the ball on 41% of graded frames. Numbers are sim_ numbers.

Prints aggregate scores only.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
import idsse_data  # noqa: E402
from sonar import deadball, deadball_features as dbf, possession, team_shape  # noqa: E402

_ORIGINAL = {"blocks": deadball.half_video_blocks, "kickoff": deadball.SECOND_HALF_KICKOFF}


def use_blocks(blocks):
    """Point sonar's BAR-ATH half blocks at this match's halves (nothing else changes)."""
    if blocks is None:
        f, kickoff = _ORIGINAL["blocks"], _ORIGINAL["kickoff"]
    else:
        def f(game=None, adjust=0):
            return blocks
        kickoff = blocks[1][1]          # no documented pre-kickoff dead spell on these matches
    deadball.half_video_blocks = f
    dbf.half_video_blocks = f
    team_shape.half_video_blocks = f
    deadball.SECOND_HALF_KICKOFF = kickoff


def mask_ball(ball, fraction, mean_gap_s=2.0, seed=0):
    """Copy of ball with seeded gaps so that about `fraction` of frames are NaN (simulated)."""
    rng = np.random.default_rng(seed)
    out = ball.copy()
    if fraction <= 0:
        return out
    gap = mean_gap_s * 25.0
    seen = gap * (1.0 - fraction) / fraction
    t, hidden = 0, False
    while t < len(out):
        length = max(1, int(rng.exponential(gap if hidden else seen)))
        if hidden:
            out[t:t + length] = np.nan
        t += length
        hidden = not hidden
    return out


# "bar-ath" degrade mode: the ball loss pattern of OUR tracker, measured label-free on BAR-ATH's
# graded frames (aggregates only): 40.7% missing in 150 gaps, nearly all longer than 5 s, gap
# length median 10.7 s, 90th percentile 33.8 s; the ball is missing on 69% of the frames Smart
# Tagging marks dead and 22% of those it marks alive. So the simulation hides the ball through a
# share of the true stoppages (from a random point inside each to its restart), then adds
# log-normal gaps fitted to those percentiles at random until the target share is missing.
# The ball also jitters: median frame-to-frame step 0.20 m against TRACAB's 0.085 m; smoothed
# noise of 0.4 m (over 0.5 s) gives 0.26 m and drops TRACAB's "still ball" share from ~20% to 13%
# (BAR-ATH reads 2.9%, partly because its placed balls are mostly unseen).
BAR_GAP_MEDIAN_S = 10.7
BAR_GAP_SIGMA = 0.90
BAR_DEAD_HIDDEN = 0.9           # share of true stoppages the ball is lost in
BAR_JITTER_M = 0.4
BAR_JITTER_FRAMES = 12


def _runs(mask):
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def mask_like_bar_ath(ball, truth, fraction, seed=0, dead_hidden=None, jitter=None):
    dead_hidden = BAR_DEAD_HIDDEN if dead_hidden is None else dead_hidden
    jitter = BAR_JITTER_M if jitter is None else jitter
    rng = np.random.default_rng(seed)
    out = ball.copy()
    if fraction <= 0:
        return out
    graded_mask = truth != "X"
    graded = np.flatnonzero(graded_mask)
    missing = np.zeros(len(out), dtype=bool)
    for a, b in _runs(truth == "D"):
        if rng.random() < dead_hidden:
            missing[a + int(rng.random() * 0.5 * (b - a)):b + int(rng.integers(0, 25))] = True
    target = fraction * len(graded)
    while missing[graded].sum() < target:
        start = int(rng.choice(graded))
        length = int(rng.lognormal(np.log(BAR_GAP_MEDIAN_S * 25), BAR_GAP_SIGMA))
        missing[start:start + length] = True
    if jitter > 0:
        k = np.ones(BAR_JITTER_FRAMES) / BAR_JITTER_FRAMES
        white = rng.normal(0.0, 1.0, size=(len(out) + BAR_JITTER_FRAMES - 1, 2))
        noise = np.stack([np.convolve(white[:, i], k, "valid") for i in range(2)], axis=1)
        out[:, :2] += (noise * jitter / noise.std()).astype(out.dtype)
    out[missing & graded_mask] = np.nan
    return out


def bar_ath_weights():
    """The release dead-ball weights, fitted on BAR-ATH weak labels as make_submission does."""
    cache = os.path.join(idsse_data.CACHE, "bar_ath_weights_{}.json".format(
        int(round(deadball.L2))))
    if os.path.exists(cache):
        with open(cache) as f:
            return np.array(json.load(f))
    from sonar import atd
    from starter import load_smart_tagging
    use_blocks(None)
    data = atd.load()
    events = load_smart_tagging.load()
    x, _ = deadball.design(dbf.compute(data))
    dead, alive = deadball.weak_labels(len(x), events)
    train = dead | alive
    w = deadball.fit_logistic(x[train], dead[train].astype(float))
    os.makedirs(idsse_data.CACHE, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(w.tolist(), f)
    return w


def prepare(match, degrade, seed, mode="bar-ath"):
    d = idsse_data.load(match)
    if mode == "bar-ath":
        d["ball"] = mask_like_bar_ath(d["ball"], d["truth"], degrade, seed=seed)
    else:
        d["ball"] = mask_ball(d["ball"], degrade, seed=seed)
    use_blocks(d["blocks"])
    features = dbf.compute(d)
    x, names = deadball.design(features)
    d["features"], d["x"] = features, x
    return d


def pipeline(d, w, dead=None, temperature=deadball.TEMPERATURE, dead_cost=deadball.SWITCH_COST,
             prior=deadball.PRIOR_DEAD, refine=True, poss_cost=possession.SWITCH_COST,
             localize=True):
    use_blocks(d["blocks"])
    n = len(d["truth"])
    if dead is None:
        llr = deadball.log_likelihood_ratio(d["x"], w)
        dead = deadball.decode(llr, n, temperature=temperature, switch_cost=dead_cost,
                               prior_dead=prior)
        if refine:
            dead, _, _ = deadball.refine(dead, d, d["features"])
    ev, details = possession.evidence(d, dead)
    labels = possession.labels(dead, ev, None, switch_cost=poss_cost)
    if localize:
        labels, _ = possession.localize(labels, dead, details["ball"], d)
    return labels


def score(truth, labels):
    gt, pred = pm.label_codes(truth), pm.label_codes(labels)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    spell = pm.spell_scores(gt, filled, runs)
    graded = truth != "X"
    cm3 = pm.scored_confusion(pm.confusion(gt[graded], pred[graded]))
    t_dead, p_dead = truth[graded] == "D", labels[graded] == "D"
    return {"spell": spell["mean_f1"], "macro": pm.macro_f1(cm3),
            "d_recall": float((t_dead & p_dead).sum() / max(t_dead.sum(), 1)),
            "d_precision": float((t_dead & p_dead).sum() / max(p_dead.sum(), 1)),
            "n_pred": spell["n_pred"], "n_gt": spell["n_gt"]}


def dfl_weights(prepared, exclude, step=5):
    """Logistic dead-ball weights fitted on the true dead flags of every match except `exclude`."""
    xs, ys = [], []
    for match, d in prepared.items():
        if match == exclude:
            continue
        graded = np.flatnonzero(d["truth"] != "X")[::step]
        xs.append(d["x"][graded])
        ys.append((d["truth"][graded] == "D").astype(float))
    return deadball.fit_logistic(np.concatenate(xs), np.concatenate(ys))


def summarise(name, rows):
    keys = ("spell", "macro", "d_recall", "d_precision")
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    per = " ".join("{:.3f}".format(r["spell"]) for r in rows)
    spells = sum(r["n_pred"] for r in rows), sum(r["n_gt"] for r in rows)
    print("  {:34s} spellF1 {:.3f}  macroF1 {:.3f}  D rec {:.2f} prec {:.2f} | spells {}/{} | {}"
          .format(name, mean["spell"], mean["macro"], mean["d_recall"], mean["d_precision"],
                  spells[0], spells[1], per))
    return mean


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--degrade", type=float, default=0.41)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("bar-ath", "random"), default="bar-ath",
                        help="bar-ath: long gaps mostly in dead time plus jitter, as our tracker; "
                             "random: short random gaps (the Metrica yardstick's simulation)")
    parser.add_argument("--matches", nargs="*", default=None)
    parser.add_argument("--grid", action="store_true")
    parser.add_argument("--poss-costs", type=float, nargs="+", default=[possession.SWITCH_COST])
    args = parser.parse_args()

    start = time.time()
    w_release = bar_ath_weights()
    matches = args.matches or idsse_data.available()
    prepared = {}
    for k, match in enumerate(matches):
        prepared[match] = prepare(match, args.degrade, args.seed + k, args.mode)
    print("{} IDSSE matches, ball masked on {:.0%} ({} mode, sim_), prepared in {:.0f} s".format(
        len(prepared), args.degrade, args.mode, time.time() - start))
    for m, d in prepared.items():
        miss, t = np.isnan(d["ball"][:, 0]), d["truth"]
        f = d["features"]
        print("  {}: missing {:.1%} (dead {:.1%}, alive {:.1%}), ball still {:.1%}".format(
            m, miss[t != "X"].mean(), miss[t == "D"].mean(), miss[(t == "H") | (t == "A")].mean(),
            np.nanmean(f["ball_speed"][t != "X"] < 0.5)))
    w_dfl = {m: dfl_weights(prepared, m) for m in prepared} if len(prepared) > 1 else {}

    variants = [
        ("release (BAR-ATH weak labels)", lambda m, d: pipeline(d, w_release)),
        ("oracle dead + release possession", lambda m, d: pipeline(d, None, dead=d["truth"] == "D")),
    ]
    weights = {"release": lambda m: w_release}
    if w_dfl:
        weights["dfl"] = lambda m: w_dfl[m]
        weights["blend"] = lambda m: 0.5 * (w_release + w_dfl[m])
        variants.append(("dfl-trained dead ball (LOMO)", lambda m, d: pipeline(d, w_dfl[m])))
        variants.append(("blend: mean of release and dfl", lambda m, d: pipeline(d, weights["blend"](m))))
    if not args.grid:
        print("\nper-match spellF1 in the order: " + " ".join(prepared))
        for name, fn in variants:
            summarise(name, [score(d["truth"], fn(m, d)) for m, d in prepared.items()])
        print("done in {:.0f} s".format(time.time() - start))
        return

    grid = [(wname, prior, dcost, pcost)
            for wname in weights
            for prior in (0.3, 0.4, 0.5)
            for dcost in (0.5, 1.0, 2.0, 4.0)
            for pcost in args.poss_costs]
    table = {}
    for g in grid:
        wname, prior, dcost, pcost = g
        rows = []
        for m, d in prepared.items():
            w = weights[wname](m)
            rows.append(score(d["truth"], pipeline(d, w, prior=prior, dead_cost=dcost,
                                                   poss_cost=pcost)))
        table[g] = rows
        summarise("{} prior {} dcost {} pcost {}".format(*g), rows)
    print("\nLeave-one-match-out: pick the best setting on six matches, score the seventh")
    picked = []
    for k, m in enumerate(prepared):
        best = max(grid, key=lambda g: np.mean([r["spell"] for j, r in enumerate(table[g])
                                                if j != k]))
        picked.append(table[best][k]["spell"])
        print("  {}: picked {} -> {:.3f}".format(m, best, table[best][k]["spell"]))
    print("  mean held-out spellF1 {:.3f}".format(float(np.mean(picked))))
    print("done in {:.0f} s".format(time.time() - start))


if __name__ == "__main__":
    main()

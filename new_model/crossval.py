"""Time-block cross-validation and audit for new_model.

Why this exists: model.py reports one temporal split (train before frame 150000, validate after).
One split cannot tell overfitting from a half-specific shift, and it cannot tell whether a
change helps on every block or only on the easy ones. This harness uses the same protocol as the
release's sonar/splits.py -- five consecutive blocks per half, 10 s either side of each test block
left out of training, graded frames only -- so numbers are comparable, but it carries its own
copy so this harness does not import sonar/splits.py. The shipped dead detector is imported from
sonar.deadball; that is the one deliberate release dependency.

    python -m new_model.crossval                 # 5-fold, shipped hybrid
    python -m new_model.crossval --half-swap     # fit on one half, test the other
    python -m new_model.crossval --dead-stage hgb  # reproduce the old dead-stage audit
    python -m new_model.crossval --leaves 7 --l2 10
    python -m new_model.crossval --penalties 8,16,32 --no-spell

What it reports, per fold and in summary:

  team AUC and logloss           agreement with the nearest-owner weak labels, train vs test gap
  dead/team AUC (--dead-stage hgb) the discarded HGB dead stage, for the audit that motivated the
                                 release detector; the shipped model does not use it
  weak macroF1                   decoded labels on the held-out block, weak-label truth
  weak spellF1                   possession-spell F1 on the held-out block, weak-label truth
  possessions / mean seconds     full-match structure for that penalty (no truth involved)
  dead share                     share of graded frames called D

The shipped dead stage is the release's fixed detector, so it has no fold-honest train/test gap:
its weak-label agreement is in-sample and is not reported as validation.  Only the team stage is
refit per fold.  The weak labels are not ground truth: this measures agreement with Smart Tagging
plus the nearest-owner proxy, and the public five minutes is not read here at all.
"""

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import possession_metrics as pm  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from new_model import atd, features, labels  # noqa: E402
from new_model.model import (  # noqa: E402
    H, A, D, X, RUNS, balanced_weights, fit, force_known_dead, nearest_team_labels,
    select_features, set_x_outside_runs, viterbi_decode, viterbi_decode_masked, weak_codes,
)
from sonar import deadball as release_deadball  # noqa: E402

FPS = 25
N_FOLDS = 5
GAP_S = 10


# ── folds (mirrors sonar/splits.py) ─────────────────────────────────────────
def graded_mask(n):
    mask = np.zeros(n, dtype=bool)
    for _, first, last in half_video_blocks(GAME):
        mask[first:last + 1] = True
    return mask


def block_ranges(n):
    out = []
    for half, first, last in half_video_blocks(GAME):
        edges = np.linspace(first, last + 1, N_FOLDS + 1).astype(np.int64)
        out += [(k, half, int(edges[k]), int(edges[k + 1]) - 1) for k in range(N_FOLDS)]
    return out


def block_ids(n):
    ids = np.full(n, -1, dtype=np.int8)
    for fold, _, first, last in block_ranges(n):
        ids[first:last + 1] = fold
    return ids


def grow(mask, frames):
    """mask widened by ``frames`` on both sides."""
    count = np.concatenate(([0], np.cumsum(mask)))
    index = np.arange(len(mask))
    return count[np.minimum(len(mask), index + frames + 1)] > count[np.maximum(0, index - frames)]


def folds(n):
    """(fold, train mask, test mask), graded frames only, GAP_S clear of each test block."""
    ids = block_ids(n)
    graded = ids >= 0
    for fold in range(N_FOLDS):
        block = ids == fold
        yield fold, graded & ~grow(block, GAP_S * FPS), graded & block


def half_folds(n):
    masks = []
    for _, first, last in half_video_blocks(GAME):
        mask = np.zeros(n, dtype=bool)
        mask[first:last + 1] = True
        masks.append(mask)
    yield 0, masks[0], masks[1]
    yield 1, masks[1], masks[0]


# ── scoring ─────────────────────────────────────────────────────────────────
def weak_spells(gt_codes, pred_codes, mask):
    """spellF1 of pred against weak labels, scored only on ``mask``.

    Frames outside the mask are X on both sides, so a held-out block is scored on its own and
    no spontaneous structure crosses into it. The weak labels are a proxy, not truth.
    """
    gt = np.full(len(gt_codes), X, dtype=np.int8)
    pred = np.full(len(pred_codes), X, dtype=np.int8)
    gt[mask] = gt_codes[mask]
    pred[mask] = pred_codes[mask]
    runs = pm.valid_runs(gt)
    if not runs:
        return float("nan"), 0, 0
    filled = pm.fill_abstentions(pred, runs)
    result = pm.spell_scores(gt, filled, runs)
    return result["mean_f1"], result["n_gt"], len(pm.possession_spells(filled, runs))


def structure(codes):
    spells = pm.possession_spells(codes, RUNS)
    durations = [stop - start for start, stop, _ in spells]
    graded = sum(stop - start for start, stop in RUNS)
    return {
        "possessions": len(spells),
        "mean_s": float(np.mean(durations)) / FPS if durations else 0.0,
        "dead_share": float(np.sum(codes == D)) / graded,
    }


def binary(y, p):
    return {
        "auc": roc_auc_score(y, p),
        "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "acc": accuracy_score(y, (p >= 0.5).astype(np.int8)),
    }


# ── one experiment ──────────────────────────────────────────────────────────
def run(fit_overrides, penalties, do_spell, dead_stage):
    F, names, _ = features.load_saved()
    dead, _, weight = labels.build()
    data = atd.load()
    owner = nearest_team_labels(data=data)
    n = len(dead)
    labelled = weight > 0
    owner_signed = np.where(owner == 0, 1, np.where(owner == 1, -1, 0))
    mechanical = weak_codes(dead, owner_signed)

    team_cols = select_features(names, "team")
    Xt = F[:, team_cols]
    if dead_stage == "hgb":
        dead_cols = select_features(names, "dead")
        Xd = F[:, dead_cols]
        print("rows {} | features dead {} team {} | labelled {}".format(
            n, len(dead_cols), len(team_cols), int(labelled.sum())))
    else:
        dead_mask, dead_details = release_deadball.detect(data, labels.load_events())
        print("rows {} | features team {} | labelled {} | dead release fixed ({} stoppages)".format(
            n, len(team_cols), int(labelled.sum()), len(dead_details["stoppages"])))
    print("fit overrides {}".format(fit_overrides or "defaults"))

    results = []
    for fold, train_mask, test_mask in folds(n):
        train = train_mask & labelled
        test = test_mask & labelled
        team_target = np.where((dead == 0) & (owner >= 0), owner, -1)
        team_train = train & (team_target >= 0)
        team_test = test & (team_target >= 0)

        y_team = (team_target == 0).astype(np.int8)
        team_model = fit(Xt[team_train], y_team[team_train],
                         balanced_weights(y_team[team_train]), **fit_overrides)
        p_home_train = team_model.predict_proba(Xt[team_train])[:, 1]
        p_home_test = team_model.predict_proba(Xt[team_test])[:, 1]
        t_train = binary(y_team[team_train], p_home_train)
        t_test = binary(y_team[team_test], p_home_test)
        p_home_full = team_model.predict_proba(Xt)[:, 1]

        row = {
            "fold": fold,
            "team_train_auc": t_train["auc"], "team_test_auc": t_test["auc"],
            "team_test_logloss": t_test["logloss"], "team_test_acc": t_test["acc"],
            "penalties": [],
        }
        if dead_stage == "hgb":
            y_dead = (dead == 1).astype(np.int8)
            dead_model = fit(Xd[train], y_dead[train], balanced_weights(y_dead[train]),
                             **fit_overrides)
            p_dead_train = dead_model.predict_proba(Xd[train])[:, 1]
            p_dead_test = dead_model.predict_proba(Xd[test])[:, 1]
            d_train = binary(y_dead[train], p_dead_train)
            d_test = binary(y_dead[test], p_dead_test)
            p_dead_full = dead_model.predict_proba(Xd)[:, 1]
            row.update({
                "dead_train_auc": d_train["auc"], "dead_test_auc": d_test["auc"],
                "dead_test_logloss": d_test["logloss"], "dead_test_acc": d_test["acc"],
            })

        for penalty in penalties:
            if dead_stage == "hgb":
                codes = viterbi_decode(p_dead_full, p_home_full, penalty, RUNS)
            else:
                codes = viterbi_decode_masked(p_home_full, dead_mask, penalty, RUNS)
            codes = force_known_dead(codes)
            codes = set_x_outside_runs(codes)
            scored = test & (mechanical != X)
            macro = (f1_score(mechanical[scored], codes[scored], average="macro",
                              labels=[H, A, D]) if scored.any() else float("nan"))
            spell, n_gt, n_pred = (weak_spells(mechanical, codes, test) if do_spell
                                   else (float("nan"), 0, 0))
            stats = structure(codes)
            row["penalties"].append({
                "penalty": penalty, "macro_f1": macro, "spell_f1": spell,
                "n_gt": n_gt, "n_pred": n_pred, **stats,
            })
        results.append(row)
        if dead_stage == "hgb":
            print("  fold {fold}: dead AUC {dead_train_auc:.3f}/{dead_test_auc:.3f} "
                  "team AUC {team_train_auc:.3f}/{team_test_auc:.3f}".format(**row))
        else:
            print("  fold {fold}: dead release (fixed) | team AUC "
                  "{team_train_auc:.3f}/{team_test_auc:.3f}".format(**row))

    print("\nmean / worst fold, weak-label truth")
    if dead_stage == "hgb":
        print("penalty   deadAUC  teamAUC  macroF1        spellF1        poss     mean_s  dead%")
        print("-" * 86)
    else:
        print("penalty   teamAUC  macroF1        spellF1        poss     mean_s  dead%")
        print("-" * 78)
    for j, penalty in enumerate(penalties):
        def col(key):
            return np.array([r["penalties"][j][key] for r in results], dtype=float)

        macro, spell = col("macro_f1"), col("spell_f1")
        poss, mean_s, dead_share = col("possessions"), col("mean_s"), col("dead_share")
        spell_mean = np.nanmean(spell) if np.isfinite(spell).any() else float("nan")
        spell_worst = np.nanmin(spell) if np.isfinite(spell).any() else float("nan")
        if dead_stage == "hgb":
            print("{:>7.1f}  {:>7.3f}  {:>7.3f}  {:.3f}/{:.3f}  {:.3f}/{:.3f}  "
                  "{:>5.0f}  {:>7.1f}  {:>5.1f}%".format(
                      penalty,
                      np.nanmean([r["dead_test_auc"] for r in results]),
                      np.nanmean([r["team_test_auc"] for r in results]),
                      np.nanmean(macro), np.nanmin(macro),
                      spell_mean, spell_worst,
                      np.nanmean(poss), np.nanmean(mean_s), 100 * np.nanmean(dead_share)))
        else:
            print("{:>7.1f}  {:>7.3f}  {:.3f}/{:.3f}  {:.3f}/{:.3f}  "
                  "{:>5.0f}  {:>7.1f}  {:>5.1f}%".format(
                      penalty,
                      np.nanmean([r["team_test_auc"] for r in results]),
                      np.nanmean(macro), np.nanmin(macro),
                      spell_mean, spell_worst,
                      np.nanmean(poss), np.nanmean(mean_s), 100 * np.nanmean(dead_share)))
    tgaps = [r["team_train_auc"] - r["team_test_auc"] for r in results]
    if dead_stage == "hgb":
        gaps = [r["dead_train_auc"] - r["dead_test_auc"] for r in results]
        print("\ntrain-minus-test AUC: dead mean {:.3f} worst {:.3f} | "
              "team mean {:.3f} worst {:.3f}".format(
                  np.mean(gaps), np.max(gaps), np.mean(tgaps), np.max(tgaps)))
    else:
        print("\ntrain-minus-test AUC: team mean {:.3f} worst {:.3f}".format(
            np.mean(tgaps), np.max(tgaps)))
        print("dead stage fixed (release detector); no fold-honest dead gap. "
              "Its weak-label agreement is in-sample.")
    print("weak labels only: Smart Tagging dead/alive plus nearest-owner team. Not ground truth.")


def run_half_swap(fit_overrides, dead_stage):
    F, names, _ = features.load_saved()
    dead, _, weight = labels.build()
    data = atd.load()
    owner = nearest_team_labels(data=data)
    n = len(dead)
    labelled = weight > 0
    team_cols = select_features(names, "team")
    Xt = F[:, team_cols]
    team_target = np.where((dead == 0) & (owner >= 0), owner, -1)

    if dead_stage == "hgb":
        dead_cols = select_features(names, "dead")
        Xd = F[:, dead_cols]
        y_dead = (dead == 1).astype(np.int8)
    else:
        print("dead stage: release detector (fixed); half-swap refits the team stage only")

    print("half-swap: fit on one half, score the other (stress test)")
    for fold, fit_mask, test_mask in half_folds(n):
        train = fit_mask & labelled
        test = test_mask & labelled
        team_train = train & (team_target >= 0)
        team_test = test & (team_target >= 0)
        team_model = fit(Xt[team_train], (team_target[team_train] == 0).astype(np.int8),
                         balanced_weights((team_target[team_train] == 0).astype(np.int8)),
                         **fit_overrides)
        p_home = team_model.predict_proba(Xt[team_test])[:, 1]
        t = binary((team_target[team_test] == 0).astype(np.int8), p_home)
        if dead_stage == "hgb":
            dead_model = fit(Xd[train], y_dead[train], balanced_weights(y_dead[train]),
                             **fit_overrides)
            p_dead = dead_model.predict_proba(Xd[test])[:, 1]
            d = binary(y_dead[test], p_dead)
            print("  half {} -> {}: dead AUC {:.3f} logloss {:.3f} | "
                  "team AUC {:.3f} logloss {:.3f}".format(
                      fold + 1, 2 - fold, d["auc"], d["logloss"], t["auc"], t["logloss"]))
        else:
            print("  half {} -> {}: team AUC {:.3f} logloss {:.3f}".format(
                fold + 1, 2 - fold, t["auc"], t["logloss"]))
    if dead_stage == "hgb":
        print("  a large dead AUC drop between directions means the halves are not the same task.")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--half-swap", action="store_true",
                        help="fit on one half and score the other instead of 5-fold")
    parser.add_argument("--dead-stage", choices=("release", "hgb"), default="release",
                        help="dead stage: fixed release detector (default) or the old HGB audit")
    parser.add_argument("--penalties", default="2,4,8,16,24,32,40,48,56,64,68,72,80,96,128",
                        help="comma-separated Viterbi switch penalties to compare")
    parser.add_argument("--no-spell", action="store_true",
                        help="skip weak-label spellF1 (faster)")
    parser.add_argument("--leaves", type=int, default=None, help="max_leaf_nodes override")
    parser.add_argument("--min-leaf", type=int, default=None, help="min_samples_leaf override")
    parser.add_argument("--l2", type=float, default=None, help="l2_regularization override")
    parser.add_argument("--iters", type=int, default=None, help="max_iter override")
    args = parser.parse_args()
    overrides = {}
    if args.leaves is not None:
        overrides["max_leaf_nodes"] = args.leaves
    if args.min_leaf is not None:
        overrides["min_samples_leaf"] = args.min_leaf
    if args.l2 is not None:
        overrides["l2_regularization"] = args.l2
    if args.iters is not None:
        overrides["max_iter"] = args.iters
    penalties = [float(value) for value in args.penalties.split(",")]
    return args, overrides, penalties


def main():
    args, overrides, penalties = parse_args()
    if args.half_swap:
        run_half_swap(overrides, args.dead_stage)
    else:
        run(overrides, penalties, not args.no_spell, args.dead_stage)


if __name__ == "__main__":
    main()

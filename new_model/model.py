"""Train the team stage and decode with the release's dead-ball detector.

Dead stage: the release's change-point detector (sonar.deadball).  It fits a logistic model to
per-frame physical evidence (ball out of bounds, ball speed, missing-run, player motion), decodes
a two-state path, then snaps stoppage edges to ball crossings, reacquisitions and strikes.  It is
reused as a fixed component, not retrained here: the HGB dead stage trained on the weak labels
overfit them (train AUC 1.000, 5-fold test mean 0.836) and its decoded D recall was the gap the
release detector closed.  Its weak-label agreement is in-sample, not a validation metric.

Team stage: home vs away on alive frames, trained on the tracking owner -- the player nearest the
ball -- because that is the scorer's team definition.  The Smart Tagging Team tag is phase-level
and disagrees with per-frame ownership; training on it was why the team stage underperformed the
nearest-player baseline.

The split is temporal.  Everything before SPLIT_FRAME is training; everything after is
validation.  No random split: consecutive frames are not independent, and a random split would
put the same possession on both sides of the fence.

The labels are Smart Tagging, not the scorer's ground truth.  The Viterbi penalty is chosen on
full-match possession structure -- mean duration near the validated release's 14.2 s (277
possessions), inside football's typical 10-15 s -- and not on weak-label macro-F1, which
reproduces the over-segmentation.  The public five minutes is never used here; decode.py uses it
only as a correctness check.

Run:
    python new_model/model.py
    python new_model/model.py --split 150000 --no-importance
"""

import argparse
import os
import sys
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import possession_metrics as pm  # noqa: E402
from games import DATA_DIR, GAME  # noqa: E402
from new_model import atd, features, labels  # noqa: E402
from sonar import deadball as release_deadball  # noqa: E402

H, A, D, X = 0, 1, 2, 3

# The two graded runs, half-open.  FAQ.md publishes them; the private ground truth has X outside.
RUNS = ((11185, 90652), (110808, 190198))
# FAQ.md: the second half starts at 113742 and the block before it is all dead.
SECOND_HALF_DEAD = (110808, 113742)

SPLIT_FRAME = 150000
# Wide enough to see the whole structure curve.  With the release dead mask, penalty 8
# over-segments at 9.0 s, 68 lands on the release's 14.2 s, and 128 is already too smooth.
# Viterbi is cheap next to training.
VITERBI_PENALTIES = (2.0, 4.0, 8.0, 16.0, 24.0, 32.0, 40.0, 48.0, 56.0, 64.0, 68.0, 72.0,
                     80.0, 96.0, 128.0)
# Full-match mean possession to aim the decoder at.  The validated release's final decode is
# 277 possessions of mean 14.2 s (sonar/POSSESSION.md), which sits inside football's typical
# 10-15 s.  It is a structure target, not a score target.
TARGET_MEAN_S = 14.2

# Nearest-player ownership flickers on deflections and crowded frames.  The reference baseline
# needs a centred majority vote to turn that signal into possessions, and its published full-match
# sweep picks +/-50 frames (2 s) as the best variant; use the same window to denoise the label.
TEAM_SMOOTH = 50
ARTIFACT = os.path.join(DATA_DIR, "cache", "{}_model.joblib".format(GAME))


def select_features(names, target):
    """Column indices for one stage.

    sp_* is dropped entirely.  sp_pre and sp_count_10s *are* the weak dead label's definition,
    and the rest of the family is set-piece timing that can reconstruct it.  A dead model trained
    with them learns the label rule, not the ball state.  The tracking features carry the signal
    on their own.

    homo_* is homography quality, not football, and scores near chance.  ball_spot_* is mostly
    NaN on the frames that matter.  ball_still and ball_speed_change are touch proxies already
    covered by time_since_touch.

    time_frame_in_half is dropped for a different reason: it resets at halftime, so a first-half
    frame and a second-half frame at the same clock offset share a value.  The model can memorise
    the first half and apply it to the second, which is exactly the cross-half leak that pushed
    the first run's training AUC to 1.000.  time_half stays; it is a half indicator, not a clock.
    """
    drop = {"ball_still", "ball_speed_change", "time_frame_in_half"}
    drop |= {name for name in names if name.startswith("sp_")}
    drop |= {name for name in names if name.startswith("homo_")}
    drop |= {name for name in names if name.startswith("ball_spot_")}
    if target == "dead":
        drop.add("ball_vx_progress")  # no dead signal; keep it for the team model
    return [j for j, name in enumerate(names) if name not in drop]


def balanced_weights(y):
    """Inverse-frequency weights so the majority class does not win by default."""
    y = np.asarray(y).astype(np.int8)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return np.where(y == 1, n0 / max(n1, 1), 1.0)


FIT_DEFAULTS = {
    "max_iter": 150,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 200,
    "l2_regularization": 2.0,
    "early_stopping": False,
    "random_state": 0,
}


def fit(X, y, weight, **overrides):
    """Fit stage classifier. ``overrides`` lets the CV harness sweep capacity without changing
    the defaults used by model.py's artifact."""
    config = dict(FIT_DEFAULTS)
    config.update(overrides)
    model = HistGradientBoostingClassifier(**config)
    model.fit(X, y, sample_weight=weight)
    return model


def binary_report(name, y, p, positive):
    pred = (p >= 0.5).astype(np.int8)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    print("\n{}".format(name))
    print("  AUC {:.3f} | logloss {:.3f} | acc {:.3f} | macroF1 {:.3f}".format(
        roc_auc_score(y, p), log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        accuracy_score(y, pred), f1_score(y, pred, average="macro")))
    print("  confusion (rows truth, cols prediction)")
    print("               pred {:>5}  pred {:>5}".format("neg", "pos"))
    print("    truth neg  {:>10}  {:>10}".format(cm[0, 0], cm[0, 1]))
    print("    truth pos  {:>10}  {:>10}".format(cm[1, 0], cm[1, 1]))
    print("  positive = {}".format(positive))


def emissions(p_dead, p_home):
    """(n, 3) log-emission probabilities for H, A, D."""
    p_dead = np.clip(np.asarray(p_dead, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    p_home = np.clip(np.asarray(p_home, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    alive = 1.0 - p_dead
    p = np.column_stack((alive * p_home, alive * (1.0 - p_home), p_dead))
    p /= p.sum(axis=1, keepdims=True)
    return np.log(p)


def viterbi_decode(p_dead, p_home, penalty, runs):
    """Most likely H/A/D sequence for the HGB audit (soft dead probability).

    A per-frame argmax has no notion of a possession: it flips on a single noisy frame, and
    spellF1 charges every flip.  The Viterbi path maximises the summed log probability with a
    fixed cost for every label switch, so a switch has to be paid for by a real change in the
    emissions.  A direct H<->A switch costs two penalties, the same as H->D->A, so the decoder
    does not prefer an invented D over a direct turnover; any switch still has to earn its cost.
    The shipped decoder is viterbi_decode_masked, which hard-constrains D to the release mask.
    """
    return viterbi_path(emissions(p_dead, p_home), penalty, runs)


def viterbi_decode_masked(p_home, dead_mask, penalty, runs):
    """Decode H/A inside the release detector's alive stretches, D where it says dead.

    The dead mask is a hard constraint, not a soft emission: the release detector already
    settles *whether* and *when* the ball is dead from physical signals and refined edges.  The
    Viterbi pass then only has to settle team switches within the alive stretches, using the
    same switch cost as the probability decoder.
    """
    p_home = np.clip(np.asarray(p_home, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    log_emission = np.empty((len(p_home), 3), dtype=np.float64)
    log_emission[:, H] = np.log(p_home)
    log_emission[:, A] = np.log(1.0 - p_home)
    log_emission[:, D] = 0.0
    log_emission[dead_mask, H] = -1e9
    log_emission[dead_mask, A] = -1e9
    log_emission[~dead_mask, D] = -1e9
    return viterbi_path(log_emission, penalty, runs)


def viterbi_path(log_emission, penalty, runs):
    """DP shared by the probability decoder and the hard-dead-mask decoder."""
    log_transition = np.full((3, 3), -float(penalty), dtype=np.float64)
    np.fill_diagonal(log_transition, 0.0)
    log_transition[H, A] = log_transition[A, H] = -2.0 * float(penalty)

    codes = np.full(len(log_emission), X, dtype=np.int8)
    for start, stop in runs:
        em = log_emission[start:stop]
        n = len(em)
        if n == 0:
            continue
        delta = np.empty((n, 3), dtype=np.float64)
        back = np.empty((n, 3), dtype=np.int8)
        delta[0] = em[0]
        back[0] = 0
        for t in range(1, n):
            scores = delta[t - 1][:, None] + log_transition
            back[t] = np.argmax(scores, axis=0)
            delta[t] = em[t] + scores[back[t], np.arange(3)]
        path = np.empty(n, dtype=np.int8)
        path[-1] = int(np.argmax(delta[-1]))
        for t in range(n - 2, -1, -1):
            path[t] = back[t + 1, path[t + 1]]
        codes[start:stop] = path
    return codes


def force_known_dead(codes):
    """The published pre-kickoff block is dead; do not let the model argue with it."""
    codes[SECOND_HALF_DEAD[0]:SECOND_HALF_DEAD[1]] = D
    return codes


def set_x_outside_runs(codes):
    """X outside the two graded runs, matching the ground truth's own abstention."""
    codes[:RUNS[0][0]] = X
    codes[RUNS[0][1]:RUNS[1][0]] = X
    codes[RUNS[1][1]:] = X
    return codes


def weak_codes(dead, team):
    """H/A/D/X codes for the weak labels, in the same alphabet as the metric."""
    codes = np.full(len(dead), X, dtype=np.int8)
    codes[dead == 1] = D
    codes[(dead == 0) & (team == 1)] = H
    codes[(dead == 0) & (team == -1)] = A
    return codes


def spell_stats(codes):
    spells = pm.possession_spells(codes, RUNS)
    lengths = [stop - start for start, stop, _ in spells]
    return len(spells), (np.mean(lengths) / 25.0 if lengths else 0.0)


def nearest_team_labels(window=TEAM_SMOOTH, data=None):
    """Per-frame team of the player nearest the ball: 0 home, 1 away, -1 unknown.

    The scorer's team label is the tracking owner, so this is the right target for the team
    stage.  The Smart Tagging Team tag is phase-level and disagrees with per-frame ownership;
    training on it was why the team stage underperformed the nearest-player baseline.  The raw
    argmax flickers, so the target is a centred majority vote over +/- ``window`` frames.
    """
    from starter.baseline import smooth

    if data is None:
        data = atd.load()
    ball = data["ball"]
    players = data["players"]
    team_of_slot = data["team"]
    n = len(ball)
    known = np.isfinite(ball[:, 0]) & np.isfinite(ball[:, 1])
    distance = np.linalg.norm(players[:, :, :2] - ball[:, None, :2], axis=2)
    distance[~np.isfinite(distance)] = np.inf
    nearest = np.argmin(distance, axis=1)
    owner = np.full(n, -1, dtype=np.int8)
    valid = known & np.isfinite(distance[np.arange(n), nearest])
    owner[valid] = team_of_slot[nearest[valid]].astype(np.int8)
    labels = smooth(owner, window)
    return np.where(labels == "H", 0, np.where(labels == "A", 1, -1)).astype(np.int8)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", type=int, default=SPLIT_FRAME,
                        help="first validation frame (default: %(default)s)")
    parser.add_argument("--no-importance", action="store_true",
                        help="skip permutation importance (faster)")
    args = parser.parse_args()

    start = time.time()
    F, names, groups = features.load_saved()
    dead, _, weight = labels.build()
    data = atd.load()
    events = labels.load_events()
    owner = nearest_team_labels(data=data)
    n = len(dead)
    frame = np.arange(n)

    labelled = weight > 0
    train = labelled & (frame < args.split)
    val = labelled & (frame >= args.split)
    team_target = np.where((dead == 0) & (owner >= 0), owner, -1)
    team_train = train & (team_target >= 0)
    team_val = val & (team_target >= 0)

    print("rows {} | features {} | labelled {} | train {} | val {}".format(
        n, F.shape[1], int(labelled.sum()), int(train.sum()), int(val.sum())))
    print("home share of team train {:.3f} | val {:.3f}".format(
        (team_target[team_train] == 0).mean(), (team_target[team_val] == 0).mean()))

    # Dead stage: the release detector, fixed.  Its weak-label agreement is in-sample because
    # the detector is fit on those labels; it is printed for provenance, not as validation.
    dead_mask, dead_details = release_deadball.detect(data, events)
    graded_mask = np.zeros(n, dtype=bool)
    for run_start, run_stop in RUNS:
        graded_mask[run_start:run_stop] = True
    dead_share = float(dead_mask[graded_mask].mean())
    weak_dead, weak_alive = dead == 1, dead == 0
    print("dead stage: release change-point detector | {} stoppages | {:.1f}% of graded dead".format(
        len(dead_details["stoppages"]), 100.0 * dead_share))
    print("  in-sample weak-label agreement (train): dead recall {:.3f} | "
          "alive false-dead {:.3f}".format(
              (dead_mask & weak_dead & train).sum() / max((weak_dead & train).sum(), 1),
              (dead_mask & weak_alive & train).sum() / max((weak_alive & train).sum(), 1)))

    team_cols = select_features(names, "team")
    Xt = F[:, team_cols]
    print("team features: {}".format(len(team_cols)))

    # Team stage: home vs away, on weak-alive, owned frames.  Target is the tracking owner,
    # not the phase-level Smart Tagging Team tag.
    y_team = (team_target == 0).astype(np.int8)
    team_model = fit(Xt[team_train], y_team[team_train],
                     balanced_weights(y_team[team_train]))
    p_home_train = team_model.predict_proba(Xt[team_train])[:, 1]
    p_home_val = team_model.predict_proba(Xt[team_val])[:, 1]
    binary_report("home vs away (train)", y_team[team_train], p_home_train, "home")
    binary_report("home vs away (validation)", y_team[team_val], p_home_val, "home")

    # Decode sweep.  The dead mask is hard; the penalty only settles team switches.  Macro-F1
    # against the weak labels stays diagnostic: the weak labels over-segment, so fitting them
    # rewards exactly the over-segmentation the scorer penalises.  Structure picks.
    p_home_full = team_model.predict_proba(Xt)[:, 1]
    owner_signed = np.where(owner == 0, 1, np.where(owner == 1, -1, 0))
    mechanical = weak_codes(dead, owner_signed)
    scored = val & (mechanical != X)
    rows = []
    for penalty in VITERBI_PENALTIES:
        codes = viterbi_decode_masked(p_home_full, dead_mask, penalty, RUNS)
        codes = force_known_dead(codes)
        codes = set_x_outside_runs(codes)
        f1 = f1_score(mechanical[scored], codes[scored], average="macro", labels=[H, A, D])
        acc = accuracy_score(mechanical[scored], codes[scored])
        n_dead = int(np.sum(codes == D))
        n_spells, mean_duration = spell_stats(codes)
        rows.append({"penalty": penalty, "macro_f1": f1, "accuracy": acc,
                     "n_dead": n_dead, "n_spells": n_spells,
                     "mean_duration": mean_duration})

    print("\nviterbi decode sweep (macroF1 is diagnostic; structure picks)")
    print("{:>7} {:>8} {:>6} {:>8} {:>9} {:>8}".format(
        "penalty", "macroF1", "acc", "n_dead", "n_spells", "mean_s"))
    print("-" * 52)
    for row in rows:
        print("{:>7.1f} {:>8.3f} {:>6.3f} {:>8d} {:>9d} {:>8.1f}".format(
            row["penalty"], row["macro_f1"], row["accuracy"],
            row["n_dead"], row["n_spells"], row["mean_duration"]))

    # The weak labels are Smart Tagging plus nearest-owner, and they fragment possession: the
    # macro-F1-optimal penalty decodes to hundreds of short spells, while the validated release
    # decodes to 277 of mean 14.2 s.  The dead mask is now the release's, so the penalty only
    # settles team switches.  Pick the penalty whose full-match mean is closest to TARGET_MEAN_S;
    # macro-F1 stays diagnostic and does not pick.
    best = min(rows, key=lambda row: (abs(row["mean_duration"] - TARGET_MEAN_S), row["penalty"]))
    print("\nchosen: viterbi penalty {:.1f}, {} possessions of mean {:.1f} s "
          "(target {:.1f} s; mechanical-val macroF1 {:.3f}, diagnostic only)".format(
              best["penalty"], best["n_spells"], best["mean_duration"], TARGET_MEAN_S,
              best["macro_f1"]))

    # Permutation importance, on the validation split.  Diagnostics, not selection.
    if not args.no_importance:
        print("\npermutation importance (validation AUC, top 15)")
        result = permutation_importance(
            team_model, Xt[team_val], y_team[team_val], scoring="roc_auc",
            n_repeats=3, random_state=0, n_jobs=1)
        order = np.argsort(result.importances_mean)[::-1][:15]
        print("\n  team model")
        for j in order:
            print("    {:<34} {:.4f} +/- {:.4f}".format(
                names[team_cols[j]], result.importances_mean[j],
                result.importances_std[j]))

    artifact = {
        "team_model": team_model,
        "team_features": [names[j] for j in team_cols],
        "dead_mask": dead_mask,
        "dead_source": "sonar.deadball.detect",
        "dead_stoppages": len(dead_details["stoppages"]),
        "dead_share": dead_share,
        "dead_refine": dead_details["refine"],
        "viterbi_penalty": best["penalty"],
        "viterbi_mean_s": best["mean_duration"],
        "viterbi_n_spells": best["n_spells"],
        "split_frame": args.split,
        "runs": RUNS,
        "second_half_dead": SECOND_HALF_DEAD,
        "mechanical_val_macro_f1": best["macro_f1"],
    }
    os.makedirs(os.path.dirname(ARTIFACT), exist_ok=True)
    joblib.dump(artifact, ARTIFACT)
    print("\nsaved -> {}".format(os.path.relpath(ARTIFACT, ROOT)))
    print("done in {:.1f} s".format(time.time() - start))


if __name__ == "__main__":
    main()

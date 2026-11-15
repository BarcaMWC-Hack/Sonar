"""Key performance metrics for Sonar's two trained models, against fixed benchmarks.

Both models are logistic regressions, trained and scored on graded frames only. Each is scored
out of sample on the team's shared folds (sonar/splits.py), then stress-tested by fitting on one
half of the match and scoring the other.

    dead-ball model    labels: weak labels from Smart Tagging (sonar/deadball.py)
    team-shape model   labels: live frames where the ball evidence is clear (sonar/team_shape.py)

Neither set of labels is the ground truth, so these numbers show how well a model carries over to
unseen play, not how often it is right about the match. The possession decoder outputs labels, not
probabilities, so it has no AUC: spellF1 and macroF1 from evaluation/run_evaluation.py judge it.

Benchmarks:
    ROC-AUC            0.7-0.8 acceptable, 0.8-0.9 excellent, above 0.9 look for leakage
    KS                 40-70 good, above 70 look for overfitting
    fitted vs held out AUC above 0.7 on every fold; KS within 10 points, mean and worst fold
    Hosmer-Lemeshow    p above 0.05

Accuracy is printed but not benchmarked, because the weak labels are imbalanced. The models are
fitted with classes reweighted to equal mass, so log loss is class-balanced too (a coin flip scores
ln 2 = 0.693), and calibration is tested after shifting the logit by the fitted part's class odds.
Calibration uses one frame every 5 s: neighbouring frames are near copies, and the test assumes
independent samples.

Prints aggregates only.

    python tools/kpm_report.py
"""

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from sonar import atd, deadball, possession, splits, team_shape  # noqa: E402
from starter import load_smart_tagging  # noqa: E402
from tools.deadball_report import auc  # noqa: E402

# Params
CALIBRATION_EVERY = 125     # frames between calibration samples (5 s at 25 fps)
CALIBRATION_GROUPS = 10     # gives 8 degrees of freedom for Hosmer-Lemeshow
AUC_FLOOR = 0.7
KS_GAP_LIMIT = 10.0
HL_ALPHA = 0.05


def ks_points(scores, labels):
    """Largest gap between the two classes' cumulative score distributions, on a 0-100 scale."""
    order = np.argsort(scores, kind="mergesort")
    ordered, y = scores[order], labels[order].astype(bool)
    positives = np.cumsum(y) / max(int(y.sum()), 1)
    negatives = np.cumsum(~y) / max(int((~y).sum()), 1)
    end_of_tie = np.append(ordered[1:] != ordered[:-1], True)
    return 100.0 * float(np.max(np.abs(positives - negatives)[end_of_tie]))


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-values))


def balanced_log_loss(logit, y):
    p = np.clip(sigmoid(logit), 1e-12, 1.0 - 1e-12)
    return 0.5 * float(np.mean(-np.log(p[y]))) + 0.5 * float(np.mean(-np.log(1.0 - p[~y])))


def hosmer_lemeshow(logit, y, prior):
    """(chi-square, p-value) over equal-sized probability groups, 8 degrees of freedom."""
    p = sigmoid(logit + prior)
    order = np.argsort(p, kind="mergesort")
    statistic = 0.0
    for group in np.array_split(order, CALIBRATION_GROUPS):
        expected = float(p[group].sum())
        observed = float(y[group].sum())
        variance = expected * (1.0 - expected / len(group))
        statistic += (observed - expected) ** 2 / max(variance, 1e-12)
    half = statistic / 2.0
    return statistic, math.exp(-half) * sum(half ** k / math.factorial(k) for k in range(4))


def metrics(logit, y):
    predicted = logit > 0
    tp, fp = int(np.sum(predicted & y)), int(np.sum(predicted & ~y))
    fn, tn = int(np.sum(~predicted & y)), int(np.sum(~predicted & ~y))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "auc": float(auc(logit, y)),
        "ks": ks_points(logit, y),
        "log_loss": balanced_log_loss(logit, y),
        "precision": precision,
        "recall": recall,
        "f1": 2.0 * precision * recall / max(precision + recall, 1e-12),
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
        "balanced_accuracy": 0.5 * (recall + tn / max(tn + fp, 1)),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def auc_band(value):
    if value < AUC_FLOOR:
        return "below acceptable"
    if value < 0.8:
        return "acceptable"
    if value <= 0.9:
        return "excellent"
    return "above 0.9: look for leakage"


def ks_band(value):
    if value < 40.0:
        return "below 40"
    if value <= 70.0:
        return "good"
    return "above 70: look for overfitting"


def verdict(ok):
    return "PASS" if ok else "FAIL"


def print_row(label, m):
    row = "    {:<9} {:>6.3f} {:>6.1f} {:>9.3f} {:>10.3f} {:>7.3f} {:>6.3f} {:>9.3f} {:>13.3f}"
    print(row.format(
        label, m["auc"], m["ks"], m["log_loss"], m["precision"], m["recall"], m["f1"],
        m["accuracy"], m["balanced_accuracy"]))


def fold_report(x, y, usable, score, l2):
    n = len(y)
    print("  Shared folds: {} time blocks per half, {} s left out either side of each test "
          "block".format(splits.N_FOLDS, splits.GAP_S))
    head = "    {:<6} {:>6} {:>6} {:>9} {:>10} {:>7} {:>6} {:>13} {:>10} {:>7}"
    print(head.format("fold", "AUC", "KS", "log loss", "precision", "recall", "F1", "balanced acc",
                      "fitted KS", "KS gap"))
    row = ("    {:<6} {:>6.3f} {:>6.1f} {:>9.3f} {:>10.3f} {:>7.3f} {:>6.3f} {:>13.3f} {:>10.1f} "
           "{:>7.1f}")
    rows, pooled_logit, pooled_y = [], [], []
    confusion = dict.fromkeys(("tp", "fp", "fn", "tn"), 0)
    for fold, fit, held_out in splits.folds(n):
        train, test = usable & fit, usable & held_out
        w = deadball.fit_logistic(x[train], y[train].astype(float), l2=l2)
        prior = math.log(y[train].sum() / (~y[train]).sum())
        fitted = metrics(score(x[train], w), y[train])
        logit = score(x[test], w)
        held = metrics(logit, y[test])
        rows.append((fitted, held))
        for key in confusion:
            confusion[key] += held[key]
        sample = np.flatnonzero(test) % CALIBRATION_EVERY == 0
        pooled_logit.append(logit[sample] + prior)
        pooled_y.append(y[test][sample])
        print(row.format(fold, held["auc"], held["ks"], held["log_loss"], held["precision"],
                         held["recall"], held["f1"], held["balanced_accuracy"], fitted["ks"],
                         abs(fitted["ks"] - held["ks"])))
    held = [r[1] for r in rows]
    gaps = [abs(r[0]["ks"] - r[1]["ks"]) for r in rows]
    mean_auc, mean_ks = np.mean([m["auc"] for m in held]), np.mean([m["ks"] for m in held])
    print("    mean: AUC {:.3f}, KS {:.1f}, log loss {:.3f} | lowest fold: AUC {:.3f}, "
          "KS {:.1f}".format(mean_auc, mean_ks, np.mean([m["log_loss"] for m in held]),
                             min(m["auc"] for m in held), min(m["ks"] for m in held)))
    print("    confusion, all held-out frames: TP {tp}  FP {fp}  FN {fn}  TN {tn}".format(
        **confusion))
    print("    bands, mean: AUC {}; KS {}".format(auc_band(mean_auc), ks_band(mean_ks)))
    lowest = min(min(r[0]["auc"], r[1]["auc"]) for r in rows)
    print("    fitted vs held out: AUC above {} on every fold: {}; KS gap mean {:.1f}: {}; "
          "worst {:.1f}: {}".format(AUC_FLOOR, verdict(lowest > AUC_FLOOR), np.mean(gaps),
                                    verdict(np.mean(gaps) <= KS_GAP_LIMIT), max(gaps),
                                    verdict(max(gaps) <= KS_GAP_LIMIT)))
    logit, labels = np.concatenate(pooled_logit), np.concatenate(pooled_y)
    statistic, p_value = hosmer_lemeshow(logit, labels, 0.0)
    print("    Hosmer-Lemeshow, pooled held-out frames, one per {} s (n={}): chi-square {:.1f}, "
          "p {:.4f}, above {}: {}".format(CALIBRATION_EVERY // 25, len(labels), statistic, p_value,
                                          HL_ALPHA, verdict(p_value > HL_ALPHA)))


def report(title, x, y, usable, score, l2):
    print(title)
    fold_report(x, y, usable, score, l2)
    print("  Stress test: fit on one half, score the other")
    for fit_half, fit, held_out in splits.half_folds(len(y)):
        test_half = 1 - fit_half
        train, test = usable & fit, usable & held_out
        w = deadball.fit_logistic(x[train], y[train].astype(float), l2=l2)
        prior = math.log(y[train].sum() / (~y[train]).sum())
        fitted = metrics(score(x[train], w), y[train])
        logit = score(x[test], w)
        held = metrics(logit, y[test])

        print("  fit half {}, score half {} ({} positive / {} negative frames held out)".format(
            fit_half + 1, test_half + 1, int(y[test].sum()), int((~y[test]).sum())))
        print("    {:<9} {:>6} {:>6} {:>9} {:>10} {:>7} {:>6} {:>9} {:>13}".format(
            "", "AUC", "KS", "log loss", "precision", "recall", "F1", "accuracy", "balanced acc"))
        print_row("fitted", fitted)
        print_row("held out", held)
        print("    confusion, held out: TP {tp}  FP {fp}  FN {fn}  TN {tn}".format(**held))
        print("    bands, held out: AUC {}; KS {}".format(auc_band(held["auc"]),
                                                       ks_band(held["ks"])))
        gap = abs(fitted["ks"] - held["ks"])
        print("    fitted vs held out: AUC drop {:+.3f}, both above {}: {}; KS gap {:.1f} points, "
              "limit {:.0f}: {}".format(
                  fitted["auc"] - held["auc"], AUC_FLOOR,
                  verdict(min(fitted["auc"], held["auc"]) > AUC_FLOOR), gap, KS_GAP_LIMIT,
                  verdict(gap <= KS_GAP_LIMIT)))
        sample = np.flatnonzero(test) % CALIBRATION_EVERY == 0
        statistic, p_value = hosmer_lemeshow(logit[sample], y[test][sample], prior)
        print("    Hosmer-Lemeshow, held out, one frame per {} s (n={}): chi-square {:.1f}, "
              "p {:.4f}, above {}: {}".format(
                  CALIBRATION_EVERY // 25, int(sample.sum()), statistic, p_value, HL_ALPHA,
                  verdict(p_value > HL_ALPHA)))
    print()


if __name__ == "__main__":
    data = atd.load()
    events = load_smart_tagging.load()
    dead, details = deadball.detect(data, events)

    report("Dead-ball model (positive = dead; labels are Smart Tagging weak labels, not truth)",
           details["x"], details["weak_dead"], details["weak_dead"] | details["weak_alive"],
           deadball.log_likelihood_ratio, deadball.L2)

    _, shape = possession.evidence(data, dead)
    ball, x = shape["ball"], shape["shape_x"]
    report("Team-shape model (positive = Barca; labels are frames with clear ball control, not "
           "truth)", x, ball > 0, team_shape.training_mask(ball, shape["alive"], x),
           team_shape.log_odds, team_shape.L2)

"""Rank the feature table against the weak labels, one feature at a time.

This is feature selection, not model selection.  For each column it reports:

  coverage   share of frames where the feature is not NaN
  auc_dead   how well the feature ranks dead (1) above alive (0), on labelled frames
  auc_team   how well it ranks home (+1) above away (-1), on labelled frames
  means      mean value in each class, so the direction is readable

AUC 0.5 is no signal; 1.0 is perfect ranking (0.0 is perfect in the opposite direction).  Because
the labels come from Smart Tagging, a high AUC means "this feature agrees with Smart Tagging",
not "this feature is correct".  Use this to drop dead weight and to sanity-check that the obvious
signals are present before you train anything.

    python new_model/audit_features.py
    python new_model/audit_features.py --top 40
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import DATA_DIR, GAME  # noqa: E402
from new_model import atd, features, labels  # noqa: E402
from starter.load_smart_tagging import load as load_events  # noqa: E402

# These columns are part of the weak dead label's definition, not independent evidence.  They are
# shown in the full ranking so the circularity is visible, then removed from the filtered ranking.
LEAKY = {"sp_pre", "sp_count_10s"}


def auc(y, x):
    """Rank AUC for binary y in {0, 1}. NaN if one class is absent."""
    y = np.asarray(y).astype(np.int8)
    x = np.asarray(x, dtype=np.float64)
    n1 = int(np.sum(y == 1))
    n0 = int(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = rankdata(x)
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def load_table():
    try:
        X, names, groups = features.load_saved()
        print("loaded cached features")
        return X, names, groups
    except FileNotFoundError:
        print("no feature cache; building it now ...")
        data = atd.load()
        X, names, groups = features.build(data)
        path = features.save(X, names, groups)
        print("cached -> {}".format(os.path.relpath(path, ROOT)))
        return X, names, groups


def audit(X, names, groups, dead, team):
    rows = []
    for j, name in enumerate(names):
        x = X[:, j]
        finite = np.isfinite(x)
        n_finite = int(finite.sum())
        coverage = n_finite / max(len(x), 1)
        mean = float(np.nanmean(x)) if n_finite else np.nan
        std = float(np.nanstd(x)) if n_finite else np.nan

        dmask = (dead != -1) & finite
        tmask = (team != 0) & finite
        auc_dead = auc(dead[dmask], x[dmask]) if dmask.any() else np.nan
        auc_team = auc((team[tmask] == 1).astype(np.int8), x[tmask]) if tmask.any() else np.nan

        rows.append({
            "name": name,
            "group": groups[j],
            "coverage": coverage,
            "mean": mean,
            "std": std,
            "n_dead": int(np.sum(dead == 1)),
            "n_alive": int(np.sum(dead == 0)),
            "n_dead_used": int(dmask.sum()),
            "n_team_used": int(tmask.sum()),
            "mean_dead": float(np.nanmean(x[dead == 1])) if (dead == 1).any() else np.nan,
            "mean_alive": float(np.nanmean(x[dead == 0])) if (dead == 0).any() else np.nan,
            "n_home": int(np.sum(team == 1)),
            "n_away": int(np.sum(team == -1)),
            "mean_home": float(np.nanmean(x[team == 1])) if (team == 1).any() else np.nan,
            "mean_away": float(np.nanmean(x[team == -1])) if (team == -1).any() else np.nan,
            "auc_dead": auc_dead,
            "auc_team": auc_team,
            "abs_auc_dead": abs(auc_dead - 0.5) if np.isfinite(auc_dead) else np.nan,
            "abs_auc_team": abs(auc_team - 0.5) if np.isfinite(auc_team) else np.nan,
        })
    return pd.DataFrame(rows)


def show(frame, column, title, n, count_col="n_dead_used"):
    print("\n" + title)
    print("{:<34} {:<7} {:>6} {:>7} {:>7} {:>9} {:>8} {:>8}".format(
        "feature", "group", "cov%", "aucD", "aucT", "n_used", "dead", "alive"))
    print("-" * 98)
    for _, row in frame.sort_values(column, ascending=False).head(n).iterrows():
        print("{:<34} {:<7} {:>5.1f}% {:>7.3f} {:>7.3f} {:>9} {:>8.3f} {:>8.3f}".format(
            row["name"], row["group"], 100 * row["coverage"],
            row["auc_dead"] if np.isfinite(row["auc_dead"]) else np.nan,
            row["auc_team"] if np.isfinite(row["auc_team"]) else np.nan,
            int(row[count_col]),
            row["mean_dead"], row["mean_alive"]))


def main():
    top = 25
    if "--top" in sys.argv:
        top = int(sys.argv[sys.argv.index("--top") + 1])

    X, names, groups = load_table()
    events = load_events()
    dead, team, weight = labels.build(events)

    print("\nfeature table: {} rows x {} columns".format(*X.shape))
    print("weak labels: {}".format(labels.summary(dead, team, weight)))

    table = audit(X, names, groups, dead, team)
    table["leaky"] = table["name"].isin(LEAKY)
    out = os.path.join(DATA_DIR, "cache", "{}_feature_audit.csv".format(GAME))
    table.to_csv(out, index=False)
    print("audit -> {}".format(os.path.relpath(out, ROOT)))

    print("\n{:<8} {:>5} {:>9} {:>9} {:>9}".format("group", "n", "mean cov", "best |aucD|", "best |aucT|"))
    print("-" * 50)
    for group in dict.fromkeys(groups):
        sub = table[table["group"] == group]
        print("{:<8} {:>5} {:>8.1f}% {:>9.3f} {:>9.3f}".format(
            group, len(sub), 100 * sub["coverage"].mean(),
            sub["abs_auc_dead"].max(), sub["abs_auc_team"].max()))

    show(table, "abs_auc_dead", "Top {} features for DEAD vs ALIVE".format(top), top)
    print("\nnote: sp_pre and sp_count_10s are the weak dead label's own definition. They rank")
    print("      high because the label was built from them. The next table removes them.")
    show(table[~table["leaky"]], "abs_auc_dead",
         "Top {} features for DEAD vs ALIVE (label-definition columns removed)".format(top), top)
    show(table, "abs_auc_team", "Top {} features for HOME vs AWAY".format(top), top,
         count_col="n_team_used")

    weak = table[table["abs_auc_dead"] < 0.02]
    sparse = table[table["coverage"] < 0.5]
    constant = table[table["std"] < 1e-9]
    low_dead = table[table["n_dead_used"] < 500]
    low_team = table[table["n_team_used"] < 500]
    print("\nflags")
    print("  no dead signal (|aucD| < 0.02): {}".format(len(weak)))
    print("  mostly missing (coverage < 50%): {}".format(len(sparse)))
    print("  constant: {}".format(len(constant)))
    print("  low support, dead (< 500 frames): {}".format(len(low_dead)))
    print("  low support, team (< 500 frames): {}".format(len(low_team)))
    if len(weak):
        print("  weak columns: {}".format(", ".join(weak["name"].tolist()[:12])))
    if len(sparse):
        print("  sparse columns: {}".format(", ".join(sparse["name"].tolist()[:12])))
    if len(constant):
        print("  constant columns: {}".format(", ".join(constant["name"].tolist())))


if __name__ == "__main__":
    main()

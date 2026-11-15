"""Decode the trained team stage into a submission.

Loads the artifact from model.py, predicts team probability on every frame, decodes H/A inside
the release detector's dead mask, applies the published second-half dead block, writes X outside
the two graded runs, and validates the file.

    python new_model/decode.py
    python new_model/decode.py --score   # also run the public 5-minute correctness check

The public score is a correctness check, not a validation score.  The public window sits inside
the training period, and the rules forbid tuning on it.  Do not iterate against it.
"""

import argparse
import os
import sys

import joblib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)

import possession_metrics as pm  # noqa: E402
from games import DATA_DIR, GAME  # noqa: E402
from new_model import features  # noqa: E402
from new_model.model import (  # noqa: E402
    RUNS, force_known_dead, set_x_outside_runs, viterbi_decode_masked,
)

ARTIFACT = os.path.join(DATA_DIR, "cache", "{}_model.joblib".format(GAME))
SUBMISSION = os.path.join(HERE, "submission.csv")
PUBLIC_GT = os.path.join(ROOT, "evaluation", "gt",
                         "{}_possession_gt_public.csv".format(GAME))
LABEL_NAMES = np.array(["H", "A", "D", "X"])


def write_submission(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(i, label)
                          for i, label in enumerate(labels))
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--score", action="store_true",
                        help="score against the public 5-minute ground truth (correctness check)")
    args = parser.parse_args()

    if not os.path.isfile(ARTIFACT):
        raise SystemExit("no model at {}; run python new_model/model.py first".format(ARTIFACT))

    artifact = joblib.load(ARTIFACT)
    X, names, groups = features.load_saved()
    column = {name: j for j, name in enumerate(names)}

    Xt = X[:, [column[name] for name in artifact["team_features"]]]
    p_home = artifact["team_model"].predict_proba(Xt)[:, 1]
    dead_mask = artifact["dead_mask"]

    codes = viterbi_decode_masked(p_home, dead_mask, artifact["viterbi_penalty"], RUNS)
    codes = force_known_dead(codes)
    codes = set_x_outside_runs(codes)

    labels = LABEL_NAMES[codes]
    path = write_submission(SUBMISSION, labels)

    counts = {name: int(np.sum(labels == name)) for name in LABEL_NAMES}
    spells = pm.possession_spells(codes, RUNS)
    durations = [stop - start for start, stop, _ in spells]
    print("wrote {}".format(os.path.relpath(path, ROOT)))
    print("rows {} | H {} | A {} | D {} | X {}".format(
        len(labels), counts["H"], counts["A"], counts["D"], counts["X"]))
    print("possessions {} | mean {:.1f} s".format(
        len(spells), np.mean(durations) / 25.0 if durations else 0.0))

    table = pm.read_label_csv(path)
    reason = pm.validate_submission(table, len(codes))
    print("format: {}".format("OK" if reason is None else "FAIL " + reason))

    if args.score:
        gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
        result = pm.evaluate(gt, codes)
        print("\npublic 5-minute correctness check (NOT a validation score)")
        print("  macroF1 {:.3f} | spellF1 {:.3f} (at IoU {:.1f} {:.3f}) | alive/dead F1 {:.3f}".format(
            result["macro_f1"], result["spell_f1_mean"], result["spell_iou_headline"],
            result["spell_f1_at_headline"], result["alive_dead_f1"]))
        print("  spells gt {} | pred {} | mean {:.1f} s | team acc (both alive) {:.3f}".format(
            result["n_gt_spells"], result["n_pred_spells"],
            result["mean_spell_duration_s"], result["team_acc_both_alive"]))
        for name in ("H", "A", "D"):
            row = result["per_class"][name]
            print("  {} precision {:.3f} recall {:.3f} F1 {:.3f} support {}".format(
                name, row["precision"], row["recall"], row["f1"], row["support"]))
        print("  the public window is inside the training period; this checks the pipeline, "
              "not the model.")


if __name__ == "__main__":
    main()

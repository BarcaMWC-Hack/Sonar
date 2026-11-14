"""Fit the dead-ball model on the open IDSSE matches' true dead flags; write the stored weights.

    python tools/idsse_fit.py            # writes sonar/deadball_idsse_weights.json

Same features and the same class-balanced L2 logistic regression as sonar/deadball.py; only the
labels change, from BAR-ATH's Smart Tagging weak labels to each IDSSE frame's BallStatus. Before
the features are computed, every match's ball is degraded the way our tracker loses it
(tools/idsse_eval.py, "bar-ath" mode, 41%, seed = match index), so the model learns from
tracking that looks like ours. One graded frame in five is used: neighbours are near copies.

The IDSSE data (Bassek et al. 2025, CC-BY 4.0) is public and is not the competition match, so
training on it is allowed; nothing here reads BAR-ATH or evaluation/gt.

Prints the weights (model parameters, not data).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import idsse_data  # noqa: E402
import idsse_eval as ie  # noqa: E402
from sonar import deadball  # noqa: E402

DEGRADE = 0.41


def main():
    matches = idsse_data.available()
    if len(matches) != len(idsse_data.MATCHES):
        sys.exit("need all {} IDSSE matches, found {}".format(len(idsse_data.MATCHES),
                                                              len(matches)))
    prepared, names = {}, None
    for k, m in enumerate(idsse_data.MATCHES):
        prepared[m] = ie.prepare(m, DEGRADE, k, "bar-ath")
        names = deadball.design(prepared[m]["features"])[1]
    w = ie.dfl_weights(prepared, exclude=None)
    out = {"names": ["bias"] + list(names), "weights": w.tolist(),
           "source": "tools/idsse_fit.py on IDSSE matches " + " ".join(idsse_data.MATCHES),
           "degrade": DEGRADE, "l2": deadball.L2}
    with open(deadball.IDSSE_WEIGHTS, "w", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    for name, value in zip(out["names"], out["weights"]):
        print("  {:16s} {:+8.4f}".format(name, value))
    print("wrote " + os.path.relpath(deadball.IDSSE_WEIGHTS, ROOT))


if __name__ == "__main__":
    main()

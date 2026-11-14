"""Fit the dead-ball model on 27 open matches' true dead flags; write the stored weights.

    python tools/open_fit.py             # writes sonar/deadball_open_weights.json

Same features and the same class-balanced L2 logistic regression as sonar/deadball.py; only the
labels change, from BAR-ATH's Smart Tagging weak labels to true dead flags from:

  - the 7 IDSSE matches (Bassek et al. 2025, CC-BY 4.0; tools/idsse_data.py): TRACAB tracking
    never loses the ball, so it is degraded first the way our tracker loses it
    (tools/idsse_eval.py, "bar-ath" mode, 41%, seed = match index)
  - the 20 SkillCorner A-League matches (SkillCorner open data, MIT; tools/skillcorner_data.py):
    broadcast tracking, used as it is, since its ball loss is real (40.7% of graded frames)

One graded frame in five is used: neighbours are near copies. Every frame counts the same, so
SkillCorner's 20 matches carry most of the weight. The data is public and none of it is the
competition match, so training on it is allowed; nothing here reads BAR-ATH or evaluation/gt.

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
import skillcorner_data  # noqa: E402
import skillcorner_eval as se  # noqa: E402
from sonar import deadball  # noqa: E402

DEGRADE = 0.41
N_SKILLCORNER = 20


def main():
    idsse = idsse_data.available()
    skillcorner = skillcorner_data.available()
    if len(idsse) != len(idsse_data.MATCHES) or len(skillcorner) != N_SKILLCORNER:
        sys.exit("need all {} IDSSE and {} SkillCorner matches, found {} and {}".format(
            len(idsse_data.MATCHES), N_SKILLCORNER, len(idsse), len(skillcorner)))
    prepared = {}
    for k, m in enumerate(idsse_data.MATCHES):
        prepared["idsse/" + m] = ie.prepare(m, DEGRADE, k, "bar-ath")
    for m in skillcorner:
        prepared["skillcorner/" + m] = se.prepare(m)
    names = deadball.design(next(iter(prepared.values()))["features"])[1]
    w = ie.dfl_weights(prepared, exclude=None)
    out = {"names": ["bias"] + list(names), "weights": w.tolist(),
           "source": "tools/open_fit.py: IDSSE " + " ".join(idsse_data.MATCHES) +
                     "; SkillCorner " + " ".join(skillcorner),
           "idsse_degrade": DEGRADE, "l2": deadball.L2}
    with open(deadball.OPEN_WEIGHTS, "w", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    for name, value in zip(out["names"], out["weights"]):
        print("  {:16s} {:+8.4f}".format(name, value))
    print("wrote " + os.path.relpath(deadball.OPEN_WEIGHTS, ROOT))


if __name__ == "__main__":
    main()

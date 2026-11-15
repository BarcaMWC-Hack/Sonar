# new_model

A possession pipeline: tracking features -> weak labels -> home/away classifier, the release's
change-point dead-ball detector as the dead stage, masked Viterbi decode -> `submission.csv`.

The dead stage is imported from `sonar/deadball.py`; that is the one deliberate dependency on the
frozen release. Copying it into `new_model/` would duplicate several hundred lines and invite
divergence. The features, weak labels, team stage, decoder and fold harness are local. The method,
the training-data disclosure and the measured results are in [TECHNICAL.md](TECHNICAL.md).

The decoder is selected on possession structure, not weak-label agreement. On the public
correctness check the hybrid now matches the release's macroF1 (0.765), exceeds its spellF1
(0.516 vs 0.400; a point estimate on 17 possessions, not a significant win) and matches its
dead-ball F1 (0.653). The public window is inside the training period, so this is a pipeline check,
not a validation score.

## What it produces

`new_model/submission.csv` — 199,319 data rows, frames `0..199318`, labels `H/A/D/X`.

| | value |
|---|---|
| md5 | `6aeaf0fce2c7adb1f109c924b33ae0e3` |
| sha256 | `f8c3c70c0d1a48e60ca8817241e8d6b3580439c846d704488eccfd3e252803ea` |
| rows | 199,319 |
| H / A / D / X | 62,806 / 36,017 / 60,034 / 40,462 |

The CSV is generated, not committed (the same policy as `evaluation/results/`). Regenerate it and
compare the hash.

## Reproduce

```bash
# Python 3.13.12; exact pins are in requirements.txt
python -m pip install -r new_model/requirements.txt

# Put the organiser-supplied tactical files in data/tactical, or:
export MTAH_DATA_DIR=/path/to/bundle        # the directory that contains tactical/

# Full pipeline: features -> model -> submission
python new_model/main.py

# From a cold cache, and fail unless the result is the recorded file:
python new_model/main.py --rebuild --no-importance --expect-md5 6aeaf0fce2c7adb1f109c924b33ae0e3

# Official format check (exactly what the scorer runs):
python evaluation/validate_submission.py new_model/submission.csv

# Public 5-minute correctness check (not a validation score; never used to tune):
python new_model/main.py --skip-features --skip-train --score
```

A cold full run takes about 40 seconds on the delivered data with `--no-importance` (features
~9 s, training ~31 s, decode ~2 s); the permutation-importance report adds about 20 s and does not
change the artifact or the hash.

The stages are runnable alone:

```bash
python -m new_model.features     # ATD + feature caches
python -m new_model.model        # train, save the artifact, print the validation report
python -m new_model.decode       # write and format-check submission.csv
python -m new_model.audit_features   # per-feature AUC against the weak labels
python -m new_model.crossval    # 5-block team CV; release dead mask fixed
python -m new_model.crossval --dead-stage hgb   # reproduce the old dead-stage audit
python -m new_model.crossval --half-swap   # fit one half, score the other
```

## Data policy

Nothing under `data/` is committed: it is the gated match bundle, and `data/*` is already ignored.
`new_model/submission.csv` is ignored too, because it is a full-match label file. The weak labels
are built at runtime from the provided Smart Tagging; they are not stored. No hand labels of the
provided match, no private ground truth and no human-produced data about the fixture are used.
Training-data details are in [TECHNICAL.md](TECHNICAL.md).

## Result, in one line

Public correctness check: **macroF1 0.765, spellF1 0.516**, 14 predicted possessions against 17
reference (14 predicted spells of mean 19.2 s vs 17 of 14.2 s), dead-ball D F1 0.653 (precision
0.951, recall 0.498), team accuracy 0.873. The release pipeline documents **macroF1 0.765,
spellF1 0.400**, dead-ball F1 0.653. The full-match structure is **276 possessions of mean 14.3 s**,
against the release's 277 of 14.2 s. The five-minute public window is under-segmented (14 vs 17),
but it is a small, training-overlapping sample; the full-match structure is the design target.

The dead stage is the change the release detector bought: the old HGB dead stage had 5-fold test
AUC 0.836 (train 1.000) and its public D recall was 0.462. The detector's fixed mask raises D F1 to
0.653 and closes the macroF1 gap. It is trained on the same weak Smart Tagging labels, so its
weak-label agreement is in-sample; the public check is an end-to-end wiring check, not an
independent sample.

The weak-label team CV is stable: 5-fold test AUC 0.980 (range 0.967-0.992), train-test gap 0.019;
half-swap 0.974/0.975. Weak-label macroF1 peaks at a low penalty (0.819 at 4) and falls to 0.799
at the chosen 68, while public spellF1 rises to 0.516: the weak labels fragment possession, so they
are a diagnostic, not the selection target.

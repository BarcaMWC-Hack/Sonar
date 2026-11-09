# Sonar | Phase 1 technical submission

Sonar assigns every tactical-video frame to home possession, away possession, dead ball or abstention. It uses tactical automatic tracking and Smart Tagging to decide when play is stopped, which team controls the ball during live play, and where possession boundaries belong.

## Possession rules and method

1. **Stoppages.** A logistic model uses motion, line crossings, ball visibility and restart geometry. Its regularization and feature set were chosen by held-out log loss between the two halves, on the weak labels described next. Weak dead labels come from intervals before automatically tagged set pieces. Weak live labels come from open-play tags clear of set pieces. A two-state sequence decoder limits flicker. The edges are refined using ball crossings, movement changes and restart kicks. The organizer-documented pre-kickoff interval is enforced in output, separately from model fitting.
2. **Team in control.** A slow ball close to a player is stronger evidence than simple proximity. A switching cost prevents the answer from flipping during a loose ball or an ordinary pass. Team spacing, measured against each half's own baseline, provides evidence when ball evidence is weak. Smart Tagging provides a fallback for a live stretch without tracking evidence, and on the submitted file it decides none of the 83 live stretches. The switching cost of 3.0 was kept over 2.0 because 2.0 adds short and quickly reversed possessions (25 more under 2 s and 26 more reversed within 3 s), not because of possession length.
3. **Change of possession.** For a decoded change during live play, the system searches a bounded interval for the new team's first touch or clear control after the old team's last local touch or control. It cannot move the change across a stoppage or neighboring change. A deflection can resemble a touch and remains a limitation. Restarts begin a new live stretch. The system emits X outside graded play and can abstain inside play when evidence is absent.

## Inputs and training disclosure

- **Organizer-supplied match assets used by the submission:** tactical tracking positions and metadata for one complete match, and its automatically generated Smart Tagging XML. The stoppage and team-shape fits use only automatic evidence. The team-shape fit uses tracking-derived control as weak supervision. Human-labeled examples used for model fitting or parameter selection from this match: **zero**.
- **Organizer public ground truth:** five minutes, used to check the fixed pipeline. It did not select a model, threshold or setting. The team did not use private ground truth or human-produced data about the fixture.
- **Other matches:** three anonymized Metrica Sports open sample games and seven Sportec/DFL IDSSE Bundesliga matches were used to explore and benchmark candidate methods. Metrica's event annotations and Sportec/DFL's frame-level ball state and ownership supplied the references. The inputs covered the available frames in those matches. Simulated tracking gaps and Smart Tagging–like inputs were used in the external tests. The submitted generator uses only the organizer-supplied match assets listed above.
- **Assets explored:** the broadcast stream was aligned in a feasibility study. The measured recovery of missing-ball frames was small and the replay cue missed its gate. Broadcast footage and tracking, homography, and pretrained models are outside the submitted pipeline.

## Checks and limitations

The fixed file has 199,319 data rows, with 82 stoppages covering 41% of graded time and 286 possessions of mean 13.1 s. On the public pipeline check it scored **0.4108 spellF1** and **0.7545 macroF1**, with **20 predicted possessions** against **17 reference possessions**. The corresponding public spellF1 values are **0.3256** for smooth50 and **0.4242** for smooth125. The full-match score is determined by the organizers using private labels. Fitting each model on one half and scoring the other, on weak labels rather than truth, gives AUC 0.81 and 0.89 for stoppages and 0.85 and 0.86 for team shape (`tools/kpm_report.py`).

The ball is missing from roughly 41% of graded frames. Short in-play fouls and throw-ins are difficult to distinguish from a brief slowdown. A ball-velocity change near a player can indicate a deflection, a clearance or control. These ambiguities can shift the boundaries of otherwise plausible possessions. On the public window the detector finds half of the dead-ball frames (recall 0.50, precision 0.86). One stoppage-model stability check and one team-shape calibration check still miss their targets. The labeled public window is too small to estimate the ranking reliably.

## Reproduce the submission

Use Python 3.11.9 and install the exact dependencies from requirements-sonar.txt. Put the organizer-supplied tactical ATD text, ATD XML and Smart Tagging XML in data/tactical, or set MTAH_DATA_DIR to the folder containing tactical. No match assets or Sonar predictions are included in the repository.

```text
python -m pip install -r requirements-sonar.txt
python -m sonar.make_submission --name Sonar --out data/release
python evaluation/validate_submission.py data/release/Sonar.csv
python tools/check_submission.py data/release/Sonar.csv
python tools/repro_check.py --existing-env --expect 12e0d605ef4e8a72ff340f390caec975
```

Expected CSV MD5 is `12e0d605ef4e8a72ff340f390caec975`. The reproduction check builds twice from independent cold caches, checks both files and compares them with the upload hash. Omit --existing-env to have the script build its own environment. The CSV is written with LF line ends on every operating system.

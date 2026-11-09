# Sonar | Phase 1 technical submission

Sonar assigns every tactical-video frame to home possession, away possession, dead ball or abstention. It uses tactical automatic tracking and Smart Tagging to decide when play is stopped, which team controls the ball during live play, and where possession boundaries belong.

## Possession rules and method

1. **Stoppages.** A logistic model uses motion, line crossings, ball visibility and restart geometry. Weak dead labels come from intervals before automatically tagged set pieces. Weak live labels come from open-play tags clear of set pieces. A two-state sequence decoder limits flicker. The edges are refined using ball crossings, movement changes and restart kicks. The organizer-documented pre-kickoff interval is enforced in output, separately from model fitting.
2. **Team in control.** A slow ball close to a player is stronger evidence than simple proximity. A switching cost prevents the answer from flipping during a loose ball or an ordinary pass. Team spacing provides evidence when ball evidence is weak. Smart Tagging provides a fallback for a live stretch without tracking evidence.
3. **Change of possession.** For a decoded change during live play, the system searches a bounded interval for the new team's first touch or clear control after the old team's last local touch or control. It cannot move the change across a stoppage or neighboring change. A deflection can resemble a touch and remains a limitation. Restarts begin a new live stretch. The system emits X outside graded play and can abstain inside play when evidence is absent.

## Inputs and training disclosure

- **Organizer-supplied match assets used by the submission:** tactical tracking positions and metadata for one complete match, and its automatically generated Smart Tagging XML. The stoppage and team-shape fits use only automatic evidence. The team-shape fit uses tracking-derived control as weak supervision. Human-labeled examples used for model fitting or parameter selection from this match: **zero**.
- **Organizer public ground truth:** five minutes, used to check the fixed pipeline. It did not select a model, threshold or setting. The team did not use private ground truth or human-produced data about the fixture.
- **Other matches:** Metrica Sports open sample games and seven Sportec/DFL IDSSE Bundesliga matches were used to explore and benchmark candidate methods. Provider event annotations or provider frame-level ball state and ownership supplied references. Simulated tracking gaps and Smart Tagging–like inputs were used in the external tests. None of the external data, fitted experimental models or experimental rules are part of the submitted generator.
- **Assets explored:** the broadcast stream was aligned in a feasibility study. The measured recovery of missing-ball frames was small and the replay cue missed its gate. Broadcast footage and tracking, homography, and pretrained models are outside the submitted pipeline.

## Checks and limitations

The fixed file has 199,319 data rows. On the public pipeline check it scored **0.4000 spellF1** and **0.7645 macroF1**, with **19 predicted possessions** against **17 reference possessions**. The corresponding public spellF1 values are **0.3256** for smooth50 and **0.4242** for smooth125. The full-match score is determined by the organizers using private labels.

The ball is missing from roughly 41% of graded frames. Short in-play fouls and throw-ins are difficult to distinguish from a brief slowdown. A ball-velocity change near a player can indicate a deflection, a clearance or control. These ambiguities can shift the boundaries of otherwise plausible possessions. The labeled public window is too small to estimate the ranking reliably.

## Reproduce the submission

Use Python 3.11.9 and install the exact dependencies from requirements-sonar.txt. Put the organizer-supplied tactical ATD text, ATD XML and Smart Tagging XML in data/tactical, or set MTAH_DATA_DIR to the folder containing tactical. No match assets or Sonar predictions are included in the repository.

```text
python -m pip install -r requirements-sonar.txt
python -m sonar.make_submission --name Sonar --out data/release
python evaluation/validate_submission.py data/release/Sonar.csv
python tools/check_submission.py data/release/Sonar.csv
python tools/repro_check.py --existing-env --expect 4eea9a9e70e90b87f7666ecd23101a43
```

Expected CSV MD5 is `4eea9a9e70e90b87f7666ecd23101a43`. The reproduction check builds twice from independent cold caches, checks both files and compares them with the upload hash. Omit --existing-env to have the script build its own environment.

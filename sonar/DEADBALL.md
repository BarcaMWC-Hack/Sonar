# Dead-ball detector (role 1)

Decides, for every graded frame, whether the ball is out of play. Everything here is computed from
the ATD and Smart Tagging; nothing reads a ground truth except the scorer, afterwards.

```
.venv\Scripts\python.exe -m sonar.make_submission        # evaluation/results/sonar.csv
.venv\Scripts\python.exe tools\deadball_report.py --grid # weights, checks, label-free sweep
.venv\Scripts\python.exe tools\deadball_diagnostics.py   # how each raw feature separates
```

All three print scores, counts and shares only, so their output is safe to paste anywhere.

## How it works

1. **Features** (`sonar/deadball_features.py`): player motion (mean speed, share moving, share
   sprinting, the fastest three), players off the pitch, kick-off line-up, ball over a line, ball
   speed from positions, long ball gaps, and where the ball sits (centre spot, corner, goal area,
   touchline).
2. **Logistic model trained on weak labels** (`sonar/deadball.py`). Dead = the 8 s before each
   Smart Tagging `SET PIECES` interval, plus 110808..113741 (FAQ.md: graded and all D). Alive =
   inside open-play phases, trimmed and kept 10 s clear of set pieces. Class-balanced, so its
   logit is a likelihood ratio; a 40% dead prior is added separately.
3. **Two-state Viterbi** per half with a switching cost, so a stoppage needs a few seconds of
   evidence and a slow moment cannot split a possession.
4. **Edge refinement**: a stoppage starts at the first frame the ball is seen over a line (else
   the start of a long ball gap); it ends at a held-then-struck ball, else at the strike just
   after the ball reappears from a long gap, else where Viterbi put it.
5. **Labels** (`sonar/make_submission.py`): dead → D; each alive stretch gets the team call from
   `sonar/possession.py` (see [POSSESSION.md](POSSESSION.md)). `--team nearest` uses the
   baseline's nearest player instead, which is what the public numbers below were measured with.

## Facts about the data that shaped it (counts only)

- Ball speed is NaN on 91% of frames where the ball's position is known, so speed is derived
  from positions.
- The tracker fills short ball gaps itself: there are no missing runs of 50-124 frames and 142
  runs of 125+. A missing ball means a long gap, which is mostly a stoppage (69% of frames
  before set pieces, 6% of public alive frames).
- Smart Tagging `SET PIECES` intervals average ~30 s and the restart falls inside them (2 of 2
  in the public window). Only 58 are in graded time and only 5 are free kicks, so the tagging
  misses most stoppages, which is why it is used as weak labels and not as the answer.

## How the settings were chosen, without the ground truth

| check | result |
|---|---|
| weak labels, fit on one half and score the other | AUC 0.82 (1→2), 0.87 (2→1) |
| dead share of graded time (a match: ~40%) | 37.1% |
| stoppages (a match: ~80-130) | 80, median 18 s |
| tagged set pieces with a predicted restart within 5 s | 50 of 58 |

Only `switch_cost / temperature` matters to the Viterbi path. The sweep in
`deadball_report.py --grid` runs from 41 stoppages (too few) to 94 with sub-3 s fragments
appearing; 25 frames per switch (one second of evidence) gives 80.

## Public 5 minutes, as a correctness check only

| | sonar | baseline_smooth50 |
|---|---|---|
| D recall / precision / F1 | 0.499 / 0.888 / 0.639 | 0.214 / 0.657 / 0.323 |
| macroF1 | 0.752 | 0.621 |
| spellF1 (5-95% bootstrap) | 0.381 (0.257-0.505) | 0.326 (0.214-0.447) |

The team call is identical in both, so the difference is the dead-ball detector. The spellF1
intervals overlap almost completely, as they must on 17 possessions. This shows the pipeline is
wired correctly, not that it is better on the full match.

## No hand labels on this match

RULES.md (clarified Nov 1) bars hand labels on the provided game for any purpose, including
validation, so there is no video review sheet. Watching the video to understand the game and
design features is fine; turning what was seen into labels, corrections or frame-specific
constants is not. Edge rules are checked only against label-free football norms and Smart
Tagging.

## Next steps

- On the public window only 3 of 20 true transitions land within 1 s, so edge timing matters
  more than catching extra stoppages.
- Add fouls and offsides: a whistle shows up as everyone decelerating together, with the ball
  inside the pitch.

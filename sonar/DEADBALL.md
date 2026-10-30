# Dead-ball detector (role 1)

Decides, for every graded frame, whether the ball is out of play. Everything here is computed from
the ATD and Smart Tagging; nothing reads a ground truth except the scorer, afterwards.

```
.venv\Scripts\python.exe -m sonar.make_submission        # evaluation/results/sonar.csv
.venv\Scripts\python.exe tools\deadball_report.py --grid # weights, checks, label-free sweep
.venv\Scripts\python.exe tools\deadball_diagnostics.py   # how each raw feature separates
.venv\Scripts\python.exe tools\make_review_sheet.py      # sheet for checking against video
```

All four print scores, counts and shares only, so their output is safe to paste anywhere. The
review sheet is the exception: it is written to `data/review/` (git-ignored) and is for teammates
with the video, never for an LLM.

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
4. **Edge refinement** from unsmoothed signals, because the Viterbi edges inherit up to 10 s of
   smoothing. `start_rule` / `end_rule` in the review sheet say which rule placed each edge.
   - **Start:** the first frame the ball is over a line (`crossing`, 29 stoppages). Otherwise
     the moment the players decelerate, i.e. the biggest drop in mean speed comparing the next
     2 s with the previous 2 s (`motion`, 37), or the ball vanishing just before that (`gap`, 13).
   - **End:** find the moment the players accelerate (biggest rise, same measure). That peak
     trails the restart, so take the *earliest* ball event from 3 s before it to 1 s after: a
     held ball struck (`kick`, 5) or the ball reappearing (`reacquire`, 27), else any ball going
     fast (`strike`, 23), else the acceleration itself (`motion`, 24). Every kind of restart is
     the first strike of a held, placed or unseen ball; later strikes are open play.
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
| dead share of graded time (a match: ~40%) | 38.2% |
| stoppages (a match: ~80-130) | 80, median 19 s |
| tagged set pieces with a predicted restart within 5 s | 51 of 58 |

Only `switch_cost / temperature` matters to the Viterbi path. The sweep in
`deadball_report.py --grid` runs from 41 stoppages (too few) to 94 with sub-3 s fragments
appearing; 25 frames per switch (one second of evidence) gives 80.

## Public 5 minutes, as a correctness check only

| | sonar | baseline_smooth50 |
|---|---|---|
| D recall / precision / F1 | 0.498 / 0.951 / 0.653 | 0.214 / 0.657 / 0.323 |
| macroF1 | 0.759 | 0.621 |
| spellF1 (5-95% bootstrap) | 0.391 (0.257-0.524) | 0.326 (0.214-0.447) |

The team call is identical in both, so the difference is the dead-ball detector. The spellF1
intervals overlap almost completely, as they must on 17 possessions. This shows the pipeline is
wired correctly, not that it is better on the full match.

`tools/timing_report.py` buckets how far each of our label changes is from the true one in the
public window (counts only). It explains what the edges mean; it is ~20 events and must not be
used to tune. The edge rules above were changed once after looking at it, because it exposed a
reasoning flaw (taking the *latest* ball event caught the first open-play pass), and not again.

## Open questions for whoever has the video

Fill in `data/review/stoppages_review.csv` for 10-20 rows of each `end_rule` and report totals
in words:

1. **Restart timing.** For each `end_rule` (`reacquire`, `strike`, `kick`, `motion`), is the
   real restart before or after `stop_video_time`, and by about how much? Same for
   `start_rule` (`crossing`, `gap`, `motion`) against the whistle or the ball going out. A
   consistent offset for one rule is a one-line fix.
2. **Missed stoppages.** Watch a few minutes of the second half: which kinds of stoppage get no
   row at all (quick throw-ins, fouls, offsides)?
3. **False stoppages.** Rows with no `nearest_set_piece`: are they real (fouls, injuries) or slow
   spells of open play?

## Next steps

- Use the review answers to correct each edge rule's offset.
- On the public window only 4 of 20 true transitions land within 1 s, so edge timing matters
  more than catching extra stoppages.
- Add fouls and offsides: a whistle shows up as everyone decelerating together, with the ball
  inside the pitch.

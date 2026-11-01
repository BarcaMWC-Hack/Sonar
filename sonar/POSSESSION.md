# Possession call (role 2, first version)

Decides H or A on every frame the dead-ball detector calls alive. A starting point for whoever
owns role 2; replace or extend it freely.

```
.venv\Scripts\python.exe -m sonar.make_submission                 # uses this model (default)
.venv\Scripts\python.exe -m sonar.make_submission --team nearest  # the baseline call, to compare
.venv\Scripts\python.exe tools\possession_diagnostics.py          # what the team signal looks like
.venv\Scripts\python.exe tools\possession_errors.py               # where switches go wrong
.venv\Scripts\python.exe tools\possession_report.py               # shape model check + sweep
```

All of these print counts and shares only.

## Why the nearest-player call fails

The nearest player to the ball is not who has the ball. During live play (counts from
`possession_diagnostics.py`):

- the raw nearest-player team **changes 25 times per live minute**;
- counted only on frames where a player is within 2 m of a ball slower than 6 m/s, it changes
  **2.6 times per live minute**, about the rate possession really changes;
- a third of tracked live frames have the ball faster than 8 m/s (in flight), and 18% are
  contested (both teams' nearest players within a metre of each other).

CHALLENGE.md says possession changes only on *controlled* possession and stays with the last team
through loose balls, so frames where nobody controls the ball should not vote.

## How it works (`sonar/possession.py`)

1. **Control-weighted evidence** per frame: full weight when the nearest player is within 1 m of
   a ball slower than 4 m/s, fading to nothing by 8 m/s or a few metres away, signed by which
   team's nearest player is closer.
2. **Team shape** (`sonar/team_shape.py`): the team with the ball spreads out. Width alone (the
   spread of the players' y) picks the team in control with AUC 0.86, and a small logistic
   model on width, length, forward speed and block position reaches 0.85-0.87 when fitted on
   one half and scored on the other. Its labels are the live frames where the ball evidence is
   clear, not any ground truth. Weighted at 0.05 per frame it is a slow push: it decides
   whether a run of touches really was a change, while the touches still set where it lands.
3. **Viterbi with a switching cost** per live stretch, on that tracking evidence: possession
   changes only when the new team has built up enough of it, and only the tracking ever decides
   *when* it changes.
4. **Smart Tagging names the team for a live stretch with no tracking evidence at all.** Phases like
   Build Up and Progression name the team in possession (83-87% agreement with the controlled
   call), but their edges are seconds off, so they never place a change. (A first version let
   them push on every untracked frame, which let tag edges move possession changes.) The
   "Defending in …" codes are mirror copies naming the other side and are ignored.

## How the settings were chosen, without the ground truth

| setting | possessions | mean | under 2 s | agrees with tags* |
|---|---|---|---|---|
| baseline nearest, ±2 s vote | 457 | 8.5 s | 110 | 66.2% |
| control 1 m, switch 1.5 | 312 | 12.4 s | 51 | 70.7% |
| **control 1 m, switch 3 (chosen)** | **265** | **14.6 s** | **30** | **72.2%** |
| control 1 m, switch 12 | 173 | 22.3 s | 0 | 75.1% |
| control 1.5 m, switch 6 | 271 | 14.3 s | 26 | 71.0% |

\*Neither the ball nor the shape evidence uses the tags, so agreement is an independent check.
It rises as possessions get longer (tags are long phases), so the count is fixed from
football's typical 10-15 s possession, and agreement only compares settings at a matched count.

Adding team shape, at switch cost 3 (from `possession_report.py`):

| shape weight | possessions | mean | switches reversed within 3 s | switches with almost no follow-up control | agrees with tags |
|---|---|---|---|---|---|
| 0 | 267 | 14.5 s | 32 | 59 | 72.3% |
| **0.05 (chosen)** | **277** | **14.2 s** | **41** | **43** | **77.4%** |
| 0.1 | 285 | 13.8 s | 50 | 48 | 77.1% |
| 0.2 | 271 | 14.5 s | 48 | 53 | 78.4% |

Shape lifts agreement by about five points at the same count. Larger weights are no clearer, and
shape lags turnovers, so the smallest weight is kept. The cost is more quick reversals (32 → 41);
some will be real turnovers won straight back, some flicker. The final file has 277 possessions
of mean 14.2 s, and Barça has 56% of live time.

`possession_errors.py` also ruled two suspects out: the tracker's straight-line gap filling
covers 3.4% of tracked live frames and never more than a second at a time, and only 3 of ~200
switches land at the end of a tracking gap.

## Public 5 minutes, as a correctness check only

| | sonar | same dead ball, baseline team call | baseline_smooth50 |
|---|---|---|---|
| possessions (truth: 17, mean 14.2 s) | 19, mean 14.2 s | 25, 10.8 s | 26, 10.8 s |
| spellF1 (5-95% bootstrap) | 0.433 (0.277-0.589) | 0.391 (0.257-0.524) | 0.326 (0.214-0.447) |
| macroF1 | 0.778 | 0.759 | 0.621 |
| right team, where both say the ball is live | 89.9% | 86.5% | 85.8% |
| true changes matched within 1 s | 3 of 20 | 4 of 20 | 2 of 20 |

The intervals overlap; this says the pipeline is right, not which version wins on the full match.
Team shape made the team call more often right here too (86.5% → 89.9% of jointly-live frames),
matching the tag-agreement gain on the full match, but did not move the timing of changes.
`tools/timing_report.py` shows most of our possession changes are still more than 3 s from a
true one, in both directions: missed and extra changes rather than a constant offset.

## Ideas not yet tried

- Infer the ball through long gaps (GhostBall): carry its last position and velocity, or use
  where the players converge.
- Make the evidence asymmetric: an interception followed by a pass between the new team's
  players is stronger evidence than one touch. This is the most direct attack on the 43
  switches the new team barely controls.
- A video check of 10-20 of our switches, split by how much follow-up control they had, would
  say whether the weak ones are real turnovers or flicker.
- Check where a change lands: with no evidence while the ball is loose, the decoder puts the
  change at the new team's first controlled touch, which matches CHALLENGE.md's "gains
  controlled possession". A video check of a dozen changes would confirm the ground truth does
  the same.

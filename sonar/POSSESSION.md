# Possession call (role 2, first version)

Decides H or A on every frame the dead-ball detector calls alive. A starting point for whoever
owns role 2; replace or extend it freely.

```
.venv\Scripts\python.exe -m sonar.make_submission                 # uses this model (default)
.venv\Scripts\python.exe -m sonar.make_submission --team nearest  # the baseline call, to compare
.venv\Scripts\python.exe tools\possession_diagnostics.py          # what the team signal looks like
.venv\Scripts\python.exe tools\possession_report.py               # label-free sweep
```

All of these print counts and shares only.

## Why the baseline call fails

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
2. **Smart Tagging fills gaps**: phases like Build Up and Progression name the team in
   possession (83-87% agreement with the controlled call). They add a small push only on frames
   where the ball gives no evidence. The "Defending in …" codes are mirror copies naming the
   other side and are ignored.
3. **Viterbi with a switching cost** per live stretch: possession changes only when the new team
   has built up enough control evidence.

## How the settings were chosen, without the ground truth

| setting | possessions | mean | under 2 s | agrees with tags* |
|---|---|---|---|---|
| baseline nearest, ±2 s vote | 464 | 8.5 s | 111 | 66.5% |
| control 1 m, switch 1.5 | 312 | 12.7 s | 47 | 70.8% |
| **control 1 m, switch 3 (chosen)** | **266** | **14.9 s** | **26** | **72.0%** |
| control 1 m, switch 12 | 175 | 22.6 s | 0 | 75.3% |
| control 1.5 m, switch 6 | 272 | 14.5 s | 22 | 70.6% |

\*Ball evidence only, so the tags are an independent check. Agreement rises as possessions get
longer (tags are long phases), so the count is fixed from football's typical 10-15 s possession,
and agreement only compares control definitions at a matched count. With the tag gap-filling
the final file has 284 possessions of mean 14.0 s, and Barça has 59% of live time.

## Public 5 minutes, as a correctness check only

| | sonar | same dead ball, baseline team call | baseline_smooth50 |
|---|---|---|---|
| possessions (truth: 17, mean 14.2 s) | 17, mean 15.7 s | 25, 10.7 s | 26, 10.8 s |
| spellF1 (5-95% bootstrap) | 0.435 (0.271-0.600) | 0.381 (0.257-0.505) | 0.326 (0.214-0.447) |
| macroF1 | 0.749 | 0.752 | 0.621 |

The intervals overlap; this says the pipeline is right, not which version wins on the full match.
What it does show: only 3 of 20 true transitions are matched within 1 s. **Edge timing is the
next thing to fix**, for possession changes and stoppages alike.

## Ideas not yet tried

- Infer the ball through long gaps (GhostBall): carry its last position and velocity, or use
  where the players converge.
- Make the evidence asymmetric: an interception followed by a pass between the new team's
  players is stronger evidence than one touch.
- Place each change at the new team's first controlled touch rather than wherever Viterbi lands.

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
   model on width, length, forward speed and block position reaches 0.85-0.86 when fitted on
   one half and scored on the other. Its labels are the live frames where the ball evidence is
   clear, not any ground truth. Weighted at 0.05 per frame it is a slow push: it decides
   whether a run of touches really was a change, while the touches still set where it lands.
3. **Viterbi with a switching cost** per live stretch, on that tracking evidence: possession
   changes only when the new team has built up enough of it, and only the tracking ever decides
   *when* it changes.
4. **Switch placement** (`localize()`): the decoder decides *whether* possession changed; this
   decides *when*. Each switch moves (at most 3 s, never past a neighbouring switch or stoppage)
   to the new team's first touch or controlled frame after the old team's last. A touch is a
   jump of more than 5 m/s in the ball's velocity, 0.24 s before vs after, with a player within
   1.5 m: 15.8 per live minute, in football's roughly 15-25 range. (Measuring the jump over
   only 0.12 s found 4 per minute, because the tracker smooths the ball's path.) This follows
   CHALLENGE.md: possession changes when the other team "gains controlled possession", and
   loose play stays with the last team. `--no-localize` turns it off.
5. **Smart Tagging names the team for a live stretch with no ball or team-shape evidence.** Team
   shape is available on every frame, so on the delivered file the tags decide none of the 83
   live stretches: 77 have ball evidence, 5 have team shape only, and 1 has neither. Labels are
   the same with and without the tags. The phases it
   reads agree with the controlled call 63-87% of the time (Build Up 87%, Progression 83%, Long Ball
   63%), but their edges are seconds off, so they never place a change. (A first version let them
   push on every untracked frame, which let tag edges move possession changes.) The "Defending in …"
   codes are mirror copies naming the other side (13-30% agreement) and are ignored.

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
a rule of thumb for typical possession length (10-15 s; the repo has no source for it), and
agreement only compares settings at a matched count.

Adding team shape, at switch cost 3 (from `possession_report.py`):

| shape weight | possessions | mean | switches reversed within 3 s | switches with almost no follow-up control | agrees with tags |
|---|---|---|---|---|---|
| 0 | 267 | 14.5 s | 32 | 59 | 72.3% |
| **0.05 (chosen)** | **277** | **14.2 s** | **41** | **43** | **77.4%** |
| 0.1 | 285 | 13.8 s | 50 | 48 | 77.1% |
| 0.2 | 271 | 14.5 s | 48 | 53 | 78.4% |

Shape lifts agreement by about five points at the same count. Larger weights are no clearer, and
shape lags turnovers, so the smallest weight is kept. The cost is more quick reversals (32 → 41);
some will be real turnovers won straight back, some flicker. The final file has 286 possessions
of mean 13.1 s, and Barça has 56% of live time.

Switch placement, from `possession_errors.py`: of 197 switches, 152 move (95 by under 0.4 s, 52
later by 0.4-3 s, 14 earlier) and 45 stay. The new team is touching or controlling the ball
within 0.2 s of the switch at 166 of them, against 91 before. That check is partly circular, as
it is what the rule aims at. Switches with almost no follow-up control drop from 43 to 36. The
possession count is unchanged (277) and tag agreement barely moves (77.4% → 77.0%).

`possession_errors.py` also ruled two suspects out: the tracker's straight-line gap filling
covers 3.4% of tracked live frames and never more than a second at a time, and only 3 of ~200
switches land at the end of a tracking gap.

## Switch cost on other games (full pipeline)

Metrica's sample games 1 and 2 are other matches, so comparing settings on them is allowed.
`tools/external_full.py` (PR #9) runs this pipeline on them, with a truth built from the event data
and dead frames taken from that truth. The numbers are simulated: they show direction, not match
results. Shape weight 0.05, first-touch placement on:

| switch cost | game 1 | game 2 | mean |
| --- | --- | --- | --- |
| 1.5 | 0.678 | 0.672 | 0.675 |
| **3.0 (chosen)** | 0.662 | 0.682 | 0.672 |
| 2.0 | 0.678 | 0.690 | 0.684 |

Held out with first-touch placement off, 2.0 is picked on both games and gains 0.005 and 0.006 over
3.0: small, and on a simulated truth. The truth starts each change at the new team's
event, which may favour first-touch placement and a lower switch cost by construction. The default
is 3.0, chosen on 2026-11-09. At 2.0 there are 326 possessions instead of 286, with 25 more under
2 s and 26 more reversed within 3 s. The choice rests on those short and reversed possessions, not
on possession length.

## Public 5 minutes, as a correctness check only

| | sonar | sonar `--no-localize` | same dead ball, baseline team call | baseline_smooth50 |
|---|---|---|---|---|
| possessions (truth: 17, mean 14.2 s) | 19, mean 14.2 s | 19, mean 14.2 s | 25, 10.8 s | 26, 10.8 s |
| spellF1 (5-95% bootstrap) | 0.400 (0.233-0.567) | 0.433 (0.277-0.589) | 0.391 (0.257-0.524) | 0.326 (0.214-0.447) |
| macroF1 | 0.765 | 0.778 | 0.759 | 0.621 |
| right team, where both say the ball is live | 87.7% | 89.9% | 86.5% | 85.8% |
| true changes matched within 1 s | 4 of 20 | 3 of 20 | 4 of 20 | 2 of 20 |

The intervals overlap; this says the pipeline is right, not which version wins on the full match.
Team shape made the team call more often right (86.5% → 89.9% of jointly-live frames), matching
the tag-agreement gain on the full match. Switch placement moved this window slightly the other
way. With 12 possession changes that is noise-sized, but it is the one open question about what
the labels mean: **during a loose ball, does the truth switch at the new team's first controlled
touch (CHALLENGE.md's wording, and what `localize()` does), or earlier, at the interception or
the old team's last touch?** We do not use the public 5 minutes or any answer key to decide it, and nothing seen in the match video
becomes a label, a correction or a frame-specific constant (RULES.md, rule 1). Placement follows the
written rule (CHALLENGE.md), the question stays open, and it is listed as a limitation.

## Ideas not yet tried

- Infer the ball through long gaps (GhostBall): carry its last position and velocity, or use
  where the players converge.
- Make the evidence asymmetric: an interception followed by a pass between the new team's
  players is stronger evidence than one touch. This is the most direct attack on the 43
  switches the new team barely controls.
- Judge the weak switches (new team barely controls the ball) on other games with an other-game
  yardstick. Watching this match to understand it is fine; turning it into labels, corrections or
  frame-specific constants is not (RULES.md, rule 1).
- Check where a change lands: with no evidence while the ball is loose, the decoder puts the
  change at the new team's first controlled touch, which matches CHALLENGE.md's "gains
  controlled possession". Left as a stated assumption.

# Sonar: testing on 7 externally graded matches (2026-11-14)

Branch `idsse-yardstick` (from `kpm-fixes` / PR #12, head 8a5108d), worktree `.wt/idsse`.
Nothing is committed or pushed yet. Every number below is an aggregate score.

## Summary

The release dead-ball model is trained on Barça–Athletic's Smart Tagging weak labels; that is Lethabo's model, and it is in PR #12. A version trained instead on the true dead/alive flags of 7 open Bundesliga matches beats it on **every one** of the 7 matches. Each match was scored by a model that never saw it. Lowering the possession switching cost from 3.0 to 2.0 adds a little more.

| Build (bar-ath simulation, 7 matches × 3 seeds) | spellF1 | macroF1 | D recall | D precision | wins vs current |
|---|---|---|---|---|---|
| Current build (release dead ball, cost 3.0) | 0.408 | 0.812 | 0.85 | 0.90 | – |
| Release dead ball, cost 2.0 | 0.419 | 0.812 | 0.85 | 0.90 | 7/7, +0.011 |
| IDSSE-trained dead ball, cost 3.0 | 0.447 | 0.855 | 0.95 | 0.92 | 7/7, +0.039 |
| **IDSSE-trained dead ball, cost 2.0 (recommended)** | **0.456** | **0.856** | 0.95 | 0.92 | **7/7, +0.048** |
| IDSSE-trained dead ball, cost 1.5 | 0.459 | 0.856 | 0.95 | 0.92 | 7/7, +0.051 |
| Ceiling: perfect dead ball, cost 2.0 | 0.531 | 0.908 | 1.00 | 1.00 | – |

The spread across seeds is at most 0.013 for every row, far smaller than the gains. With the old random-gap simulation the gap is larger still: 0.333 for the current build against 0.433 for the recommended one.

## The data

- **Source:** the IDSSE dataset (Bassek, Rein, Weber and Memmert, *Scientific Data*, 2025; CC-BY 4.0). It has 7 Bundesliga matches of TRACAB tracking at 25 Hz, taken from the public Hugging Face mirror `pysport/idsse-data` (2.6 GB).
- **Location:** saved to `data/external/idsse/`, which git ignores.
- **Truth:** every frame carries `BallStatus` (dead or alive) and `BallPossession` (home or away). That is the same kind of truth the competition derives from its corrected tracking. So unlike the Metrica yardstick, which uses event-derived proxy truth and an oracle dead ball, this tests the dead-ball detector and the possession calls together, on the real spellF1 scorer.
- **Fit:** dead share across the 7 matches is 40.7–51.7%, and player-speed medians match Barça–Athletic's (1.4–1.7 against 1.5 m/s).
- **Rules:** these are other games, so tuning and training on them is allowed. None of it is Barça–Athletic data, and nothing read `evaluation/gt/`.

## Making the test look like our tracker

The ball in the IDSSE data is never missing, so I measured how our tracker loses it on Barça–Athletic. This uses label-free aggregates only:

- 40.7% of graded frames are missing, in only 150 gaps with a median length of 10.7 s; nearly all gaps are longer than 5 s.
- The ball is missing on 69% of the frames Smart Tagging marks dead and 22% of those it marks alive.
- Our ball jitters: the median frame-to-frame step is 0.20 m, against TRACAB's 0.085 m.

The old Metrica simulation used short random 2 s gaps, which looks nothing like this. The new **bar-ath** mode works like this:

- It hides the ball through 90% of the true stoppages.
- It adds log-normal gaps fitted to the measured lengths until 41% of frames are missing. That gives missing shares of 67–74% in dead frames and 10–23% in live frames, close to the measured split.
- It adds 0.4 m of smoothed jitter, which brings the "ball still" share down from about 20% to about 4%; ours is 2.9%.

## What changed in the code

None of this changes the default build: the release build still writes `Sonar.csv` with MD5 `f3b7993ffb94d6fa5c6fae8ee17e4fd6`.

- `tools/idsse_data.py` parses and caches the matches.
- `tools/idsse_eval.py` scores builds leave-one-match-out, with `--grid` and `--mode bar-ath|random`.
- `tools/idsse_fit.py` fits the dead-ball model on all 7 matches and writes `sonar/deadball_idsse_weights.json`. It uses the same features and the same L2 = 100 logistic regression; only the labels are new.
- `tools/idsse_transfer.py` is a label-free check on Barça–Athletic.
- `sonar/deadball.py`: `detect(..., weights="idsse")` uses the stored weights.
- `sonar/make_submission.py`: new options `--deadball weak|idsse` and `--possession-cost`.

## Candidate submissions for Barça–Athletic

These come from the label-free checks; no ground truth was used.

| Candidate | Command flags | Dead share | Stoppages | Possessions | Rule checks | MD5 |
|---|---|---|---|---|---|---|
| Current build (`release.csv`) | – | 41.1% | 82 | 286 | 9/9 | f3b7993f… |
| `idsse.csv` | `--deadball idsse` | 47.3% | 103 | 287 | 9/9 | 3699cb49… |
| **`idsse_p2.csv`** | `--deadball idsse --possession-cost 2` | 47.3% | 103 | 335 | 9/9 | e5b31069… |

The IDSSE-trained detector agrees with Barça–Athletic's Smart Tagging labels with an AUC of 0.816 without ever being trained on them. The release model scores 0.859 there, but in-sample. That suggests the model carries over. Its 103 stoppages also sit nearer a real match's roughly 100–120 than 82 does.

## Why cost 2.0 and not 1.5

1.5 is best by 0.003 here, but DFL's possession truth is choppier than the competition's is likely to be:

- **DFL:** about 4.9 possessions per minute.
- **Competition:** the public 5 minutes show about 3.4 per minute.
- **Our builds:** 2.7 per minute at cost 3.0 and 3.2 at cost 2.0.

The Metrica yardstick also ranked 2.0 at or above 3.0. So 2.0 is the safer step.

## Caveats

- The masking is a simulation fitted from aggregates; the real tracker may differ in ways the numbers do not capture.
- The IDSSE data is Bundesliga and TRACAB, and the stored weights put a large weight on still-ball and out-of-bounds features. If our tracker's ball behaves differently near stoppages, the gain on Barça–Athletic will be smaller than measured here.
- This is not tuned on, or checked against, the public 5 minutes, per the rules and our own rule.
- Reproducing the stored weights needs the public IDSSE download. The weights JSON is committed so the submission regenerates without it.

## Decisions for the team

1. Should the submission switch to `--deadball idsse --possession-cost 2`? Lethabo should review, since it replaces the weak-label training in his dead-ball model.
2. Should this go up as a PR into `kpm-fixes`/`main` before the freeze? `docs/TECHNICAL.md` would also need a paragraph on the IDSSE training data and its CC-BY attribution.

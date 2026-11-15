# Sonar: testing on 27 open graded matches (2026-11-14)

Branch `idsse-yardstick` (PR #13, into `kpm-fixes`). Every number below is an aggregate score; no Barça–Athletic ground truth was used anywhere.

## Summary

**Recommendation: build the submission with `--deadball open` and change nothing else.**

The release dead-ball model is trained on Barça–Athletic's Smart Tagging weak labels; that is Lethabo's model. The new one uses the same features and the same logistic model, but is trained on the **true** dead/alive flags of 27 open matches.

- **Broadcast tracking (our kind):** on SkillCorner's 20 matches, scored leave-one-match-out, it wins **20 of 20**. spellF1 rises from **0.367 to 0.402** (+0.035, standard error 0.003).
- **Simulated tracking:** on the 7 IDSSE matches, with our ball loss simulated, it wins **7 of 7** (+0.039 at cost 3.0).
- **Switch cost:** stays at **3.0**. An earlier draft recommended 2.0; that was wrong, and the broadcast matches reversed it.

## The two datasets

| | IDSSE (Bundesliga) | SkillCorner (A-League) |
|---|---|---|
| Matches | 7 | 20 |
| Tracking | TRACAB, 25 Hz, ball never lost | **Broadcast**, 10 Hz (resampled to 25), ball lost for real |
| Dead truth | `BallStatus` on every frame | not inside a phase of play |
| Possession truth | `BallPossession` on every frame | the frame's possession group |
| Ball missing | 0%, so I simulate ours | **40.7%** (dead 79%, alive 7%), ours is 40.7% |
| Possessions per match | ~470 | ~345, ours ~290, public 5 min suggest ~360 |
| Licence | CC-BY 4.0 (Bassek et al., *Sci. Data*, 2025) | MIT (SkillCorner open data) |

SkillCorner is the closest stand-in for Barça–Athletic we have: the same kind of tracking, the same rate of ball loss, and a similar number of possessions. Both datasets sit in `data/external/` (git-ignored) and are other games, so the rules allow training and tuning on them.

## Results on SkillCorner: 20 broadcast matches, leave one match out

| Dead-ball weights | Cost 3.0 | Cost 2.5 | Cost 2.0 | Wins vs current (cost 3.0) |
|---|---|---|---|---|
| Release (current build) | **0.367** | 0.363 | 0.360 | – |
| IDSSE only (never saw SkillCorner) | 0.392 | 0.389 | 0.383 | 19/20, +0.025 |
| Other 19 SkillCorner matches | 0.401 | 0.397 | 0.391 | 20/20, +0.034 |
| **IDSSE + other 19 SkillCorner** | **0.402** | 0.398 | 0.392 | **20/20, +0.035** |
| Ceiling: perfect dead ball | 0.493 | 0.488 | – | – |

- **macroF1:** 0.789 for the current build, 0.829 for the new one.
- **Dead frames caught:** 82% for the current build, 92% for the new one, with precision staying at 0.92.
- **Higher costs:** 3.5 to 5.0 add at most +0.006, which is not enough to move.
- **Decoder settings:** the prior and dead-ball switching cost are best as they are (0.40 / 1.0).
- **First-touch placement:** keeping it helps a little (+0.002).

## Results on IDSSE: 7 matches, simulated ball loss, 3 seeds

Simulation: long gaps placed mostly in stoppages, plus jitter, matched to Barça–Athletic's measured 150 gaps (median 10.7 s), so that 41% of frames are missing.

| Build | spellF1 | Wins vs current |
|---|---|---|
| Current build (release, cost 3.0) | 0.408 | – |
| IDSSE-trained (leave one out), cost 3.0 | 0.447 | 7/7 |
| IDSSE-trained (leave one out), cost 2.0 | 0.456 | 7/7 |

IDSSE prefers lower costs because its possession truth changes hands about 40% more often than SkillCorner's or the competition's. That is why the cost-2.0 idea did not survive the broadcast matches.

## The candidate submission for Barça–Athletic

These come from the label-free checks; no ground truth was used.

| | Current build | `--deadball open` |
|---|---|---|
| MD5 | f3b7993f… | **89a47208f3290db232ed237ec4f29cfc** (identical over 2 runs) |
| Dead share | 41.1% | 48.0% |
| Stoppages | 82 | 97 |
| Possessions | 286 | 277 |
| Rule checks | 9/9 | 9/9 |
| Weak-label AUC | 0.859 (in-sample) | 0.816 (never fitted on these tags) |

## Code

- `tools/idsse_data.py` and `tools/skillcorner_data.py` are the loaders, each with a cache.
- `tools/idsse_eval.py` and `tools/skillcorner_eval.py` are the scorers; every comparison is leave-one-match-out.
- `tools/open_fit.py` fits on all 27 matches and writes `sonar/deadball_open_weights.json`.
- `sonar/deadball.py`: `detect(..., weights="open")`.
- `sonar/make_submission.py`: new options `--deadball weak|open` and `--possession-cost` (default 3.0).
- The default build is unchanged.

## Caveats

- Neither dataset's definition of possession is exactly the competition's.
- SkillCorner's dead flag comes from its phases of play, not from tracking.
- The new model calls more time dead on Barça–Athletic (48.0%, near the top of the 30–50% band). The open matches average about 46% dead, so this is plausible, but it is the main risk.
- This is not checked against the public 5 minutes, by design.

## Decision (2026-11-14)

The team adopted it: `--deadball open` is now the default in `sonar/make_submission.py`, and `--deadball weak` rebuilds the earlier file (`f3b7993f…`).

- **New expected MD5:** `89a47208f3290db232ed237ec4f29cfc`.
- **Documents:** `docs/TECHNICAL.md` now describes the open-data training and credits both datasets.
- **Public 5-minute check,** run after the choice: spellF1 0.379 for the new build against 0.411 for the earlier one; macroF1 0.761 against 0.755. On 17 possessions that gap is far inside the noise (about ±0.32), and per the rules it was not used to choose.
- **Review:** Lethabo is the reviewer on PR #13.

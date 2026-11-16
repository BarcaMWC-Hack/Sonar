# Lethabo's contributions to Team Sonar (Phase 1)

Author: Lethabo (GitHub `LethaboMH14`), role 3: pipeline, reproducibility, technical document and
evaluation. Period: 1-15 November 2026. Every number from other games is marked `sim_`: it uses proxy
or simulated truth on open data, never the provided match's labels.

## 1. What I delivered

| What | Where | Status |
|---|---|---|
| Synced the challenge docs to the organisers' stricter Oct 22-23 rules (no hand labels on the match for any purpose; public 5 min for checks only) | PR #8 | merged |
| Code reviews of the dead-ball, possession and document PRs, with reproduced numbers | comments on #1, #2, #7 | done |
| Other-game yardstick: score possession variants on Metrica's open games with the official spellF1 | PR #9, `tools/external_eval.py`, `tools/external_full.py` | merged |
| Compliance fixes: dead-ball model trained on Smart Tagging only; tools that read the public ground truth gated behind `--public-check`; video-check proposals removed; data cache keyed by content hash; training-data statement; clean-environment reproduction check `tools/repro_check.py` | PR #10 | merged |
| Honest yardstick on 7 Sportec/IDSSE matches with real frame-level truth, our own predicted dead periods and noise calibrated to our match | `experiments/lethabo/dfl_yardstick.py` | done |
| Broadcast-to-tactical alignment study and the replay/close-up dead-ball cue | `experiments/lethabo/broadcast_*.py`, section "Assets explored" of `docs/TECHNICAL.md` | done, both dropped |
| Learned decoders (GRU, logistic regression, blend) and their confirmation tests | `experiments/lethabo/learned_possession.py`, `confirm_sportec.py`, `ten_games.py`, `hedge.py` | done, dropped |
| Parameter sweeps (81 settings locally; 2,025 settings on Kaggle) and a 3-game, 2-proxy protocol | `experiments/lethabo/external_params.py`, `kaggle_sweep.py`, `joint_report.py`, `external_joint.py` | done, defaults kept |
| Label-free edge-case census of the submission | `experiments/lethabo/edge_census.py` | done |
| No `X` on graded frames: write the label the scorer would carry forward | branch `release-no-x`, PR into `idsse-yardstick` | proposed |
| Submission package and criteria audit | section 5 below | done |

## 2. Decisions and why

| # | Decision | Why |
|---|---|---|
| 1 | Claim role 3 (pipeline, reproducibility, document) | Roles 1 and 2 already had working PRs; the unowned risk was reproducibility and compliance. |
| 2 | Close my early plan PR (#6) | The team's code had overtaken it; a plan that disagreed with the code would confuse reviewers. |
| 3 | Sync the rules docs to upstream (#8) | Our copies were the Oct 15 versions; the Oct 22-23 versions forbid hand labels on the match for any purpose and tuning on the public 5 minutes. |
| 4 | Every change must pass a test fixed before any number is seen; otherwise it is dropped | There is no leaderboard and the public window has only 17 possessions (the scorer says gaps under ~0.33 are noise), so a gain we cannot test is a guess. |
| 5 | Never tune on the public 5 minutes; gate the tools that read them | RULES.md allows the window for pipeline checks only. |
| 6 | Train the dead-ball model on Smart Tagging only, not on the FAQ's pre-kickoff frames (#10) | Reviewers flagged the FAQ frames as a risky training label; they still decide those frames as a published fact. Effect: 2.9% of frames changed. |
| 7 | Key the data cache by file content, not size and time (#10) | A changed input could silently reuse a stale cache and break byte-for-byte reproduction. |
| 8 | Keep the switch cost at 3.0 | Held out across games, 1.5-2.0 was mixed (+0.012 on one game, -0.004 on another). The team later reached the same conclusion on the broadcast-like SkillCorner matches. |
| 9 | Keep the default thresholds (no change from the 2,025-setting sweep) | Leave-one-game-out mean gain -0.009 and the stricter proxy negative: no setting generalised. |
| 10 | Drop the learned decoders | Trained on one provider they gained (+0.07 on held-out Metrica games) but lost on the other (-0.04 to -0.06 on 7 Bundesliga matches); a 10-game model split the same way (7/7 Bundesliga wins, 0/3 Metrica). A half-weight blend still lost 0.034-0.039 off-provider. They learn a provider's convention, so they cannot be trusted on our match. |
| 11 | Stop the broadcast ball-fill | The streams align precisely (first half +160.5 s, second half +1076 s, players within 0.3-0.6 m), but an aligned broadcast ball exists for only 2% of our missing-ball frames: the broadcast camera loses the ball at the same moments. |
| 12 | Drop the replay/close-up dead-ball cue | It failed its gate: 76% of cue time in our stoppages (needed 80%) and only 24% of cue stretches in live time looked like stoppages (needed 60%); the broadcast also cuts to close-ups during play. |
| 13 | Keep the possession call unchanged after testing 5 variants | Persistence 5/10 frames, shape only with the ball, merging spells under 1/2 s: all within -0.014 to -0.001 of the release on the Sportec yardstick; none passed. |
| 14 | Propose writing explicit labels instead of `X` on graded frames | The scorer carries `X` forward (the last label in that half, which can be D), so spellF1 is identical for any truth and macroF1 cannot fall. Verified: the new file equals the scorer's own fill of the release file on every graded frame. |
| 15 | Package the team's adopted release plus decision 14 as the submission | The release (PR #13) beat the earlier build on 20 of 20 SkillCorner matches; decision 14 is provably no worse. Nothing untested goes in two days before the deadline. |

## 3. Key evidence (sim_)

- **Ours against the official baselines on 7 Sportec matches, realistic noise, our own predicted dead periods** (`results/dfl_yardstick.csv`): ours 0.461 spellF1; official nearest-player baselines 0.285 (+-25 frames), 0.290 (+-50, the one behind the 0.2433 floor), 0.213 (+-125). Ours wins all 7.
- **Where the remaining points are**: with true dead periods ours rises to 0.533; with the true team to 0.830. The team call and switch timing hold most of the headroom. Caveat: the simulator drops the ball exactly in true stoppages, so dead-ball detection looks easier there than on our match.
- **Three Metrica games** (dead periods from the proxy, possession only): ours 0.629 against 0.384 / 0.489 / 0.471 for nearest-player votes of 25 / 50 / 125 frames.
- **Tracking statistics, label-free**: our match drops the ball like Metrica's samples (40.7% missing, almost all in gaps of 5 s or more); Sportec tracking never drops it. This is why provider matters.

Corrections to my own earlier evidence, found in review and fixed or relabelled: an event-based proxy mis-assigned some restarts; early "41% masked" tests really had 60-65% missing; a game-3 loader mapped players to the wrong teams; two cache and data-mutation bugs in the Kaggle scripts. Results produced before each fix were rerun or discarded.

## 4. Where the experiment code runs

The scripts in `experiments/lethabo/` were run from my working branches, pushed for provenance as `lethabo/yardstick-v2`, `lethabo/external-yardstick`, `lethabo/gap-inference` and `lethabo/ball-thresholds`. None of them is meant to be merged: `lethabo/ball-thresholds` holds the rejected threshold candidate. Some scripts import helpers or off-by-default options that exist only on `lethabo/yardstick-v2` (for example `possession.localize(..., persist=...)`), so run them from that branch. Open-data runs used private Kaggle notebooks with code and open data only; the match data never left the team's machines in any public form, and none of it is in this repository.

## 5. Submission package and criteria audit (Nov 15)

| Requirement (SUBMISSION.md) | Status |
|---|---|
| `Sonar.csv`: header `frame,label`, 199,319 rows, labels H/A/D/X | **Pass**: official validator; 9/9 rule checks |
| Regenerates byte for byte | **Pass**: two clean-environment cold builds give MD5 `2748a02c58b59f5ae30e940dac7a4745` (Python 3.11.9, pinned requirements) |
| Named after the team: `Sonar.csv`, `Sonar.pdf` or `.md` | **Pass** (`.md`) |
| Public code repository | **Not yet**: the repository is private; an organisation owner must make it public before submitting |
| No match data in the repository | **Pass**: full history scanned on all branches; only `data/README.md` and the organisers' own public files |
| Technical document covers approach, assets used and dropped, training data, possession rules, limitations, reproduction | **Pass** after the `release-no-x` update |
| Upload through the Google Form (URL by email), team name as registered, Google account, one response | **To do** by a team member, by Nov 16 (deadline Nov 17; hour unconfirmed) |
| Validate the exact uploaded file | **To do** at upload: `python evaluation/validate_submission.py Sonar.csv` |

Expected numbers on the public pipeline check (not used for any choice): spellF1 0.3789, macroF1 0.7612, identical to the release because none of the 150 filled frames fall in those 5 minutes.

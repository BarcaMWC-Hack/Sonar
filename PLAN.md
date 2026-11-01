# Plan: Team Sonar — Phase 1 ball-possession (spellF1)
_Draft plan, 2026-11-01._

## Goal
Emit H/A/D (X only under the rule below) for all 199,319 tactical frames, maximising `spellF1`, with `macroF1` as tie-break. Graded windows: frames 11,185–90,651 and 110,808–190,197; evaluation reports both windows separately, the CSV still covers every frame.
Dates: registration closes Nov 10; freeze Nov 14; submit Nov 15–16 (cutoff Nov 17).

## Data and rules posture (from RULES.md)
- Submitted labels must come from an algorithm. No hand-written or hand-corrected submission, no private ground truth.
- Small hand-labelled sets for training/validation are allowed by RULES.md. Default for this plan: do not hand-label this match at all; ask the organisers before any exception. Other-match labels need provenance, rights, organiser access and documentation.
- The 5 public minutes are available for pipeline checks only, not tuning.
- Match data stays local; nothing goes to hosted LLMs or public GitHub.

## Approach
1. Reproduce the four shipped baselines with the local scorer. Public 5-min spellF1: smooth125 0.4242, smooth50 0.3256, smooth25 0.2868, raw 0.0378. These are not comparable to the full-match 0.2433 reference.
2. Evidence layer (deterministic): team-level ball distance, ball speed, tracking quality, Smart Tagging as a weak, N/A-aware prior. Loader: `kloppy`.
3. Control model (frozen fallback): nearest-team with minimum-duration smoothing and carry-through of the last possessor over missing-ball frames. This is what we submit if no credible selection data exists.
4. Candidate model: 3-state H/A/D semi-Markov/HMM. Specify transitions, duration priors and D-restart behaviour before running. Adopt only if it beats the control under the validation protocol.
5. `D` rule: require corroborating stoppage evidence (e.g. restart tags, sustained non-play cues), never missing ball plus no motion alone, since that also describes live-play occlusion.
6. `X` rule: default never. Allowed only by an observable confidence rule (e.g. short tracking gap with stable last-possessor margin), tested at transitions and at half starts, because the scorer forward-fills X within a half, back-fills it at the start of a half, and macroF1 counts it as a miss. Boundary checks cover both fills.
7. Validation: match-level paired comparison on independently labelled other matches if obtainable; possession-level results exploratory only. Three gates, all on match-level paired differences: Wilcoxon signed-rank test (p<0.05); bootstrap 95% CI of the mean paired difference (fixed seed, 10,000 resamples over matches) excluding zero; paired effect size (rank-biserial correlation of the signed differences). Any gate failing means "no significant difference". Smoothing being the main lever is a hypothesis to test, not a finding.
8. Reproducibility: pinned dependencies in a container image (fixed Python minor version, platform and thread settings recorded in the repo), no randomness (or fixed seeds), explicit tie-breaks, fixed CSV serialization, input/config hashes, and a byte-identical regeneration test in a clean environment.
9. Deliverables checklist: team-named CSV passing `validate_submission.py`; public code repo that regenerates it; short technical document (approach, assets, training labels, loose-ball/restart rules, limitations, reproduction steps).

## Risks / open questions
- Private Kaggle storage permitted? Default: local.
- Roster (3–5) and registration state unconfirmed.
- No independent labelled matches identified; without them the control ships.
- Official submission link and exact cutoff hour to confirm from the participant email.

## Out of scope
Broadcast-video alignment, re-detecting players, any hand correction of outputs.

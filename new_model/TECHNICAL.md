# new_model — technical document

A possession pipeline for the Barcelona–Athletic Club tactical video. Every one of the 199,319
frames gets `H`, `A`, `D` or `X`.

`new_model` is an alternative to the frozen `sonar/` release in this branch, not a replacement for
it. Both produce a valid submission. This document describes only `new_model`, the code that
actually generated `new_model/submission.csv`.

## Summary

- **Approach.** 113 tracking-derived features feed a home/away gradient-boosted classifier. The
  dead stage is the release's change-point detector (`sonar/deadball.py`), a logistic model on
  physical evidence with a two-state Viterbi and refined stoppage edges. A masked Viterbi decoder
  settles H/A inside the detector's alive stretches; D is the detector's mask. The decoder penalty
  is chosen on full-match possession structure, not on agreement with the weak labels. Output is
  masked to the two graded runs.
- **Labels.** Weak labels only: Smart Tagging phases for the release detector and the dead/alive
  audit, and the nearest tracked player to the ball for home/away. No human label of this match,
  no private ground truth, no external data.
- **Result.** Byte-reproducible file (`md5 6aeaf0fce2c7adb1f109c924b33ae0e3`). On the public
  correctness check it scores **macroF1 0.765, spellF1 0.516**, matching the release's macroF1 and
  dead-ball F1 (0.653) and exceeding its spellF1 (0.400). Full-match structure is **276
  possessions of mean 14.3 s**, against the release's 277 of 14.2 s. The five-minute window is
  under-segmented (14 predicted vs 17 reference); the full-match structure is the design target.
- **Dependency.** The dead detector is imported from the release, not copied. That is the one
  deliberate coupling: duplicating the detector would create two versions of the same physics.

## Approach

### 1. Features (`features.py`, 113 columns, label-free)

All features are computed for all 199,319 frames, including halftime and the ungraded edges.

| family | n | what it measures |
|---|---|---|
| `ball` | 24 | visibility, position, speed, out-of-play distance, zones, restart spots |
| `geo` | 26 | distances and control weights for the nearest home/away players, pressure counts |
| `shape` | 24 | team width, depth, compactness, forward speed, players ahead of the ball and in the box |
| `motion` | 11 | player speeds, sprint/moving shares, players off the pitch |
| `time` | 13 | rolling windows, time since the ball/touch was seen, half indicator |
| `sp` | 11 | set-piece active/pre-restart flags, timing, type one-hots |
| `homo` | 4 | homography validity and reprojection error |

Coordinates are converted from the normalised tracking to metres on the real 105.3 × 68.0 m pitch.
Team orientation is read from where each team actually stands in each half, so no feature depends
on a hard-coded attacking direction.

### 2. Weak labels (`labels.py`)

- **Dead.** The 8 s before each Smart Tagging `SET PIECES` interval, plus the organiser-published
  second-half pre-kickoff block `110808–113741`. The set-piece interval itself is not labelled
  dead, because the restart is inside it.
- **Alive and team.** Frames inside a Smart Tagging possession phase whose `Team` tag names
  Barcelona or Athletic Club: `BUILD UP`, `PROGRESSION`, `CREATING CHANCES`, `ATTACKING
  TRANSITION`, `BALL IN FINAL THIRD`, `BALL IN THE BOX`, `LONG BALL`. Each phase is trimmed by
  1.5 s at both ends. `DEFENDING …`, `DEFENSIVE TRANSITION`, `GOALS` and `Team = N/A` are dropped.
- **Guard.** Alive labels within 10 s of a set-piece boundary are dropped and given weight 0,
  because those phase edges are seconds off.
- **Team target for the team stage.** Not the Smart Tagging `Team` tag, which is phase-level. The
  target is the nearest tracked player to the ball, majority-voted over ±50 frames (2 s). That is
  the available proxy for the tracking's own owner, which is what the ground truth is derived from.
- **Amount.** 73,454 labelled frames (14,467 dead, 58,987 alive). Temporal split at frame 150,000:
  57,662 training and 15,792 validation frames. The team stage has 46,728 training and 12,259
  validation frames with a non-zero team target. The split is temporal, never random: adjacent
  frames are not independent.

### 3. Stage 1 — dead vs alive (`sonar/deadball.py`, imported)

The dead stage is the release's fixed change-point detector, not a model retrained here. It builds
per-frame physical evidence from the provided ATD -- ball out of bounds, ball speed, ball-known
share, missing-run, mean/top player speed, sprint share, players outside the pitch and own half,
restart spots -- fits a logistic model on the Smart Tagging dead/alive weak labels, decodes a
two-state path, and snaps each stoppage edge to a ball crossing, reacquisition or strike. On this
match it calls **60,034 dead frames (37.8% of the graded runs)** across **84 stoppages**, with
in-sample weak-label agreement on training frames of dead recall 0.793 and alive false-dead 0.108.
Those agreement numbers are in-sample: the detector is fit on the same weak labels it is scored
against. They are provenance, not validation.

The published second-half block `110808–113741` is forced dead after decoding; the detector
already includes it, and the forced block is a guard, not a second signal.

**Why the previous stage was replaced.** The earlier HGB dead stage (150 iterations, 15 leaves,
leaf 200, L2 2.0, inverse-frequency weights, 90 tracking features) reached training AUC 1.000 and
5-fold test AUC 0.802–0.885 (mean 0.836), gap 0.164 mean / 0.198 worst; half-swap 0.756/0.768. A
capacity/regularization sweep (7 leaves, L2 10, min-leaf 1000) left test AUC at 0.822–0.842. It was
fitting the Smart Tagging set-piece proxy, and its decoded D recall on the public window was 0.462
(D F1 0.590). The release detector's D F1 on the same window is 0.653. The audit is reproducible
with `python -m new_model.crossval --dead-stage hgb`; the shipped path does not train it.

### 4. Home vs away

The HGB defaults above (150 iterations, learning rate 0.05, 15 leaves, leaf size 200, L2 2.0,
`random_state=0`) with inverse-frequency weights, trained only on weak-alive frames with a non-zero
smoothed owner.

`select_features` drops features that would let the model reconstruct the weak label rather than
the ball state: all `sp_*` (set-piece timing is the weak dead label's own definition), all `homo_*`
(calibration quality, not football), all `ball_spot_*` (mostly missing where it matters),
`ball_still`, `ball_speed_change`, `time_frame_in_half` (resets at halftime and invites a
cross-half leak), and, for the dead audit only, `ball_vx_progress`.

Single-split validation AUC 0.975, accuracy 0.924, logloss 0.207. Five-block CV mean test AUC 0.980
(0.967-0.992), train-test gap 0.019. Half-swap 0.974/0.975. This stage is stable and is not the
bottleneck.

### 5. Decode (`decode.py`)

The release detector's mask is a hard constraint: D where it says dead, H/A only inside its alive
stretches. A three-state Viterbi pass then settles the team label inside each graded run, decoded
separately per run. Every label switch costs a fixed penalty. A direct `H <-> A` switch costs two
penalties, the same as `H -> D -> A`, so the decoder does not invent a `D` just to avoid a direct
turnover; D is available only where the detector put it. The penalty therefore controls how much
team evidence is needed to switch, not whether the ball is dead.

The penalty is chosen on **full-match possession structure**, not on weak-label macroF1. The
validated release decodes to 277 possessions of mean 14.2 s (`sonar/POSSESSION.md`), inside
football's typical 10-15 s; the target here is 14.2 s. The sweep is `2..128` with 68 included; the
penalty whose full-match mean duration is closest to 14.2 s is chosen. It chose **68.0**: 276
possessions, mean 14.3 s, 2 sub-2 s spells.

The first version (HGB dead stage) chose the smallest penalty within 0.003 macroF1 of the best on
the weak validation labels, and it chose 8.0. That produced 486 possessions of mean 8.5 s, 27
public possessions against 17, and spellF1 0.291. MacroF1 is reported in the sweep as a diagnostic,
but it does not choose: the weak labels fragment possession, so fitting them rewards the
over-segmentation the scorer penalises. With the release dead mask, penalty 8 still over-segments
(440 possessions of mean 9.0 s).

After decoding:

- the published second-half block `110808–113741` is forced to `D`;
- `X` is written outside the two graded runs `11185–90651` and `110808–190197`;
- the file is format-checked with the same validator the scorer uses.

## Validation harness and label audit

`new_model/crossval.py` mirrors the release's `sonar/splits.py` protocol without importing it:
five consecutive blocks per half, every frame within 10 s of a test block excluded from training,
graded frames only. It reports mean and worst fold, because one temporal split can hide a
half-specific shift.

The shipped dead stage is the release detector, fixed, so it has no fold-honest train/test gap:
only the team stage is refit per fold. The team stage is stable:

| stage | train AUC | 5-fold test AUC | gap (mean/worst) | half-swap |
|---|---|---|---|---|
| home vs away (shipped) | 0.999 | 0.967-0.992 (mean 0.980) | 0.019 / 0.033 | 0.974 / 0.975 |
| dead vs alive (HGB audit, discarded) | 1.000 | 0.802-0.885 (mean 0.836) | 0.164 / 0.198 | 0.756 / 0.768 |

A team-stage control block (ball height, closing rates, contest gap, nearest speed, lane occupancy)
was tried and dropped: 5-fold test AUC stayed 0.980 and weak spellF1 was 0.590 vs 0.594, so it
added complexity without a fold-honest gain.

The HGB dead stage was the weak one, and it was not weak because it needed more rows: a capacity
sweep (7 leaves, L2 10, min-leaf 1000) left test AUC at 0.822-0.842 and the gap at 0.158-0.176.
The half-swap result is symmetric, so it was not a first-half/second-half direction problem. It
was fitting the Smart Tagging set-piece proxy, and that proxy does not transfer perfectly to a
held-out block. That is why the shipped pipeline uses the release detector instead; the HGB row is
reproduced with `--dead-stage hgb` and is not a shipped component.

The structural sweep is the evidence for the decoder choice. Structure is decoded by the shipped
artifact with the release dead mask; weak spellF1 is the 5-fold mean against the weak labels. The
penalty was chosen on structure (closest to 14.2 s). The public check was run at penalties 48, 64
and 68 during the rework; 48 returned 0.485 and 64/68 returned 0.516. It is a correctness check,
not the selector:

| penalty | full-match possessions | mean | sub-2 s | weak spellF1 (CV mean) | public spellF1 |
|---|---|---|---|---|---|
| 8 | 440 | 9.0 s | 60 | 0.679 | not scored |
| 32 | 350 | 11.3 s | 21 | 0.640 | not scored |
| 48 | 311 | 12.7 s | 6 | 0.615 | 0.485 |
| 64 | 288 | 13.7 s | 3 | 0.599 | not scored |
| **68** | **276** | **14.3 s** | **2** | **0.594** | **0.516** |
| 96 | 236 | 16.7 s | 1 | 0.557 | not scored |

The weak-label metric and the public check still move in opposite directions across this range:
weak spellF1 is best at the over-segmented penalty 4 (0.688), public spellF1 is 0.516 at 68 and
0.485 at 48. That is why weak-label spellF1 is not the selection signal. The 5-fold team refits
decode slightly smoother than the shipped artifact (270 possessions, 14.6 s at penalty 68); that is
a robustness check, not the selection input. The public check remains a correctness check, not a
validation score: the window sits inside the period the team stage trains on, and inside the weak
labels the release detector was fit on.

## Possession rules implemented

- **Ball alive.** `D` is the release detector's refined dead mask, plus the organiser-documented
  pre-kickoff block. A frame in a graded run is otherwise `H` or `A`. The mask is a hard constraint:
  the H/A decoder cannot overwrite it.
- **Who has it.** The team of the nearest tracked player to the ball, smoothed over 2 s, is the
  team-stage target and the mechanical validation reference. The classifier predicts a per-frame
  team probability from geometry and shape, and the decoder imposes persistence.
- **Loose balls.** There is no explicit loose-ball state. A loose ball keeps the previous label
  because the 2 s majority vote and the switch penalty make a one-frame change expensive.
- **Challenges.** There is no explicit challenge detector. A challenge only changes possession if
  the new owner's evidence survives the smoothing and pays the decoder's switch cost.
- **Restarts.** The detector's stoppage edges are snapped to the ball crossing, reacquisition or
  strike, so the dead interval ends when the ball is actually back in play. The 8 s weak-label
  window is used for the release detector's training and the dead/alive audit, not as the shipped
  boundary.
- **Dead-ball ownership.** Not predicted. `D` is a single state; the tracking's dead-ball team is
  discarded, as the challenge asks.
- **X.** Used outside graded play. It is never used to fill a gap inside a possession.

## Training data

Every labelled dataset that touches this pipeline:

| data | game | source / who labelled it | how much | used for |
|---|---|---|---|---|
| Tactical ATD tracking | provided match | organiser's automatic tracker | all 199,319 frames | every feature |
| ATD XML metadata | provided match | organiser | 46 track slots | home/away slot map |
| Smart Tagging XML | provided match | organiser, algorithmically inferred | 638 instances; 336 possession phases; 59 set pieces | release dead detector's training labels, dead/alive audit, set-piece features |
| Nearest-player owner | provided match | derived by this code from the ATD, no person | 46,728 train / 12,259 val frames | team-stage target |
| Published dead block and graded runs | provided match | organiser documentation (`FAQ.md`) | 2,934 forced-dead frames, two runs | output masking |
| Public 5-minute ground truth | provided match | organiser | 17 possessions / 7,500 graded frames | **correctness check only** |
| External matches | none | — | — | **not used** |

**Human labels used to fit or select anything: zero.** No hand labels, hand corrections or
human-produced data about the provided match are used, for training, validation or tuning. The
public five minutes is read only by the `--score` flag; it does not select a feature, threshold,
penalty or split. The weak labels were produced by algorithms, not by a person watching video.

**Assets tried and dropped.** Homography quality features were computed and then dropped from the
team matrix and the dead audit (calibration quality, not football). Ball-spot codes and the Smart
Tagging possession-team tag were dropped for the reasons above. Broadcast video was not used.

## Results

Full match, structural counts:

| | new_model |
|---|---|
| rows | 199,319 |
| H / A / D / X | 62,806 / 36,017 / 60,034 / 40,462 |
| possessions, mean length | 276, 14.3 s |
| dead share of graded frames | 37.8% |

Public 5-minute correctness check (not a validation score):

| metric | new_model | release `docs/TECHNICAL.md` |
|---|---|---|
| macroF1 | 0.765 | 0.765 |
| spellF1 (IoU 0.5) | 0.516 (0.581) | 0.400 |
| D precision / recall / F1 | 0.951 / 0.498 / 0.653 | — / — / 0.653 |
| predicted possessions (truth 17) | 14 | 19 |
| mean predicted possession | 19.2 s | 14.2 s |
| team accuracy on both-alive frames | 0.873 | 87.7% (jointly live) |

The public window is only 17 possessions, and the release reports a 5-95% spellF1 interval of
0.233-0.567; the 0.516 vs 0.400 difference is a point estimate, not a significant win. The
structural match (276 vs 277, 14.3 vs 14.2 s) is the stronger evidence. The evaluator's binary
alive/dead F1 is 0.939, but that is a different metric from the D-class F1 in the release table and
is not compared here. The window is under-segmented (14 vs 17), which is the visible cost of the
smoother full-match structure.

## Limitations

1. **Dead-boundary recall is still the visible gap.** The release detector matches the release's
   D F1 on the public window (0.653) and the full-match structure (276 / 14.3 s vs 277 / 14.2 s),
   but public D recall is 0.498: it misses half the dead frames the reference calls dead. The
   public window is under-segmented (14 predicted vs 17 reference, mean 19.2 s vs 14.2 s). This is
   the release's known dead-boundary behaviour, not a new model failure; it is the main open
   issue if the scorer weights D recall.
2. **The dead stage is a fixed import.** `new_model` imports `sonar.deadball`; it does not train or
   tune the detector. That is deliberate (one physics implementation, no copy divergence), but it
   means the dead stage cannot be re-validated fold-honestly here: the detector is fit on the full
   match's weak labels, including the public window. The team stage is the only stage this harness
   refits. The HGB audit (train 1.000, test 0.836) is why the HGB stage was discarded, not a
   current component.
3. **Owner proxy.** The nearest-player target flickers on deflections and crowded frames; the 2 s
   majority vote hides that rather than resolving it. A deflection can still create or destroy a
   possession.
4. **Missing ball.** The ball is absent on roughly 41% of graded frames. There is no ball
   interpolation through gaps; those frames rely on shape and motion.
5. **Weak-label metrics are not the target.** The 5-fold weak spellF1 peaks at the over-segmented
   penalty 4 (0.688) and falls to 0.594 at the shipped 68, while public spellF1 is 0.516.
   Weak-label agreement cannot rank this pipeline against the release, and it cannot choose the
   decoder.
6. **Public window inside training.** The five minutes lie inside the period the team stage trains
   on and inside the weak labels the release detector was fit on, which is why it is a wiring check
   and not a score.

## Reproduction

```bash
python -m pip install -r new_model/requirements.txt   # Python 3.13.12, exact pins
# data/tactical/ holds the organiser files, or set MTAH_DATA_DIR to the bundle root
python new_model/main.py --rebuild --no-importance \
    --expect-md5 6aeaf0fce2c7adb1f109c924b33ae0e3
```

The cold-cache run rebuilds the ATD parse, the 113-column feature table, the release dead mask, the
team model and the CSV. Verified: two independent cold builds produced the same file, `md5
6aeaf0fce2c7adb1f109c924b33ae0e3`, `sha256
f8c3c70c0d1a48e60ca8817241e8d6b3580439c846d704488eccfd3e252803ea`. The official
`evaluation/validate_submission.py` reports `OK, 199319 rows, frame 0..199318`.

No match asset, weak-label table or generated prediction is committed. `data/*` and
`new_model/submission.csv` are git-ignored; the labels are rebuilt at runtime from the provided
Smart Tagging.

# Sonar — Phase 1 Technical Document

Draft, 2026-11-01. Living copy: the team's Claude Doc; this file is the version that ships with the code.

## Summary

Sonar labels every frame of Barcelona vs Athletic Club with a three-stage pipeline built only on the provided tracking and Smart Tagging: a dead-ball detector, a control-based possession model, and edge placement from ball and player events. No setting was chosen on the 5-minute public ground truth; every choice rests on label-free checks against how a football match behaves.

- **Full match:** 80 stoppages covering 38% of graded time, and 277 possessions of mean 14.2 s, both close to a real match.
- **Public 5 minutes (correctness check only):** dead-ball F1 0.65 against 0.32 for the published baseline, macroF1 0.77 against 0.62, spellF1 0.40 against 0.33. All spellF1 intervals overlap on 17 possessions.
- **Reproducibility:** one command rebuilds the submission byte for byte, including from a cold cache.

## Problem and data

Each of the 199,319 tactical-video frames gets H, A, D or X; 158,857 frames are graded. Ranking is on spellF1 (possessions matched by temporal IoU 0.3-0.7). Being 1 s late on every edge costs about a third of the score.

| Fact (counted on the delivered file) | Count | Consequence |
| --- | --- | --- |
| Ball position missing | 41% of graded frames | Design for gaps |
| Missing-ball runs of 2-5 s | 0 (142 runs of 5 s+) | Tracker fills short gaps; a missing ball means a long gap, mostly a stoppage |
| Ball speed empty where position known | 91% | Speed derived from positions |
| Nearest-player team changes | 25 per live minute | Nearest player is not possession |
| Same, only when a player has a slow ball | 2.6 per live minute | Close to the real change rate |
| Smart Tagging set pieces in graded time | 58 (5 free kicks) | Weak labels, far from complete |

## Approach

Inputs (ATD, Smart Tagging) → dead ball → who has it → switch timing → frame labels. The public ground truth is used only to check the pipeline is wired correctly.

## Dead-ball detection (`sonar/deadball.py`)

80 stoppages, 38.2% of graded time, median 19 s.

1. Features: player speeds, moving/sprinting shares, players off the pitch, kick-off line-up, ball over a line, ball speed, long ball gaps, restart spots.
2. Logistic model on weak labels: dead = 8 s before each tagged set piece plus frames 110,808-113,741 (FAQ.md states these are graded and all D); alive = open-play phases clear of set pieces. Cross-half AUC 0.82 / 0.87.
3. Two-state Viterbi, one switch = one second of evidence.
4. Edges from unsmoothed signals: start at line crossing, deceleration or ball vanishing; end at the earliest ball event around the players' acceleration.

## Possession (`sonar/possession.py`, `sonar/team_shape.py`)

277 possessions of mean 14.2 s (baseline 457 of 8.5 s).

1. Control evidence: nearest player within 1 m of a ball under 4 m/s, fading by 8 m/s.
2. Team shape: width alone AUC 0.86; shape model 0.85-0.87 across halves; weight 0.05 per frame.
3. Viterbi with switch cost 3 decides whether possession changed.
4. Each switch moves to the new team's first touch or control (touch = >5 m/s ball velocity jump within 1.5 m of a player; 15.8 per live minute).
5. Smart Tagging names the team only for a live stretch with no ball or team-shape evidence. Team shape is available on every frame, so on the delivered file this reaches at most one of the 81 live stretches (8 stretches have no ball evidence and take their team from shape).

## Validation

| Full match, no ground truth | Baseline nearest | Sonar |
| --- | --- | --- |
| Possessions, mean length | 457, 8.5 s | 277, 14.2 s |
| Agreement with Smart Tagging team | 66.2% | 77.0% |
| Switches the new team barely controls | 239 | 36 |

| Public 5 minutes | baseline_smooth50 | Sonar |
| --- | --- | --- |
| spellF1 (5-95%) | 0.326 (0.21-0.45) | 0.400 (0.23-0.57) |
| macroF1 | 0.621 | 0.765 |
| Dead-ball F1 | 0.323 | 0.653 |
| Possessions (truth 17) | 26 | 19 |

Other games (Metrica sample games 1 and 2, with a simulated truth): switch cost 2.0 scores highest on both, but picking the best setting on one game and testing it on the other is mixed, so the default stays at 3.0 (see `sonar/POSSESSION.md`).

## Data used for training and validation

No hand labels were made on the provided match, and none on any other game. Every dataset that touches the model:

| Data | Game | Labelled by | Used for |
| --- | --- | --- | --- |
| Automatic tracking (ATD) | provided | the organisers' tracker (RULES.md: automatic tracking is given) | all features |
| Smart Tagging (phases, set pieces, team tags) | provided | inferred by an algorithm ("the Smart Tagging is inferred too", CHALLENGE.md) | dead-ball weak labels (training); the team fallback (at most one stretch) |
| All-dead interval 110,808-113,741 | provided | stated by the organisers in FAQ.md, not by us | dead-ball weak labels (training) |
| Team-shape fit | provided | computed from the tracking itself (frames where ball control is clear), no external labels | team shape (fitted on the same match) |
| Public 5-minute ground truth | provided | the organisers | the scorer's correctness check only; never used to choose a setting |
| Metrica open sample games 1 and 2: tracking and events | other games | Metrica's published sample data; we did not produce these labels | comparing possession settings in `tools/external_full.py` (PR #9); not training data for the submission |

## Reproducibility and data handling

```
python -m pip install -r requirements-sonar.txt
python -m sonar.make_submission
python tools/check_submission.py
```

Python 3.11.9 with numpy 2.4.6 and pandas 3.0.6 only (pinned in `requirements-sonar.txt`), no randomness, weights rounded to 4 decimals. No match data committed or given to any hosted model; scripts print only scores, counts and shares.

## Limitations and next steps

- Only 4 of 20 public changes fall within 1 s; whether the truth flips at first controlled touch is unconfirmed (`--no-localize` keeps the earlier placement). Reading the public labels to understand that rule is the window's stated purpose; it is not used to tune anything.
- Short throw-ins and fouls likely under-detected.
- No ball inference through long gaps yet.
- No hand labels or hand corrections on this match are used anywhere, including for validation (RULES.md, rule 1). Edge rules are checked only against label-free football norms and Smart Tagging, which is algorithmic.

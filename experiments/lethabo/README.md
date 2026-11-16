# Lethabo's experiments (archive, not part of the submission build)

These are the scripts behind the decisions in `docs/CONTRIBUTIONS_LETHABO.md`. None of them is imported by
`sonar/make_submission.py`, and none of them changes the uploaded file. All numbers they print are `sim_`:
proxy or simulated truth on open data, never the provided match's labels.

| Script | Question it answered | Outcome |
|---|---|---|
| `dfl_yardstick.py` | How does the full pipeline (predicted dead periods, realistic noise) compare with the official baselines on 7 Sportec matches with frame-level truth? | ours 0.461 vs 0.285 / 0.290 / 0.213; 7 of 7 |
| `team_call_test.py` | Do persistence, shape-only-with-ball or short-spell merging improve the team call? | none passes the pre-set gate |
| `edge_census.py` | Label-free counts of risky situations in our submission | 37/84 stoppages with a moving ball; 31 spells under 2 s; 15 switches with no ball evidence |
| `broadcast_align.py`, `broadcast_cue.py` | Can the broadcast stream fill ball gaps or flag stoppages? | aligned, but fills 2% of gaps; cue fails its gate |
| `domain_stats.py` | Which open provider does our match resemble? | Metrica-like ball loss (40.7% missing) |
| `learned_possession.py`, `confirm_sportec.py`, `ten_games.py`, `hedge.py` | Do learned decoders beat the rule decoder off-provider? | no: gains on one provider, losses on the other |
| `external_params.py`, `kaggle_sweep.py`, `joint_report.py`, `external_joint.py`, `external_game3.py` | Do any thresholds or switch costs generalise across games? | no; defaults kept |
| `external_gaps.py`, `external_gapshape.py`, `gaps.py` | Should switches be placed differently inside ball gaps? | no gain |

`results/` holds the CSV outputs on open data only. Data comes from public sources (Sportec/IDSSE via
kloppy, Metrica sample data, SkillCorner open data); paths are read from `MTAH_DATA_DIR` or downloaded at run
time. Several scripts need options that exist only on branch `lethabo/yardstick-v2`; run them from there.

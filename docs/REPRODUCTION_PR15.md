# PR #15 reproduction report — reviewer machine

**PR:** #15, “Lethabo: contributions, decision log and experiment archive”
**Branch:** `lethabo-contributions`
**Commit:** `f25e183`
**Base:** `772d66b`; `main` is now `26b6809`
**Date:** 2026-11-16
**Scope:** verify the claims and reproducibility of PR #15 on this machine under Python 3.11.9.

## 1. Verdict

PR #15 adds only `docs/CONTRIBUTIONS_LETHABO.md` and `experiments/lethabo/`; it changes no `sonar/` code and nothing the submission build imports. **It cannot change the submission’s efficacy.** The submission audit in the doc refers to the separate `release-no-x` branch, not to code in #15.

Under Python 3.11.9:

- The `release-no-x` submission audit reproduces **exactly**: MD5 `2748a02c58b59f5ae30e940dac7a4745`, validator 9/9, public `macroF1 0.7612`, `spellF1 0.3789`.
- The archived numerical claims are consistent with the committed CSVs: the Sportec yardstick, the learned-decoder results, the 2,025-setting sweep, the edge census, and the 40.7% ball-loss figure all check out.
- The experiment package is **not reproducible from the PR branch**. The archived scripts are location-broken, the helpers are spread across four provenance branches, the main external yardstick cannot fetch its data today, `torch` is unpinned/absent, the open data is not in the repository, and `dfl_yardstick.csv` is malformed.

**Recommendation:** safe to merge as documentation after correcting the stale audit and the reproducibility claims. Do not describe it as a runnable, self-contained reproduction package.

## 2. Machine and environment

- Linux x86_64, NixOS devshell; workspace `/mnt/nixfiles/projects/Sonar`.
- Data present: `data/tactical/` (ATD, Smart Tagging, homography, no video).
- Data absent: `data/external/` (Metrica/Sportec), broadcast video.
- Python: **CPython 3.11.9** created with `uv` at `/tmp/opencode/py311`; numpy 2.4.6, pandas 3.0.6 from `requirements-sonar.txt`; `kloppy==3.18.0` installed for this test; no `torch`.
- No repository files were changed. Test worktrees: `/tmp/opencode/lethabo-yardstick` (`origin/lethabo/yardstick-v2`) and `/tmp/opencode/release-no-x` (`origin/release-no-x`).

## 3. Python 3.11.9 — what worked

### 3.1 Clean reproduction check

`tools/repro_check.py --expect 2748a02c58b59f5ae30e940dac7a4745` on `origin/release-no-x`:

```text
MD5 run 1: 2748a02c58b59f5ae30e940dac7a4745
MD5 run 2: 2748a02c58b59f5ae30e940dac7a4745
PASS: both independent cold builds match the expected upload file.
```

The tool also reported 199,319 rows, the expected label counts, and **9 of 9** release checks.

### 3.2 Public check

`evaluation/run_evaluation.py --results /tmp/opencode/py311out --gt public --no-detail`:

- `macroF1 0.7612`
- `spellF1 0.3789`
- 21 predicted possessions, mean 12.8 s; ground truth 17 possessions, 14.2 s.

Exact match with the PR’s expected numbers.

### 3.3 Experiment checks

- `edge_census` on `origin/lethabo/yardstick-v2`: 37/84 moving-ball stoppages, 31 spells under 2 s, 15 switches with no ball evidence; dead share 0.379, 284 spells, mean 13.9 s, 200 H↔A switches.
- `joint_report`: 2,025 settings × 3 games, mean `-0.0094`, min `-0.0212`, `ADOPT NO`.
- `pandas.read_csv` on `dfl_yardstick.csv`: `ParserError` at line 3 (see section 4.3).
- Archived `dfl_yardstick.py --smoke`: `ModuleNotFoundError: external_eval`.
- Archived `edge_census.py`: `games`/`fill_graded_abstentions` failures.
- Provenance `dfl_yardstick.py --smoke` with `kloppy==3.18.0`: `XMLSyntaxError: Document is empty`.

### 3.4 Caveat on “cold”

`repro_check.py` reports cold builds, but I set `MTAH_DATA_DIR` to the existing provided data directory. That means the ATD parse cache was not necessarily deleted between the two runs. The two outputs were byte-identical; a fully cold-cache confirmation should use a copied `data/` tree and clear its `cache/` between builds.

## 4. Python 3.11.9 — what did not work

### 4.1 Archived scripts do not run from the PR branch

- `experiments/lethabo/dfl_yardstick.py --smoke`
  → `ModuleNotFoundError: No module named 'external_eval'`
- `experiments/lethabo/edge_census.py`
  → `ModuleNotFoundError: No module named 'games'`
- Same script with `PYTHONPATH=$PWD`
  → `AttributeError: module 'sonar.make_submission' has no attribute 'fill_graded_abstentions'`

Cause: the archive is in `experiments/lethabo/`, but the scripts compute `ROOT` as `experiments/` and their docstrings still say `python tools/...`. The helpers are in `tools/` on provenance branches, not in `experiments/lethabo`. The PR branch’s `sonar` also predates `fill_graded_abstentions`.

### 4.2 The main external yardstick cannot be rerun

`tools/dfl_yardstick.py --smoke` from `origin/lethabo/yardstick-v2`, with `kloppy==3.18.0` installed:

```text
lxml.etree.XMLSyntaxError: Document is empty, line 1, column 1
```

The failure is inside `kloppy.sportec.load_open_tracking_data`; the open Sportec endpoint returns an empty document. The 0.461-vs-baselines result is verified from the committed CSV, not independently regenerated.

### 4.3 Malformed result CSV

`pandas.read_csv('experiments/lethabo/results/dfl_yardstick.csv')` fails:

```text
ParserError: Error tokenizing data. C error: Expected 6 fields in line 3, saw 7
```

`dfl_yardstick.py:246` writes the `method` field unquoted; `ours, true dead` and `our dead, true team` contain commas. The numbers are correct if parsed by hand; the committed artifact is not machine-readable.

### 4.4 Missing dependencies and data

- `torch` is absent and unpinned; `learned_possession.py`, `ten_games.py`, `hedge.py` need it. The learned results were checked from the CSVs only.
- `data/external/` is absent. `external_game3.py` hardcodes `data/external/metrica`; `kaggle_sweep.py` reads `data/external/metrica`. The README’s claim that paths come from `MTAH_DATA_DIR` is not true for those scripts.
- The 2,025-setting sweep was run in private Kaggle notebooks; only the result CSV is archived. `joint_report.py` verifies the decision, not the sweep.
- `broadcast_align/cue` need video; the local bundle has none. `domain_stats --local` needs Metrica data. Those claims were not rerun.

## 5. Efficacy summary

| Claim | Source | Result on this machine | Status |
|---|---|---|---|
| Sportec yardstick 0.461 vs 0.285/0.290/0.213; 7/7 | `dfl_yardstick.csv` | 0.46065 vs 0.28534/0.29026/0.21324; 7/7 | Supported; CSV malformed |
| True dead 0.533; true team 0.830 | same | 0.53251; 0.82983 | Supported |
| Learned decoders 7/7 Bundesliga, 0/3 Metrica | `learned_ten_games.csv` | 7/7; 0/3 | Supported |
| Learned loses -0.04 to -0.06 off-provider | `learned_sportec_confirm.csv` | -0.0439 to -0.0594 (`gru@5`) | Supported |
| Learned gains +0.07 in-provider | `learned_metrica_logo.csv` | +0.0727 clean, +0.0530 masked | Supported |
| No threshold generalises | `threshold_sweep_2025.csv` | 2,025 × 3; mean -0.0094; ADOPT NO | Supported |
| 37/84 moving-ball stoppages; 31 short spells; 15 no-evidence switches | `edge_census` | 37/84; 31; 15 | Supported on provenance branch |
| 40.7% ball loss | tracking | 0.4069 graded; 0.4306 all frames | Supported |
| Submission MD5 / public score | `release-no-x` | `2748...`; 0.7612 / 0.3789 | Verified under 3.11.9 |
| Broadcast alignment / cue gates | prose only | no video; not run | Not verified |

## 6. Reproducibility summary

| Item | From PR branch | From provenance branch | Verdict |
|---|---|---|---|
| `joint_report.py` | Yes | Yes | Reproducible from CSV |
| `edge_census.py` | No: missing `games`, `fill_graded_abstentions` | Yes on `yardstick-v2` | Reproducible only on provenance branch |
| `dfl_yardstick.py` | No: missing `external_eval` | `--help` yes; `--smoke` fails at empty Sportec XML | Not reproducible today |
| `learned_possession.py`, `ten_games.py`, `hedge.py` | No: no `torch` pin/install | Not run | CSV claims only |
| `broadcast_align.py`, `broadcast_cue.py` | No video | Not run | Not reproducible here |
| `domain_stats.py` | No Metrica data | Not run | Not reproducible here |
| `threshold_sweep_2025.csv` | Yes | Yes | Result verified; sweep itself private |
| `release-no-x` submission | N/A | Yes | Reproducible under 3.11.9 |

## 7. Submission package and criteria audit

For the **`release-no-x`** branch, which is the branch the doc describes:

- `Sonar.csv` header, 199,319 rows, labels H/A/D/X: pass.
- Regenerates byte-for-byte: pass; MD5 `2748a02c58b59f5ae30e940dac7a4745` on two 3.11.9 builds.
- Validator: 9/9 checks pass.
- Public check: `macroF1 0.7612`, `spellF1 0.3789`.
- Public repository: the doc says “Not yet: private”, but `gh repo view` reports **PUBLIC**. The audit is stale on this point.
- No match data in history: `git rev-list --all --objects` found no tactical/broadcast video or ATD match assets; only the organiser public GT and `data/README.md`.
- Technical document, upload form: not tested here.

For PR #15’s own branch, the base release build is `8623c439...` with 150 graded X frames. That is **not** the audited `2748...` file. The audit belongs to `release-no-x`; the PR text should say that explicitly.

## 8. What should be fixed before calling this reproducible

1. Add a runnable entrypoint from the PR branch: either move the helpers into `experiments/lethabo/` or add the right `sys.path`/`ROOT` and update the `tools/...` docstrings.
2. Put all scripts and off-by-default options on one branch, or add a script-to-branch manifest. The README currently points only to `yardstick-v2`; `external_params`, `kaggle_sweep`, `ten_games`, `hedge`, `external_gaps`, and `external_gapshape` live elsewhere.
3. Add an experiment requirements file: `kloppy` version, `torch` version, and any other optional dependencies.
4. Add a data manifest: where Metrica/Sportec come from, how to fetch them, and which scripts honour `MTAH_DATA_DIR`.
5. Quote the `dfl_yardstick` `method` field (`csv.writer` or explicit quoting) and regenerate the CSV.
6. Commit the `edge_census`, broadcast, and domain-stat outputs as CSVs/JSON, or state that only prose was retained.
7. Update the audit: repository is public; the audited file is `release-no-x`; the PR is based on `772d66b` while `main` is `26b6809`.
8. Re-run the Sportec yardstick with a working data source, or document that it is a frozen CSV and cannot be regenerated.

## Appendix: key commands

```bash
# PR
gh pr checkout 15
git diff --stat main...HEAD

# release-no-x audit, 3.11.9
uv venv --python 3.11.9 /tmp/opencode/py311
uv pip install --python /tmp/opencode/py311/bin/python -r requirements-sonar.txt
cd /tmp/opencode/release-no-x
MTAH_DATA_DIR=/mnt/nixfiles/projects/Sonar/data \
  /tmp/opencode/py311/bin/python tools/repro_check.py \
  --expect 2748a02c58b59f5ae30e940dac7a4745

# committed sweep
/tmp/opencode/py311/bin/python experiments/lethabo/joint_report.py \
  experiments/lethabo/results/threshold_sweep_2025.csv

# census on the provenance branch
cd /tmp/opencode/lethabo-yardstick
MTAH_DATA_DIR=/mnt/nixfiles/projects/Sonar/data \
  /tmp/opencode/py311/bin/python tools/edge_census.py
```

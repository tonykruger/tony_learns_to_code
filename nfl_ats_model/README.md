# NFL Against-the-Spread Model Research

Independent NFL ATS modeling experiments (001–008), including power ratings, EPA features, QB continuity auditing, confidence ranking, regression scaling diagnostics, and walk-forward comparisons.

## Status

**Research only.** No reliable profitable betting edge has been demonstrated. Historical closing-line returns are hypothetical, and 2023–2025 results have been inspected repeatedly and are not a pristine holdout.

## Scripts

- `nfl_walters_experiment_001.py` — initial model
- `nfl_walters_experiment_002.py` — EPA/success-rate features
- `nfl_walters_experiment_003.py` — independent margin model
- `e3_auto_qb1_v4_fixed.py` — frozen QB continuity audit
- `nfl_experiment_004_top6.py` — top-six edge ranking
- `nfl_experiment_005_confidence.py` — confidence ranking
- `nfl_experiment_006_audit.py` — margin compression and underdog bias audit
- `nfl_experiment_007_regression.py` — scaling and regularization comparisons
- `nfl_experiment_008_walkforward.py` — opponent/recency features and season walk-forward

## Running

Use Python 3.10+ on Windows or another platform. Install required libraries according to imports in each script (typically `pandas`, `numpy`, `scikit-learn`, and `pyarrow` for parquet inputs). Scripts expect locally prepared historical play-by-play, games, and/or previous experiment CSV outputs; those large inputs are not included in this source-only package.

From the project directory, for example:

```powershell
py -m pip install pandas numpy scikit-learn pyarrow
py -u nfl_experiment_008_walkforward.py
```

Some experiments require output files from earlier experiments. Check script headers and filenames for exact input expectations.

## Data and evaluation limitations

- Do not treat retrospective closing spreads as guaranteed executable odds.
- Training-set accuracy and returns are not independent validation.
- Historical QB continuity labels do not establish pregame knowledge or injury-free starts.
- Keep raw NFL play-by-play datasets and personal/local files outside Git.

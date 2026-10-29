"""Experiment 003 v4: freeze all-game predictions before QB1 cohort analysis.

Run: py -u e3_auto_qb1_v4.py
Requires e3_auto_qb1_v3.py, e3_auto_qb1_v2.py, e3_qb1_fixed.py,
         nfl_walters_experiment_002.py in the same directory.

IMPORTANT: Provisional QB1 continuity uses POSTGAME data. No pregame betting
ROI is inferred from that subset. Historical closing lines are not executable
odds. Previously inspected 2023-25 test years are not pristine holdout data.
"""
from pathlib import Path
import importlib
import numpy as np
import pandas as pd
import e3_auto_qb1_v3 as v3
import nfl_walters_experiment_002 as old

ROOT = Path(__file__).resolve().parent
PRED = ROOT / 'nfl_experiment_003_v4_frozen_predictions.csv'
LABELED = ROOT / 'nfl_experiment_003_v4_labeled_predictions.csv'
SUMMARY = ROOT / 'nfl_experiment_003_v4_cohort_summary.csv'
COUNTS = ROOT / 'nfl_experiment_003_v4_cohort_counts.csv'
MODELS = ('independent', 'player', 'market')


def frozen_predictions():
    """Run the original e3 regression only ONCE, on the full game set."""
    e3 = importlib.import_module('e3_qb1_fixed') if (ROOT/'e3_qb1_fixed.py').exists() else importlib.import_module('nfl_walters_experiment_003')
    e3.filter_qb1_games = lambda games: games.copy()
    print('\nFitting all-game models ONCE (2015-2020 training); no cohort refitting.', flush=True)
    e3.run()
    src = ROOT/'nfl_experiment_003_predictions.csv'
    if not src.exists():
        raise FileNotFoundError(f'Model did not create {src.name}')
    x = pd.read_csv(src, low_memory=False)
    needed = {'season','week','home','away','actual_home_margin','market_home_margin','split'}
    for m in MODELS:
        needed |= {f'{m}_pick',f'{m}_result'}
    missing = needed-set(x.columns)
    if missing:
        raise ValueError(f'Frozen predictions missing: {sorted(missing)}')
    if x.duplicated(['season','week','home','away']).any():
        raise ValueError('Duplicate season/week/home/away in frozen predictions')
    x.to_csv(PRED,index=False)
    return x


def join_labels(pred, games):
    """Join feature rows to schedule/QB1 labels by their shared game identity."""
    keys = ['season', 'week', 'home', 'away']
    labels = games[['season', 'week', 'home_team', 'away_team',
                    'provisional_qb1_game', 'verified_qb1_game',
                    'home_qb_status', 'away_qb_status']].copy()
    labels = labels.rename(columns={'home_team': 'home', 'away_team': 'away'})
    pred = pred.copy()
    for frame in (pred, labels):
        frame['season'] = pd.to_numeric(frame['season'], errors='raise').astype(int)
        frame['week'] = pd.to_numeric(frame['week'], errors='raise').astype(int)
        for col in ('home', 'away'):
            frame[col] = frame[col].astype(str).str.strip().str.upper()
    if labels.duplicated(keys).any():
        raise ValueError('Duplicate game identities in QB1 labels')
    if pred.duplicated(keys).any():
        raise ValueError('Duplicate game identities in frozen predictions')
    x = pred.merge(labels, on=keys, how='left', validate='one_to_one', indicator=True)
    if (x['_merge'] != 'both').any():
        missing = x.loc[x['_merge'] != 'both', keys].head(10).to_dict('records')
        raise ValueError(f"{(x['_merge'] != 'both').sum()} predictions lack QB1 labels; examples: {missing}")
    x = x.drop(columns='_merge')
    for c in ('provisional_qb1_game', 'verified_qb1_game'):
        x[c] = x[c].fillna(False).astype(bool)
    x.to_csv(LABELED, index=False)
    return x


def metric(df, cohort, split, model):
    picks=df[f'{model}_pick'].fillna('PASS').astype(str).str.upper()
    bets=df.loc[picks.ne('PASS')]
    results=bets[f'{model}_result'].fillna('UNKNOWN').astype(str).str.upper()
    if not results.isin(['WIN','LOSS','PUSH']).all():
        raise ValueError(f'Invalid graded results for {cohort} / {split} / {model}')
    w=int(results.eq('WIN').sum()); l=int(results.eq('LOSS').sum()); p=int(results.eq('PUSH').sum())
    profit=w*(100/110)-l
    pred_col={'independent':'independent_margin','player':'independent_player_margin','market':'market_margin'}[model]
    mae=float((pd.to_numeric(df.actual_home_margin)-pd.to_numeric(df[pred_col])).abs().mean()) if len(df) else np.nan
    return dict(cohort=cohort,split=split,model=model,games=len(df),bets=len(bets),wins=w,losses=l,pushes=p,
                win_pct_ex_push=round(100*w/(w+l),2) if w+l else np.nan,
                units_profit=round(profit,2),roi_pct=round(100*profit/len(bets),2) if len(bets) else np.nan,
                margin_mae=round(mae,3) if np.isfinite(mae) else np.nan)


def compare(x):
    cohorts={'all_games':pd.Series(True,index=x.index),
             'provisional_postgame':x.provisional_qb1_game,
             'not_provisional_postgame':~x.provisional_qb1_game,
             'verified_postgame':x.verified_qb1_game}
    rows=[]; count_rows=[]
    for split in ('train','validation','test'):
        sx=x.loc[x.split.eq(split)]
        for name, mask in cohorts.items():
            group=sx.loc[mask.reindex(sx.index,fill_value=False)]
            count_rows.append(dict(split=split,cohort=name,games=len(group)))
            if group.empty: continue
            for model in MODELS:
                rows.append(metric(group,name,split,model))
    out=pd.DataFrame(rows)
    out.to_csv(SUMMARY,index=False)
    pd.DataFrame(count_rows).to_csv(COUNTS,index=False)
    print('\nFROZEN PREDICTIONS: same fitted models, picks and grading in every cohort\n',flush=True)
    print(out.to_string(index=False),flush=True)
    print('\nCohort counts:\n'+pd.DataFrame(count_rows).to_string(index=False),flush=True)
    # Partition consistency: every all-game bet must be in exactly one of the two disjoint groups.
    for split in ('train','validation','test'):
        for model in MODELS:
            z=out[(out.split==split)&(out.model==model)].set_index('cohort')
            for col in ('games','bets','wins','losses','pushes'):
                total=int(z.loc['all_games',col]); parts=int(z.loc['provisional_postgame',col])+int(z.loc['not_provisional_postgame',col])
                if total!=parts: raise AssertionError(f'Partition mismatch: {split} {model} {col}: {total}!={parts}')
    print('\nPASS: all-games totals equal provisional + nonprovisional totals for every split/model.',flush=True)
    return out


def main():
    games=old.load_games()
    audit=v3.build_audit(games)
    labeled_games=v3.label_games(games,audit)
    pred=frozen_predictions()
    labeled=join_labels(pred,labeled_games)
    compare(labeled)
    print('\nFILES:',PRED.name,LABELED.name,SUMMARY.name,COUNTS.name,flush=True)
    print('CAUTION: QB1 continuity classification is retrospective, NOT a pregame eligibility rule.',flush=True)
    print('No independently verified pregame QB1 starts / injury exits are inferred from PBP.',flush=True)
    print('Market is historical CLOSING spread; not timestamped, executable odds.',flush=True)
    print('Player model is identical to independent when no verified player inputs are provided.',flush=True)
    print('2023-2025 already examined: not an untouched test set.',flush=True)

if __name__=='__main__':
    main()

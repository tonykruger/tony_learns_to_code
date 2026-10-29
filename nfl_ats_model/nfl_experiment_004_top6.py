"""Experiment 004: rank the top six weekly NFL spread picks from FROZEN Experiment 003 predictions.

Run: py -u nfl_experiment_004_top6.py
Input: nfl_experiment_003_v4_frozen_predictions.csv (or nfl_experiment_003_predictions.csv).
Does not fit models, fetch data, or select games based on outcomes/postgame QB information.
Historical closing spreads are NOT verified executable pregame odds.
The 2023-25 period has already been inspected and is NOT a pristine holdout.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
SOURCES=[ROOT/'nfl_experiment_003_v4_frozen_predictions.csv',ROOT/'nfl_experiment_003_predictions.csv']
MODEL='independent' # player currently duplicates independent; market makes zero bets
TOP_N=6
PRICE_WIN=100/110 # illustrative -110; not verified odds
MIN_EDGE=1.5
FEATURES=['home_off_epa','away_def_epa','away_off_epa','home_def_epa',
          'home_off_success','away_def_success','away_off_success','home_def_success']


def summarize(frame,group):
    n=len(frame)
    w=int(frame.result.eq('WIN').sum()); l=int(frame.result.eq('LOSS').sum()); p=int(frame.result.eq('PUSH').sum())
    profit=w*PRICE_WIN-l
    return dict(group=group,games=n,wins=w,losses=l,pushes=p,
                win_pct=round(100*w/(w+l),2) if w+l else np.nan,
                units=round(profit,2),roi_pct=round(100*profit/n,2) if n else np.nan,
                avg_abs_edge=round(frame.abs_edge.mean(),3) if n else np.nan,
                avg_abs_margin_error=round(frame.abs_margin_error.mean(),3) if n else np.nan)


def main():
    src=next((p for p in SOURCES if p.exists()),None)
    if src is None:
        raise FileNotFoundError('Place nfl_experiment_003_v4_frozen_predictions.csv beside this script first.')
    x=pd.read_csv(src,low_memory=False)
    required={'season','week','home','away','market_home_margin','actual_home_margin',
              'independent_margin','independent_pick','independent_result','split'}
    missing=required-set(x.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    x=x.copy()
    for c in ('season','week','market_home_margin','actual_home_margin','independent_margin'):
        x[c]=pd.to_numeric(x[c],errors='raise')
    x['edge']=x.independent_margin-x.market_home_margin
    x['abs_edge']=x.edge.abs()
    x['pick']=x.independent_pick.fillna('PASS').astype(str).str.upper()
    x['result']=x.independent_result.fillna('PASS').astype(str).str.upper()
    x['abs_margin_error']=(x.independent_margin-x.actual_home_margin).abs()
    x['picked_team']=np.where(x.pick.eq('HOME'),x.home,np.where(x.pick.eq('AWAY'),x.away,''))
    x['side_cover_margin']=np.where(x.pick.eq('HOME'),x.actual_home_margin-x.market_home_margin,
                              np.where(x.pick.eq('AWAY'),x.market_home_margin-x.actual_home_margin,np.nan))
    x['split']=x.split.astype(str).str.lower()
    if x.duplicated(['season','week','home','away']).any(): raise ValueError('Duplicate games in frozen file')
    if not x.pick.isin(['HOME','AWAY','PASS']).all(): raise ValueError('Unknown pick value')
    bets=x[x.pick.ne('PASS')].copy()
    if not bets.result.isin(['WIN','LOSS','PUSH']).all(): raise ValueError('Invalid bet grading')
    if (bets.abs_edge+1e-8<MIN_EDGE).any():
        print('NOTE: Some existing picks have edge below 1.5 points; retaining original picks.')
    # Deterministic ranking: no result, final score, or postgame QB feature is involved.
    bets=bets.sort_values(['season','week','abs_edge','home','away'],ascending=[True,True,False,True,True]).copy()
    bets['weekly_rank']=bets.groupby(['season','week']).cumcount()+1
    bets['selection']=np.where(bets.weekly_rank<=TOP_N,'top6','remaining')
    bets['profit_units']=np.select([bets.result.eq('WIN'),bets.result.eq('LOSS')],[PRICE_WIN,-1.],default=0.)
    bets['actual_cover']=bets.result.eq('WIN')
    cols=['season','week','date','split','weekly_rank','selection','home','away','picked_team','pick',
          'independent_margin','market_home_margin','edge','abs_edge','actual_home_margin',
          'side_cover_margin','result','profit_units','abs_margin_error']
    cols=[c for c in cols if c in bets.columns]+[c for c in FEATURES if c in bets.columns]
    bets[cols].to_csv(ROOT/'nfl_experiment_004_all_ranked_bets.csv',index=False)
    top=bets[bets.selection.eq('top6')].copy()
    top[cols].to_csv(ROOT/'nfl_experiment_004_weekly_top6.csv',index=False)
    summary=[]
    for split in ['train','validation','test']:
        z=bets[bets.split.eq(split)]
        for group,frame in [('top6',z[z.selection.eq('top6')]),('remaining',z[z.selection.eq('remaining')]),('all_bets',z)]:
            summary.append(dict(split=split,**summarize(frame,group)))
    result=pd.DataFrame(summary)
    result.to_csv(ROOT/'nfl_experiment_004_summary.csv',index=False)
    weekly=[]
    for (season,week),group in bets.groupby(['season','week']):
        for label,sub in [('top6',group[group.selection.eq('top6')]),('remaining',group[group.selection.eq('remaining')])]:
            weekly.append(dict(season=int(season),week=int(week),split=str(group.split.iloc[0]),**summarize(sub,label)))
    pd.DataFrame(weekly).to_csv(ROOT/'nfl_experiment_004_weekly_results.csv',index=False)
    rank_rows=[]
    for split in ['train','validation','test']:
        z=top[top.split.eq(split)]
        for rank in range(1,TOP_N+1):
            rank_rows.append(dict(split=split,rank=rank,**summarize(z[z.weekly_rank.eq(rank)],'rank')))
    pd.DataFrame(rank_rows).to_csv(ROOT/'nfl_experiment_004_rank_reliability.csv',index=False)
    # Investigate feature patterns without making causal claims or tuning a selector on test outcomes.
    diagnostic=[]
    for split in ['train','validation','test']:
        z=top[top.split.eq(split)]
        for feature in FEATURES:
            if feature not in z: continue
            a=pd.to_numeric(z[feature],errors='coerce')
            for outcome in ['WIN','LOSS']:
                vals=a[z.result.eq(outcome)].dropna()
                diagnostic.append(dict(split=split,feature=feature,outcome=outcome,
                                       games=len(vals),mean=round(vals.mean(),5) if len(vals) else np.nan,
                                       median=round(vals.median(),5) if len(vals) else np.nan))
    pd.DataFrame(diagnostic).to_csv(ROOT/'nfl_experiment_004_feature_diagnostics.csv',index=False)
    print('\nEXPERIMENT 004: TOP SIX BY ABSOLUTE MODEL/MARKET DISAGREEMENT',flush=True)
    print(f'Input: {src.name}; no refitting; same picks and grading.',flush=True)
    print(result.to_string(index=False),flush=True)
    print('\nTOP SIX BY RANK, TEST YEARS (2023-2025):',flush=True)
    print(pd.DataFrame(rank_rows).query("split=='test'").to_string(index=False),flush=True)
    print('\nEXAMPLE: LATEST HISTORICAL WEEK',flush=True)
    last=top.sort_values(['season','week']).iloc[-1][['season','week']]
    sample=top[(top.season==last.season)&(top.week==last.week)]
    print(sample[['weekly_rank','home','away','picked_team','edge','result']].to_string(index=False),flush=True)
    print('\nFiles: nfl_experiment_004_weekly_top6.csv, nfl_experiment_004_all_ranked_bets.csv,',flush=True)
    print('nfl_experiment_004_summary.csv, nfl_experiment_004_weekly_results.csv,',flush=True)
    print('nfl_experiment_004_rank_reliability.csv, nfl_experiment_004_feature_diagnostics.csv',flush=True)
    print('\nCAUTIONS: Rank is model disagreement, NOT calibrated certainty. Closing lines are retrospective.',flush=True)
    print('2023-25 is previously inspected, not pristine. -110 grading is hypothetical; no verified bettable prices.',flush=True)
    print('Features explain associations, not causes. No postgame QB filter used for selection.',flush=True)

if __name__=='__main__':
    main()

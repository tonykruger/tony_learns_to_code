"""Experiment 005: rank up to six weekly picks by learned cover reliability.
Run: py -u nfl_experiment_005_confidence.py
Uses Experiment 004 individual bet CSV; does not refit game-margin model.
2015-20 trains reliability layer; 2021-22 chooses a prespecified feature set
by log loss; 2023-25 is descriptive, previously inspected (not pristine).
All scores use only listed pregame-derived columns; no outcomes in features.
Historical closing lines are NOT timestamp-verified executable odds.
"""
from pathlib import Path
import numpy as np
import pandas as pd
try:
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, brier_score_loss
except ImportError as exc:
    raise SystemExit('Install dependencies: py -m pip install pandas numpy scikit-learn') from exc

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'nfl_experiment_004_all_ranked_bets.csv'
if not SOURCE.exists():
    candidates=[ROOT/'Pasted text(5).txt']
    SOURCE=next((p for p in candidates if p.exists()),SOURCE)
if not SOURCE.exists():
    raise SystemExit('Missing nfl_experiment_004_all_ranked_bets.csv beside this script.')
df=pd.read_csv(SOURCE)
required={'season','week','split','home','away','pick','result','profit_units','edge','abs_edge',
          'market_home_margin','independent_margin','home_off_epa','away_off_epa',
          'home_def_epa','away_def_epa','home_off_success','away_off_success',
          'home_def_success','away_def_success'}
missing=required-set(df.columns)
if missing: raise SystemExit(f'Missing columns: {sorted(missing)}')
if df.duplicated(['season','week','home','away']).any():
    raise SystemExit('Duplicate games: expected one pick per season/week/matchup')
df=df[df.result.isin(['WIN','LOSS','PUSH'])].copy()
df['is_win']=df.result.eq('WIN').astype(int)
# No postgame variables are used in any confidence features.
sign=np.where(df['pick'].eq('HOME'),1.0,-1.0)
df['pick_home']=sign==1
df['selected_off_epa']=np.where(df.pick_home,df.home_off_epa,df.away_off_epa)
df['opponent_off_epa']=np.where(df.pick_home,df.away_off_epa,df.home_off_epa)
df['selected_def_epa']=np.where(df.pick_home,df.home_def_epa,df.away_def_epa)
df['opponent_def_epa']=np.where(df.pick_home,df.away_def_epa,df.home_def_epa)
df['selected_off_success']=np.where(df.pick_home,df.home_off_success,df.away_off_success)
df['opponent_off_success']=np.where(df.pick_home,df.away_off_success,df.home_off_success)
df['selected_def_success']=np.where(df.pick_home,df.home_def_success,df.away_def_success)
df['opponent_def_success']=np.where(df.pick_home,df.away_def_success,df.home_def_success)
df['market_pick_margin']=sign*df.market_home_margin
df['model_pick_margin']=sign*df.independent_margin
df['market_favorite']=df.market_pick_margin>0
df['off_matchup_epa']=df.selected_off_epa-df.opponent_def_epa
df['def_matchup_epa']=df.opponent_off_epa-df.selected_def_epa
df['off_matchup_success']=df.selected_off_success-df.opponent_def_success
df['def_matchup_success']=df.opponent_off_success-df.selected_def_success
df['epa_agreement']=((df.off_matchup_epa>0).astype(int)+(df.def_matchup_epa<0).astype(int))
df['success_agreement']=((df.off_matchup_success>0).astype(int)+(df.def_matchup_success<0).astype(int))
df['week_number']=pd.to_numeric(df.week,errors='coerce')
# Prespecified alternatives, limited to avoid a large search for a lucky strategy.
CANDIDATES={
 'edge_only':['abs_edge'],
 'market_context':['abs_edge','market_pick_margin','market_favorite','week_number'],
 'matchup_context':['abs_edge','market_pick_margin','market_favorite','week_number',
                    'off_matchup_epa','def_matchup_epa','off_matchup_success',
                    'def_matchup_success','epa_agreement','success_agreement']
}
train=df[df.split.eq('train') & df.result.ne('PUSH')]
valid=df[df.split.eq('validation') & df.result.ne('PUSH')]
if len(train)<100 or len(valid)<30: raise SystemExit('Insufficient train/validation samples')
models={}; scores=[]
for name,features in CANDIDATES.items():
    m=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),
                    LogisticRegression(C=0.2,max_iter=2000,random_state=17))
    m.fit(train[features].astype(float),train.is_win)
    p=np.clip(m.predict_proba(valid[features].astype(float))[:,1],1e-5,1-1e-5)
    scores.append({'candidate':name,'validation_log_loss':log_loss(valid.is_win,p),
                   'validation_brier':brier_score_loss(valid.is_win,p),
                   'validation_games':len(valid)})
    models[name]=m
score_df=pd.DataFrame(scores).sort_values(['validation_log_loss','candidate'])
chosen=score_df.iloc[0]['candidate']
model=models[chosen]; features=CANDIDATES[chosen]
# Apply chosen model to ALL rows. Selection uses only prediction-time columns.
df['confidence_score']=model.predict_proba(df[features].astype(float))[:,1]
df['confidence_rank']=df.groupby(['season','week'])['confidence_score'].rank(method='first',ascending=False).astype(int)
df['confidence_group']=np.where(df.confidence_rank<=6,'top6','remaining')
df['edge_rank']=df.groupby(['season','week'])['abs_edge'].rank(method='first',ascending=False).astype(int)
df['edge_group']=np.where(df.edge_rank<=6,'top6','remaining')

def summarize(rows,method,group):
    wins=int((rows.result=='WIN').sum()); losses=int((rows.result=='LOSS').sum()); pushes=int((rows.result=='PUSH').sum())
    n=len(rows); units=float(rows.profit_units.sum()); decisions=wins+losses
    return {'method':method,'group':group,'games':n,'wins':wins,'losses':losses,
            'pushes':pushes,'win_pct':round(100*wins/decisions,2) if decisions else np.nan,
            'units':round(units,2),'roi_pct':round(100*units/n,2) if n else np.nan,
            'avg_abs_edge':round(float(rows.abs_edge.mean()),3) if n else np.nan,
            'avg_abs_margin_error':round(float(rows.abs_margin_error.mean()),3) if n else np.nan}

summaries=[]
for split,part in df.groupby('split',sort=False):
    for method,col in [('edge_baseline','edge_group'),('learned_confidence','confidence_group')]:
        for group in ['top6','remaining']:
            summaries.append({'split':split,**summarize(part[part[col].eq(group)],method,group)})
        summaries.append({'split':split,**summarize(part,method,'all')})
summary=pd.DataFrame(summaries)
weekly=df[df.confidence_group.eq('top6')].sort_values(['season','week','confidence_rank'])
weekly.to_csv(ROOT/'nfl_experiment_005_weekly_top6.csv',index=False)
df.to_csv(ROOT/'nfl_experiment_005_all_scored.csv',index=False)
summary.to_csv(ROOT/'nfl_experiment_005_comparison.csv',index=False)
score_df.to_csv(ROOT/'nfl_experiment_005_validation_candidates.csv',index=False)
rankrows=[]
for split,part in weekly.groupby('split'):
    for rank,sub in part.groupby('confidence_rank'):
        rankrows.append({'split':split,'rank':rank,**summarize(sub,'learned_confidence','rank')})
pd.DataFrame(rankrows).to_csv(ROOT/'nfl_experiment_005_rank_reliability.csv',index=False)
print('\nEXPERIMENT 005: CONFIDENCE RANKING (NOT CALIBRATED WIN PROBABILITY)')
print('Input:',SOURCE.name,'| rows:',len(df))
print('\nPrespecified candidates (chosen ONLY on 2021-22 log loss):')
print(score_df.to_string(index=False))
print('\nSelected:',chosen,'| features:',', '.join(features))
print('\nTop-six comparison:')
print(summary[summary.group.eq('top6')].to_string(index=False))
print('\n2023-25 confidence-rank performance:')
print(pd.DataFrame(rankrows).query("split=='test'").to_string(index=False))
print('\nOutput files: nfl_experiment_005_weekly_top6.csv, nfl_experiment_005_all_scored.csv,')
print(' nfl_experiment_005_comparison.csv, nfl_experiment_005_validation_candidates.csv, nfl_experiment_005_rank_reliability.csv')
print('\nCAUTIONS: This is a reliability research layer trained on IN-SAMPLE 2015-20 base-model picks.')
print('Predicted confidence scores are NOT validated/calibrated probabilities.')
print('The 2023-25 data was previously inspected; results are exploratory, NOT a fresh holdout.')
print('Only existing bet rows are ranked; weeks with fewer than six eligible bets remain shorter.')
print('Historical closing spreads are not timestamped available prices; -110 returns are hypothetical.')

"""Experiment 007: diagnose unscaled Ridge and compare prespecified alternatives.
Run: py -u nfl_experiment_007_regression.py
Requires nfl_walters_experiment_002.py plus its original nflverse source data.
2015-20 train, 2021-22 validation, 2023-25 descriptive only (previously inspected).
Closing spreads are retrospective, not verified executable odds.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import nfl_walters_experiment_002 as old

ROOT=Path(__file__).resolve().parent
features=old.BASE_COLS
print('EXPERIMENT 007 — REGRESSION SCALE AUDIT')
print('Original Ridge alpha: 100, no standardization. Home field constant:',old.HFA)
print('Loading ALL games, including those that never qualified as bets...')
games=old.load_games()
eff=old.load_efficiency()
data=old.make_features(games,eff,{})
assert not data.empty
assert not data.duplicated(['season','week','home','away']).any(), 'Duplicate games'
train=data[data.season<=2020].copy()
valid=data[data.season.between(2021,2022)].copy()
test=data[data.season.between(2023,2025)].copy()
if min(len(train),len(valid),len(test))==0: raise SystemExit('Missing a train, validation or test period')
X=train[features].astype(float)
y=train.actual_home_margin-old.HFA
print('\nTRAINING FEATURE SCALES')
print(pd.DataFrame({'std':X.std(),'min':X.min(),'max':X.max()}).round(5).to_string())
models={
 'original_unscaled_alpha100':Ridge(alpha=100.0),
 'unscaled_alpha1':Ridge(alpha=1.0),
 'scaled_alpha100':make_pipeline(StandardScaler(),Ridge(alpha=100.0)),
 'scaled_alpha1000':make_pipeline(StandardScaler(),Ridge(alpha=1000.0)),
 'scaled_alpha10000':make_pipeline(StandardScaler(),Ridge(alpha=10000.0)),
}
rows=[]; pred_frames=[]; coef_rows=[]
for name,model in models.items():
 model.fit(X,y)
 ridge=model[-1] if hasattr(model,'steps') else model
 for f,c in zip(features,ridge.coef_):coef_rows.append({'model':name,'feature':f,'coefficient':float(c),'coefficient_scale':'standardized' if name.startswith('scaled') else 'raw'})
 for split,frame in [('train',train),('validation',valid),('test',test)]:
  z=frame[['season','week','home','away','market_home_margin','actual_home_margin']].copy()
  z['split']=split;z['model']=name
  z['pred_margin']=old.HFA+model.predict(frame[features].astype(float))
  z['abs_error']=(z.pred_margin-z.actual_home_margin).abs()
  z['market_abs_error']=(z.market_home_margin-z.actual_home_margin).abs()
  z['edge']=z.pred_margin-z.market_home_margin
  z['abs_edge']=z.edge.abs()
  z['pick']=np.where(z.edge>=1.5,'HOME',np.where(z.edge<=-1.5,'AWAY','PASS'))
  z['is_underdog']=np.where(z.pick=='HOME',z.market_home_margin<0,np.where(z.pick=='AWAY',z.market_home_margin>0,False))
  z['cover_margin']=np.where(z.pick=='HOME',z.actual_home_margin-z.market_home_margin,z.market_home_margin-z.actual_home_margin)
  z['result']=np.where(z.pick=='PASS','PASS',np.where(z.cover_margin>0,'WIN',np.where(z.cover_margin<0,'LOSS','PUSH')))
  z['weekly_rank']=z[z.pick!='PASS'].groupby(['season','week']).abs_edge.rank(method='first',ascending=False).reindex(z.index)
  z['top6']=z.weekly_rank.le(6).fillna(False)
  for group,subset in [('all_games',z),('eligible_bets',z[z.pick!='PASS']),('top6',z[z.top6])]:
   w=int((subset.result=='WIN').sum());l=int((subset.result=='LOSS').sum());p=int((subset.result=='PUSH').sum());n=len(subset)
   units=w*(100/110)-l
   rows.append({'model':name,'split':split,'group':group,'games':n,'wins':w,'losses':l,'pushes':p,
    'win_pct':round(100*w/(w+l),2) if w+l else np.nan,'roi_pct':round(100*units/n,2) if n else np.nan,
    'mae':round(float(subset.abs_error.mean()),3),'market_mae':round(float(subset.market_abs_error.mean()),3),
    'mean_abs_pred':round(float(subset.pred_margin.abs().mean()),3),'std_pred':round(float(subset.pred_margin.std()),3),
    'underdog_pct':round(100*float(subset.is_underdog.mean()),2) if group!='all_games' else np.nan})
  pred_frames.append(z)
summary=pd.DataFrame(rows)
coefs=pd.DataFrame(coef_rows)
preds=pd.concat(pred_frames,ignore_index=True)
summary.to_csv(ROOT/'nfl_experiment_007_summary.csv',index=False)
coefs.to_csv(ROOT/'nfl_experiment_007_coefficients.csv',index=False)
preds.to_csv(ROOT/'nfl_experiment_007_all_games_predictions.csv',index=False)
print('\nALL GAME MARGIN DIAGNOSTICS')
print(summary[summary.group=='all_games'][['split','model','games','mae','market_mae','mean_abs_pred','std_pred']].to_string(index=False))
print('\nTOP SIX BETS (HYPOTHETICAL -110)')
print(summary[summary.group=='top6'][['split','model','games','wins','losses','pushes','win_pct','roi_pct','underdog_pct']].to_string(index=False))
print('\nCOEFFICIENTS')
print(coefs.pivot(index='feature',columns='model',values='coefficient').round(4).to_string())
print('\nInterpretation: compare 2021-22 validation; 2023-25 is previously inspected and NOT a fresh holdout.')
print('Do not select a winner using 2023-25 performance. This is a diagnostic, not an executable betting system.')
print('Files: nfl_experiment_007_summary.csv, nfl_experiment_007_coefficients.csv, nfl_experiment_007_all_games_predictions.csv')

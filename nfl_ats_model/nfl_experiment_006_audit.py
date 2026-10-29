from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parent
p=ROOT/'nfl_experiment_005_all_scored.csv'
if not p.exists(): p=ROOT/'nfl_experiment_004_all_ranked_bets.csv'
d=pd.read_csv(p)
for c in ['independent_margin','market_home_margin','actual_home_margin','edge','abs_edge','profit_units']:
 d[c]=pd.to_numeric(d[c],errors='coerce')
d['picked_underdog']=np.where(d['pick'].eq('HOME'),d.market_home_margin<0,d.market_home_margin>0)
d['picked_favorite']=np.where(d['pick'].eq('HOME'),d.market_home_margin>0,d.market_home_margin<0)
d['market_abs_margin']=d.market_home_margin.abs()
d['model_abs_margin']=d.independent_margin.abs()
d['model_error']=d.independent_margin-d.actual_home_margin
d['market_error']=d.market_home_margin-d.actual_home_margin
d['market_fav_bin']=pd.cut(d.market_abs_margin,[-.001,2.5,6.5,10.5,np.inf],labels=['0-2.5','3-6.5','7-10.5','11+'])
d['edge_bin']=pd.cut(d.abs_edge,[-.001,2.5,5,8,np.inf],labels=['<=2.5','2.5-5','5-8','8+'])
d['side']=np.select([d.picked_underdog,d.picked_favorite],['underdog','favorite'],default='pickem')
d['market_favorite_home']=np.where(d.market_home_margin>0,'home',np.where(d.market_home_margin<0,'away','pickem'))
d['model_favorite_home']=np.where(d.independent_margin>0,'home',np.where(d.independent_margin<0,'away','pickem'))
d['favorite_disagreement']=d.market_favorite_home.ne(d.model_favorite_home)
def summary(g):
 w=(g.result=='WIN').sum(); l=(g.result=='LOSS').sum(); p=(g.result=='PUSH').sum(); n=len(g)
 return pd.Series({'games':n,'wins':w,'losses':l,'pushes':p,'win_pct_ex_push':round(100*w/(w+l),2) if w+l else np.nan,'roi_pct':round(100*g.profit_units.sum()/n,2) if n else np.nan,'mean_model_margin':round(g.independent_margin.mean(),3),'mean_market_margin':round(g.market_home_margin.mean(),3),'mean_actual_margin':round(g.actual_home_margin.mean(),3),'model_mae':round(g.model_error.abs().mean(),3),'market_mae':round(g.market_error.abs().mean(),3),'mean_model_abs_margin':round(g.model_abs_margin.mean(),3),'mean_market_abs_margin':round(g.market_abs_margin.mean(),3),'mean_abs_edge':round(g.abs_edge.mean(),3),'underdog_pct':round(100*g.picked_underdog.mean(),2),'opposite_favorite_pct':round(100*g.favorite_disagreement.mean(),2),'model_bias':round(g.model_error.mean(),3),'market_bias':round(g.market_error.mean(),3)})
parts=[]
for name,cols in [('overall',[]),('by_split',['split']),('by_season',['season']),('by_split_side',['split','side']),('by_split_market_favorite_size',['split','market_fav_bin']),('by_split_edge_size',['split','edge_bin']),('by_split_favorite_disagreement',['split','favorite_disagreement']),('by_split_pick',['split','pick'])]:
 if cols:
  for keys,g in d.groupby(cols,observed=True,dropna=False):
   if not isinstance(keys,tuple):keys=(keys,)
   row={'analysis':name,**dict(zip(cols,map(str,keys))),**summary(g).to_dict()};parts.append(row)
 else:parts.append({'analysis':name,**summary(d).to_dict()})
out=pd.DataFrame(parts)
out.to_csv(ROOT/'nfl_experiment_006_audit_summary.csv',index=False)
d.to_csv(ROOT/'nfl_experiment_006_audit_games.csv',index=False)
print('EXPERIMENT 006: FROZEN MODEL DIAGNOSTIC — NO REFITTING')
print('Source:',p.name,'| rows:',len(d))
for name in ['by_split','by_split_side','by_split_market_favorite_size','by_split_favorite_disagreement']:
 print('\n'+name.upper())
 cols={'by_split':['split'],'by_split_side':['split','side'],'by_split_market_favorite_size':['split','market_fav_bin'],'by_split_favorite_disagreement':['split','favorite_disagreement']}[name]
 print(out[out.analysis==name][cols+['games','win_pct_ex_push','roi_pct','model_mae','market_mae','mean_model_abs_margin','mean_market_abs_margin','underdog_pct']].to_string(index=False))
print('\nFiles: nfl_experiment_006_audit_summary.csv, nfl_experiment_006_audit_games.csv')
print('CAUTION: Only previously selected bets, not all NFL games; retrospective closing spreads; 2023-25 previously inspected.')

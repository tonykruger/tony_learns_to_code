"""Experiment 008: opponent adjustment, recency, season walk-forward audit.
Run: py -u nfl_experiment_008_walkforward.py
Requires nfl_walters_experiment_002.py, nfl_games_source.csv and 2015-2025 PBP parquet.
No market spread or game outcome is used as a model feature. Closing lines are
retrospective proxies, not verified executable odds. Test period is NOT pristine.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import nfl_walters_experiment_002 as old

ROOT = Path(__file__).resolve().parent
HALF_LIFE = 4.0
PRIOR_PLAYS = 120.0
MIN_HISTORY_SEASONS = 2
CANDIDATES = ('baseline','recent','opponent','recent_opponent')
FEATURES = old.BASE_COLS


def make_dataset(games, eff):
    # One record per team-week-side. Every feature at week W uses weeks < W.
    weekly = eff.copy()
    weekly['season'] = weekly.season.astype(int)
    weekly['week'] = weekly.week.astype(int)
    schedule = games[['season','week','home_team','away_team']].drop_duplicates()
    opp_rows = pd.concat([
        schedule.rename(columns={'home_team':'team','away_team':'opponent'}),
        schedule.rename(columns={'away_team':'team','home_team':'opponent'})
    ], ignore_index=True)
    weekly = weekly.merge(opp_rows, on=['season','week','team'], how='left', validate='many_to_one')
    if weekly.opponent.isna().any():
        print('WARNING: some efficiency rows lack a matched regular-season opponent; excluded from opponent adjustment.')
    results=[]
    for season, season_games in games.groupby('season',sort=True):
        season_eff = weekly[weekly.season==season].copy()
        for week, group in season_games.groupby('week',sort=True):
            if week < old.MIN_WEEK: continue
            prior=season_eff[season_eff.week<week].copy()
            if prior.empty: continue
            # League mean, by side and metric, from completed games only.
            league={}
            for side in ('off','def'):
                z=prior[prior.side==side]
                for metric in ('epa','success'):
                    league[(side,metric)]=float(np.average(z[metric],weights=z.n)) if len(z) else (0.0 if metric=='epa' else .45)
            # Opponent strengths are estimated from OTHER completed games, using
            # the same strictly-prior-to-target-week data snapshot.
            strengths={}
            for side in ('off','def'):
                z=prior[prior.side==side]
                for team, h in z.groupby('team'):
                    for metric in ('epa','success'):
                        mu=league[(side,metric)]
                        strengths[(team,side,metric)]=float((np.dot(h[metric],h.n)+PRIOR_PLAYS*mu)/(h.n.sum()+PRIOR_PLAYS))
            teams=set(group.home_team)|set(group.away_team)
            ratings={}
            for team in teams:
                for side in ('off','def'):
                    h=prior[(prior.team==team)&(prior.side==side)].copy()
                    for metric in ('epa','success'):
                        mu=league[(side,metric)]
                        opp_side='def' if side=='off' else 'off'
                        for candidate in CANDIDATES:
                            if h.empty:
                                val=mu
                            else:
                                recency = candidate in ('recent','recent_opponent')
                                adjust = candidate in ('opponent','recent_opponent')
                                w=h.n.to_numpy(dtype=float)
                                if recency:
                                    w=w*np.exp(-np.log(2)*(week-h.week.to_numpy(dtype=float)-1)/HALF_LIFE)
                                raw=h[metric].to_numpy(dtype=float)
                                if adjust:
                                    # Opponent quality from snapshot available before target game.
                                    # Shrunk towards league mean; not a full iterative SRS model.
                                    opp=np.array([strengths.get((o,opp_side,metric),mu) for o in h.opponent])
                                    raw=raw-(opp-mu)
                                val=float((np.dot(raw,w)+PRIOR_PLAYS*mu)/(w.sum()+PRIOR_PLAYS))
                            ratings[(team,side,metric,candidate)]=val
            for _,g in group.iterrows():
                base={'season':int(season),'week':int(week),'home':g.home_team,'away':g.away_team,
                      'actual_home_margin':float(g.home_score-g.away_score),
                      'market_home_margin':float(g.market_home_margin)}
                for candidate in CANDIDATES:
                    row=dict(base);row['candidate']=candidate
                    for side in ('off','def'):
                        for metric in ('epa','success'):
                            row[f'home_{side}_{metric}']=ratings[(g.home_team,side,metric,candidate)]
                            row[f'away_{side}_{metric}']=ratings[(g.away_team,side,metric,candidate)]
                    results.append(row)
        print(f'Built pregame features for {season}',flush=True)
    return pd.DataFrame(results)


def score(z):
    z=z.copy()
    z['abs_error']=(z.pred_margin-z.actual_home_margin).abs()
    z['market_abs_error']=(z.market_home_margin-z.actual_home_margin).abs()
    z['edge']=z.pred_margin-z.market_home_margin
    z['pick']=np.where(z.edge>=old.EDGE,'HOME',np.where(z.edge<=-old.EDGE,'AWAY','PASS'))
    z['abs_edge']=z.edge.abs()
    z['cover_margin']=np.where(z.pick=='HOME',z.actual_home_margin-z.market_home_margin,z.market_home_margin-z.actual_home_margin)
    z['result']=np.where(z.pick=='PASS','PASS',np.where(z.cover_margin>0,'WIN',np.where(z.cover_margin<0,'LOSS','PUSH')))
    z['is_underdog']=np.where(z.pick=='HOME',z.market_home_margin<0,np.where(z.pick=='AWAY',z.market_home_margin>0,False))
    z['weekly_rank']=z[z.pick!='PASS'].groupby(['season','week']).abs_edge.rank(method='first',ascending=False).reindex(z.index)
    z['top6']=z.weekly_rank.le(6).fillna(False)
    return z


def summarize(z,group):
    if group=='top6': z=z[z.top6]
    elif group=='eligible': z=z[z.pick!='PASS']
    w=int((z.result=='WIN').sum()); l=int((z.result=='LOSS').sum()); p=int((z.result=='PUSH').sum())
    units=w*(100/110)-l
    return {'group':group,'games':len(z),'wins':w,'losses':l,'pushes':p,
            'win_pct':round(100*w/(w+l),2) if w+l else np.nan,
            'roi_pct':round(100*units/len(z),2) if len(z) else np.nan,
            'mae':round(float(z.abs_error.mean()),3) if len(z) else np.nan,
            'market_mae':round(float(z.market_abs_error.mean()),3) if len(z) else np.nan,
            'underdog_pct':round(100*float(z.is_underdog.mean()),2) if len(z) and group!='all_games' else np.nan}


def run():
    print('EXPERIMENT 008 — PRE-GAME OPPONENT/RECENCY FEATURES + SEASON WALK-FORWARD',flush=True)
    print('Fixed: scaled Ridge alpha=100; opponent adjustment uses target-week prior snapshot; recency half-life 4 weeks.')
    games=old.load_games(); eff=old.load_efficiency()
    data=make_dataset(games,eff)
    if data.empty: raise SystemExit('No eligible games')
    if data.duplicated(['season','week','home','away','candidate']).any():
        raise SystemExit('Duplicate game-candidate records')
    if data[FEATURES].isna().any().any(): raise SystemExit('Missing feature values')
    # Fit once before each season; never fit on current/future season outcomes.
    preds=[]; summary=[]
    for candidate in CANDIDATES:
        cd=data[data.candidate==candidate]
        for season in sorted(cd.season.unique()):
            train=cd[cd.season<season]
            if train.season.nunique()<MIN_HISTORY_SEASONS: continue
            hold=cd[cd.season==season].copy()
            model=make_pipeline(StandardScaler(),Ridge(alpha=100.0))
            model.fit(train[FEATURES],train.actual_home_margin-old.HFA)
            hold['pred_margin']=old.HFA+model.predict(hold[FEATURES])
            hold['training_through_season']=int(season-1)
            hold['split']='train_oos' if season<=2020 else ('validation' if season<=2022 else 'test_exploratory')
            preds.append(score(hold))
    if not preds: raise SystemExit('Insufficient historical seasons')
    all_pred=pd.concat(preds,ignore_index=True)
    for (candidate,split), frame in all_pred.groupby(['candidate','split'],sort=True):
        for group in ('all_games','eligible','top6'):
            summary.append({'candidate':candidate,'split':split,**summarize(frame,group)})
    summ=pd.DataFrame(summary)
    all_pred.to_csv(ROOT/'nfl_experiment_008_all_predictions.csv',index=False)
    summ.to_csv(ROOT/'nfl_experiment_008_summary.csv',index=False)
    all_pred[all_pred.top6].to_csv(ROOT/'nfl_experiment_008_weekly_top6.csv',index=False)
    print('\nALL GAMES: lower MAE is better')
    print(summ[summ.group=='all_games'][['candidate','split','games','mae','market_mae']].to_string(index=False))
    print('\nTOP SIX (retrospective closing spread; hypothetical -110)')
    print(summ[summ.group=='top6'][['candidate','split','games','wins','losses','pushes','win_pct','roi_pct','underdog_pct']].to_string(index=False))
    print('\nCAUTIONS: 2023-25 previously inspected; no fresh holdout. Market lines not verified executable.')
    print('Opponent adjustment is an approximate opponent-strength correction, not a complete opponent-adjusted EPA estimator.')
    print('Fixed parameters are exploratory choices, not proven optimal. No in-season retraining.')
    print('Files: nfl_experiment_008_summary.csv, nfl_experiment_008_all_predictions.csv, nfl_experiment_008_weekly_top6.csv')

if __name__=='__main__': run()

"""NFL experiment 001: chronological opponent-adjusted margin ratings.
Run: pip install pandas numpy; python nfl_walters_experiment_001.py
Data: nflverse (CC BY 4.0), https://github.com/nflverse/nflverse-data
Uses historical CLOSING spreads as an exploratory benchmark, not executable 24h odds.
"""
import numpy as np
import pandas as pd
from pathlib import Path

URL = 'https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv'
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'nfl_games_source.csv'
EDGE = 1.5
ODDS = -110
HOME_ADV = 1.5  # fixed illustrative assumption; sensitivity analysis required
RIDGE = 12.0    # fixed regularization, not optimized on test set
DECAY = 0.985   # older games downweighted, not optimized on test set

def load():
    if SOURCE.exists():
        df = pd.read_csv(SOURCE)
    else:
        try:
            df = pd.read_csv(URL)
            df.to_csv(SOURCE, index=False)
        except Exception as exc:
            raise SystemExit(f'Download failed: {exc}\nDownload {URL} manually to {SOURCE}')
    required = ['season','week','game_type','gameday','home_team','away_team','home_score','away_score','spread_line']
    missing = [c for c in required if c not in df]
    if missing: raise SystemExit(f'Missing columns: {missing}')
    df = df[(df.season.between(2015,2025)) & (df.game_type=='REG')].copy()
    df['gameday'] = pd.to_datetime(df.gameday)
    df = df.dropna(subset=['home_score','away_score','spread_line'])
    df = df.sort_values(['season','week','gameday','home_team']).reset_index(drop=True)
    return df

def fit_ratings(past, teams):
    # Every row: margin = home rating - away rating + fixed home advantage.
    # Ridge constrains ratings to zero and solves disconnected early-season samples.
    if past.empty: return dict.fromkeys(teams, 0.0)
    ids = {t:i for i,t in enumerate(teams)}
    X = np.zeros((len(past),len(teams)))
    for i,(h,a) in enumerate(zip(past.home_team,past.away_team)):
        X[i,ids[h]]=1; X[i,ids[a]]=-1
    y = (past.home_score-past.away_score-HOME_ADV).to_numpy(float)
    age = np.arange(len(past)-1,-1,-1)
    w = DECAY ** age
    coef = np.linalg.solve(X.T@(w[:,None]*X)+RIDGE*np.eye(len(teams)),X.T@(w*y))
    coef -= coef.mean()
    return dict(zip(teams,coef))

def run():
    df=load()
    teams=sorted(set(df.home_team)|set(df.away_team))
    predictions=[]; snapshots=[]
    # All games in a week use a single snapshot from strictly earlier weeks.
    for (season,week),games in df.groupby(['season','week'],sort=True):
        prior=df[(df.season<season)|((df.season==season)&(df.week<week))]
        # Train on current season and last season only, avoiding obsolete rosters.
        prior=prior[prior.season>=season-1]
        ratings=fit_ratings(prior,teams)
        for team,rating in ratings.items():
            snapshots.append({'season':season,'week':week,'team':team,'power_rating':round(rating,3)})
        for _,g in games.iterrows():
            pred=ratings[g.home_team]-ratings[g.away_team]+HOME_ADV
            # nflverse spread_line is the favorite's positive magnitude; translate
            # to signed home margin using home_spread_odds? NOT reliable for side.
            # Determine favorite from home_spread_line if present, otherwise cannot
            # infer favored team from unsigned spread_line alone.
            if 'home_spread_line' in g.index and pd.notna(g['home_spread_line']):
                market=-float(g['home_spread_line'])
            else:
                # nflverse `spread_line` convention: home-team implied margin,
                # positive if home favorite, negative if away favorite.
                market=float(g.spread_line)
            diff=pred-market
            side='HOME' if diff>=EDGE else ('AWAY' if diff<=-EDGE else 'PASS')
            actual=float(g.home_score-g.away_score)
            cover=actual-market if side=='HOME' else market-actual
            result='PASS' if side=='PASS' else ('WIN' if cover>0 else ('PUSH' if cover==0 else 'LOSS'))
            predictions.append({'season':season,'week':week,'date':str(g.gameday.date()),'home':g.home_team,'away':g.away_team,
              'model_home_margin':round(pred,3),'market_home_margin':market,'edge':round(abs(diff),3),
              'pick':side,'actual_home_margin':actual,'result':result,
              'split':'train' if season<=2020 else ('validation' if season<=2022 else 'test')})
    p=pd.DataFrame(predictions)
    # Critical data sanity check: inspect market signs before trusting any results.
    p.to_csv(ROOT/'nfl_experiment_001_predictions.csv',index=False)
    pd.DataFrame(snapshots).to_csv(ROOT/'nfl_experiment_001_weekly_ratings.csv',index=False)
    rows=[]
    for split,g in p.groupby('split'):
        bets=g[g.pick!='PASS']; w=(bets.result=='WIN').sum(); l=(bets.result=='LOSS').sum(); push=(bets.result=='PUSH').sum()
        profit=w*(100/110)-l
        rows.append({'split':split,'games':len(g),'bets':len(bets),'wins':w,'losses':l,'pushes':push,
          'win_pct_ex_push':round(100*w/(w+l),2) if w+l else np.nan,
          'units_profit':round(profit,2),'roi_pct':round(100*profit/len(bets),2) if len(bets) else np.nan})
    summary=pd.DataFrame(rows)
    summary.to_csv(ROOT/'nfl_experiment_001_summary.csv',index=False)
    print(summary.to_string(index=False))
    print('WARNING: Verify signed spread_line convention against actual game lines before interpreting performance.')
    print('Exploratory closing-line comparison only; no timestamped pregame odds.')

if __name__=='__main__': run()

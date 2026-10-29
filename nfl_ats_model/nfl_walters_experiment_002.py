"""NFL Experiment 002: Week-5+ current-season efficiency and injury-aware spreads.

Install: python -m pip install pandas numpy pyarrow scikit-learn
Run: python nfl_walters_experiment_002.py

Public data: nflverse-data releases. Historical spreads are closing-line proxies,
NOT proven 24-hour-before-kickoff executable odds. Historical injury inputs are
optional and must be supplied with authentic timestamps and estimated point impacts.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import brier_score_loss, mean_absolute_error

ROOT = Path(__file__).resolve().parent
GAMES_URL = 'https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv'
PBP_URL = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet'
START, END = 2015, 2025
MIN_WEEK, EDGE, HFA, ODDS = 5, 1.5, 1.5, -110
# Data-availability proxy: injury updates must be timestamped before game date;
# this is NOT a 24-hour pregame cutoff without exact kickoff timestamps.
INJURY_FILE = ROOT / 'nfl_injury_adjustments_002.csv'


def read_or_download(path, url, parquet=False):
    if not path.exists():
        print(f'Downloading {path.name} ...', flush=True)
        try:
            df = pd.read_parquet(url) if parquet else pd.read_csv(url)
            if parquet: df.to_parquet(path, index=False)
            else: df.to_csv(path, index=False)
        except Exception as e:
            raise SystemExit(f'Could not download {url}: {e}\nDownload it manually to {path}')
    return pd.read_parquet(path) if parquet else pd.read_csv(path)


def load_games():
    df = read_or_download(ROOT / 'nfl_games_source.csv', GAMES_URL)
    need = ['season','week','game_type','gameday','home_team','away_team','home_score','away_score','spread_line']
    if any(c not in df for c in need): raise ValueError(f'Missing schedule fields: {set(need)-set(df)}')
    df = df[df.season.between(START, END) & df.game_type.eq('REG')].copy()
    df['gameday'] = pd.to_datetime(df.gameday)
    df = df.dropna(subset=['home_score','away_score','spread_line'])
    # IMPORTANT: nflverse spread_line is home-team perspective, positive home favored.
    # A supplied home_spread_line, if available, is a home-team bet line (negative when favored).
    df['market_home_margin'] = df.spread_line.astype(float)
    if 'home_spread_line' in df:
        mask = df.home_spread_line.notna()
        df.loc[mask,'market_home_margin'] = -df.loc[mask,'home_spread_line'].astype(float)
    df['actual_home_margin'] = df.home_score - df.away_score
    return df.sort_values(['season','week','gameday']).reset_index(drop=True)


def load_efficiency():
    records = []
    for year in range(START, END+1):
        file = ROOT / f'play_by_play_{year}.parquet'
        pbp = read_or_download(file, PBP_URL.format(season=year), parquet=True)
        required = ['season','week','season_type','posteam','defteam','epa','play_type','success']
        if any(c not in pbp for c in required):
            raise ValueError(f'PBP {year}: missing {set(required)-set(pbp)}')
        plays = pbp[(pbp.season_type=='REG') & pbp.play_type.isin(['pass','run'])].copy()
        plays = plays.dropna(subset=['posteam','defteam','epa','success'])
        # Cap EPA at the 1st/99th percentiles, computed only from earlier
        # completed weeks in this same season. Week 1 uses a fixed conservative
        # bound; never learn clipping limits from future games.
        for week, current in plays.groupby('week',sort=True):
            prior = plays[plays.week < week]['epa'].astype(float)
            lo, hi = (prior.quantile(.01), prior.quantile(.99)) if len(prior)>=300 else (-5.0, 5.0)
            current = current.copy()
            current['epa_robust'] = current.epa.astype(float).clip(lo,hi)
            for side, field in [('off','posteam'),('def','defteam')]:
                t = current.groupby(field).agg(epa=('epa_robust','mean'), success=('success','mean'),n=('epa_robust','size'))
                for team, r in t.iterrows():
                    records.append({'season':year,'week':int(week),'team':team,'side':side,
                                    'epa':float(r.epa),'success':float(r.success),'n':int(r.n)})
        print(f'Processed PBP {year}',flush=True)
    return pd.DataFrame(records)


def team_features(history, teams):
    # Aggregate prior weeks ONLY; empirical shrinkage towards league average.
    out={}
    for side in ('off','def'):
        h=history[history.side==side]
        # Weighted averages by number of qualifying plays, not by game count.
        if len(h):
            agg=h.assign(epa_sum=h.epa*h.n, success_sum=h.success*h.n).groupby('team').agg(
                epa_sum=('epa_sum','sum'), success_sum=('success_sum','sum'), n=('n','sum'))
            league_epa=h.eval('epa*n').sum()/h.n.sum()
            league_succ=h.eval('success*n').sum()/h.n.sum()
        else:
            agg=pd.DataFrame(); league_epa=0.;league_succ=.45
        for team in teams:
            if team in agg.index:
                n=float(agg.loc[team,'n']); e=float(agg.loc[team,'epa_sum']);s=float(agg.loc[team,'success_sum'])
            else: n=e=s=0.
            # Prior of 120 plays; no previous-season team ratings.
            out[(team,side,'epa')]=(e+120*league_epa)/(n+120)
            out[(team,side,'success')]=(s+120*league_succ)/(n+120)
    return out


def injury_points(games):
    if not INJURY_FILE.exists():
        pd.DataFrame(columns=['season','week','team','player','as_of_utc','delta_points','source']).to_csv(INJURY_FILE,index=False)
        print(f'Created empty injury template: {INJURY_FILE.name}. NO injury effects applied.')
        return {}
    x=pd.read_csv(INJURY_FILE)
    if x.empty: return {}
    needed={'season','week','team','player','as_of_utc','delta_points','source'}
    if not needed.issubset(x): raise ValueError(f'Injury file missing {needed-set(x)}')
    x['as_of_utc']=pd.to_datetime(x.as_of_utc,utc=True,errors='raise')
    x['delta_points']=pd.to_numeric(x.delta_points,errors='raise')
    if x.duplicated(['season','week','team','player']).any():
        raise ValueError('Duplicate injury entries per team/player/week. Keep one latest known adjustment.')
    # Positive delta_points means lineup becomes stronger; negative weaker.
    # Values must be *incremental* versus performance already in prior-week EPA.
    out={}
    for _,r in x.iterrows():
        matches=games[(games.season==r.season)&(games.week==r.week)&((games.home_team==r.team)|(games.away_team==r.team))]
        if len(matches)!=1: raise ValueError(f'Injury row has no unique game: {r.team} {r.season} W{r.week}')
        # Date-only schedule cannot verify precise kickoff. Conservative date-bound
        # validation ensures no information from game day or later is admitted.
        deadline=pd.Timestamp(matches.iloc[0].gameday.date(),tz='UTC')
        if r.as_of_utc >= deadline:
            raise ValueError(f'Injury timestamp not strictly before game date: {r.player} {r.as_of_utc}')
        key=(int(r.season),int(r.week),r.team)
        out[key]=out.get(key,0.)+float(r.delta_points)
    return out


def make_features(games, eff, injuries):
    rows=[]
    for (season,week), group in games.groupby(['season','week'],sort=True):
        if week < MIN_WEEK: continue
        h=eff[(eff.season==season)&(eff.week<week)]
        teams=set(group.home_team)|set(group.away_team)
        f=team_features(h,teams)
        for _,g in group.iterrows():
            a,b=g.home_team,g.away_team
            row={'season':int(season),'week':int(week),'date':str(g.gameday.date()),'home':a,'away':b,
                 'market_home_margin':float(g.market_home_margin),'actual_home_margin':float(g.actual_home_margin),
                 'injury_home':injuries.get((season,week,a),0.),'injury_away':injuries.get((season,week,b),0.)}
            for side in ('off','def'):
                for metric in ('epa','success'):
                    row[f'home_{side}_{metric}']=f[(a,side,metric)]
                    row[f'away_{side}_{metric}']=f[(b,side,metric)]
            rows.append(row)
    return pd.DataFrame(rows)


BASE_COLS=['home_off_epa','away_def_epa','away_off_epa','home_def_epa',
           'home_off_success','away_def_success','away_off_success','home_def_success']


def run():
    games=load_games(); eff=load_efficiency(); injuries=injury_points(games)
    data=make_features(games,eff,injuries)
    data['injury_net']=data.injury_home-data.injury_away
    # Use 2015-2020 to learn efficiency->margin mapping; 2021-22 to calibrate
    # cover probabilities. This is a frozen historical comparison, not fully
    # walk-forward retraining; within-season features use strictly prior weeks.
    train=data[data.season<=2020].copy()
    val=data[data.season.between(2021,2022)].copy()
    test=data[data.season>=2023].copy()
    # Fit separate base and injury models. Injury impact is user-supplied
    # independent adjustment; we do not pretend missing injury data is measured.
    model=Ridge(alpha=100.0)
    model.fit(train[BASE_COLS],train.actual_home_margin-HFA)
    for frame in (train,val,test):
        frame['base_home_margin']=HFA+model.predict(frame[BASE_COLS])
        frame['injury_home_margin']=frame.base_home_margin+frame.injury_net
        frame['raw_edge']=frame.injury_home_margin-frame.market_home_margin
        frame['edge']=frame.raw_edge.abs()
        frame['pick']=np.where(frame.raw_edge>=EDGE,'HOME',np.where(frame.raw_edge<=-EDGE,'AWAY','PASS'))
        frame['signed_cover_margin']=np.where(frame.pick=='HOME',frame.actual_home_margin-frame.market_home_margin,
                                              frame.market_home_margin-frame.actual_home_margin)
        frame['result']=np.where(frame.pick=='PASS','PASS',np.where(frame.signed_cover_margin>0,'WIN',
                                 np.where(frame.signed_cover_margin<0,'LOSS','PUSH')))
    # Calibrate ONLY on validation (no test leakage). Fit cover probability
    # as a function of predicted absolute edge, not a fabricated confidence.
    eligible=val[(val.pick!='PASS') & (val.result!='PUSH')]
    calibrator=None
    if len(eligible)>=30 and eligible.result.nunique()==2:
        calibrator=LogisticRegression(C=1.0)
        calibrator.fit(eligible[['edge']],(eligible.result=='WIN').astype(int))
    frames=[]
    for label,frame in [('train',train),('validation',val),('test',test)]:
        frame['split']=label
        frame['estimated_cover_probability']=np.nan
        if calibrator is not None:
            mask=frame.pick!='PASS'
            frame.loc[mask,'estimated_cover_probability']=calibrator.predict_proba(frame.loc[mask,['edge']])[:,1]
        frames.append(frame)
    result=pd.concat(frames,ignore_index=True)
    result.to_csv(ROOT/'nfl_experiment_002_predictions.csv',index=False)
    summary=[]
    for split,frame in result.groupby('split'):
        bets=frame[frame.pick!='PASS']; w=int((bets.result=='WIN').sum());l=int((bets.result=='LOSS').sum());p=int((bets.result=='PUSH').sum())
        profit=w*(100/110)-l
        row={'split':split,'games':len(frame),'bets':len(bets),'wins':w,'losses':l,'pushes':p,
             'win_pct_ex_push':round(100*w/(w+l),2) if w+l else np.nan,
             'units_profit':round(profit,2),'roi_pct':round(100*profit/len(bets),2) if len(bets) else np.nan,
             'margin_mae':round(mean_absolute_error(frame.actual_home_margin,frame.injury_home_margin),3),
             'market_mae':round(mean_absolute_error(frame.actual_home_margin,frame.market_home_margin),3)}
        nonpush=bets[bets.result!='PUSH']
        if len(nonpush) and nonpush.estimated_cover_probability.notna().all():
            row['brier']=round(brier_score_loss((nonpush.result=='WIN').astype(int),nonpush.estimated_cover_probability),4)
        summary.append(row)
    summ=pd.DataFrame(summary)
    summ.to_csv(ROOT/'nfl_experiment_002_summary.csv',index=False)
    print('\n'+summ.to_string(index=False))
    print('\nNOTE: Closing spreads are exploratory proxies, not verified executable pregame lines.')
    print('Injury data:', 'Provided; check provenance and incremental impact values.' if injuries else 'NONE; zero adjustments, injury-aware framework only.')
    print('Calibration trained on 2021-2022 validation; 2023-2025 was previously inspected in Experiment 001, so not pristine.')
    print('Check spread sign manually against known games before interpreting results.')

if __name__=='__main__': run()

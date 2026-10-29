"""NFL Experiment 003 — player/replacement-aware market comparison.

Run in the same folder as Experiment 002:
    py -m pip install pandas numpy pyarrow scikit-learn
    py nfl_walters_experiment_003.py

Experiment 002 provides leakage-controlled, current-season team efficiency.
Player adjustments require independently sourced, timestamped ratings and expected
snap shares in nfl_player_lineups_003.csv; an empty file means NO PLAYER EFFECTS.
The script NEVER substitutes invented player values, backup identities or odds.
Closing lines in nflverse schedules are *not* timestamped betting quotes. Results
are exploratory and not evidence of realizable betting returns.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, brier_score_loss
import nfl_walters_experiment_002 as old

ROOT=Path(__file__).resolve().parent
LINEUP_FILE=ROOT/'nfl_player_lineups_003.csv'
MIN_EDGE=1.5
QB_AUDIT_FILE=ROOT/'nfl_qb1_game_audit_003.csv'
QB_AUDIT_COLUMNS=['season','week','team','designated_qb1','actual_starter',
                  'qb1_started','qb1_injury_exit_no_return','verified','source']


def filter_qb1_games(games):
    """Strict, externally verified two-team QB1 audit; never infer injury from snap count.

    Audit is retrospective: results describe QB1-completed games only, NOT a
    pregame executable betting strategy (injury exits cannot be known pregame).
    """
    if not QB_AUDIT_FILE.exists():
        pd.DataFrame(columns=QB_AUDIT_COLUMNS).to_csv(QB_AUDIT_FILE,index=False)
    audit=pd.read_csv(QB_AUDIT_FILE,dtype={'season':'Int64','week':'Int64'})
    if audit.empty:
        raise ValueError(f'QB1-only filter requires verified audit rows in {QB_AUDIT_FILE.name}. '
                         'Created blank template. No results produced: unknown games are NOT assumed eligible.')
    missing=set(QB_AUDIT_COLUMNS)-set(audit.columns)
    if missing: raise ValueError(f'QB1 audit missing columns: {sorted(missing)}')
    if audit[['season','week','team']].isna().any().any():
        raise ValueError('QB1 audit has missing season/week/team identifiers')
    if audit.duplicated(['season','week','team']).any():
        raise ValueError('Duplicate QB1 audit team-game rows')
    for col in ['qb1_started','qb1_injury_exit_no_return','verified']:
        values=audit[col].astype('string').str.strip().str.lower()
        if not values.isin(['true','false','1','0']).all():
            raise ValueError(f'{col}: use true/false (no missing/unknown values)')
        audit[col]=values.isin(['true','1'])
    audit['designated_qb1']=audit['designated_qb1'].astype('string').str.strip()
    audit['actual_starter']=audit['actual_starter'].astype('string').str.strip()
    audit['source']=audit['source'].astype('string').str.strip()
    audit['eligible_team']=(audit['verified'] & audit['qb1_started'] &
                            ~audit['qb1_injury_exit_no_return'] &
                            audit['designated_qb1'].notna() &
                            audit['actual_starter'].notna() &
                            audit['designated_qb1'].eq(audit['actual_starter']) &
                            audit['source'].notna() & audit['source'].ne(''))
    home=audit[['season','week','team','eligible_team']].rename(
        columns={'team':'home_team','eligible_team':'home_qb1_ok'})
    away=audit[['season','week','team','eligible_team']].rename(
        columns={'team':'away_team','eligible_team':'away_qb1_ok'})
    n=len(games)
    out=games.merge(home,on=['season','week','home_team'],how='left',validate='many_to_one')
    out=out.merge(away,on=['season','week','away_team'],how='left',validate='many_to_one')
    eligible=out['home_qb1_ok'].fillna(False) & out['away_qb1_ok'].fillna(False)
    print(f'QB1-only audit: {int(eligible.sum())} eligible of {n} scheduled games; '
          f'{n-int(eligible.sum())} excluded or unverified.')
    return out.loc[eligible, games.columns].copy()
# Deliberately do not use a retrospective closing line to claim a pregame edge.
COLUMNS=['season','week','team','position','player','scenario','expected_snap_share',
         'player_points','rating_as_of_utc','availability_as_of_utc','source','rating_basis']


def load_lineups(games):
    if not LINEUP_FILE.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(LINEUP_FILE,index=False)
        print('Created player lineup template:',LINEUP_FILE.name)
    x=pd.read_csv(LINEUP_FILE)
    if x.empty:
        print('NO VERIFIED PLAYER RATINGS: injury and replacement adjustments disabled.')
        return {},False
    missing=set(COLUMNS)-set(x.columns)
    if missing: raise ValueError(f'Player lineup CSV missing columns: {sorted(missing)}')
    x=x.copy()
    x['scenario']=x.scenario.astype(str).str.lower().str.strip()
    if not x.scenario.isin(['baseline','expected']).all():
        raise ValueError('scenario must be baseline or expected')
    for c in ['season','week']:
        x[c]=pd.to_numeric(x[c],errors='raise').astype(int)
    for c in ['expected_snap_share','player_points']:
        x[c]=pd.to_numeric(x[c],errors='raise')
    if (~x.expected_snap_share.between(0,1)).any():
        raise ValueError('expected_snap_share must be between 0 and 1')
    for c in ['rating_as_of_utc','availability_as_of_utc']:
        x[c]=pd.to_datetime(x[c],utc=True,errors='raise')
        if x[c].isna().any(): raise ValueError(f'{c} contains missing dates')
    if x[['source','rating_basis','player','position','team']].isna().any().any():
        raise ValueError('Source, rating_basis, player, position and team are mandatory')
    if x.duplicated(['season','week','team','position','player','scenario']).any():
        raise ValueError('Duplicate player in one scenario; combine expected snap share')
    # Date-only schedule: conservative cutoff before the date of the game, not
    # an invented precise kickoff or 24-hour betting cutoff.
    dates={}
    for r in games.itertuples():
        for team in [r.home_team,r.away_team]:
            dates[(int(r.season),int(r.week),team)]=pd.Timestamp(r.gameday.date(),tz='UTC')
    for r in x.itertuples():
        key=(r.season,r.week,r.team)
        if key not in dates: raise ValueError(f'No matching scheduled game: {key}')
        if r.rating_as_of_utc>=dates[key] or r.availability_as_of_utc>=dates[key]:
            raise ValueError(f'Player data not available before game date: {key} {r.player}')
    # Explicit full depth-chart scenarios required for each adjusted position.
    # Baseline is the lineup represented in prior-week team efficiency; expected
    # is the upcoming game lineup. A missing player is not implicitly worth zero.
    impacts={}
    for key,g in x.groupby(['season','week','team','position']):
        scenarios=set(g.scenario)
        if scenarios!={'baseline','expected'}:
            raise ValueError(f'Need both baseline and expected for {key}')
        for scenario,rows in g.groupby('scenario'):
            total=rows.expected_snap_share.sum()
            if not np.isclose(total,1.0,atol=.025):
                raise ValueError(f'{key} {scenario}: snap shares sum to {total:.3f}, expected 1')
        base=g[g.scenario=='baseline']; expected=g[g.scenario=='expected']
        b=float((base.expected_snap_share*base.player_points).sum())
        e=float((expected.expected_snap_share*expected.player_points).sum())
        tkey=key[:3]
        impacts[tkey]=impacts.get(tkey,0.)+e-b
    print(f'Loaded {len(x)} player scenario rows for {len(impacts)} team-games.')
    print('WARNING: Player values are external inputs; point scale and baseline validity require independent audit.')
    return impacts,True


def grade(df, pred, prefix):
    edge=df[pred]-df.market_home_margin
    df[f'{prefix}_raw_edge']=edge
    df[f'{prefix}_pick']=np.select([edge>=MIN_EDGE,edge<=-MIN_EDGE],['HOME','AWAY'],default='PASS')
    signed=np.where(df[f'{prefix}_pick']=='HOME',df.actual_home_margin-df.market_home_margin,
                    df.market_home_margin-df.actual_home_margin)
    df[f'{prefix}_result']=np.where(df[f'{prefix}_pick']=='PASS','PASS',
                    np.where(signed>0,'WIN',np.where(signed<0,'LOSS','PUSH')))


def metrics(frame,split,model):
    bets=frame[frame[f'{model}_pick']!='PASS']
    results=bets[f'{model}_result']
    w=int((results=='WIN').sum()); l=int((results=='LOSS').sum()); p=int((results=='PUSH').sum())
    profit=w*100/110-l
    return {'split':split,'model':model,'games':len(frame),'bets':len(bets),
            'wins':w,'losses':l,'pushes':p,'win_pct_ex_push':round(100*w/(w+l),2) if w+l else np.nan,
            'units_profit':round(profit,2),'roi_pct':round(100*profit/len(bets),2) if len(bets) else np.nan,
            'margin_mae':round(mean_absolute_error(frame.actual_home_margin,frame[{'independent':'independent_margin','player':'independent_player_margin','market':'market_margin'}[model]]),3)}


def run():
    games=old.load_games()
    games=filter_qb1_games(games)
    if games.empty: raise ValueError('No verified QB1-completed games available to model.')
    eff=old.load_efficiency()
    impacts,has_players=load_lineups(games)
    data=old.make_features(games,eff,{})
    data['player_home']=data.apply(lambda r:impacts.get((int(r.season),int(r.week),r.home),0.),axis=1)
    data['player_away']=data.apply(lambda r:impacts.get((int(r.season),int(r.week),r.away),0.),axis=1)
    data['player_net']=data.player_home-data.player_away
    data['has_player_inputs']=data.apply(lambda r:(int(r.season),int(r.week),r.home) in impacts or
                                         (int(r.season),int(r.week),r.away) in impacts,axis=1)
    # Frozen team-only regression from the original training period.
    train=data[data.season<=2020]
    if train.empty: raise ValueError('No QB1-eligible training games (2015-2020). Populate the audit first.')
    model=Ridge(alpha=100.)
    model.fit(train[old.BASE_COLS],train.actual_home_margin-old.HFA)
    data['independent_margin']=old.HFA+model.predict(data[old.BASE_COLS])
    data['independent_player_margin']=data.independent_margin+data.player_net
    # Market-anchored adjustment: learn residuals against closing lines using
    # TRAIN only, never validation/test results. This is a *diagnostic* using
    # retrospective closing lines, not an executable betting strategy.
    # Ridge shrinkage discourages aggressive corrections to the market.
    data['model_market_disagreement']=data.independent_player_margin-data.market_home_margin
    xcols=['model_market_disagreement','player_net']
    residual_model=Ridge(alpha=5000.)
    residual_model.fit(data.loc[data.season<=2020,xcols],
                       data.loc[data.season<=2020,'actual_home_margin']-
                       data.loc[data.season<=2020,'market_home_margin'])
    data['market_margin']=data.market_home_margin+residual_model.predict(data[xcols])
    # Compare matched games; market anchor should not be interpreted as an
    # independent margin forecast because it already uses the market line.
    for name,pred in [('independent','independent_margin'),
                      ('player','independent_player_margin'),('market','market_margin')]:
        grade(data,pred,name)
    data['split']=np.select([data.season<=2020,data.season<=2022],['train','validation'],default='test')
    data.to_csv(ROOT/'nfl_experiment_003_predictions.csv',index=False)
    rows=[]
    for split,frame in data.groupby('split'):
        for name in ['independent','player','market']:
            rows.append(metrics(frame,split,name))
    summary=pd.DataFrame(rows)
    summary.to_csv(ROOT/'nfl_experiment_003_summary.csv',index=False)
    print('\n'+summary.to_string(index=False))
    print('\nMarket line: HISTORICAL CLOSING SPREAD, not verified timestamped available odds.')
    print('Player inputs:', 'USER-SUPPLIED; not independently verified.' if has_players else 'NONE; player model equals independent model.')
    print('No calibrated betting probabilities or claimed +EV: those require properly timed odds and a separate calibration evaluation.')
    print('2023-2025 was already examined in earlier experiments: NOT a pristine holdout.')
    print('The market model is trained on closing-line residuals, not a live wagering recommendation.')
    print('QB1-only cohort: both designated QB1s started and neither had a verified injury exit without return.')
    print('IMPORTANT: QB1 injury-exit exclusions use postgame information; results are retrospective and selection-biased.')
    print('Outputs: nfl_experiment_003_predictions.csv and nfl_experiment_003_summary.csv')

if __name__=='__main__':
    run()

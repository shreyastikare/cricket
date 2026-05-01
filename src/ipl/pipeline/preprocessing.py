import numpy as np
import pandas as pd
from datetime import datetime

import ipl

PATH = 'data/raw/IPL.csv'


def _apply_dynamic_remaining_ball_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def numeric_col(col: str, default_value = np.nan) -> pd.Series:
        if col in out.columns:
            return pd.to_numeric(out[col], errors = 'coerce')
        return pd.Series(default_value, index = out.index, dtype = 'float64')

    team_runs = numeric_col('team_runs')
    team_balls = numeric_col('team_balls').fillna(0)
    team_wicket = numeric_col('team_wicket')
    innings = numeric_col('innings')
    runs_target = numeric_col('runs_target')
    overs = numeric_col('overs')
    balls_per_over = numeric_col('balls_per_over')
    balls_per_over = balls_per_over.where(balls_per_over > 0, 6).fillna(6)

    total_balls = np.rint(overs * balls_per_over)
    total_balls = pd.Series(total_balls, index = out.index, dtype = 'float64')
    total_balls = total_balls.where(total_balls > 0, 120).fillna(120)

    powerplay_total = np.rint(total_balls * 0.30)
    middle_total = np.rint(total_balls * 0.50)

    out['run_rate'] = team_runs / team_balls * 6
    out['required_runs'] = np.where(innings == 2, runs_target - team_runs, np.nan)
    out['balls_remaining'] = (total_balls - team_balls).clip(lower = 0)
    out['wickets_remaining'] = 10 - team_wicket
    out['required_run_rate'] = np.where(innings == 2, out['required_runs'] / out['balls_remaining'] * 6, np.nan)

    out['powerplay_balls_remaining'] = (powerplay_total - team_balls).clip(lower = 0)
    out['middle_balls_remaining'] = (
        (powerplay_total + middle_total - team_balls).clip(lower = 0) - out['powerplay_balls_remaining']
    ).clip(lower = 0)
    out['death_balls_remaining'] = (
        out['balls_remaining'] - out['middle_balls_remaining'] - out['powerplay_balls_remaining']
    ).clip(lower = 0)

    return out


def load_data(path = PATH):
    # import data and parse date-time
    df = pd.read_csv(path, low_memory = False)
    df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d')
    
    df['overs_phase'] = np.where(df['over'] >= 16, 'death', np.where(df['over'] <= 5, 'powerplay', 'middle'))
    df = _apply_dynamic_remaining_ball_features(df)
    
    # designate if each delivery was a boundary or not (for the batter)
    df['four'] = (df['runs_batter'] == 4) & (df['runs_not_boundary'] == False)
    df['six'] = (df['runs_batter'] == 6) & (df['runs_not_boundary'] == False)
    df['dot'] = ((df['runs_batter'] == 0) & (df['valid_ball'] == 1))

    # count dismissals for each batter (not counting retired hurt)
    dismissals = df[(df['innings'] <= 2) & df['player_out'].notna() & (df['wicket_kind'] != 'retired hurt')][['match_id', 'player_out']]
    dismissals['batter_out_in_innings'] = True

    # create a new column 'batter_out_in_innings' that indicates whether the batter's wicket was taken during the match
    df = pd.merge(left = df, right = dismissals, left_on = ['match_id', 'batter'], right_on = ['match_id', 'player_out'], how = 'left', suffixes = ('', '_y'))
    df['batter_out_in_innings'] = df['batter_out_in_innings'].fillna(False)

    # put a match winner when team wins via superover
    df['match_won_by'] = np.where(
        df['superover_winner'].notna(),
        df['superover_winner'],
        df['match_won_by']
    )

    # playoff indicator
    df['playoff_match'] = (df['stage'] != 'Unknown').astype(int)
    df['season_post_2008'] = df['year'] - 2008
    
    # bin deliveries into valid / wide / no ball
    df['delivery_type'] = np.where(
        df['valid_ball'] == 1, 
        'valid', 
        np.where(
            df['extra_type'] == 'wides',
            'wide',
            np.where(
                df['extra_type'].isin(['noballs', 'legbyes, noballs', 'byes, noballs']),
                'noball',
                None
            )
        )
    )
    
    # wicket indicator
    df['wicket_taken'] = df['wicket_kind'].notna().astype(int)
    df['striker_out'] = (df['batter'] == df['player_out']).fillna(False).astype(int)
    df['free_hit'] = df.groupby(['match_id', 'innings'])['delivery_type'].shift(1).eq('noball').fillna(False).astype(int)

    # batsman and non-striker runs/balls faced
    df['non_striker_runs'] = 1
    df['non_striker_balls'] = 1

    # partnership stats
    df['partnership_runs'] = df.groupby(['match_id', 'batting_partners', 'innings'])['runs_total'].cumsum()
    df['partnership_balls'] = df.groupby(['match_id', 'batting_partners', 'innings'])['balls_faced'].cumsum()

    # fix venue and city names, replace low count cities with 'Other' or 'South Africa'
    df.loc[df['venue'] == 'Sharjah Cricket Stadium', 'city'] = 'Sharjah'
    df.loc[df['venue'] == 'Dubai International Cricket Stadium', 'city'] = 'Dubai'

    # cities_SA = ['Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein']
    # cities_low_count = ['Rajkot', 'Mohali', 'Indore', 'Ranchi', 'Cuttack', 'Raipur', 'Kochi', 'Guwahati', 'Kanpur', 'Nagpur']
    # df['city'] = df['city'].replace('Navi Mumbai', 'Mumbai')
    # df['city'] = df['city'].replace('New Chandigarh', 'Chandigarh')
    # df['city'] = df['city'].replace(cities_SA, 'South Africa')
    # df['city'] = df['city'].replace(cities_low_count, 'Other')
    # df['city'] = df['city'].replace('Lucknow', 'Other')
    
    # fix team names
    team_name_fixes = {
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
        'Kings XI Punjab': 'Punjab Kings',
        'Rising Pune Supergiant': 'Rising Pune Supergiants'
    }
    # df = df.replace(team_name_fixes)

    # drop irrelevant columns
    drop_cols = [
        'id', 
        'match_type', 
        'event_name', 
        'gender', 
        'team_type', 
        'balls_per_over', 
        'overs', 
        'match_number', 
        'ball_no', 
        'review_batter', 
        'team_reviewed', 
        'umpire', 
        'umpires_call', 
        'season', 
        'player_out_y'
    ]
    df = df.drop(drop_cols, axis = 1)

    return df


def load_data_live(path = PATH):
    # import data and parse date-time
    df = pd.read_csv(path, low_memory = False)
    df['date'] = pd.to_datetime(df['date'], format = '%Y-%m-%d')

    # infer fields that can be known in a live match feed
    if 'year' not in df.columns:
        df['year'] = df['date'].dt.year.astype(int)
    else:
        inferred_year = df['date'].dt.year
        df['year'] = pd.to_numeric(df['year'], errors = 'coerce').fillna(inferred_year).astype(int)

    if 'stage' not in df.columns:
        df['stage'] = 'Unknown'
    df['stage'] = df['stage'].fillna('Unknown')

    if 'runs_target' not in df.columns:
        df['runs_target'] = np.nan
    first_innings_targets = (
        df[df['innings'] == 1]
        .groupby('match_id')['team_runs']
        .max()
        .add(1)
    )
    second_innings_mask = df['innings'] == 2
    df['runs_target'] = pd.to_numeric(df['runs_target'], errors = 'coerce')
    df.loc[second_innings_mask, 'runs_target'] = df.loc[second_innings_mask, 'runs_target'].fillna(
        df.loc[second_innings_mask, 'match_id'].map(first_innings_targets)
    )

    if 'balls_faced' not in df.columns:
        wides_mask = df['extra_type'].fillna('').astype(str).str.contains('wides', na = False)
        df['balls_faced'] = np.where(wides_mask, 0, 1).astype(int)
    else:
        wides_mask = df['extra_type'].fillna('').astype(str).str.contains('wides', na = False)
        inferred_balls = pd.Series(np.where(wides_mask, 0, 1), index = df.index)
        df['balls_faced'] = pd.to_numeric(df['balls_faced'], errors = 'coerce').fillna(inferred_balls).astype(int)

    inferred_partners = df[['batter', 'non_striker']].apply(
        lambda row: (
            str(tuple(sorted([row['batter'], row['non_striker']])))
            if pd.notna(row['batter']) and pd.notna(row['non_striker'])
            else np.nan
        ),
        axis = 1
    )
    if 'batting_partners' not in df.columns:
        df['batting_partners'] = inferred_partners
    else:
        df['batting_partners'] = df['batting_partners'].fillna(inferred_partners)

    df['overs_phase'] = np.where(df['over'] >= 16, 'death', np.where(df['over'] <= 5, 'powerplay', 'middle'))
    df = _apply_dynamic_remaining_ball_features(df)

    # designate if each delivery was a boundary or not (for the batter)
    df['four'] = (df['runs_batter'] == 4) & (df['runs_not_boundary'] == False)
    df['six'] = (df['runs_batter'] == 6) & (df['runs_not_boundary'] == False)
    df['dot'] = ((df['runs_batter'] == 0) & (df['valid_ball'] == 1))

    # playoff indicator
    df['playoff_match'] = (df['stage'] != 'Unknown').astype(int)
    df['season_post_2008'] = df['year'] - 2008

    # bin deliveries into valid / wide / no ball
    df['delivery_type'] = np.where(
        df['valid_ball'] == 1,
        'valid',
        np.where(
            df['extra_type'] == 'wides',
            'wide',
            np.where(
                df['extra_type'].isin(['noballs', 'legbyes, noballs', 'byes, noballs']),
                'noball',
                None
            )
        )
    )

    # wicket indicator
    df['wicket_taken'] = df['wicket_kind'].notna().astype(int)
    df['striker_out'] = (df['batter'] == df['player_out']).fillna(False).astype(int)
    df['free_hit'] = df.groupby(['match_id', 'innings'])['delivery_type'].shift(1).eq('noball').fillna(False).astype(int)

    # batsman and non-striker runs/balls faced
    df['non_striker_runs'] = 1
    df['non_striker_balls'] = 1

    # partnership stats
    df['partnership_runs'] = df.groupby(['match_id', 'batting_partners', 'innings'])['runs_total'].cumsum()
    df['partnership_balls'] = df.groupby(['match_id', 'batting_partners', 'innings'])['balls_faced'].cumsum()

    # fix venue and city names, replace low count cities with 'Other' or 'South Africa'
    if 'venue' in df.columns and 'city' in df.columns:
        df['city'] = df['city'].astype('object')
        df.loc[df['venue'] == 'Sharjah Cricket Stadium', 'city'] = 'Sharjah'
        df.loc[df['venue'] == 'Dubai International Cricket Stadium', 'city'] = 'Dubai'

        cities_SA = ['Cape Town', 'Port Elizabeth', 'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley', 'Bloemfontein']
        cities_low_count = ['Rajkot', 'Mohali', 'Indore', 'Ranchi', 'Cuttack', 'Raipur', 'Kochi', 'Guwahati', 'Kanpur', 'Nagpur']
        df['city'] = df['city'].replace('Navi Mumbai', 'Mumbai')
        df['city'] = df['city'].replace('New Chandigarh', 'Chandigarh')
        df['city'] = df['city'].replace(cities_SA, 'South Africa')
        df['city'] = df['city'].replace(cities_low_count, 'Other')
        df['city'] = df['city'].replace('Lucknow', 'Other')

    # fix team names
    team_name_fixes = {
        'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
        'Kings XI Punjab': 'Punjab Kings',
        'Rising Pune Supergiant': 'Rising Pune Supergiants'
    }
    df = df.replace(team_name_fixes)

    # drop irrelevant / unavailable-in-live columns
    drop_cols = [
        'id',
        'match_type',
        'event_name',
        'gender',
        'team_type',
        'balls_per_over',
        'overs',
        'match_number',
        'ball_no',
        'review_batter',
        'team_reviewed',
        'umpire',
        'umpires_call',
        'season',
        'player_out_y',
        'match_won_by',
        'superover_winner',
        'player_of_match',
        'win_outcome',
        'result_type',
        'method',
        'batter_out_in_innings'
    ]
    df = df.drop(columns = drop_cols, errors = 'ignore')

    return df


def match_list(df):
    required_cols = ['match_id', 'innings', 'team_runs', 'team_wicket', 'team_balls', 'run_rate', 'balls_remaining']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f'match_list requires load_data/load_data_live derived columns; missing: {missing_cols}'
        )

    def summarize_innings(innings_no: int, prefix: str) -> pd.DataFrame:
        innings_df = df[df['innings'] == innings_no].copy()
        innings_base = (
            innings_df[['match_id', 'team_runs', 'team_wicket', 'team_balls', 'run_rate', 'balls_remaining']]
            .groupby('match_id')
            .last()
        )

        innings_base['team_balls'] = pd.to_numeric(innings_base['team_balls'], errors = 'coerce').fillna(0)
        innings_base['team_wicket'] = pd.to_numeric(innings_base['team_wicket'], errors = 'coerce').fillna(0)
        innings_base['team_runs'] = pd.to_numeric(innings_base['team_runs'], errors = 'coerce').fillna(0)
        innings_base['run_rate'] = pd.to_numeric(innings_base['run_rate'], errors = 'coerce')
        innings_base['balls_remaining'] = pd.to_numeric(innings_base['balls_remaining'], errors = 'coerce').fillna(0)

        innings_total_balls = innings_base['team_balls'] + innings_base['balls_remaining']
        innings_total_balls = innings_total_balls.where(innings_total_balls > 0, 120).fillna(120)

        powerplay_total = np.rint(innings_total_balls * 0.30)
        middle_total = np.rint(innings_total_balls * 0.50)

        innings_base['team_balls_adj'] = np.where(
            innings_base['team_wicket'] == 10,
            innings_total_balls,
            innings_base['team_balls']
        )

        innings_base['powerplay_balls'] = np.minimum(powerplay_total, innings_base['team_balls'])
        innings_base['middle_balls'] = np.maximum(
            0,
            np.minimum(powerplay_total + middle_total, innings_base['team_balls']) - innings_base['powerplay_balls']
        )
        innings_base['death_balls'] = np.maximum(
            0,
            innings_base['team_balls'] - innings_base['powerplay_balls'] - innings_base['middle_balls']
        )

        progression = innings_df[['match_id', 'team_balls', 'team_runs', 'team_wicket']].copy()
        progression['team_balls'] = pd.to_numeric(progression['team_balls'], errors = 'coerce').fillna(0)
        progression['team_runs'] = pd.to_numeric(progression['team_runs'], errors = 'coerce').fillna(0)
        progression['team_wicket'] = pd.to_numeric(progression['team_wicket'], errors = 'coerce').fillna(0)
        progression = progression.sort_values(['match_id', 'team_balls'], kind = 'mergesort')
        progression = progression.groupby(['match_id', 'team_balls'], as_index = False).last()

        phase_stats_rows = []
        for match_id in innings_base.index:
            match_progression = progression[progression['match_id'] == match_id]
            pp_cutoff = powerplay_total.loc[match_id]
            middle_cutoff = (powerplay_total + middle_total).loc[match_id]

            pp_state = match_progression[match_progression['team_balls'] <= pp_cutoff]
            middle_state = match_progression[match_progression['team_balls'] <= middle_cutoff]

            pp_runs = 0 if pp_state.empty else float(pp_state.iloc[-1]['team_runs'])
            pp_wicket = 0 if pp_state.empty else float(pp_state.iloc[-1]['team_wicket'])
            middle_runs_cum = 0 if middle_state.empty else float(middle_state.iloc[-1]['team_runs'])
            middle_wicket_cum = 0 if middle_state.empty else float(middle_state.iloc[-1]['team_wicket'])

            total_runs = float(innings_base.loc[match_id, 'team_runs'])
            total_wickets = float(innings_base.loc[match_id, 'team_wicket'])

            phase_stats_rows.append({
                'match_id': match_id,
                'powerplay_runs': pp_runs,
                'powerplay_wicket': pp_wicket,
                'middle_runs': max(0, middle_runs_cum - pp_runs),
                'middle_wicket': max(0, middle_wicket_cum - pp_wicket),
                'death_runs': max(0, total_runs - middle_runs_cum),
                'death_wicket': max(0, total_wickets - middle_wicket_cum)
            })

        if phase_stats_rows:
            phase_stats = pd.DataFrame(phase_stats_rows).set_index('match_id')
        else:
            phase_stats = pd.DataFrame(
                0,
                index = innings_base.index,
                columns = [
                    'powerplay_runs',
                    'powerplay_wicket',
                    'middle_runs',
                    'middle_wicket',
                    'death_runs',
                    'death_wicket'
                ]
            )
        innings_base = innings_base.join(phase_stats)
        innings_base = innings_base.drop(columns = ['balls_remaining'])

        rename_map = {
            'team_runs': f'{prefix}_runs',
            'team_wicket': f'{prefix}_wickets',
            'team_balls': f'{prefix}_balls',
            'team_balls_adj': f'{prefix}_balls_adj',
            'powerplay_balls': f'{prefix}_powerplay_balls',
            'middle_balls': f'{prefix}_middle_balls',
            'death_balls': f'{prefix}_death_balls',
            'run_rate': f'{prefix}_run_rate',
            'powerplay_runs': f'{prefix}_powerplay_runs',
            'powerplay_wicket': f'{prefix}_powerplay_wicket',
            'middle_runs': f'{prefix}_middle_runs',
            'middle_wicket': f'{prefix}_middle_wicket',
            'death_runs': f'{prefix}_death_runs',
            'death_wicket': f'{prefix}_death_wicket'
        }
        return innings_base.rename(rename_map, axis = 1)

    match_first_innings = summarize_innings(1, 'first_innings')
    match_second_innings = summarize_innings(2, 'second_innings')

    match_info_cols = [
        'match_id', 
        'date', 
        'batting_team', 
        'bowling_team', 
        'player_of_match', 
        'match_won_by', 
        'win_outcome', 
        'toss_winner', 
        'toss_decision', 
        'venue', 
        'city', 
        'year', 
        'superover_winner',
        'result_type', 
        'method', 
        'event_match_no', 
        'stage',
        'playoff_match'
    ]
    match_list = (
        df[match_info_cols]
        .groupby('match_id')
        .first()
        .rename(columns = {
            'batting_team': 'bat_first',
            'bowling_team': 'bowl_first'
        })
    )

    match_list = pd.merge(left = match_list, right = match_first_innings, on = 'match_id')
    match_list = pd.merge(left = match_list, right = match_second_innings, on = 'match_id')
    
    # indicator whether batting team won the match
    match_list['batting_team_won'] = (match_list['bat_first'] == match_list['match_won_by']).astype(int)
    match_list['bowling_team_won'] = (match_list['bowl_first'] == match_list['match_won_by']).astype(int)
    
    match_list['batting_team_won_toss'] = (match_list['toss_decision'] == 'bat').astype(int)
    match_list['bowling_team_won_toss'] = (match_list['toss_decision'] == 'bowl').astype(int)
    match_list['batting_team_NRR'] = match_list['first_innings_run_rate'] - match_list['second_innings_run_rate']
    match_list['bowling_team_NRR'] = match_list['second_innings_run_rate'] - match_list['first_innings_run_rate']
    
    match_list = match_list.reset_index()
    return match_list


def team_lag_features(X: pd.DataFrame, n_lags = 3) -> pd.DataFrame:
    '''Given a filtered match list dataframe, create lag features for the team's stats in the prior few games. Helper function for prior_match_stats.'''
    if n_lags <= 0:
        return X
    
    X = X.sort_values(['team', 'date'])
    
    X['total_runs'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['runs'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['total_balls'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['balls'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['total_balls_adj'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['balls_adj'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['total_runs_conceded'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_runs'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['total_balls_bowled'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_balls'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['total_balls_bowled_adj'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_balls_adj'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['total_wickets_taken'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_wickets'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    # runs conceded per wicket
    X['prior_avg_runs_conceded_per_wicket'] = (X['total_runs_conceded'] / X['total_wickets_taken']).where(X['total_wickets_taken'] > 0, np.nan)
    
    # run rates and NRR
    X['prior_run_rate'] = (X['total_runs'] / X['total_balls'] * 6).where(X['total_balls'] > 0, np.nan)
    X['opp_prior_run_rate'] = (X['total_runs_conceded'] / X['total_balls_bowled'] * 6).where(X['total_balls_bowled'] > 0, np.nan)
    
    ############ PRIOR BATTING STATS ################
    
    # POWERPLAY BATTING STATS
    X['prior_powerplay_total_runs_scored'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['powerplay_runs'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['prior_powerplay_avg_runs_scored'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['powerplay_runs'].shift(1).rolling(n_lags, min_periods = 1).mean(), include_groups = False)
    )
    
    X['prior_powerplay_total_wickets_lost'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['powerplay_wicket'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['prior_powerplay_avg_wickets_lost'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['powerplay_wicket'].shift(1).rolling(n_lags, min_periods = 1).mean(), include_groups = False)
    )
    
    X['prior_powerplay_balls_played'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['powerplay_balls'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    # MIDDLE OVERS BATTING STATS
    X['prior_middle_total_runs_scored'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['middle_runs'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['prior_middle_avg_runs_scored'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['middle_runs'].shift(1).rolling(n_lags, min_periods = 1).mean(), include_groups = False)
    )
    
    X['prior_middle_total_wickets_lost'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['middle_wicket'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['prior_middle_avg_wickets_lost'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['middle_wicket'].shift(1).rolling(n_lags, min_periods = 1).mean(), include_groups = False)
    )
    
    X['prior_middle_balls_played'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['middle_balls'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    # DEATH OVERS BATTING STATS
    X['prior_death_total_runs_scored'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['death_runs'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['prior_death_avg_runs_scored'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['death_runs'].shift(1).rolling(n_lags, min_periods = 1).mean(), include_groups = False)
    )
    
    X['prior_death_total_wickets_lost'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['death_wicket'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['prior_death_avg_wickets_lost'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['death_wicket'].shift(1).rolling(n_lags, min_periods = 1).mean(), include_groups = False)
    )
    
    X['prior_death_balls_played'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['death_balls'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    ########## PRIOR BOWLING STATS ############
    
    # POWERPLAY BOWLING STATS
    X['prior_powerplay_total_runs_conceded'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_powerplay_runs'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['prior_powerplay_avg_runs_conceded'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_powerplay_runs'].shift(1).rolling(n_lags, min_periods = 1).mean(), include_groups = False)
    )
    
    X['prior_powerplay_balls_bowled'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_powerplay_balls'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['prior_powerplay_total_wickets_taken'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_powerplay_wicket'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['prior_powerplay_avg_wickets_taken'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_powerplay_wicket'].shift(1).rolling(n_lags, min_periods = 1).mean(), include_groups = False)
    )
    
    # MIDDLE BOWLING STATS
    X['prior_middle_total_runs_conceded'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_middle_runs'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['prior_middle_avg_runs_conceded'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_middle_runs'].shift(1).rolling(n_lags, min_periods = 1).mean(), include_groups = False)
    )
    
    X['prior_middle_balls_bowled'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_middle_balls'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['prior_middle_total_wickets_taken'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_middle_wicket'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['prior_middle_avg_wickets_taken'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_middle_wicket'].shift(1).rolling(n_lags, min_periods = 1).mean(), include_groups = False)
    )
    
    # DEATH BOWLING STATS
    X['prior_death_total_runs_conceded'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_death_runs'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['prior_death_avg_runs_conceded'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_death_runs'].shift(1).rolling(n_lags, min_periods = 1).mean(), include_groups = False)
    )
    
    X['prior_death_balls_bowled'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_death_balls'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['prior_death_total_wickets_taken'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_death_wicket'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['prior_death_avg_wickets_taken'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['opp_death_wicket'].shift(1).rolling(n_lags, min_periods = 1).mean(), include_groups = False)
    )
    
    ######## RUN RATES ################
    
    # total NRR
    X['prior_NRR'] = (
        (X['total_runs'] / X['total_balls_adj'] * 6).where(X['total_balls_adj'] > 0, np.nan)
        - (X['total_runs_conceded'] / X['total_balls_bowled_adj'] * 6).where(X['total_balls_bowled_adj'] > 0, np.nan)
    ).fillna(0)
    
    X['prior_powerplay_NRR'] = (
        (X['prior_powerplay_total_runs_scored'] / X['prior_powerplay_balls_played'] * 6).where(X['prior_powerplay_balls_played'] > 0, np.nan)
        - (X['prior_powerplay_total_runs_conceded'] / X['prior_powerplay_balls_bowled'] * 6).where(X['prior_powerplay_balls_bowled'] > 0, np.nan)
    ).fillna(0)
    
    X['prior_middle_NRR'] = (
        (X['prior_middle_total_runs_scored'] / X['prior_middle_balls_played'] * 6).where(X['prior_middle_balls_played'] > 0, np.nan)
        - (X['prior_middle_total_runs_conceded'] / X['prior_middle_balls_bowled'] * 6).where(X['prior_middle_balls_bowled'] > 0, np.nan)
    ).fillna(0)
    
    X['prior_death_NRR'] = (
        (X['prior_death_total_runs_scored'] / X['prior_death_balls_played'] * 6).where(X['prior_death_balls_played'] > 0, np.nan)
        - (X['prior_death_total_runs_conceded'] / X['prior_death_balls_bowled'] * 6).where(X['prior_death_balls_bowled'] > 0, np.nan)
    ).fillna(0)
    
    # matches played, won, and lost
    X['total_wins'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['match_won'].shift(1).rolling(n_lags, min_periods = 1).sum(), include_groups = False)
    )
    
    X['total_played'] = (
        X.groupby(['team', 'year'], group_keys = False)
        .apply(lambda x: x['match_won'].shift(1).rolling(n_lags, min_periods = 1).count(), include_groups = False)
    )
    
    X['prior_ratio_won'] = (X['total_wins'] / X['total_played']).where(X['total_played'] > 0, np.nan)
    
    lag_features = [col for col in X.columns if col.startswith('prior')]
    cols_to_keep = [
        'match_id',
        'date',
        'team',
        *lag_features
    ]
    
    X = X[cols_to_keep]
    return X


def prior_match_stats(X: pd.DataFrame, n_lags = 3) -> pd.DataFrame:
    '''Creates features for prior matches for both the batting and bowling team.'''
    X = X[X['result_type'] != 'no result']
    
    X1 = X.copy()
    X2 = X.copy()
    
    cols_to_rename_1 = {
        'bat_first': 'team',
        'bowl_first': 'opp',
        **{col: col.replace('first_innings_', '') for col in X1.columns if col.startswith('first_innings_')},
        **{col: col.replace('batting_team_', '') for col in X1.columns if col.startswith('batting_team_')},
        **{col: col.replace('second_innings_', 'opp_') for col in X1.columns if col.startswith('second_innings_')},
        **{col: col.replace('bowling_team_', 'opp_') for col in X1.columns if col.startswith('bowling_team_')},
    }
    X1 = X1.rename(columns = cols_to_rename_1)
    
    X1['match_won'] = (X1['match_won_by'] == X1['team']).astype(int)
    X1['toss_won'] = (X1['toss_winner'] == X1['team']).astype(int)
    X1['batted_first'] = 1
    
    cols_to_drop_1 = [
        'win_outcome',
        'player_of_match',
        'match_won_by',
        'win_outcome',
        'toss_winner',
        'toss_decision',
        'venue',
        'city',
        'superover_winner',
        'result_type',
        'method',
        'event_match_no',
        'stage',
        'won_toss',
        'won',
        'opp_NRR',
        'playoff_match'
    ]
    X1 = X1.drop(columns = cols_to_drop_1)
    
    cols_to_rename_2 = {
        'bat_first': 'opp',
        'bowl_first': 'team',
        **{col: col.replace('first_innings_', 'opp_') for col in X2.columns if col.startswith('first_innings_')},
        **{col: col.replace('batting_team_', 'opp_') for col in X2.columns if col.startswith('batting_team_')},
        **{col: col.replace('second_innings_', '') for col in X2.columns if col.startswith('second_innings_')},
        **{col: col.replace('bowling_team_', '') for col in X2.columns if col.startswith('bowling_team_')},
    }
    X2 = X2.rename(columns = cols_to_rename_2)
    
    X2['match_won'] = (X2['match_won_by'] == X2['team']).astype(int)
    X2['toss_won'] = (X2['toss_winner'] == X2['team']).astype(int)
    X2['batted_first'] = 0
    
    cols_to_drop_2 = [
        'win_outcome',
        'player_of_match',
        'match_won_by',
        'win_outcome',
        'toss_winner',
        'toss_decision',
        'venue',
        'city',
        'superover_winner',
        'result_type',
        'method',
        'event_match_no',
        'stage',
        'opp_won_toss',
        'opp_won',
        'opp_NRR',
        'playoff_match'
    ]
    X2 = X2.drop(columns = cols_to_drop_2)  
    
    X_cat = pd.concat([X1, X2], ignore_index = True).sort_values('date', ascending = True).reset_index(drop = True)
    X_lag = team_lag_features(X_cat, n_lags = n_lags)
    
    features = [
        'date',
        'team',
        'prior_run_rate',
        'prior_avg_runs_conceded_per_wicket',
        'prior_NRR', 
        'prior_powerplay_NRR',
        'prior_middle_NRR',
        'prior_death_NRR',
        'prior_ratio_won'
    ]
    
    batting_team_rename = {col: 'batting_team_' + col for col in features if col.startswith('prior')}
    bowling_team_rename = {col: 'bowling_team_' + col for col in features if col.startswith('prior')}
    
    X_lag_bat = X_lag[features].rename(columns = batting_team_rename)
    X_lag_bowl = X_lag[features].rename(columns = bowling_team_rename)
    
    X_final = X.merge(X_lag_bat, left_on = ['bat_first', 'date'], right_on = ['team', 'date'], how = 'inner')
    X_final = X_final.merge(X_lag_bowl, left_on = ['bowl_first', 'date'], right_on = ['team', 'date'], how = 'inner')
    X_final = X_final.drop(columns = ['team_x', 'team_y'])
    
    return X_final


def preprocess_first_innings(X_df: pd.DataFrame, match_list: pd.DataFrame, full = True, min_overs = 0, max_overs = 20):
    X = X_df.copy()

    required_cols = [
        'run_rate',
        'wickets_remaining',
        'balls_remaining',
        'powerplay_balls_remaining',
        'middle_balls_remaining',
        'death_balls_remaining'
    ]
    missing_cols = [col for col in required_cols if col not in X.columns]
    if missing_cols:
        raise ValueError(
            f'preprocess_first_innings requires load_data/load_data_live derived columns; missing: {missing_cols}'
        )

    cols_to_keep = [
        'match_id',
        'team_runs',
        'team_wicket',
        'team_balls',
        'run_rate',
        'batter',
        'bowler',
        'innings',
        'batter_runs',
        'balls_remaining',
        'wickets_remaining',
        'powerplay_balls_remaining',
        'middle_balls_remaining',
        'death_balls_remaining',
        'player_out', 
        'extra_type', 
        'wicket_kind', 
        'wicket_taken'
    ]
    if 'ball_id' in X.columns:
        cols_to_keep.append('ball_id')
    X = X[X['innings'] == 1][cols_to_keep]
    if min_overs > 0:
        X = X[(X['team_balls'] > (min_overs * 6)) & (X['team_balls'] <= (max_overs * 6))]
    else: 
        X = X[X['team_balls'] <= (max_overs * 6)]
    
    X = X.merge(match_list, on = 'match_id', how = 'inner')
    X['runs_delta'] = X['first_innings_runs'] - X['team_runs']
    
    X['season_post_2008'] = X['year'] - 2008
    
    X['wicket_x_run_rate_x_powerplay_balls'] = X['wickets_remaining'] * X['run_rate'] * X['powerplay_balls_remaining']
    X['wicket_x_run_rate_x_middle_balls'] = X['wickets_remaining'] * X['run_rate'] * X['middle_balls_remaining']
    X['wicket_x_run_rate_x_death_balls'] = X['wickets_remaining'] * X['run_rate'] * X['death_balls_remaining']

    X['run_rate_x_powerplay_balls'] = X['run_rate'] * X['powerplay_balls_remaining']
    X['run_rate_x_middle_balls'] = X['run_rate'] * X['middle_balls_remaining']
    X['run_rate_x_death_balls'] = X['run_rate'] * X['death_balls_remaining']
    
    X['wicket_x_powerplay_balls'] = X['wickets_remaining'] * X['powerplay_balls_remaining']
    X['wicket_x_middle_balls'] = X['wickets_remaining'] * X['middle_balls_remaining']
    X['wicket_x_death_balls'] = X['wickets_remaining'] * X['death_balls_remaining']
    
    X['sqrt_wicket_x_powerplay_balls'] = np.sqrt(X['wickets_remaining'] * X['powerplay_balls_remaining'])
    X['sqrt_wicket_x_middle_balls'] = np.sqrt(X['wickets_remaining'] * X['middle_balls_remaining'])
    X['sqrt_wicket_x_death_balls'] = np.sqrt(X['wickets_remaining'] * X['death_balls_remaining'])
    
    X['season_post_2008_x_powerplay_balls'] = X['season_post_2008'] * X['powerplay_balls_remaining']
    X['season_post_2008_x_middle_balls'] = X['season_post_2008'] * X['middle_balls_remaining']
    X['season_post_2008_x_death_balls'] = X['season_post_2008'] * X['death_balls_remaining']
    
    X['batting_team_prior_powerplay_NRR_x_powerplay_balls'] = X['batting_team_prior_powerplay_NRR'] * X['powerplay_balls_remaining']
    X['batting_team_prior_middle_NRR_x_middle_balls'] = X['batting_team_prior_middle_NRR'] * X['middle_balls_remaining']
    X['batting_team_prior_death_NRR_x_death_balls'] = X['batting_team_prior_death_NRR'] * X['death_balls_remaining']
    
    X['bowling_team_prior_powerplay_NRR_x_powerplay_balls'] = X['bowling_team_prior_powerplay_NRR'] * X['powerplay_balls_remaining']
    X['bowling_team_prior_middle_NRR_x_middle_balls'] = X['bowling_team_prior_middle_NRR'] * X['middle_balls_remaining']
    X['bowling_team_prior_death_NRR_x_death_balls'] = X['bowling_team_prior_death_NRR'] * X['death_balls_remaining']
    
    X['batting_team_prior_run_rate_x_powerplay_balls'] = X['batting_team_prior_run_rate'] * X['powerplay_balls_remaining']
    X['batting_team_prior_run_rate_x_middle_balls'] = X['batting_team_prior_run_rate'] * X['middle_balls_remaining']
    X['batting_team_prior_run_rate_x_death_balls'] = X['batting_team_prior_run_rate'] * X['death_balls_remaining']
    
    # completed_first_innings = X[(X['year'] <= 2021) & (X['innings'] == 1) & (X['team_balls'] == 120)]['match_id'].unique()
    # validation_first_innings = X[X['year'] > 2021]['match_id'].unique()
    # first_innings_list = list(set(completed_first_innings) | set(validation_first_innings))
    # X = X[X['match_id'].isin(first_innings_list)]
    if full:
        params = ipl.load_resource_params()
    else:
        params = ipl.load_resource_training_params()
    X['resource'] = ipl.resource_function(X, params)
    
    return X


def preprocess_second_innings(df: pd.DataFrame, match_list: pd.DataFrame):
    X = df.copy()

    required_cols = [
        'required_runs',
        'required_run_rate',
        'wickets_remaining',
        'balls_remaining',
        'powerplay_balls_remaining',
        'middle_balls_remaining',
        'death_balls_remaining'
    ]
    missing_cols = [col for col in required_cols if col not in X.columns]
    if missing_cols:
        raise ValueError(
            f'preprocess_second_innings requires load_data/load_data_live derived columns; missing: {missing_cols}'
        )
    
    # first innings parallel progress
    X_1 = X[X['innings'] == 1][['match_id', 'team_runs', 'wickets_remaining', 'team_balls']]
    X_1 = X_1.groupby(['match_id', 'team_balls']).last().reset_index()
    X_1 = X_1.rename(columns = {
        'team_runs': 'first_innings_current_runs',
        'wickets_remaining': 'first_innings_current_wickets_remaining'
    })
    
    # second innings 
    cols_to_keep = [
        'match_id',
        'required_runs',
        'required_run_rate',
        'batter',
        'bowler',
        'innings',
        'batter_runs',
        'team_runs',
        'team_wicket',
        'team_balls',
        'balls_remaining',
        'wickets_remaining',
        'powerplay_balls_remaining',
        'middle_balls_remaining',
        'death_balls_remaining',
        'player_out', 
        'extra_type', 
        'wicket_kind', 
        'wicket_taken'
    ]
    if 'ball_id' in X.columns:
        cols_to_keep.append('ball_id')
    X = X[X['innings'] == 2][cols_to_keep]
    X = X.merge(X_1, on = ['match_id', 'team_balls'], how = 'inner')
    X = X.merge(match_list, on = 'match_id', how = 'inner')
    
    X['season_post_2008'] = X['year'] - 2008
    
    # remove issues where required runs or RRR could be negative, and cap the RRR at 50
    X = X[X['required_runs'] > 0]
    X['required_run_rate'] = X['required_run_rate'].replace(-np.inf, 0).clip(lower = 0)
    X['required_run_rate'] = X['required_run_rate'].replace(np.inf, 50).clip(upper = 50)
    
    X['current_NRR'] = (X['team_runs'] - X['first_innings_current_runs']) / X['team_balls'] * 6
    X['current_NRR'] = X['current_NRR'].replace([-np.inf, np.inf], 0).fillna(0)
    
    # features for second innings
    X['wicket_x_balls_remaining'] = X['wickets_remaining'] * X['balls_remaining']
    
    X['wicket_x_powerplay_balls'] = X['wickets_remaining'] * X['powerplay_balls_remaining']
    X['wicket_x_middle_balls'] = X['wickets_remaining'] * X['middle_balls_remaining']
    X['wicket_x_death_balls'] = X['wickets_remaining'] * X['death_balls_remaining']
    
    X['batting_team_prior_powerplay_NRR_x_powerplay_balls'] = X['batting_team_prior_powerplay_NRR'] * X['powerplay_balls_remaining']
    X['batting_team_prior_middle_NRR_x_middle_balls'] = X['batting_team_prior_middle_NRR'] * X['middle_balls_remaining']
    X['batting_team_prior_death_NRR_x_death_balls'] = X['batting_team_prior_death_NRR'] * X['death_balls_remaining']
    
    X['bowling_team_prior_powerplay_NRR_x_powerplay_balls'] = X['bowling_team_prior_powerplay_NRR'] * X['powerplay_balls_remaining']
    X['bowling_team_prior_middle_NRR_x_middle_balls'] = X['bowling_team_prior_middle_NRR'] * X['middle_balls_remaining']
    X['bowling_team_prior_death_NRR_x_death_balls'] = X['bowling_team_prior_death_NRR'] * X['death_balls_remaining']
    
    X['wicket_x_batting_team_runs_conceded_per_wicket'] = X['wickets_remaining'] * X['batting_team_prior_avg_runs_conceded_per_wicket']
    
    return X


def second_innings_weights(X, y):
    matches_won = X.groupby('match_id')['bowling_team_won'].first().sum()
    total_matches = len(X.groupby('match_id')['bowling_team_won'].first())
    pi_1 = matches_won / total_matches
    pi_0 = 1 - pi_1
    
    won_balls = X['bowling_team_won'].sum()
    total_balls = len(X)
    rho_1 = won_balls / total_balls
    rho_0 = 1 - rho_1
    
    w = (pi_1 / pi_0) / (rho_1 / rho_0)
    
    weights = np.where(y == 1, w, 1)
    return weights


def preprocess_cv(X: pd.DataFrame, y):
    X = X.reset_index(drop = True)
    cv_splits = []
    val_years = range(2016, 2022)
    
    for year in val_years:
        train_fold_idx = X.index[X['year'] < year].to_list()
        val_fold_idx = X.index[X['year'] == year].to_list()
        cv_splits.append((train_fold_idx, val_fold_idx))
    
    X_train = X[X['year'] <= 2021]
    y_train = X[X['year'] <= 2021][y].to_numpy()
    
    X_test = X[(X['year'] > 2021) & (X['year'] <= 2025)]
    y_test = X[(X['year'] > 2021) & (X['year'] <= 2025)][y].to_numpy()
    
    return X_train, y_train, X_test, y_test, cv_splits



import numpy as np
import pandas as pd
from datetime import datetime
import ipl


def get_baseline(X_df, model):
    # NOTE: this will need to be updated if the model is changed.
    dummy = X_df.iloc[[0]].copy()
    

    dummy['balls_remaining'] = ((dummy['balls_remaining'] + 1) // 6) * 6
    dummy['team_runs'] = 0
    dummy['team_wicket'] = 0
    dummy['team_balls'] = 0
    dummy['wickets_remaining'] = 10
    
    params = ipl.load_resource_params()
    dummy['resource'] = ipl.resource_function(dummy, params)

    return model.predict_proba(dummy)[0, 1]


def calculate_impact(X_df, model=None):
    X = X_df.copy()
    X = X.sort_values([
        'date',
        'match_id',
        'innings',
        'team_balls',
        'team_runs',
        'team_wicket'
    ])
    X['prob_bowling_diff'] = 100 * (
        X.groupby('match_id')['y_prob']
        .diff()
        .fillna(0)
    )
    if model is not None:
        baselines = pd.Series(
            {
                match_id: get_baseline(X.loc[group_index], model)
                for match_id, group_index in X.groupby('match_id', sort=False).groups.items()
            }
        )
        first_rows = X.groupby('match_id', sort=False).head(1).index
        X.loc[first_rows, 'prob_bowling_diff'] = 100 * (
            X.loc[first_rows, 'y_prob']
            - X.loc[first_rows, 'match_id'].map(baselines)
        )
    X['prob_batting_diff'] = - X['prob_bowling_diff']
    X['bowler_delta'] = np.where(
        X['innings'] == 1,
        X['prob_bowling_diff'],
        - X['prob_bowling_diff']
    )
    X['batter_delta'] = np.where(
        (X['extra_type'].isna()) | (X['batter_runs'] > 0),
        np.where(
            X['innings'] == 1,
            X['prob_batting_diff'],
            X['prob_bowling_diff']
        ),
        0
    )
    X.loc[X['wicket_taken'] == 1, ['batter_delta']] = 0

    return X


def aggregate_impact(X_impact):
    def _player_key_name(frame: pd.DataFrame, role: str) -> pd.DataFrame:
        out = frame.copy()
        name_col = role
        player_id_col = f'{role}_player_id'
        canonical_col = f'{role}_canonical_name'

        names = out[name_col].fillna('').astype(str).str.strip()
        if player_id_col in out.columns:
            player_id = pd.to_numeric(out[player_id_col], errors = 'coerce')
        else:
            player_id = pd.Series(np.nan, index = out.index)

        if canonical_col in out.columns:
            canonical = out[canonical_col].fillna('').astype(str).str.strip()
        else:
            canonical = pd.Series('', index = out.index)

        resolved = player_id.notna()
        player_id_text = player_id.astype('Int64').astype(str)
        out['_player_key'] = np.where(resolved, 'id:' + player_id_text, 'name:' + names)
        out['_player_name'] = np.where(resolved & (canonical != ''), canonical, names)
        out = out[out['_player_name'] != '']
        return out

    batter_impact = _player_key_name(X_impact, 'batter')
    batter_impact = batter_impact.groupby(['_player_key', '_player_name']).agg({
        'batter_delta': 'sum',
        'batting_team': 'first'
        }).reset_index().rename(columns = {'_player_key': '_PlayerKey', '_player_name': 'Player', 'batting_team': 'Team'})

    bowler_impact = _player_key_name(X_impact, 'bowler')
    bowler_impact = bowler_impact.groupby(['_player_key', '_player_name']).agg({
            'bowler_delta': 'sum',
            'bowling_team': 'first'
        }).reset_index().rename(columns = {'_player_key': '_PlayerKey', '_player_name': 'Player', 'bowling_team': 'Team'})

    total_impact = pd.merge(
        left = batter_impact, 
        right = bowler_impact,
        on = '_PlayerKey',
        how = 'outer',
        suffixes = ['_bat', '_bowl']
    )
    total_impact['Player'] = np.where(
        total_impact['Player_bat'].notna() & (total_impact['Player_bat'] != ''),
        total_impact['Player_bat'],
        total_impact['Player_bowl']
    )
    total_impact['batter_delta'] = total_impact['batter_delta'].fillna(0)
    total_impact['bowler_delta'] = total_impact['bowler_delta'].fillna(0)
    total_impact['impact_score'] = total_impact['batter_delta'] + total_impact['bowler_delta']
    
    total_impact = total_impact.rename(
        columns = {
            'batter_delta': 'Batting Impact',
            'bowler_delta': 'Bowling Impact',
            'impact_score': 'Total Impact'
        }
    )
    total_impact['Team'] = np.where(
        total_impact['Team_bat'].isna(),
        total_impact['Team_bowl'],
        total_impact['Team_bat']
    )
    total_impact = total_impact[total_impact['Total Impact'] != 0]
    total_impact = total_impact[['Player', 'Team', 'Batting Impact', 'Bowling Impact', 'Total Impact']]
    return total_impact

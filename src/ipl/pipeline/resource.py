import ipl
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from datetime import datetime

from scipy.optimize import curve_fit

RESOURCE_TRAIN_PATH = 'src/ipl/models/resource_train.joblib'
RESOURCE_PATH = 'src/ipl/models/resource.joblib'


def preprocess_resource(X_df: pd.DataFrame, match_list: pd.DataFrame, min_overs = 0, max_overs = 20):
    X = X_df.copy()
    
    cols_to_keep = [
        'match_id',
        'team_runs',
        'team_wicket',
        'team_balls',
        'innings',
        'balls_remaining',
        'wickets_remaining',
    ]
    
    X = X[X['innings'] == 1][cols_to_keep]
    if min_overs > 0:
        X = X[(X['team_balls'] > (min_overs * 6)) & (X['team_balls'] <= (max_overs * 6))]
    else: 
        X = X[X['team_balls'] <= (max_overs * 6)]
    
    X = X.merge(match_list, on = 'match_id', how = 'inner')
    X['runs_delta'] = X['first_innings_runs'] - X['team_runs']
    
    cols_to_keep = [
        'match_id',
        'team_runs',
        'team_wicket',
        'team_balls',
        'innings',
        'balls_remaining',
        'wickets_remaining',
        'runs_delta'
    ]
    return X


# train on training set only
def resource_train_partial(path = RESOURCE_TRAIN_PATH):
    def R_w(x, R_0, beta):
        return R_0 * (1 - np.exp(-beta * x))
    
    df = ipl.load_data()
    match_list = ipl.match_list(df)
    matches_with_prior = ipl.prior_match_stats(match_list)
    
    X = preprocess_resource(df, matches_with_prior, min_overs=0, max_overs=20)
    X_train, y_train, X_test, y_test, cv_splits = ipl.preprocess_cv(X, 'runs_delta')
    
    params = [np.array([1, 0])]
    for w in range(1, 11):
        Xw = X_train[X_train['wickets_remaining'] == w]
        
        popt, _ = curve_fit(
            R_w,
            Xw['balls_remaining'].to_numpy(),
            Xw['runs_delta'].to_numpy(),
            p0 = [150, 0.01],
            bounds = ([0, 0], [300, 1])
        )
        params.append(popt)
    
    params = np.array(params)
    joblib.dump(params, path)
    return


def load_resource_training_params(path = RESOURCE_TRAIN_PATH):
    return joblib.load(path)

# re-train on full training + validation set
def resource_train_full(path = RESOURCE_PATH):
    def R_w(x, R_0, beta):
        return R_0 * (1 - np.exp(-beta * x))
    
    df = ipl.load_data()
    match_list = ipl.match_list(df)
    matches_with_prior = ipl.prior_match_stats(match_list)
    
    X = preprocess_resource(df, matches_with_prior, min_overs=0, max_overs=20)
    
    params = [np.array([1, 0])]
    for w in range(1, 11):
        Xw = X[X['wickets_remaining'] == w]
        
        popt, _ = curve_fit(
            R_w,
            Xw['balls_remaining'].to_numpy(),
            Xw['runs_delta'].to_numpy(),
            p0 = [150, 0.01],
            bounds = ([0, 0], [300, 1])
        )
        params.append(popt)
    
    params = np.array(params)
    joblib.dump(params, path)
    return


def load_resource_params(path = RESOURCE_PATH):
    return joblib.load(path)


def resource_function(df, params):
    w = df['wickets_remaining'].to_numpy(dtype=int)
    b = df['balls_remaining'].to_numpy(dtype=float)
    
    R_0 = params[w, 0]
    beta = params[w, 1]
    
    return R_0 * (1 - np.exp(-beta * b))


def resource_one_over(df, params):
    # shows the number of runs projected in the next 6 legal balls
    w = df['wickets_remaining'].to_numpy(dtype=int)
    
    R_0 = params[w, 0]
    beta = params[w, 1]
    
    return R_0 * (1 - np.exp(-beta * 6))
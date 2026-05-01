import ipl
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

from scipy.optimize import curve_fit

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


FIRST_INNINGS_GLM_TRAIN_PATH = 'src/ipl/models/first_innings_glm_train.joblib'
FIRST_INNINGS_GLM_PATH = 'src/ipl/models/first_innings_glm.joblib'


def first_innings_glm_train_partial(path = FIRST_INNINGS_GLM_TRAIN_PATH):
    df = ipl.load_data()
    match_list = ipl.match_list(df)
    matches_with_prior = ipl.prior_match_stats(match_list)
    
    X = ipl.preprocess_first_innings(df, matches_with_prior, full = False, min_overs = 0, max_overs = 20)
    X_train, y_train, X_test, y_test, cv_splits = ipl.preprocess_cv(X, 'bowling_team_won')

    features = [
        'team_runs',
        'playoff_match', 
        'season_post_2008',
        'bowling_team_won_toss',
        'batting_team_prior_powerplay_NRR',
        'batting_team_prior_middle_NRR',
        'batting_team_prior_death_NRR',
        'bowling_team_prior_powerplay_NRR',
        'bowling_team_prior_middle_NRR',
        'bowling_team_prior_death_NRR',
        'batting_team_prior_powerplay_NRR_x_powerplay_balls',
        'batting_team_prior_middle_NRR_x_middle_balls',
        'batting_team_prior_death_NRR_x_death_balls',
        'bowling_team_prior_powerplay_NRR_x_powerplay_balls',
        'bowling_team_prior_middle_NRR_x_middle_balls',
        'bowling_team_prior_death_NRR_x_death_balls',
        'resource'
    ]

    cat_features = [
        col for col in features
        if is_string_dtype(X[col])
    ]

    binary_features = [col for col in features if X[col].nunique() <= 2]

    num_features = [
        col for col in features
        if (col not in cat_features)
        and (col not in binary_features)
    ]

    first_innings_num = Pipeline(
        steps = [
            ('impute', SimpleImputer(strategy = 'median')),
            ('scale', StandardScaler())
        ]
    )

    first_innings_dummy = Pipeline(
        steps = [
            ('dummy', OneHotEncoder())
        ]
    )

    first_innings_other = Pipeline(
        steps = [
            ('impute', SimpleImputer(strategy = 'median'))
        ]
    )

    first_innings_scaler = ColumnTransformer(
        transformers = [
            ('transform', first_innings_num, num_features),
            ('dummy', first_innings_dummy, cat_features),
            ('no_transform', first_innings_other, binary_features)
        ],
        remainder = 'drop'
    )

    first_innings_glm = Pipeline(
        steps = [
            ('preprocess', first_innings_scaler),
            ('model', LogisticRegressionCV(
                Cs = 20,
                l1_ratios = np.linspace(0, 1, 21),
                cv = cv_splits,
                solver  = 'saga',
                scoring = 'neg_log_loss',
                max_iter = 5000,
                random_state = 33,
                n_jobs = -1,
                use_legacy_attributes = True
            ))
        ]
    )

    first_innings_glm.fit(X_train, y_train)
    
    Path('src/ipl/models').mkdir(exist_ok = True)
    joblib.dump(first_innings_glm, path)
    
    return


def first_innings_glm_train_load(path = FIRST_INNINGS_GLM_TRAIN_PATH):
    first_innings_glm = joblib.load(path)
    return first_innings_glm


def first_innings_glm_train_full(path = FIRST_INNINGS_GLM_PATH):
    df = ipl.load_data()
    match_list = ipl.match_list(df)
    matches_with_prior = ipl.prior_match_stats(match_list)
    
    X = ipl.preprocess_first_innings(df, matches_with_prior, full = True, min_overs = 0, max_overs = 20)
    y = X['bowling_team_won']

    first_innings_glm_train = first_innings_glm_train_load()
    
    best_C = first_innings_glm_train.named_steps['model'].C_[0]
    best_l1_ratio = first_innings_glm_train.named_steps['model'].l1_ratio_[0]

    features = [
        'team_runs',
        'playoff_match', 
        'season_post_2008',
        'bowling_team_won_toss',
        'batting_team_prior_powerplay_NRR',
        'batting_team_prior_middle_NRR',
        'batting_team_prior_death_NRR',
        'bowling_team_prior_powerplay_NRR',
        'bowling_team_prior_middle_NRR',
        'bowling_team_prior_death_NRR',
        'batting_team_prior_powerplay_NRR_x_powerplay_balls',
        'batting_team_prior_middle_NRR_x_middle_balls',
        'batting_team_prior_death_NRR_x_death_balls',
        'bowling_team_prior_powerplay_NRR_x_powerplay_balls',
        'bowling_team_prior_middle_NRR_x_middle_balls',
        'bowling_team_prior_death_NRR_x_death_balls',
        'resource'
    ]

    cat_features = [
        col for col in features
        if is_string_dtype(X[col])
    ]

    binary_features = [col for col in features if X[col].nunique() <= 2]

    num_features = [
        col for col in features
        if (col not in cat_features)
        and (col not in binary_features)
    ]

    first_innings_num = Pipeline(
        steps = [
            ('impute', SimpleImputer(strategy = 'median')),
            ('scale', StandardScaler())
        ]
    )

    first_innings_dummy = Pipeline(
        steps = [
            ('dummy', OneHotEncoder())
        ]
    )

    first_innings_other = Pipeline(
        steps = [
            ('impute', SimpleImputer(strategy = 'median'))
        ]
    )

    first_innings_scaler = ColumnTransformer(
        transformers = [
            ('transform', first_innings_num, num_features),
            ('dummy', first_innings_dummy, cat_features),
            ('no_transform', first_innings_other, binary_features)
        ],
        remainder = 'drop'
    )

    first_innings_glm = Pipeline(
        steps = [
            ('preprocess', first_innings_scaler),
            ('model', LogisticRegression(
                C = best_C,
                l1_ratio = best_l1_ratio,
                solver  = 'saga',
                max_iter = 5000,
                random_state = 33,
                n_jobs = -1,
            ))
        ]
    )

    first_innings_glm.fit(X, y)
    
    Path('src/ipl/models').mkdir(exist_ok = True)
    joblib.dump(first_innings_glm, path)
    
    return


def first_innings_glm_load(path = FIRST_INNINGS_GLM_PATH):
    first_innings_glm = joblib.load(path)
    return first_innings_glm

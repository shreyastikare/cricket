import ipl
import joblib
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

SECOND_INNINGS_GLM_TRAIN_PATH = 'src/ipl/models/second_innings_glm_train.joblib'
SECOND_INNINGS_GLM_PATH = 'src/ipl/models/second_innings_glm.joblib'


def second_innings_glm_train_partial(path = SECOND_INNINGS_GLM_TRAIN_PATH):
    df = ipl.load_data()
    match_list = ipl.match_list(df)
    matches_with_prior = ipl.prior_match_stats(match_list)
    
    X = ipl.preprocess_second_innings(df, matches_with_prior)
    X_train, y_train, X_test, y_test, cv_splits = ipl.preprocess_cv(X, 'bowling_team_won')
    weights = ipl.second_innings_weights(X_train, y_train)
    
    features = [
        'required_runs',
        'required_run_rate',
        'playoff_match', 
        'season_post_2008',
        'first_innings_current_runs',
        'first_innings_current_wickets_remaining',
        'current_NRR',
        'wicket_x_balls_remaining',
        'wickets_remaining',
        'powerplay_balls_remaining',
        'middle_balls_remaining',
        'death_balls_remaining',
        'wicket_x_powerplay_balls',
        'wicket_x_middle_balls',
        'wicket_x_death_balls',
        'batting_team_prior_powerplay_NRR_x_powerplay_balls',
        'batting_team_prior_middle_NRR_x_middle_balls',
        'batting_team_prior_death_NRR_x_death_balls',
        'bowling_team_prior_powerplay_NRR_x_powerplay_balls',
        'bowling_team_prior_middle_NRR_x_middle_balls',
        'bowling_team_prior_death_NRR_x_death_balls',
        'wicket_x_batting_team_runs_conceded_per_wicket',
        'batting_team_prior_avg_runs_conceded_per_wicket'
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
    
    second_innings_num = Pipeline(
        steps = [
            ('impute', SimpleImputer(strategy = 'median')),
            ('scale', StandardScaler())
        ]
    )
    
    second_innings_dummy = Pipeline(
        steps = [
            ('dummy', OneHotEncoder())
        ]
    )
    
    second_innings_other = Pipeline(
        steps = [
            ('impute', SimpleImputer(strategy = 'median'))
        ]
    )
    
    second_innings_scaler = ColumnTransformer(
        transformers = [
            ('transform', second_innings_num, num_features),
            ('dummy', second_innings_dummy, cat_features),
            ('no_transform', second_innings_other, binary_features)
        ],
        remainder = 'drop'
    )
    
    second_innings_glm = Pipeline(
        steps = [
            ('preprocess', second_innings_scaler),
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
    
    second_innings_glm.fit(X_train, y_train, model__sample_weight = weights)
    
    Path('src/ipl/models').mkdir(exist_ok = True)
    joblib.dump(second_innings_glm, path)
    
    return


def second_innings_glm_train_load(path = SECOND_INNINGS_GLM_TRAIN_PATH):
    second_innings_glm = joblib.load(path)
    return second_innings_glm


# re-train the model on the full training + validation set
def second_innings_glm_train_full(path = SECOND_INNINGS_GLM_PATH):
    df = ipl.load_data()
    match_list = ipl.match_list(df)
    matches_with_prior = ipl.prior_match_stats(match_list)
    
    X = ipl.preprocess_second_innings(df, matches_with_prior)
    y = X['bowling_team_won']
    weights = ipl.second_innings_weights(X, y)
    
    second_innings_glm_train = second_innings_glm_train_load()
    
    best_C = second_innings_glm_train.named_steps['model'].C_[0]
    best_l1_ratio = second_innings_glm_train.named_steps['model'].l1_ratio_[0]
    
    features = [
        'required_runs',
        'required_run_rate',
        'playoff_match', 
        'season_post_2008',
        'first_innings_current_runs',
        'first_innings_current_wickets_remaining',
        'current_NRR',
        'wicket_x_balls_remaining',
        'wickets_remaining',
        'powerplay_balls_remaining',
        'middle_balls_remaining',
        'death_balls_remaining',
        'wicket_x_powerplay_balls',
        'wicket_x_middle_balls',
        'wicket_x_death_balls',
        'batting_team_prior_powerplay_NRR_x_powerplay_balls',
        'batting_team_prior_middle_NRR_x_middle_balls',
        'batting_team_prior_death_NRR_x_death_balls',
        'bowling_team_prior_powerplay_NRR_x_powerplay_balls',
        'bowling_team_prior_middle_NRR_x_middle_balls',
        'bowling_team_prior_death_NRR_x_death_balls',
        'wicket_x_batting_team_runs_conceded_per_wicket',
        'batting_team_prior_avg_runs_conceded_per_wicket'
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
    
    second_innings_num = Pipeline(
        steps = [
            ('impute', SimpleImputer(strategy = 'median')),
            ('scale', StandardScaler())
        ]
    )
    
    second_innings_dummy = Pipeline(
        steps = [
            ('dummy', OneHotEncoder())
        ]
    )
    
    second_innings_other = Pipeline(
        steps = [
            ('impute', SimpleImputer(strategy = 'median'))
        ]
    )
    
    second_innings_scaler = ColumnTransformer(
        transformers = [
            ('transform', second_innings_num, num_features),
            ('dummy', second_innings_dummy, cat_features),
            ('no_transform', second_innings_other, binary_features)
        ],
        remainder = 'drop'
    )
    
    second_innings_glm = Pipeline(
        steps = [
            ('preprocess', second_innings_scaler),
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
    
    second_innings_glm.fit(X, y, model__sample_weight = weights)
    
    Path('src/ipl/models').mkdir(exist_ok = True)
    joblib.dump(second_innings_glm, path)
    
    return


def second_innings_glm_load(path = SECOND_INNINGS_GLM_PATH):
    second_innings_glm = joblib.load(path)
    return second_innings_glm



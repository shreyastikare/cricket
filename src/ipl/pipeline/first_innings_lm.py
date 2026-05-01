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
from sklearn.linear_model import ElasticNetCV

FIRST_INNINGS_LM_PATH = 'src/ipl/models/first_innings_lm.joblib'


def first_innings_lm_train(path = FIRST_INNINGS_LM_PATH):
    df = ipl.load_data()
    match_list = ipl.match_list(df)
    matches_with_prior = ipl.prior_match_stats(match_list)
    
    X_first = ipl.preprocess_first_innings(df, matches_with_prior, min_overs = 6, max_overs = 20)
    X_train, y_train, X_test, y_test, cv_splits= ipl.preprocess_cv(X_first, 'runs_delta')
    
    features = [
        'playoff_match',
        'season_post_2008',
        'run_rate_x_middle_balls',
        'run_rate_x_death_balls',
        'wickets_remaining',
        'middle_balls_remaining', 
        'death_balls_remaining', 
        'wicket_x_middle_balls',
        'wicket_x_death_balls',
        'wicket_x_run_rate_x_middle_balls',
        'wicket_x_run_rate_x_death_balls',
        # 'batting_team_prior_middle_NRR_x_middle_balls',
        # 'batting_team_prior_death_NRR_x_death_balls',
        # 'bowling_team_prior_middle_NRR_x_middle_balls',
        # 'bowling_team_prior_death_NRR_x_death_balls',
    ]
    
    cat_features = [
        col for col in features
        if is_string_dtype(X_first[col])
    ]
    
    binary_features = [col for col in features if X_first[col].nunique() <= 2]
    
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
            ('no_transform', first_innings_other, binary_features),
        ],
        remainder = 'drop'
    )
    
    first_innings_lm = Pipeline(
        steps = [
            ('preprocess', first_innings_scaler),
            ('model', ElasticNetCV(
                l1_ratio = np.linspace(0.05, 1, 20),
                alphas = [0.1, 0.5, 1, 5, 10, 50, 100, 200, 500],
                cv = cv_splits,
                )
            )
        ]
    )
    
    first_innings_lm.fit(X_train, y_train)
    
    Path('src/ipl/models').mkdir(exist_ok = True)
    joblib.dump(first_innings_lm, path)
    
    return


def first_innings_lm_load(path = FIRST_INNINGS_LM_PATH):
    first_innings_lm = joblib.load(path)
    return first_innings_lm

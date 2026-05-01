import ipl

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
import random

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV, LogisticRegressionCV

import sqlite3
import joblib

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go


class Team():
    def __init__(self, id):
        return
    
    def team_stats(self, id):
        return
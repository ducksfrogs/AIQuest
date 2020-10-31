import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.preprocessing import LabelEncoder
from itertools import product


data = pd.read_pickle('data.pkl')

data = data[[
    'month_idx',
    'shop_id',
    'item_id',
    'item_cnt_month',
    'cat_id',
    'type_code',
    'item_cnt_month_lag_1',
    'item_cnt_month_lag_2',
    'item_cnt_month_lag_3',
    'item_cnt_month_lag_6',
    'item_cnt_month_lag_12',
    'delta_price_lag',
    'delta_revenue_lag_1',
    'month',
    'days',
    'item_last_sale',
    'item_shop_first_sale',
    'item_first_sale'
]]

X_train = data[data.month_idx < 22].drop(['item_cnt_month'], axis=1)
y_train = data[data.month_idx < 22]['item_cnt_month']
X_valid = data[data.month_idx == 22].drop(['item_cnt_month'], axis=1)
y_valid  = data[data.month_idx == 22]['item_cnt_month']
X_test =

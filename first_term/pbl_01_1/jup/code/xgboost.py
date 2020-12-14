import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from xgboost import XGBRegressor
from xgboost import plot_importance

import gc

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

X_train = data[data.month_idx < 21].drop(['item_cnt_month'], axis=1)
y_train = data[data.month_idx < 21]['item_cnt_month']
X_valid = data[data.month_idx == 21].drop(['item_cnt_month'], axis=1)
y_valid  = data[data.month_idx == 21]['item_cnt_month']
X_test = data[data.month_idx ==22].drop(['item_cnt_month'], axis=1)

ts = time.time()

model = XGBRegressor(
    max_depth=8,
    n_estimators = 1000,
    min_child_weight = 300,
    colsample_bytree=0.8,
    eta=0.3,
    seed=42
)

model.fit(X_train, y_train,eval_metric="rmse", eval_set=[(X_train, y_train),(X_valid, y_valid)],
    verbose=True,
    early_stopping_rounds=10)

time.time() - ts

y_pred = model.predict(X_valid).clip(0,20)
y_test = model.predict(X_test).clip(0,20)

submission = pd.DataFrame({
    "ID": test.index,
    "item_cnt_month": y_test
})

submission.to_csv('xgb_submission.csv', index=False )

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.preprocessing import LabelEncoder
from itertools import product

import pickle


sales_train = pd.read_csv('../data/sales_history.csv')
items = pd.read_csv('../data/item_categories.csv')
cat_name = pd.read_csv('../data/category_names.csv')
test = pd.read_csv('../data/test.csv', index_col=0)
submit = pd.read_csv(',,/data/sample_submission.csv')


#index change


train.columns = 'date shop_id item_id price item_cnt_day'.split()
test.columns = 'idx item_id shop_id'.split()
cat_name.columns = 'cat_id cat_name'.split()
item_cat.columns = 'item_id cat_id'.split()

plt.figure(figsize=(10,4))
sns.boxplot(x=train.item_cnt_day)

plt.figure(figsize=(10,4))
sns.boxplot(x=train.price)

train = train[train.price <200000]
train = train[train.item_cnt_day < 120]

train.loc[train.item_cnt_day < 0, 'item_cnt_day'] = 0

def add_monthly_idx(x):
    if x >= '2018-01-01' and x < '2018-02-01':
        return 0
    elif x >= '2018-02-01' and x < '2018-03-01':
        return 1
    elif x >= '2018-03-01' and x < '2018-04-01':
        return 2
    elif x >= '2018-04-01' and x < '2018-05-01':
        return 3
    elif x >= '2018-05-01' and x < '2018-06-01':
        return 4
    elif x >= '2018-06-01' and x < '2018-07-01':
        return 5
    elif x >= '2018-07-01' and x < '2018-08-01':
        return 6
    elif x >= '2018-08-01' and x < '2018-09-01':
        return 7
    elif x >= '2018-09-01' and x < '2018-10-01':
        return 8
    elif x >= '2018-10-01' and x < '2018-11-01':
        return 9
    elif x >= '2018-11-01' and x < '2018-12-01':
        return 10
    elif x >= '2018-12-01' and x < '2019-01-01':
        return 11
    elif x >= '2019-01-01' and x < '2019-02-01':
        return 12
    elif x >= '2019-02-01' and x < '2019-03-01':
        return 13
    elif x >= '2019-03-01' and x < '2019-04-01':
        return 14
    elif x >= '2019-04-01' and x < '2019-05-01':
        return 15
    elif x >= '2019-05-01' and x < '2019-06-01':
        return 16
    elif x >= '2019-06-01' and x < '2019-07-01':
        return 17
    elif x >= '2019-07-01' and x < '2019-08-01':
        return 18
    elif x >= '2019-08-01' and x < '2019-09-01':
        return 19
    elif x >= '2019-09-01' and x < '2019-10-01':
        return 20
    elif x >= '2019-10-01' and x < '2019-11-01':
        return 21
    else:
        return 22

train['month_idx'] = train['date'].apply(add_monthly_idx)

#Feature engineering

cat_name['sprit'] = cat_name['cat_name'].str.split('-')
cat_name['type'] = cat_name['sprit'].map(lambda x: x[0].strip())
cat_name['type-code'] = LabelEncoder().fit_transform(cat_name['type'])
cat_name = cat_name[['cat_id','type_code']]

grid = []
for block_num in

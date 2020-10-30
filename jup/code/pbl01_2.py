import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.preprocessing import LabelEncoder
from itertools import product


train = pd.read_csv('../data/sales_history.csv')
item_cat = pd.read_csv('../data/item_categories.csv')
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

matrix = []
cols = ['month_idx','shop_id', 'item_id']

for i in range(22):
    sales = train[train.date==i]
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int64'))

matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['month_idx'] = matrix['month_idx'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int64)
matrix.sort_values(cols, inplace=True)

train['revenue'] = train['price'] * train['item_cnt_day']


group = train.groupby(['month_idx', 'shop_id', 'item_id']).agg({'item_cnt_day':['sum']})
group.columns= ['item_cnt_month']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
                                .clip(0,20)
                                .astype(np.float16))

#Test set

test['month_idx'] = 22
test['month_idx'] = test['month_idx'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int64)

matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0,inplace=True)

matrix = pd.merge(matrix, item_cat, on=['item_id'], how='left')
matrix = pd.merge(matrix, cat_name, on=['cat_id'], how='left')

#Target lags

def lag_feature(df, lags, col):
    tmp = df[['month_idx', 'shop_id','item_id', col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['month_idx','shop_id', 'item_id', col+'_lag_'+str(i)]
        shifted['month_idx'] += i
        df = pd.merge(df, shifted, on=['month_idx','shop_id','item_id'], how='left')
    return df

matrix = lag_feature(matrix, [1,2,3,6,12], 'item_cnt_month')

#Mean encorded feature


group = matrix.groupby(['month_idx']).agg({'item_cnt_month':['mean']})
group.columns = ['date_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['month_idx'], how='left')
matrix['date_avg_item_cnt'] = matrix['date_avg_item_cnt'].astype(np.float16)
ma

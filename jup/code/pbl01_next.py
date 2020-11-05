import numpy as np
import pandas as pd


sales_data = pd.read_csv("../data/sales_history.csv")
item_cat = pd.read_csv("../data/item_categories.csv")
cats = pd.read_csv('../data/category_names.csv')
sample_submission =pd.read_csv('../data/sample_submission.csv')
test_data =pd.read_csv('../data/test.csv')

sales_data.columns = 'date shop_id item_id price item_cnt_day'.split()
item_cat.columns = 'item_id cat_id'.split()
cats.columns = 'cat_id cat_name'.split()
test.columns = 'idx item_id shop_id'.split()

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

sales_data['date_block_num'] = sales_data['date'].apply(add_monthly_idx)

def basic_eda(df):
    print("-------TOP 5 RECORDS")
    print(df.head(5))
    print('------INFO----------')
    print(df.info())
    print('------Descrive------')
    print(df.describe())
    print('-----Columns--------')
    print(df.columns)
    print('-----Data type -----')
    print(df.dtypes)
    print('-----Missing values -----')
    print(df.isnull().sum())
    print('----NULL Values--------')
    print(df.isna().sum())
    print('-----Shape of Data------')
    print(df.shape)


print("=================Sales Data ==================Data tData typeype")
basic_eda(sales_data)
print("=================Test Data ==================")
basic_eda(test_data)
print("=================Iten Categories ==================")
basic_eda(item_cat)
print("=================Items ==================")
basic_eda(cats)
print("=================Sample Submmision ==================")
basic_eda(sample_submission)



dataset  = sales_data.pivot_table(index=['shop_id','item_id',],values=['item_cnt_day'], columns=['date_block_num'], fill_value=0, aggfunc='sum')

dataset.reset_index(inplace=True)

dataset = pd.merge(test_data, dataset, on=['item_id','shop_id'],how='left')

dataset.fillna(0, inplace=True)

dataset.drop(['shop_id','item_id','ID'],inplace=True, axis=1)

X_train = np.expand_dims(dataset.values[:,:-1], axis=2)
y_train = dataset.values[:, -1:]

X_test = np.expand_dims(dataset.values[:,1:], axis=2)

print(X_train.shape, y_train.shape, X_test.shape)

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(units=64,input_shape=(21,1)))
model.add(Dropout(0.4))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mean_squared_error'])

model.fit(X_train, y_train, batch_size=4096, epochs=10)

submission_pfs = model.predict(X_test)
submission = pd.DataFrame({'ID':test_data['ID'], 'item_cnt_month':submission_pfs.ravel()})
submission.to_csv('submission_pfs.csv', index=False)

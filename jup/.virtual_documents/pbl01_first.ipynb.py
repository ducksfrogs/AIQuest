import pandas as pd
import numpy as np
get_ipython().run_line_magic("matplotlib", " inline")


df = pd.read_csv('../data/sales_history.csv')
test = pd.read_csv('../data/test.csv',index_col=0)


df.head()


df.tail()


test.columns = 'id shop_id'.split()


df.columns = 'date shop_id id price quantity'.split()


df.columns


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



df['month_idx'] =  df['date'].apply(add_monthly_idx)


df.info()


df.tail()


df = df[df['id'].isin(test['id'])]


monthly_sold = df.groupby(['month_idx','shop_id','id'],as_index=False).agg({'price': 'mean', 'quantity': 'sum'})


monthly_sold.iloc[:100]


monthly_sold.tail()


monthly_sold[22:44]


monthly_sold['price'].hist()


month_cnt('2018-02-01')


df.index


df.head()




import pandas as pd
import numpy as np
get_ipython().run_line_magic("matplotlib", " inline")


df = pd.read_csv('../data/sales_history.csv', index_col='日付', parse_dates=True)
test = pd.read_csv('../data/test.csv',index_col=0)


df.head()


df.tail()


test.columns = 'id shop_id'.split()


df.columns = 'shop_id id price quantity'.split()


df.columns


df.info()


df = df[df['id'].isin(test['id'])]


monthly_sold = df.groupby(['shop_id','id']).resample(rule="M").agg({'price': 'mean', 'quantity': 'sum'})


monthly_sold.info()


monthly_sold[22:44]


def month_cnt(date):
    for i in date:
        if i >= '2018-01-01' and i < '2018-02-01':
            return 0
        elif i >= '2018-02-01' and i < '2018-03-01':
            return 1
        else:
            return 4
    


month_cnt('2018-02-01')


df.index


df.head()




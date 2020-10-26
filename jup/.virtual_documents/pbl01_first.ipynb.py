import pandas as pd
import numpy as np
get_ipython().run_line_magic("matplotlib", " inline")


df = pd.read_csv('../data/sales_history.csv', index_col='日付', parse_dates=True)
test = pd.read_csv('../data/test.csv',index_col=0)


df.head()


test.columns = 'id shop_id'.split()


df.columns = 'shop_id id price quantity'.split()


df.columns


df.info()


df = df[df['id'].isin(test['id'])]


df.groupby(['shop_id','id']).sum()


df.month


df.year




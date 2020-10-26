import pandas as pd
import numpy as np
get_ipython().run_line_magic("matplotlib", " inline")


df = pd.DataFrame([[4, 9]] * 3, columns=['A', 'B'])


df


df['C'] = df.apply(np.sum, axis=1)


df


def add(x):
        if x >=1:
            return 2 
        else:
            return 4


lI = [15, 8, 10]


df["mmm"] = df['A'].apply(add)


df




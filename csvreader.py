import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')

x = df['Age']
x = x.dropna()
x = x.values
print(x)
print('最大値:{}, 最小値:{}, 平均:{}, 中央値:{:.2f}, 標準偏差:{:.2f}'.format(
    np.max(x),
    np.min(x),
    np.mean(x),
    np.median(x),
    np.std(x)))



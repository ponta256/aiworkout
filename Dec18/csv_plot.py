
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import statistics

# df = pd.read_csv('HR_0.csv')
df = pd.read_csv('HR_1.csv')
# df = pd.read_csv('out.csv')

age = df['Age in Yrs.']

# ヒストグラム表示
plt.xlim([0, 80])
plt.hist(age, bins=40)
plt.show()


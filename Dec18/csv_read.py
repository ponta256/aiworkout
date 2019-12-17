
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics

df = pd.read_csv('HR_0.csv')
# df = pd.read_csv('HR_1.csv')
# df = pd.read_csv('out.csv')

# データ数 (行数)
num = len(df)
print('データ数', num)

# 特定の列を取得
age = df['Age in Yrs.']

min =  min(age)
max =  max(age)
mean = statistics.mean(age)     # 平均
median = statistics.median(age)	# 中央値
stdev = statistics.stdev(age)   # 標準偏差

print('最小値', min)
print('最大値', max)
print('平均', mean)
print('中央値', median)
print('標準偏差', stdev)

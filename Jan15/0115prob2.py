from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
import numpy as np


boston = load_boston()

x = boston['data']  # 物件の情報
y = boston['target']  # 家賃

x = x[:, [4]] # NOx濃度

plt.scatter(x, y, c='gray', alpha=0.5) # データプロット
plt.show()

model = LinearRegression()
model.fit(x, y)

a = model.coef_
b = model.intercept_

print('a={}, b={}'.format(a, b))

plt.scatter(x, y, c='gray', alpha=0.5) # データプロット
lreg_y = a * x + b # clf.predict(X_RM)の出力値と同じ
plt.plot(x, lreg_y, 'r') # 回帰直線プロット
plt.show()

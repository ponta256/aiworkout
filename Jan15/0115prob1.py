from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pickle
import numpy as np


boston = load_boston()

x = boston['data']  # 物件の情報
y = boston['target']  # 家賃

x = x[:, [4]] # NOx濃度

print(x,y)
model = LinearRegression()
model.fit(x, y)

a = model.coef_
b = model.intercept_

print('a={}, b={}'.format(a, b))

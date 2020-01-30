from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

car = pd.read_csv('imports-85.csv', header=None)

length = car[10].values.tolist()
width = car[11].values.tolist()
height = car[12].values.tolist()
price = car[25].values.tolist()

x = []
for l, w, h in zip(length, width, height):
    x.append([l,w,h])

# priceには'?'が含まれるのでクレンジングする
y = [int(p) if p != '?' else 0 for p in price ]

model = LinearRegression()
model.fit(x, y)

a = model.coef_
b = model.intercept_

print('a={}, b={}'.format(a, b))


from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

car = pd.read_csv('imports-85.csv', header=None)

engine_size = car[16].values.tolist()
price = car[25].values.tolist()

x = []
for v in engine_size:
    x.append([v])

y = []
for v in price:
    if v.isdecimal():
        y.append(int(v))    
    else:
        y.append(0)


plt.scatter(x, y, c='gray', alpha=0.5) 
plt.show()
        
model = LinearRegression()
model.fit(x, y)

a = model.coef_
b = model.intercept_

print('a={}, b={}'.format(a, b))

plt.scatter(x, y, c='gray', alpha=0.5)
lreg_y = a * x + b 
plt.plot(x, lreg_y, 'r')
plt.show()

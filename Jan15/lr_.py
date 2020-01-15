import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

boston = load_boston()

x = boston['data']  # 物件の情報
x = x[:, 5]
y = boston['target']  # 家賃

plt.scatter(x, y, c='gray', alpha=0.5) # データプロット
plt.show()

alpha = 0.04 
itera = 10000
cost = np.zeros(itera)
b = 1 #bの初期値
a = 1 #aの初期値
m = len(y)

for i in range(itera):
    cost[i] = (1/(2*m))*np.sum(np.square(b+a*x-y))
    b = b - alpha*(1/(m))*np.sum(b+a*x-y)
    a = a - alpha*(1/(m))*np.sum((b+a*x-y)*x)
    
print(b, a)

plt.xkcd()
plt.scatter(x, y, c='gray', alpha=0.5) # データプロット
lreg_y = a * x + b
plt.plot(x, lreg_y, 'r') # 回帰直線プロット
plt.show()

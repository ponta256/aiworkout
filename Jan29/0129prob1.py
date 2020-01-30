
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
import pandas as pd

def main():
    car = pd.read_csv('imports-85.csv', header=None)
    engine_size = car[16].values.tolist()    
    length = car[10].values.tolist()
    width = car[11].values.tolist()
    height = car[12].values.tolist()

    price = car[25].values.tolist()
    price = [int(p) if p != '?' else 0 for p in price ]

    features = []
    for e, l, w, h in zip(engine_size, length, width, height):
        features.append([e,l,w,h])
    
    pca = PCA(n_components=2)
    pca.fit(features)

    transformed = pca.fit_transform(features)

    carclass = ['H' if price[i]>20000 else 'M' if price[i]>10000 else 'L' for i in range(len(price))]
    carclass = np.array(carclass)

    # クルマを値段によって三分類する
    for c in ['H','M','L']:    
        plt.scatter(transformed[carclass==c, 0], transformed[carclass==c, 1], label=c)

    plt.title('principal component')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    plt.legend(loc='upper left')

    print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
    print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))

    plt.show()


if __name__ == '__main__':
    main()

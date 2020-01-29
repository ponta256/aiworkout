#
# https://blog.amedama.jp/entry/2017/04/02/130530 から引用
#

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets

def main():
    dataset = datasets.load_iris()

    features = dataset.data
    targets = dataset.target

    # 主成分分析する
    pca = PCA(n_components=2)
    pca.fit(features)

    # 分析結果を元にデータセットを主成分に変換する
    transformed = pca.fit_transform(features)

    # 主成分をプロットする
    for label in np.unique(targets):
        plt.scatter(transformed[targets == label, 0],
                    transformed[targets == label, 1])
    plt.title('principal component')
    plt.xlabel('pc1')
    plt.ylabel('pc2')

    print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
    print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))

    plt.show()


if __name__ == '__main__':
    main()

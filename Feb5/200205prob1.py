from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

# lr = LogisticRegression(solver='liblinear', multi_class='auto')
lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=10000)
predict = lr.fit(X_train, y_train)

score = predict.score(X_test, y_test)
print('Test set score: {}'.format(score))

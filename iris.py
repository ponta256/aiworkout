from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


iris = load_iris()
x = iris.data
y = iris.target

#define a function that prints the iris' classification based on the algorithm's output
def classifyiris(z):
    if z[0] == 0:
        print("The iris is setosa.\n")
    elif z[0] == 1:
        print("The iris is versicolor.\n")
    else:
        print("The iris is virginica.\n")

#Using the K Nearest Neighbor Algorithm
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(x,y)
z = knn.predict([[3,5,4,2]])
print("Using the k nearest neighbor algorithm =", knn.predict([[3,5,4,2]]))
classifyiris(z)


#Using the Logistic Regression Algorithm
lr = LogisticRegression(multi_class='auto', solver='liblinear')
lr.fit(x,y)
z = lr.predict([[3,5,4,2]])
print("Using the Logistic Regression algorithm =", lr.predict([[3,5,4,2]]))
classifyiris(z)

#Using the Decision Tree
decision_tree = tree.DecisionTreeClassifier(criterion='gini')
decision_tree.fit(x,y)
z = decision_tree.predict([[3,5,4,2]])
print("Using the Decision Tree =", decision_tree.predict([[3,5,4,2]]))
classifyiris(z)

#Using Support Vector Classification
svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm.fit(x,y)
z = svm.predict([[3,5,4,2]])
print("Using the Support Vector Classification =", svm.predict([[3,5,4,2]]))
classifyiris(z)

#Using Random Forest
random_forest = RandomForestClassifier(n_estimators=10)
random_forest.fit(x,y)
z = random_forest.predict([[3,5,4,2]])
print("Using the Random Forest Classification =", random_forest.predict([[3,5,4,2]]))
classifyiris(z)

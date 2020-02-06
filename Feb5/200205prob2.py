import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split

mr = pd.read_csv("mushroom.csv", header=None)

label = []
data = []
attr_list = []
for row_index, row in mr.iterrows():
    label.append(row.iloc[0])
    row_data = []
    for v in row.iloc[1:]:
        row_data.append(ord(v))
    data.append(row_data)

# print(data[0])
# print(label)

X_train, X_test, y_train, y_test = train_test_split(data, label)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

predict = rf.predict(X_test)
score = metrics.accuracy_score(y_test, predict)
print('Test set score: {}'.format(score))


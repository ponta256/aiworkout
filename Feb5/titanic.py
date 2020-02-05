import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# read train data
df = pd.read_csv('train.csv')
x = df[['Pclass', 'Sex', 'Age']]
x = x.replace('male', 0)
x = x.replace('female', 1)
x = x.fillna({'Sex':0, 'Age':30})
y = df['Survived']

lr = LogisticRegression(solver='liblinear')
svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
rf = RandomForestClassifier(n_estimators=100)

# train
# print(x)
# print(y)

lr.fit(x, y)
svm.fit(x, y)
rf.fit(x, y)

# read test data
df = pd.read_csv('test.csv')
x = df[['Pclass', 'Sex', 'Age']]
x = x.replace('male', 0)
x = x.replace('female', 1)
x = x.fillna({'Sex':0, 'Age':30})
df = pd.read_csv('gt.csv')
y = df['Survived']

# predict
cc_lr = 0
for i in range(len(x)):
    z = lr.predict([x.iloc[i, :]])
    if z[0] == y[i]:
        cc_lr += 1

cc_svm = 0        
for i in range(len(x)):
    z = svm.predict([x.iloc[i, :]])
    if z[0] == y[i]:
        cc_svm += 1
        
cc_rf = 0        
for i in range(len(x)):
    z = rf.predict([x.iloc[i, :]])
    if z[0] == y[i]:
        cc_rf += 1

print('LR {}/{} ({})'.format(cc_lr, len(x), cc_lr/len(x)))
print('SVM {}/{} ({})'.format(cc_svm, len(x), cc_svm/len(x)))
print('RF {}/{} ({})'.format(cc_rf, len(x), cc_rf/len(x)))

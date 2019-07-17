import pandas as pd
from sklearn.linear_model import LogisticRegression

# read train data
df = pd.read_csv('train.csv')
x = df[['Pclass', 'Sex', 'Age']]
x = x.replace('male', 0)
x = x.replace('female', 1)
x = x.fillna({'Sex':0, 'Age':30})
y = df['Survived']
lr = LogisticRegression(solver='liblinear')

# train
lr.fit(x,y)

# read test data
df = pd.read_csv('test.csv')
x = df[['Pclass', 'Sex', 'Age']]
x = x.replace('male', 0)
x = x.replace('female', 1)
x = x.fillna({'Sex':0, 'Age':30})
df = pd.read_csv('gt.csv')
y = df['Survived']

# predict
cc = 0
for i in range(len(x)):
    z = lr.predict([x.iloc[i, :]])
    if z[0] == y[i]:
        cc += 1

print('{}/{} ({})'.format(cc, len(x), cc/len(x)))
    

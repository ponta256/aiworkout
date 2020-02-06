import pandas as pd

'''
0,1,2,3,4,5
10,11,12,13,14,15
20,21,22,23,24,25
30,31,32,33,34,35
40,41,42,43,44,45
50,51,52,53,54,55
60,61,62,63,64,65
70,71,72,73,74,75
80,81,82,83,84,85
90,91,92,93,94,95
'''

# CSVデータの読み込み
m = pd.read_csv('sample.csv', header=None)

# print(m)
print('データのタイプ', type(m))
print('行数', len(m))
print('列数', len(m.columns))

# インデックスが0から始まることに注意 = 0行目、0列目から数える
print(m.iloc[:, 0]) # 0列目
print(m.iloc[:, 2]) # 2列目
print(m.iloc[:, 1:4]) # 1列目~3列目
print(m.iloc[:, 2:]) # 2列目~
print(m.iloc[:, -1]) # 最終列
print(m.iloc[:, [2,3]]) # 2, 3列目

print(m.iloc[0, :]) # 0行目
print(m.iloc[[3,4], :]) # 3,4行目

print(m.iloc[2,3]) # 2行3列



#### 文字を適当に数値化する例 ####

x = ['a','b','c','a','c','c']


# 1. 強引に変換する方法 (not recommended)
for c in x:
    if c == 'a':
        print(0)
    elif c == 'b':
        print(1)
    elif c == 'c':
        print(2)        

# 2. 変換テーブルを使う方法
t = {'a':0, 'b':1, 'c':2}
for c in x:
    print(t[c])

# 3. ord()を使う方法 -> 文字コードに変換してくれる (ASCIIコード)
for c in x:
    print(ord(c))

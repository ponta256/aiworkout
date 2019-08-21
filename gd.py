import numpy as np

def unit(x):
    thresh = 0.5
    x = np.array(x)
    x[x>=thresh] = 1
    x[x<thresh] = -1
    return x

def forward(w, x):
    return unit(np.dot(w, x))

def train(w, x, t, lr):
    y = forward(x, w)
    # 正解じゃなかったときに weightを誤差を減らす方向に動かす
    if(t != y):
        w += lr * t * x
    return w

# 'AND'
def main():
    lr = 0.1
    max_epoch = 100

    # train data
    X = np.array([[1,0,0], [1,0,1], [1,1,0], [1,1,1]], dtype=np.float32) 
    t = np.array([-1,-1,-1,1], dtype=np.float32) # label

    # initial weight - please try the other values
    w  = np.array([0.2,0.5,0.3], dtype=np.float32)

    for e in range(max_epoch):
        for i in range(t.size):
            w = train(w, X[i], t[i], lr)

    # test
    X = np.array([[1,1,1], [0,0,0], [0,1,0]], dtype=np.float32)
    t = np.array([1,-1,-1], dtype=np.float32)
    
    y = unit(np.sum(w*X, 1))
    print("output：", y)
    print("grand truth：", t)
    print("weight：", w)

if __name__ == "__main__":
    main()

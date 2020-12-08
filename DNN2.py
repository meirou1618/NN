#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DNN実装
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

##### データの取得
#クラス数を定義
m = 5

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape([60000, 28*28])
x_train = x_train[y_train < m,:]

x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape([10000, 28*28])
x_test = x_test[y_test < m,:]

y_train = y_train[y_train < m]
y_train = to_categorical(y_train, m)

y_test = y_test[y_test < m]
y_test = to_categorical(y_test, m)

n, d = x_train.shape
n_test, _ = x_test.shape

np.random.seed(123)

##### 活性化関数, 誤差関数, 順伝播, 逆伝播
#シグモイド関数
def sigmoid(x):
    sig=1/(1+np.e**(-x))
    return sig,(1-sig)*sig

#ReLU
def ReLU(x):
    return x*(x>0),1*(x>0)

# ソフトマックス関数
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

# クロスエントロピー
def CrossEntoropy(x, y):
    return -np.sum(y*np.log(x))

# 順伝播
def forward(x, w, fncs):
    z,zz=fncs(np.dot(w,x))
    za=[1]
    za.extend(z)
    return za,zz

# 逆伝播
def backward(w, delta, derivative):
    return np.dot(np.delete(derivative,0,1).T,delta)*w

##### 中間層のユニット数とパラメータの初期値
q = 200
wnum=3
w=[]
w[0]=np.append(w,np.random.normal(0,0.3,size=(q,d+1)))
for i in range(1,wnum-1):
    w = np.append(w,np.random.normal(0, 0.3, size=(q+i, q+i)))

w = np.append(w,np.random.normal(0, 0.3, size=(m, q+wnum-1)))

########## 確率的勾配降下法によるパラメータ推定
num_epoch = 50

eta = 10**(-2)

e = []
e_test = []
error = []
error_test = []

for epoch in range(0, num_epoch):
    index = np.random.permutation(n)
    
    eta_t = eta/(epoch+1) 
    for i in index:
        xi = np.append(1, x_train[i, :])
        yi = y_train[i, :]


        ##### 順伝播
        z=[]
        wx=[]
        zc,wxc=forward(xi,w[0],ReLU)
        z=np.append(z,zc)
        wx=np.append(wx,wxc)

        for i in range(1,wnum-1):
            zc,wxc=forward(z[i-1],w[i],sigmoid)
            z=np.append(z,zc)
            wx=np.append(wx,wxc)

        z=np.append(z,softmax(np.dot(w[wnum-1],z[wnum-2])))
        
        ##### 誤差評価
        e.append(CrossEntoropy(z[wnum-1], yi))
        ##### 逆伝播
        delta1=[]
        delta2=[]
        delta1=np.append(delta1,softmax(np.dot(w[wnum-1],z[wnum-2]))-yi)
        delta2=np.append(delta2,np.array([delta1[0]]))

        for i in range(1,wnum):
            delta1=np.append(delta1,backward(wx[wnum-i-1],delta1[i-1],w[wnum-i]))
            delta2=np.append(delta2,np.array[delta1[i]])
        
        ##### パラメータの更新
        w[0]=w[0]-eta_t*np.outer(delta2[wnum-1],xi)

        for i in range(1,wnum):
            w[i]=w[i]-eta_t*np.outer(delta2[wnum-i-1],z[i-1])

    ##### エポックごとの訓練誤差: eの平均をerrorにappendする
    error.append(sum(e)/n)
    e = []
    
    ##### test error
    for j in range(0, n_test):
        xi = np.append(1, x_test[j, :])
        yi = y_test[j, :]

        ##### テスト誤差: 誤差をe_testにappendする
        z=[]
        wx=[]
        zc,wxc=forward(xi,w[0],ReLU)
        z=np.append(z,zc)
        wx=np.append(wx,wxc)

        for i in range(1,wnum-1):
            zc,wxc=forward(z[i-1],w[i],sigmoid)
            z=np.append(z,zc)
            wx=np.append(wx,wxc)

        z=np.append(z,softmax(np.dot(w[wnum-1],z[wnum-2])))

        e_test.append(CrossEntoropy(z[wnum-1], yi))

    ##### エポックごとの訓練誤差: e_testの平均をerror_testにappendする
    error_test.append(sum(e_test)/n_test)
    e_test = []

########## 誤差関数のプロット
plt.clf()
plt.plot(error, label="training", lw=3)     #青線
plt.plot(error_test, label="test", lw=3)     #オレンジ線
plt.grid()
plt.legend(fontsize =16)
plt.savefig("./error.pdf", bbox_inches='tight', transparent=True)

########## 確率が高いクラスにデータを分類
prob = []
for j in range(0, n_test):    
    xi = np.append(1, x_test[j, :])
    yi = y_test[j, :]
    
    # テストデータに対する順伝播: 順伝播の結果をprobにappendする
    z=[]
    wx=[]
    zc,wxc=forward(xi,w[0],ReLU)
    z=np.append(z,zc)
    wx=np.append(wx,wxc)

    for i in range(1,wnum-1):
        zc,wxc=forward(z[i-1],w[i],sigmoid)
        z=np.append(z,zc)
        wx=np.append(wx,wxc)

    z=np.append(z,softmax(np.dot(w[wnum-1],z[wnum-2])))

    prob.append(z[wnum-1])

predict = np.argmax(prob, 1)

##### confusion matrixと誤分類結果のプロット
ConfMat = np.zeros((m, m))
for i in range(m):
    idx_true = (y_test[:, i]==1)
    for j in range(m):
        idx_predict = (predict==j)
        ConfMat[i, j] = sum(idx_true*idx_predict)
        if j != i:
            for l in np.where(idx_true*idx_predict == True)[0]:
                plt.clf()
                D = np.reshape(x_test[l, :], (28, 28))
                sns.heatmap(D, cbar =False, cmap="Blues", square=True)
                plt.axis("off")
                plt.title('{} to {}'.format(i, j))
                plt.savefig("./misslabeled{}.pdf".format(l), bbox_inches='tight', transparent=True)

plt.clf()
fig, ax = plt.subplots(figsize=(5,5),tight_layout=True)
fig.show()
sns.heatmap(ConfMat.astype(dtype = int), linewidths=1, annot = True, fmt="1", cbar =False, cmap="Blues")
ax.set_xlabel(xlabel="Predict", fontsize=18)
ax.set_ylabel(ylabel="True", fontsize=18)
plt.savefig("./confusion.pdf", bbox_inches="tight", transparent=True)

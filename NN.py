#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3層NN
"""
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist

##### データの取得
#クラス数
m = 4

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

##### シグモイド関数, 誤差関数
def ReLU(x):#f
    # ReLUとその微分
    return x*(x>0),1*(x>0)

def softmax(x):#g
    # ソフトマックス関数
    return np.exp(x)/np.sum(np.exp(x))

def CrossEntoropy(x, y):
    # クロスエントロピー
    return -np.sum(y*np.log(x))

def forward(x, w, fncs):
    # 順伝播
    z,zz=fncs(np.dot(w,x))
    za=[1]
    za.extend(z)
    return za,zz

def backward(w, delta, derivative):
    # 逆伝播
    return np.dot(np.delete(derivative,0,1).T,delta)*w

#####中間層のユニット数とパラメータの初期値
q = 200
w = np.random.normal(0, 0.3, size=(q, d+1))
v = np.random.normal(0, 0.3, size=(m, q+1))

########## 確率的勾配降下法によるパラメータ推定
e = []
e_test = []
error = []
error_test = []

num_epoch = 10

eta = 0.1

for epoch in range(0, num_epoch):
    index = np.random.permutation(n)
    
    eta_t = eta/(epoch + 1)
    for i in index:       
        xi = np.append(1, x_train[i, :])
        yi = y_train[i, :]
        
        ##### 順伝播 
        z1,wx=forward(xi,w,ReLU)
        z2=softmax(np.dot(v,z1))
        ##### 誤差評価
        e.append(CrossEntoropy(z2, yi))
        
        ##### 逆伝播
        delta1=softmax(np.dot(v,z1))-yi
        delta11=np.array([delta1])

        delta2=backward(wx,delta1,v)
        delta22=np.array([delta2])
        ##### パラメータの更新
        v=v-eta_t*np.outer(delta11,z1)
        w=w-eta_t*np.outer(delta22,xi)
        
    ##### training error
    error.append(sum(e)/n)
    e = []
    
    ##### test error
    for j in range(0, n_test):        
        xi = np.append(1, x_test[j, :])
        yi = y_train[j, :]
        
        z1, u1 = forward(xi, w, ReLU)
        z2 = softmax(np.dot(v, z1))
        
        e_test.append(CrossEntoropy(z2, yi))
        
    error_test.append(sum(e_test)/n_test)
    e_test = []

########## 誤差関数のプロット
plt.clf()
plt.plot(error, label="training", lw=3)     #青線
plt.plot(error_test, label="test", lw=3)     #オレンジ線
plt.grid()
plt.legend(fontsize =16)
plt.savefig("./error.pdf")

########## 確率が高いクラスにデータを分類
##### モデルの出力を評価
prob = []
for j in range(0, n_test):    
    xi = np.append(1, x_test[j, :])
    yi = y_train[j, :]
    
    z1, u1 = forward(xi, w, ReLU)
    z2 = softmax(np.dot(v, z1))
    
    prob.append(z2)

predict = np.argmax(prob, 1)

# confusion matrix
ConfMat = np.zeros((m, m))
for i in range(len(predict)):
    if y_test[i][0]==1:
        ConfMat[0][predict[i]] +=1
    elif y_test[i][1]==1:
        ConfMat[1][predict[i]] +=1
    elif y_test[i][2]==1:
        ConfMat[2][predict[i]] +=1
    elif y_test[i][3]==1:
        ConfMat[3][predict[i]] +=1


plt.clf()
fig, ax = plt.subplots(figsize=(5,5),tight_layout=True)
fig.show()
sns.heatmap(ConfMat.astype(dtype = int), linewidths=1, annot = True, fmt="1", cbar =False, cmap="Blues")
ax.set_xlabel(xlabel="Predict", fontsize=18)
ax.set_ylabel(ylabel="True", fontsize=18)
plt.savefig("./confusion.pdf", bbox_inches="tight", transparent=True)

# 誤分類結果のプロット
for i in range(m):
    idx_true = (y_test[:, i]==1)
    for j in range(m):
        if j != i:
            idx_predict = (predict==j)
            for l in np.where(idx_true*idx_predict == True)[0]:
                plt.clf()
                D = np.reshape(x_test[l, :], (28, 28))
                sns.heatmap(D, cbar =False, cmap="Blues", square=True)
                plt.axis("off")
                plt.title('{} to {}'.format(i, j))
                plt.savefig("./misslabeled{}.pdf".format(l), bbox_inches='tight', transparent=True)


#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2020/07/22
# @Author : humaohai
# @File   : nnfeature.py
# @desc   : 神经网络构造特征

"""
结论：神经网络抽取隐层特征有一定的效果。
风险点：目前只测试了逻辑回归分类，其有效果比较好解释，其实整体就是一个神经网络的改写版，
       但对树模型的分类是否有用未测试。也没有理论支撑，需要测试。
"""

from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.models import Model
from keras.initializers import RandomNormal
from keras.utils.generic_utils import get_custom_objects
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


X,y = load_iris(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=102,test_size=0.5)
lr1 = LogisticRegression()
lr1.fit(X=x_train,y=y_train)
print('--------------逻辑回归直接拟合----------------')
print('训练集：')
print(lr1.score(X=x_train,y=y_train))
print('测试集：')
print(lr1.score(X=x_test,y=y_test))


# 为特征交叉可解释性自定义y=x函数
# def at(x):
#     return x
#
# get_custom_objects().update({'at': Activation(at)})

model = Sequential()
model.add(Dense(16, activation='relu', input_dim=4,kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=666)))
model.add(Dense(8, activation='relu',name="Dense_1",kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=666)))
model.add(Dense(3, activation='softmax',name="Dense_2",kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=666)))
model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
model.fit(x_train, y_train, epochs=400, batch_size=32,verbose=0)

dense1_layer_model = Model(inputs=model.input,
outputs=model.get_layer('Dense_1').output)

dense1_train = dense1_layer_model.predict(x_train)
print('经过隐层前后的特征对比')
print(x_train[0])
print(dense1_train[0])  # relu激活函数负数会直接变成0
lr = LogisticRegression()
lr.fit(X=dense1_train,y=y_train)

print('--------------抽取隐层特征后逻辑回归----------------')
print('训练集：')
print(lr.score(X=dense1_train,y=y_train))
print('测试集：')
dense1_test = dense1_layer_model.predict(x_test)
print(lr.score(X=dense1_test,y=y_test))



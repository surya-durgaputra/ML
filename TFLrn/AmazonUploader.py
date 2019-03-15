# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:00:18 2019

@author: DevAccessa
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

housing_bunch = fetch_california_housing()
housing_target = np.c_[housing_bunch.data,housing_bunch.target]
column_names = np.concatenate((housing_bunch.feature_names,["target"]))
housing = pd.DataFrame(housing_target,columns=column_names)

# since we are manually doing the normal equation, we do like so
m,n = housing_bunch.data.shape
scaler = StandardScaler()
scaled_housing_data = scaler.fit_transform(housing_bunch.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m,1)),scaled_housing_data]
#X = tf.constant(housing_data_plus_bias,dtype=tf.float32,name="X")
#y = tf.constant(housing_bunch.target.reshape(-1,1),dtype=tf.float32,name="y")
#XT = tf.transpose(X)
#theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)



X = tf.constant(scaled_housing_data_plus_bias,dtype=tf.float32,name="X")
y = tf.constant(housing_bunch.target.reshape(-1,1),dtype=tf.float32,name="y")

theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0), name="theta")

y_pred = tf.matmul(X,theta,name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
#gradients = 2/m*tf.matmul(tf.transpose(X),error)
gradients = tf.gradients(mse,[theta])[0]
learning_rate = 0.01
n_epochs = 1000
training_op = tf.assign(theta,theta-learning_rate*gradients)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE=",mse.eval())
        sess.run(training_op)
        
    best_theta = theta.eval()
    
#print("Best theta:", best_theta)


reset_graph()

a = tf.Variable(0.2, name="a")
b = tf.Variable(0.3, name="b")
z = tf.constant(0.0, name="z0")
for i in range(100):
    z = a * tf.cos(z + i) + z * tf.sin(b - i)

grads = tf.gradients(z, [a,b])
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    print(z.eval())
    print(sess.run(grads))
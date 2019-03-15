
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

housing_bunch = fetch_california_housing()
housing_target = np.c_[housing_bunch.data,housing_bunch.target]
column_names = np.concatenate((housing_bunch.feature_names,["target"]))
housing = pd.DataFrame(housing_target,columns=column_names)

# since we are manually doing the normal equation, we do like so
m,n = housing_bunch.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)),housing_bunch.data]
#X = tf.constant(housing_data_plus_bias,dtype=tf.float32,name="X")
#y = tf.constant(housing_bunch.target.reshape(-1,1),dtype=tf.float32,name="y")
#XT = tf.transpose(X)
#theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT,X)),XT),y)

scaler = StandardScaler()
scaled_housing_data_plus_bias = scaler.fit_transform(housing_data_plus_bias)

X = tf.constant(scaled_housing_data_plus_bias,dtype=tf.float32,name="X")
y = tf.constant(housing_bunch.target.reshape(-1,1),dtype=tf.float32,name="y")

theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0), name="theta")

y_pred = tf.matmul(X,theta,name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m*tf.matmul(tf.transpose(X),error)
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
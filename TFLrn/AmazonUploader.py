# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 11:00:18 2019

@author: DevAccessa
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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


def localized_code():
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
        
    # using tensorboard
    reset_graph()



from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir,now)

n_epochs = 10
batch_size = 100
learning_rate = 0.01

n_batches = int(np.ceil(m/batch_size))

X = tf.placeholder(tf.float32, shape=(None,n+1),name="X")
y = tf.placeholder(tf.float32, shape=(None,1),name="y")
theta = tf.Variable(tf.random_uniform([n+1,1],-1.0,1.0,seed=42,name="theta"))
y_pred = tf.matmul(X,theta,name="predictions")
#create a name scope
with tf.name_scope("loss") as scope:
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE',mse)
file_writer = tf.summary.FileWriter(logdir,tf.get_default_graph())


def fetch_batch(epoch, batch_index, batch_size, n_batches, data, target):
    total_rows = np.size(target)
    np.random.seed(epoch*n_batches + batch_index )
    indices = np.random.randint(total_rows,size=batch_size)
    X_batch = data[indices]
    y_batch = target.reshape(-1,1)[indices]
    return X_batch, y_batch

with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            X_batch, y_batch = fetch_batch(epoch,batch_index,
                                           batch_size,n_batches,
                                           scaled_housing_data_plus_bias,
                                           housing_bunch.target)
            if batch_index % 10 == 0:
                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
                step = epoch * n_batches + batch_index
                file_writer.add_summary(summary_str, step)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            
    best_theta = theta.eval()

file_writer.close()


#reset_graph()
#init = tf.global_variables_initializer()
#def relu(X):
#    with tf.name_scope("relu"):
#        w_shape = (int(X.get_shape()[1]), 1)                          # not shown in the book
#        w = tf.Variable(tf.random_normal(w_shape), name="weights")    # not shown
#        b = tf.Variable(0.0, name="bias")                             # not shown
#        z = tf.add(tf.matmul(X, w), b, name="z")                      # not shown
#        return tf.maximum(z, 0., name="max")                          # not shown
#
#n_features = 3
#X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
#relus = [relu(X) for i in range(5)]
#output = tf.add_n(relus, name="output")
#file_writer = tf.summary.FileWriter("logs/relu2", tf.get_default_graph())
#
#X_sample = np.random.normal(size=(1,n_features))
#with tf.Session() as sess:
#    sess.run(init)
#    sess.run(output,feed_dict={X:X_sample})
#file_writer.close()

from sklearn.datasets import make_blobs
data = make_blobs(n_samples=50,n_features=2,centers=2,random_state=75)

reset_graph()

# simple linear regression
sample_size = 1000000
batch_size = 10
n_epochs = 10
x_data = np.linspace(0,10,sample_size)+np.random.uniform(-1.5,1.5,sample_size)
y_label = np.linspace(0,10,sample_size)+np.random.uniform(-1.5,1.5,sample_size)

xdf = pd.DataFrame(x_data,columns=['X'])
ydf = pd.DataFrame(y_label,columns=['Y'])
x_y = pd.concat((xdf,ydf),axis=1)

m,c = np.random.rand(2)
print('original slope:',m,' original intercept:',c)
m = tf.Variable(m,dtype=tf.float32)
c = tf.Variable(c,dtype=tf.float32)

xph = tf.placeholder(tf.float32,[batch_size])
yph = tf.placeholder(tf.float32,[batch_size])

y_predicted = m*xph + c

loss = tf.reduce_sum(tf.square(y_predicted-yph))
    
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(n_epochs):
        rand_index = np.random.randint(len(x_data),size=batch_size)
        x_batch = x_data[rand_index]
        y_batch = y_label[rand_index]
        sess.run(train,feed_dict={xph:x_batch,yph:y_batch})
    final_slope,final_intercept = sess.run([m,c])    
    print('slope:',final_slope,' intercept:',final_intercept)
x_y.sample(n=250).plot(kind='scatter',x='X',y='Y')
y_final = [final_slope*x + final_intercept for x in x_data]
plt.plot(x_data,y_final,'r')
# end simple linear regression

# Estimator API : simple example
feat_cols = [tf.feature_column.numeric_column('x',shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

x_train, x_eval, y_train, y_eval = train_test_split(x_data,y_label,
                                                    test_size=0.3, 
                                                    random_state=101)






    
    





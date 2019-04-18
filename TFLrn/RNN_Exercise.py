# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 13:14:10 2019

@author: DevAccessa
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from time import strptime, strftime
from sklearn.preprocessing import MinMaxScaler

data_df = pd.read_csv('monthly-milk-production.csv')
data_df["Month"] = data_df.Month.apply(lambda x:strftime('%Y-%m-%d',
       strptime(x,'%Y-%m-%d %H:%M:%S')))
data_df = data_df.set_index(pd.to_datetime(data_df.Month))
data_df = data_df.drop(columns=['Month'],axis=1)



class Dataset():

    def __init__(self, data):
        self._data = data
        self._epochs_completed = 0
        self._num_of_samples = data.shape[0]

    def create_train_test_data(self):
        train_set = self._data.head(156)
        test_set = self._data.tail(12)

        scaler = MinMaxScaler()

        self._train_set = pd.DataFrame(scaler.fit_transform(
                train_set[['Milk Production']]), columns=['Milk Production'],
                index=train_set.index)
        self._test_set = pd.DataFrame(scaler.transform(
                test_set[['Milk Production']]), columns=['Milk Production'],
                index=test_set.index)

    @property
    def data(self):
        return self._data

    def next_batch(self, batch_size, steps):
        high = self._num_of_samples - steps
        low = 0
        rand_start = np.random.randint(low=low, high=high)
        sliced_df = self._train_set.iloc[rand_start:rand_start+steps+1]
        print('rand_start:',rand_start)
        print('high:',high)
        return np.array(sliced_df.iloc[:-1]['Milk Production']).reshape(-1,steps,1), np.array(
                sliced_df.iloc[1:]['Milk Production']).reshape(-1,steps,1)


def next_batch(training_data,batch_size,steps):
    # Grab a random starting point for each batch
    high = len(training_data)-(steps+1)
    rand_start = np.random.randint(low=0,high=high)
    # Create Y data for time series in the batches
    y_batch = np.array(training_data.iloc[rand_start:
        rand_start+steps+1]).reshape(-1,steps+1)
    #the returned y values will be just one time step apart
    # in other words returned values will only be one month apart
    # later we will predict 1 whole year (12 time steps) of y value
    return y_batch[:,:-1].reshape(-1,steps,1),y_batch[:,1:].reshape(-1,steps,1)

### setting up the RNN Model
# define the constants
#Just one feature, the time series
num_inputs = 1
#num of steps in each batch
num_time_steps = 12
# 100 neuron layer.. play with this
num_neurons = 100
#just one output, predicted time series
num_outputs = 1

## you can also try increasing the iterations but decreasing the learning rate
# learning rate ... play with this
learning_rate = 0.03
#how many iterations to go through(training steps), play with this
num_train_iterations = 4000
#size of the batch of data. We will feed in one batch at a time.
batch_size = 1

# Create placeholders for X and y. 
# The shape for these placeholders will be (None, steps, num_inputs),
# (None, steps, num_outputs)
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])

### Now we will create the NN layer

# Also play around with GRUCell
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicLSTMCell(num_units=num_neurons, activation=tf.nn.relu),
    output_size=num_outputs)

#now we will get the output and the states of these rnn cells
#we'll use a convenience function provided by tf.nn called dynamic_rnn
outputs,states = tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)

# LOSS FUNCTION
# we'll use mse as our loss function
loss = tf.reduce_mean(tf.square(outputs-y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

#SESSION
#this line needed if running on GPU. Not needed if running on CPU
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)

#lets create a saver function to save the model and use it later
saver = tf.train.Saver()

# Run a tf.Session that trains on the batches created by next_batch
# function. Also add an a loss evaluation for every 100 training
# iterations. Remember to save your model after you are done training.
d = Dataset(data_df)
d.create_train_test_data()

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    
    for iteration in range(num_train_iterations):
        X_batch, y_batch = d.next_batch(batch_size,num_time_steps)
        sess.run(train,feed_dict={X:X_batch,y:y_batch})
        
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X:X_batch,y:y_batch})
            print(iteration,"\tMSE",mse)
            
    saver.save(sess,"./rnn_time_series_codealong")
        
# lets restore the model I just saved
with tf.Session() as sess:
    saver.restore(sess,"./rnn_time_series_codealong")
    # Create a numpy array for your genreative seed from the last 12 months of the 
    # training set data. Hint: Just use tail(12) and then pass it to an np.array
    X_seed = np.array(d._test_set.iloc[-num_time_steps:]['Milk Production']).reshape(-1,num_time_steps,1)

    #Now create a loop for predicting values for 12 months
    for iteration in range(12):
        y_pred = sess.run(outputs,feed_dict={X:X_seed[-num_time_steps:0]})
        

    
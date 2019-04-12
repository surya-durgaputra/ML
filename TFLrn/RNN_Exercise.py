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

train_set = data_df.head(156)
test_set = data_df.head(12)

scaler = MinMaxScaler()

train_set = scaler.fit_transform(train_set)
test_set = scaler.transform(test_set)

def next_batch(training_data,batch_size,steps):
    # Grab a random starting point for each batch
    high = len(training_data)-steps
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
#size of the batch of data
batch_size = 1


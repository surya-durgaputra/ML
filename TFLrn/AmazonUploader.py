# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 12:45:05 2019

@author: DevAccessa
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

################## MANUAL RNN #################################
def manual_rnn_only_for_demo():
    #CONSTANTS
    num_inputs = 2
    num_neurons = 3
    
    #PLACEHOLDERS
    x0 = tf.placeholder(tf.float32,[None,num_inputs])
    x1 = tf.placeholder(tf.float32,[None,num_inputs])
    
    #VARIABLES
    Wx = tf.Variable(tf.random_normal(shape=[num_inputs,num_neurons]))
    Wy = tf.Variable(tf.random_normal(shape=[num_neurons,num_neurons]))
    b = tf.Variable(tf.zeros([1,num_neurons]))
    
    #GRAPHS
    y0 = tf.tanh(tf.matmul(x0,Wx)+b)
    y1 = tf.tanh(tf.matmul(y0,Wy) + tf.matmul(x1,Wx) +b)
    
    init = tf.global_variables_initializer()
    
    ##CREATE DATA
    #timestep 0
    x0_batch = np.array([ [0,1], [2,3], [4,5] ])
    #timestep 1
    x1_batch = np.array([ [100,101], [102,103], [104,105] ])
    
    with tf.Session() as sess:
        sess.run(init)
        
        y0_output_vals, y1_output_vals = sess.run([y0,y1], feed_dict={x0:x0_batch,x1:x1_batch})
        
    print('y0_output_vals: ',y0_output_vals)
    print('y1_output_vals: ',y1_output_vals)
    
    '''
    this manual scenario is not practical.
    Here we need to type out Wx, Wy,b, x , y, x_batche, y_batche
    for each timestep manually. This is not practical for like 100 timesteps.
    here we showed only two timesteps.
    '''        

#manual_rnn_only_for_demo()
    
##############################################################

## RNN WITH TF ###
# target is to produce portions of a sine wave
# based on the input and some labeled output,
# our model is going to learn and then predict
# values in the future. We know that the target
# values are from sine wave (y_true) but the model
# will learn this
class TimeSeriesData():
    def __init__(self,num_points,xmin,xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = (xmax-xmin)/num_points
        self.x_data = np.linspace(xmin,xmax,num_points)
        # the y_true is a sinusoid
        self.y_true = np.sin(self.x_data)
        
    def ret_true(self,x_series):
        return np.sin(x_series)
    
    #return_batch_ts means return_batch_timeseries
    def next_batch(self,batch_size,steps,return_batch_ts=False):
        # Grab a random starting point for each batch
        rand_start = np.random.rand(batch_size,1)
        # Convert to be on time series
        ts_start = rand_start * (self.xmax - self.xmin - (steps*self.resolution))
        # Create batch time series on the x-axis
        batch_ts = ts_start + np.arange(0.0,steps+1) * self.resolution
        print('rand_start:',rand_start)
        print('ts_start:',ts_start)
        print('resolution:',self.resolution)
        print('batch_ts:',batch_ts)
        # Create the Y data for the time series x axis from previous step
        y_batch = np.sin(batch_ts)
        # FORMATTING for RNN
        if return_batch_ts:
            return y_batch[:,:-1].reshape(-1,steps,1),y_batch[:,1:].reshape(-1,steps,1), batch_ts
        else:
            return y_batch[:,:-1].reshape(-1,steps,1),y_batch[:,1:].reshape(-1,steps,1)
        
        
ts_data = TimeSeriesData(250,0,10)
#plt.plot(ts_data.x_data,ts_data.y_true)
num_time_steps = 30

y1,y2,ts = ts_data.next_batch(1,num_time_steps,True)

class Rect:
    l = 9
    def inisde(self):
        print('attttr:',Rect.l)
print('attr:',Rect.l)
r = Rect()
r.inisde()
        
        
        
        
        
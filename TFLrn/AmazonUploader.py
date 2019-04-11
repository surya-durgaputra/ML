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
#        print('rand_start:',rand_start)
#        print('ts_start:',ts_start)
#        print('resolution:',self.resolution)
#        print('batch_ts:',batch_ts)
        # Create the Y data for the time series x axis from previous step
        y_batch = np.sin(batch_ts)
        # FORMATTING for RNN
        if return_batch_ts:
            return y_batch[:,:-1].reshape(-1,steps,1),y_batch[:,1:].reshape(-1,steps,1),batch_ts
        else:
            return y_batch[:,:-1].reshape(-1,steps,1),y_batch[:,1:].reshape(-1,steps,1)
        
        
ts_data = TimeSeriesData(250,0,10)
#plt.plot(ts_data.x_data,ts_data.y_true)
num_time_steps = 30

y1,y2,ts = ts_data.next_batch(1,num_time_steps,True)
plt.plot(ts_data.x_data,ts_data.y_true,label='sin(t)')
plt.plot(ts.flatten()[1:],y2.flatten(),'*',label='Single training instance')
plt.legend()
plt.tight_layout()

#TRAINING DATA
train_inst = np.linspace(5,5+ts_data.resolution*(num_time_steps+1),
                         num_time_steps+1)

plt.figure()
plt.title('A TRAINING INSTANCE')
plt.plot(train_inst[:-1],ts_data.ret_true(train_inst[:-1]),
         'bo',markersize=15,alpha=0.5,label='INSTANCE')
plt.plot(train_inst[1:],ts_data.ret_true(train_inst[1:]),
         'ko',markersize=8,label='TARGET')

#CREATING THE MODEL
tf.reset_default_graph()
num_inputs = 1
num_neurons = 100 #the number of inputs in our layer
num_outputs = 1
learning_rate = 0.0001 # play around with this. Use this for BasicRNNCell
#learning_rate = 0.01 #use this for GRUCell and LSTMCell
num_train_iterations = 2000
batch_size = 1

#PLACEHOLDERS
X = tf.placeholder(tf.float32,[None,num_time_steps,num_inputs])
y = tf.placeholder(tf.float32,[None,num_time_steps,num_outputs])

#RNN CELL LAYER
cell = tf.contrib.rnn.BasicRNNCell(num_units=num_neurons,activation=tf.nn.relu)
#cell = tf.contrib.rnn.GRUCell(num_units=num_neurons,activation=tf.nn.relu)
#cell = tf.contrib.rnn.LSTMCell(num_units=num_neurons,activation=tf.nn.relu)

#since we are using 100 neurons in our layer, but are actually creating only
#one output (instead of 100), we will need to use an output projection wrapper
cell = tf.contrib.rnn.OutputProjectionWrapper(cell,output_size=num_outputs)
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

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(init)
    
    for iteration in range(num_train_iterations):
        X_batch, y_batch = ts_data.next_batch(batch_size,num_time_steps)
        sess.run(train,feed_dict={X:X_batch,y:y_batch})
        
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X:X_batch,y:y_batch})
            print(iteration,"\tMSE",mse)
            
    saver.save(sess,"./rnn_time_series_codealong")
        
# lets restore the model I just saved
with tf.Session() as sess:
    saver.restore(sess,"./rnn_time_series_codealong")
    X_new = np.sin(np.array(train_inst[:-1].reshape(
            -1,num_time_steps,num_inputs)))
    y_pred = sess.run(outputs,feed_dict={X:X_new})


#PLOTTING
plt.figure()
plt.title('TESTING THE MODEL')
#TRAINING INSTANCE
plt.plot(train_inst[:-1],np.sin(train_inst[:-1]),'bo',markersize=15,
         alpha=0.5,label='TRAINING INST')
plt.plot(train_inst[1:],np.sin(train_inst[1:]),'ko',markersize=10,
         label='TARGET')
plt.plot(train_inst[1:],y_pred[0,:,0],'r.',markersize=10,
         label='PREDICTIONS')        
        
plt.legend()
plt.tight_layout()

#### GENERATING NEW SEQUENCES
with tf.Session() as sess:
    saver.restore(sess,'./rnn_time_series_codealong')
    
    #SEED WITH ZEROS
    zero_seed_seq = [0. for _ in range(num_time_steps)]
    for iteration in range(len(ts_data.x_data)-num_time_steps):
        X_batch = 
        
        
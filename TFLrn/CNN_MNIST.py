# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 16:31:10 2019

@author: DevAccessa
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)

#mnist now contains the MNIST dataset as a specialized tensorflow Dataset class
# this class has many convenience methods built into it


#print(mnist.train.images)# this will print the array of all training images as array. These are flattened .ie. [784] pixels array per image
#print(mnist.train.num_examples)# this will print how many training examples are there (55000 in this case)
#print(mnist.test.num_examples)# this will print how many test examples are there (10000 in this case)
#print(mnist.validation.num_examples)# this will print how many validation examples are there (5000 in this case)

# lets visualize the images, using matplotlib
# note: we'll need to reshape the images to 28x28 to visualize
# we DONT need to reshape for training the model
single_image = mnist.train.images[1].reshape(28,28)
#plt.imshow(single_image)#this will show it in color
#plt.imshow(single_image,cmap='gist_gray')#this will show it in gray-scale. This is the image we'll be training on
#also note that the dataset has already been normalized for us: single_image.min()=0.0 and single_image.max()=1.0

# basic_approach will solve MNIST task with a very simple linear approach (no hidden layers)
def basic_approach():
    # CREATE PLACEHOLDERS
    x = tf.placeholder(tf.float32,shape=(None,784))
    # CREATE VARIABLES
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))
    # CREATE GRAPH OPERATIONS
    z = tf.matmul(x,W) + b
    # LOSS FUNCTION
    y_true = tf.placeholder(tf.float32,[None,10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=z))
    # OPTIMIZER
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
    train = optimizer.minimize(cross_entropy)
    # CREATE SESSION
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # mnist dataset from tensorflow comes with a nice convenience method for batch data called next_data
        for step in range(1000):
            batch_x,batch_y = mnist.train.next_batch(100) #100 samples at a time
            #in our own datasets (not downloaded from tensorflow), lot of our time 
            # will be spent in cleaning datasets and adding these convenience methods
            sess.run(train,feed_dict={x:batch_x,y_true:batch_y})
        # EVALUATE THE MODEL
        # note argmax will return the predicted label because of the fact
        # that we have used one-hot encoding and softmax regression
        correct_prediction = tf.equal(tf.argmax(z,1),tf.argmax(y_true,1))
        #note: (tf.argmax(z,1),tf.argmax(y_true,1)) .. will return something like (3,4)
        # i.e. predicted was 3 but actually was 4 .. for that sample.
        #tf.equal(3,4) will be FALSE
        # convert the list of booleans to 0 and 1 so it is easier to analyze mathematically.. like get mean etc
        # note tf provides a convenience function tf.cast that can cast boolean to floats (true -> 1.0; false -> 0.0)
        # note that since our results are just 1 and 0, accuracy is just mean like below
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
        print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))
        
basic_approach()
# we see an accuracy of around 91% (low 90s) wwhich is OK
# we will see that using CNN we can get accuracy in high 90s

# cnn_approach will solve MNIST task with much better approach using Convolutional Neural Network
def cnn_approach(steps=300):
    ### lets create some helper functions    
    ## INIT WEIGHTS : helper function
    # shape depends on the tensor
    def init_weights(shape):
        init_random_dist = tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(init_random_dist)
    ## INIT BIAS : helper function
    def init_bias(shape):
        init_bias_vals = tf.constant(0.1,shape=shape)
        return tf.Variable(init_bias_vals)
    ## CONV2D : helper function : take a tensor and take a filter and apply as a convolution
    # there is a nice helper function in tf itself that computes 2d convolution
    # it takes in an input tensor and an input kernel or filter tensor and then
    # performs convolution on it depending on what strides and what padding you provide.
    # So in the helper function, we will create a wrapper around it to setup the parameters.
    # x: is the input set of image tensors
    # x ---> [batch,H,W,Channels] - batch is no of images in batch, H: height, W:Width, Channels: no of channels
    # W ---> [filter H, filter W, channels IN, channels OUT] - filter height, filter width, no. of Channels coming in
    # no. of Channels going out
    def conv2d(x,W):
        return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
    ## POOLING : helper function
    # there is already a tensorflow function for pooling
    # we will just create a wrapper around it
    def max_pool_2by2(x):
        # x ---> [batch,height,width,channels]
        # k size means kernel size
        # k_size -> size of the window for each dimension of the input tensor
        # dimensions of input tensor - batch, height of image, image width, channel
        # I only want to do pooling along height and width of the individual image
        # [1,2,2,1]
        #The kernel size ksize will typically be [1, 2, 2, 1] if you have a 2x2 window over which you take the maximum. 
        #On the batch size dimension and the channels dimension, 
        #ksize is 1 because we don't want to take the maximum over multiple examples, or over multiples channels.
        return tf.nn.max_pool(x,ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
    
    ### now lets create the functions that will help us create the layers
    # CONVOLUTIONAL LAYER
    def convolutional_layer(input_x, shape):
        W = init_weights(shape)
        b = init_bias([shape[3]])#biases run along the 4th dimension..corresponding to no. of features (or neurons in that layer)
        return tf.nn.relu(conv2d(input_x,W)+b)
    
    # now we will make the normal (fully-connected) layer
    # NORMAL (FULLY CONNECTED) LAYER
    def normal_full_layer(input_layer, size):
        input_size = int(input_layer.get_shape()[1])
        W = init_weights([input_size,size])
        b = init_bias([size])
        return tf.matmul(input_layer,W) + b
    
    ### lets now create our CNN
    #PLACEHOLDERS
    x = tf.placeholder(tf.float32,shape=[None,784])
    y_true = tf.placeholder(tf.float32, shape=[None,10])# one hot encoded. 10 for 10 digits
    
    # LAYERS
    x_image = tf.reshape(x,[-1,28,28,1])#image is gray scale. so only one color channel
    # 1st convolutional layer
    # we are going to compute 32 features for each 5x5 patch
    convo_1 = convolutional_layer(x_image,shape=[5,5,1,32])# so the weight tensor is 5,5,1,32
    #5x5 is the patch size. 1 is the channel (grayscale..so 1) and 32 corresponds to the no. of actual
    #features we are computing. So, 32 is the no. of output channels
    #now lets do the pooling layer attached to this convolution layer
    convo_1_pooling = max_pool_2by2(convo_1)
    
    # 2nd convolutional layer. Input 32 and output 64 (here we will be computing 64 features)
    convo_2 = convolutional_layer(convo_1_pooling, shape=[5,5,32,64])
    convo_2_pooling = max_pool_2by2(convo_2)
    
    convo_2_flat = tf.reshape(convo_2_pooling,[-1,7*7*64])
    full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat,1024))#we decided to have 1024 neurons in this layer
    
    #DROPOUT layer : to avoid overfitting
    # we randomly drop neurons during training
    # for example: hold_prob = 0.5 means each neuron has 50% chance of being retained
    # (conversely, each neuron has 50% chance of being dropped).
    # so with hold_prob=0.5, we will randomly drop half of our neurons
    hold_prob = tf.placeholder(tf.float32)
    full_one_dropout = tf.nn.dropout(full_layer_one,keep_prob=hold_prob)
    
    y_pred = normal_full_layer(full_one_dropout,10)
    
    #LOSS FUNCTION
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
    
    #OPTIMIZER
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(cross_entropy)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        shape= [5,5,1,32]
        print(sess.run(tf.truncated_normal(shape,stddev=0.1)))
        
        for i in range(steps):
            batch_x, batch_y = mnist.train.next_batch(50)
            sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
            
            if i%100 == 0:
                print("ON STEP: {}".format(i))
                print("ACCURACY: ")
                
                matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
                acc = tf.reduce_mean(tf.cast(matches,tf.float32))
                print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels,hold_prob:1.0}))
                print('\n')
               
cnn_approach(steps=1)
#increase steps to 5000 to see accuracy hit 99%
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
def cnn_approach():
    
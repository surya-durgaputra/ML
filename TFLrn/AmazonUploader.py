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
plt.imshow(single_image,cmap='gist_gray')#this will show it in gray-scale. This is the image we'll be training on
#also note that the dataset has already been normalized for us: single_image.min()=0.0 and single_image.max()=1.0

def basic_approach():
    # CREATE PLACEHOLDERS
    x = tf.placeholder(tf.float32,shape=(None,784))
    # CREATE VARIABLES
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros[10])
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
    with tf.Session as sess:
        sess.run(init)
        # mnist dataset from tensorflow comes with a nice convenience method for batch data called next_data
        for step in range(1000):
            batch_x,batch_y = mnist.train.next_batch(100) #100 samples at a time
            #in our own datasets (not downloaded from tensorflow), lot of our time 
            # will be spent in cleaning datasets and adding these convenience methods
            sess.run(train,feed_dict={x:batch_x,y_true:batch_y})
        
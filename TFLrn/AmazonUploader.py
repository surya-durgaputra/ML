# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:54:29 2019

@author: DevAccessa
"""
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np

#CIFAR = r"D:\VS\Pt\ML\TFLrn\CIFAR-10_data\cifar-10-batches-py\"
CIFAR = "D:\\VS\\Pt\\ML\\TFLrn\\CIFAR-10_data\\cifar-10-batches-py\\"

def normalize_rgb_images(collection):
    return (collection - collection.min())/(collection.max() - collection.min())

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        cifar_dict = pickle.load(fo, encoding='bytes')
    return cifar_dict

dirs = ['batches.meta','data_batch_1','data_batch_2',
        'data_batch_3','data_batch_4','data_batch_5',
        'test_batch']
all_data = [0,1,2,3,4,5,6]

for i,directory in zip(all_data,dirs):
    all_data[i] = unpickle(CIFAR+directory)
    
batch_meta = all_data[0]
data_batch1 = all_data[1]
data_batch2 = all_data[2]
data_batch3 = all_data[3]
data_batch4 = all_data[4]
data_batch5 = all_data[5]
test_batch = all_data[6]


#X = data_batch1[b'data']
#X = X.reshape(10000,3,32,32).transpose(0,2,3,1).astype('uint8')
# see  explanation here for transpose: https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c
#plt.imshow(X[0])


    


class CifarHelper():
    def __init__(self):
        self.i = 0
        
        #Grabs a list of all data batches for training
        self.all_train_batches = [data_batch1,data_batch2,data_batch3,data_batch4,data_batch5]
        # Grabs a list of all the test batches (really just one batch)
        self.test_batch = [test_batch]
        
        # Intialize some empty variables for later on
        self.training_images = None
        self.training_images_scaled = None
        self.training_labels = None
        
        self.test_images = None
        self.test_images_scaled = None
        self.test_labels = None
    
    
        
    def set_up_images(self):
        print('Setting up training images and labels')
        #Vertically stack all training images
        self.training_images = np.vstack(batch[b'data'] for batch in self.all_train_batches)
        train_len = len(self.training_images)
        
        #Reshape and normalize training images
        self.training_images = self.training_images.reshape(train_len,3,32,32).transpose(0,2,3,1).astype(b'uint8')
        self.training_images_scaled = normalize_rgb_images(self.training_images)
        
        # One hot Encodes the training labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        onehotencoder_training = OneHotEncoder(sparse=False, categories='auto')
        self.training_labels = onehotencoder_training.fit_transform(np.hstack([d[b"labels"] for d in self.all_train_batches]).reshape(-1,1))
        
        print('Setting up test images and labels')
        #Vertically stack all training images
        self.test_images = np.vstack(batch[b'data'] for batch in self.test_batch)
        train_len = len(self.test_images)
        
        #Reshape and normalize training images
        self.test_images = self.test_images.reshape(train_len,3,32,32).transpose(0,2,3,1).astype(b'uint8')
        self.test_images_scaled = normalize_rgb_images(self.test_images)
        
        # One hot Encodes the training labels (e.g. [0,0,0,1,0,0,0,0,0,0])
        onehotencoder_test = OneHotEncoder(sparse=False, categories='auto')
        self.test_labels = onehotencoder_test.fit_transform(np.hstack([d[b"labels"] for d in self.test_batch]).reshape(-1,1))
        
    def next_batch(self, batch_size):
        x = self.training_images_scaled[self.i:self.i+batch_size].reshape(batch_size,32,32,3)
        y = self.training_labels[self.i:self.i+batch_size]
        self.i = (self.i + batch_size) % len(self.training_images_scaled)
        return x,y
    
def cnn_approach(steps=10, graph_interval = 5):
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
    x = tf.placeholder(tf.float32,shape=[None,32,32,3])
    y_true = tf.placeholder(tf.float32, shape=[None,10])# one hot encoded. 10 for 10 digits
    
    ### LAYERS
    # 1st convolutional layer
    # we are going to compute 32 features for each 5x5 patch
    convo_1 = convolutional_layer(x,shape=[4,4,3,32])# so the weight tensor is 4,4,3,32
    #4x4 is the patch size. 3 is the channel (rgb..so 3) and 32 corresponds to the no. of actual
    #features we are computing. So, 32 is the no. of output channels
    #now lets do the pooling layer attached to this convolution layer
    convo_1_pooling = max_pool_2by2(convo_1)
    
    # 2nd convolutional layer. Input 32 and output 64 (here we will be computing 64 features)
    convo_2 = convolutional_layer(convo_1_pooling, shape=[4,4,32,64])
    convo_2_pooling = max_pool_2by2(convo_2)
    
    convo_2_flat = tf.reshape(convo_2_pooling,[-1,8*8*64])
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
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    train = optimizer.minimize(cross_entropy)
    
    init = tf.global_variables_initializer()
    ch = CifarHelper()
    ch.set_up_images()
    with tf.Session() as sess:
        print('Total steps:{}'.format(graph_interval))
        sess.run(init)
        
        for i in range(steps):
            batch_x, batch_y = ch.next_batch(100)
            sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})
            
            if i%graph_interval == 0:
                print("ON STEP: {}".format(i))
                print('Accuracy is:')
                matches = tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
                acc = tf.reduce_mean(tf.cast(matches,tf.float32))
                
                print(sess.run(acc,feed_dict={x:ch.test_images_scaled,y_true:ch.test_labels,hold_prob:1.0}))
                print('\n')
                
cnn_approach(5000,100)
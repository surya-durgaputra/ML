# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:54:29 2019

@author: DevAccessa
"""
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np

#CIFAR = r"D:\VS\Pt\ML\TFLrn\CIFAR-10_data\cifar-10-batches-py\"
CIFAR = "D:\\VS\\Pt\\ML\\TFLrn\\CIFAR-10_data\\cifar-10-batches-py\\"

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
        self.training_labels = None
        
        self.test_images = None
        self.test_labels = None
        
    def set_up_images(self):
        print('Setting up training images and labels')
        #Vertically stack all training images
        self.training_images = np.vstack(batch[b'data'] for batch in self.all_train_batches)

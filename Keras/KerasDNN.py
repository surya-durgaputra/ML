# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:38:49 2019

@author: DevAccessa
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import *

def rhead(x, nrow = 6, ncol = 4):
    pd.set_option('display.expand_frame_repr', False)
    seq = np.arange(0, len(x.columns), ncol)
    for i in seq:
        print(x.loc[range(0, nrow), x.columns[
                range(i, min(i+ncol, len(x.columns)))]])
    pd.set_option('display.expand_frame_repr', True)
    
sales_training_df = pd.read_csv('sales_data_training.csv')
sales_test_df = pd.read_csv('sales_data_test.csv')

scaler = MinMaxScaler(feature_range=(0,1))

scaled_training_df = pd.DataFrame(scaler.fit_transform(sales_training_df),
                               columns=sales_training_df.columns,
                               index=sales_training_df.index)
scaled_test_df = pd.DataFrame(scaler.transform(sales_test_df),
                           columns=sales_test_df.columns,
                           index=sales_test_df.index)

X_train = scaled_training_df.drop(['total_earnings'], axis=1)
X_test = scaled_test_df.drop(['total_earnings'], axis=1)
y_train = scaled_training_df['total_earnings']
y_test = scaled_test_df['total_earnings']

# Define the model
# our model will take 9 inputs and predict one value
model = Sequential()
model.add(Dense(50, input_dim=9, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
#output layer
# our predicted value for earnings should be a linear value
# so we use a linear activation function
model.add(Dense(1, activation='linear'))
# final step to building a model is to compile the model
# to compile it, we need to provode it the loss function
model.compile(loss="mse", optimizer="adam")

# Train the model
model.fit(
        X_train,
        y_train,
        epochs=50,
        shuffle=true,
        verbose=2 # this will tell keras to print more detailed info
        )
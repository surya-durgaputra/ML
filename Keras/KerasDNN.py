# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 14:38:49 2019

@author: DevAccessa
"""

import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import *
from tensorflow.keras.callbacks import TensorBoard
import time

NAME = "DNN-sample-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

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
print(
      "Note: total_earnings values were "
      "scaled by multiplying by {:.10f} and adding {:.6f}".
      format(scaler.scale_[8], scaler.min_[8]))
X_train = scaled_training_df.drop(['total_earnings'], axis=1).values
X_test = scaled_test_df.drop(['total_earnings'], axis=1).values
y_train = scaled_training_df[['total_earnings']].values
y_test = scaled_test_df[['total_earnings']].values

# Define the model
# our model will take 9 inputs and predict one value
# giving names to layer is only for displaying in Tensorboard
model = Sequential()
model.add(Dense(50, input_dim=9, activation='relu', name='layer_1'))
model.add(Dense(100, activation='relu', name='layer_2'))
model.add(Dense(50, activation='relu', name='layer_3'))
#output layer
# our predicted value for earnings should be a linear value
# so we use a linear activation function
model.add(Dense(1, activation='linear', name='output_layer'))
# final step to building a model is to compile the model
# to compile it, we need to provode it the loss function
model.compile(loss="mse", optimizer="adam")

## Train the model
model.fit(
       X_train,
       y_train,
       epochs=50,
       shuffle=True, # shuffle data randomly.
       #NNs perform best on randomly shuffled data
       verbose=2 # this will tell keras to print more detailed info
       # during trainnig to know what is going on
       )

#run the test dataset
test_error_rate = model.evaluate(X_test, y_test, verbose=0)
print(
      "The mean squared error (MSE) for the test data is : {}".format(
              test_error_rate))

## After evaluating the model against a test dataset, we will now predict 
## using the model
# note: the data in proposed_new_product.csv is already scaled. 
# so we are not scaling it
X = pd.read_csv('proposed_new_product.csv')
# Make a prediction with the neural network
prediction = model.predict(X)
# Grab just the first element of the first prediction (since we only have one)
# keras always assumes that we are going to use multiple predictions with 
# multiple values for each prediction
prediction_scalar = prediction[0][0]

# Re-scale the data from the 0-to-1 range back to dollars
# These constants are from when the data was originally
# scaled down to the 0-to-1 range

reverse_scale = MinMaxScaler()
reverse_scale.min_, reverse_scale.scale_ = scaler.min_[8], scaler.scale_[8]
print("Earnings Prediction for Proposed Product - ${}".
      format(reverse_scale.inverse_transform(prediction).flatten()[0]))


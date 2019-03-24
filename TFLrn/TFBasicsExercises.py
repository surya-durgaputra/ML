#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:57:42 2019

@author: helios
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
    
def rhead(x, nrow = 6, ncol = 4):
    pd.set_option('display.expand_frame_repr', False)
    seq = np.arange(0, len(x.columns), ncol)
    for i in seq:
        print(x.loc[range(0, nrow), x.columns[range(i, min(i+ncol, len(x.columns)))]])
    pd.set_option('display.expand_frame_repr', True)

def regression_exercise():  
    df = pd.read_csv('cal_housing_clean.csv')
    x_data = df.drop('medianHouseValue',axis=1)
    y_label = df['medianHouseValue']
    X_train, X_test, y_train, y_test = train_test_split(x_data,y_label,
                                                            test_size=0.3, 
                                                            random_state=101)
    
    #scale the features. 
    # recreate DF since MInMaxScaler returns numpy arrays
    mm_scaler = MinMaxScaler()
    x_train2 = pd.DataFrame(mm_scaler.fit_transform(X_train))
    x_test2 = pd.DataFrame(mm_scaler.transform(X_test))
    x_train2.columns = x_test2.columns = X_train.columns.values
    x_train2.index = X_train.index.values
    x_test2.index = X_test.index.values
    
    X_train = x_train2
    X_test = x_test2
    
    #Create feature columns
    median_age = tf.feature_column.numeric_column('housingMedianAge')
    total_rooms = tf.feature_column.numeric_column('totalRooms')
    total_bedrooms = tf.feature_column.numeric_column('totalBedrooms')
    population = tf.feature_column.numeric_column('population')
    households = tf.feature_column.numeric_column('households')
    median_income = tf.feature_column.numeric_column('medianIncome')
    
    feat_cols = [median_age,total_rooms,total_bedrooms,
                 population,households,median_income]
    
    #create input functions
    input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,
                                                     batch_size=10,
                                                     num_epochs=1000,
                                                     shuffle=True)
    train_input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,
                                                     batch_size=10,
                                                     num_epochs=1,
                                                     shuffle=False)
    eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,
                                                     batch_size=10,
                                                     num_epochs=1,
                                                     shuffle=False)
    dnn_model = tf.estimator.DNNRegressor(hidden_units=[10,10,10],
                                          feature_columns=feat_cols,
                                          activation_fn=tf.nn.relu)
    #now lets train the model
    dnn_model.train(input_fn=input_func,steps=20000)
    #evaluate
    training_results = dnn_model.evaluate(train_input_func)
    results = dnn_model.evaluate(eval_input_func)
    
    #training_results: 
    ##{'average_loss': 55839347000.0, 'loss': 558316200000.0, 'global_step': 1000}
    ##{'average_loss': 55745642000.0, 'loss': 557379300000.0, 'global_step': 20000}
    #results: 
    ##{'average_loss': 56626800000.0, 'loss': 565537340000.0, 'global_step': 1000}
    ##{'average_loss': 56532570000.0, 'loss': 564596240000.0, 'global_step': 20000}
    
    #prediction
    pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,
                                                     batch_size=10,
                                                     num_epochs=1,
                                                     shuffle=False)
    predictions = dnn_model.predict(pred_input_func)
    
    y_pred = [pred["predictions"] for pred in predictions]
    
    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    print('RMSE: ',rmse)
    
df = pd.read_csv('census_data.csv')
df.income_bracket = (df.income_bracket==' <=50K').astype(int)
x_data = df.drop('income_bracket',axis=1)
y_label = df['income_bracket']
X_train, X_test, y_train, y_test = train_test_split(x_data,y_label,
                                                        test_size=0.3, 
                                                        random_state=42)
#create feature columns for tf.estimator
age = tf.feature_column.numeric_column('age')
workclass = tf.feature_column.categorical_column_with_hash_bucket('workclass',hash_bucket_size=50000)
workclass = tf.feature_column.embedding_column(workclass,dimension=9)

education = tf.feature_column.categorical_column_with_hash_bucket('education',hash_bucket_size=50000)
education = tf.feature_column.embedding_column(education,dimension=16)

education_num = tf.feature_column.numeric_column('education_num')
marital_status = tf.feature_column.categorical_column_with_hash_bucket('marital_status',hash_bucket_size=50000)
marital_status = tf.feature_column.embedding_column(marital_status,dimension=7)

occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation',hash_bucket_size=50000)
occupation = tf.feature_column.embedding_column(occupation,dimension=15)

relationship = tf.feature_column.categorical_column_with_hash_bucket('relationship',hash_bucket_size=50000)
relationship = tf.feature_column.embedding_column(relationship,dimension=6)

race = tf.feature_column.categorical_column_with_hash_bucket('race',hash_bucket_size=50000)
race = tf.feature_column.embedding_column(race,dimension=5)

gender = tf.feature_column.categorical_column_with_hash_bucket('gender',hash_bucket_size=50000)
gender = tf.feature_column.embedding_column(gender,dimension=2)

capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')
native_country = tf.feature_column.categorical_column_with_hash_bucket('native_country',hash_bucket_size=50000)
native_country = tf.feature_column.embedding_column(native_country,dimension=42)

feat_cols = [age,workclass,education,education_num,marital_status,
             occupation,relationship,race,gender,capital_gain,
             capital_loss,hours_per_week,native_country]

# create input function
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,
                                                 batch_size=1000,
                                                 num_epochs=1000,
                                                 shuffle=True)
train_input_func = tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,
                                                 batch_size=1000,
                                                 num_epochs=1,
                                                 shuffle=False)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,
                                                 batch_size=1000,
                                                 num_epochs=1,
                                                 shuffle=False)

#create model
dnn_model = tf.estimator.DNNRegressor(hidden_units=[10,10,10],
                                      feature_columns=feat_cols,
                                      model_dir='models/TFBasicsExercisesDNNClassifier',
                                      activation_fn=tf.nn.relu)

#now lets train the model
dnn_model.train(input_fn=input_func,steps=1000)
#evaluate
training_results = dnn_model.evaluate(train_input_func)
test_results = dnn_model.evaluate(eval_input_func)

#prediction
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,
                                                 batch_size=len(X_test),
                                                 shuffle=False)
predictions_gen = dnn_model.predict(pred_input_func)
y_pred = [int(np.rint(pred["predictions"])) for pred in list(predictions_gen)]

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

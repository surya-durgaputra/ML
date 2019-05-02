#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:23:41 2019

@author: helios
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras import backend as K

def rhead(x, nrow = 6, ncol = 4):
    pd.set_option('display.expand_frame_repr', False)
    seq = np.arange(0, len(x.columns), ncol)
    for i in seq:
        print(x.loc[range(0, nrow), x.columns[
                range(i, min(i+ncol, len(x.columns)))]])
    pd.set_option('display.expand_frame_repr', True)

titanic_df = pd.read_csv('titanic/train.csv')
titanic_test_df = pd.read_csv('titanic/test.csv')
seed = 42

useless_cols = ['Name','PassengerId','Ticket']
titanic_df = titanic_df.drop(useless_cols, axis=1)
passengerIds_test = titanic_test_df[['PassengerId']]
titanic_test_df = titanic_test_df.drop(useless_cols, axis=1)
def clean_feature_columns(df, feature_columns):
    """ check if feature columns have any NaNs.
    Feature columns are used in groupby.
    They should not have any nans.
    """
    clean_cols = []
    for col in feature_columns:
        if df[col].isnull().any():
            continue
        else:
            clean_cols.append(col)
    return clean_cols

def fill_numeric_col(df, feature_columns, target_column):
    filled_col_name = target_column + '_filled'
    if df[target_column].isnull().any():
        df[filled_col_name] = df.groupby(feature_columns)[target_column]\
            .transform(lambda x: x.fillna(np.nan if pd.isnull(x.median()) else x.median()))
        while True:
            if df[filled_col_name].isnull().any():
                if len(feature_columns) > 1:
                    del feature_columns[-1]
                    df[filled_col_name] = df.groupby(feature_columns)[filled_col_name]\
            .transform(lambda x: x.fillna(np.nan if pd.isnull(x.median()) else x.median()))
                else:
                    df[filled_col_name] = df[filled_col_name].fillna(df[target_column].median(dropna=True))
                    break
            else:
                break
        return df
    else:
        df[filled_col_name] = df[target_column]
        return df

def fill_categoric_col(df, feature_columns, target_column):
    filled_col_name = target_column + '_filled'
    if df[target_column].isnull().any():
        df[filled_col_name] = df.groupby(feature_columns)[target_column]\
            .transform(lambda x: x.fillna(np.nan if x.count()<=0 else x.mode()[0]))
        while True:
            if df[filled_col_name].isnull().any():
                if len(feature_columns) > 1:
                    del feature_columns[-1]
                    df[filled_col_name] = df.groupby(feature_columns)[filled_col_name]\
            .transform(lambda x: x.fillna(np.nan if x.count()<=0 else x.mode()[0]))
                else:
                    df[filled_col_name] = df[filled_col_name].fillna(df[target_column].mode(dropna=True))
                    break
            else:
                break
        return df
    else:
        df[filled_col_name] = df[target_column]
        return df

titanic_df = fill_numeric_col(titanic_df,clean_feature_columns(titanic_df,['Sex', 'Pclass','Parch']),'Age')
titanic_df = fill_categoric_col(titanic_df,clean_feature_columns(titanic_df,['Sex', 'Pclass','Parch']),'Embarked')
titanic_df = fill_categoric_col(titanic_df,clean_feature_columns(titanic_df,['Sex', 'Pclass','Parch']),'Cabin')

titanic_test_df = fill_numeric_col(titanic_test_df,clean_feature_columns(titanic_test_df,['Sex', 'Pclass','Parch']),'Age')
titanic_test_df = fill_categoric_col(titanic_test_df,clean_feature_columns(titanic_test_df,['Sex', 'Pclass','Parch','SibSp']),'Embarked')
titanic_test_df = fill_categoric_col(titanic_test_df,clean_feature_columns(titanic_test_df,['Sex', 'Pclass','Parch','SibSp','Fare']),'Cabin')

#titanic_df['under15'] = titanic_df['Age_filled'].apply(under15)
#titanic_test_df['under15'] = titanic_test_df['Age_filled'].apply(under15)
#titanic_df['young'] = titanic_df['Age_filled'].apply(young)
#titanic_test_df['young'] = titanic_test_df['Age_filled'].apply(young)

y = titanic_df[['Survived']]
X = titanic_df.drop(['Survived','Age','Embarked','Cabin'], axis=1)
X = pd.get_dummies(X, columns=['Sex','Pclass','Cabin_filled','Embarked_filled'])

titanic_test = titanic_test_df.drop(['Age','Embarked','Cabin'], axis=1)
titanic_test = pd.get_dummies(titanic_test_df, columns=['Sex','Pclass','Cabin_filled','Embarked_filled'])


scaler = MinMaxScaler(feature_range=(0,1))

X_train,X_test,y_train,y_test = train_test_split(X,
                                                 y,
                                                 test_size=0.30,
                                                 random_state=seed)

X_train = pd.DataFrame(scaler.fit_transform(X_train),
                               columns=X_train.columns,
                               index=X_train.index)

X_test = pd.DataFrame(scaler.transform(X_test),
                           columns=X_test.columns,
                           index=X_test.index)



def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def model():
    """build the Keras model callback"""
    model = Sequential()
    model.add(Dense(10, input_dim=159, activation='tanh', name='layer_1'))
    model.add(Dense(5, activation='tanh', name='layer_2'))
    model.add(Dense(5, activation='tanh', name='layer_3'))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='Zeros', name='output_layer'))
    
    model.compile(loss="binary_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy', f1_m,precision_m, recall_m])
    return model

estimator = KerasClassifier(
        build_fn=model,
        epochs=1, batch_size=20,
        verbose=2)
kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X_train, y_train, cv=kfold)
print("Model Performance: mean: %.2f%% std: (%.2f%%)" % (results.mean()*100, results.std()*100))
model = model()
model.fit(
       X_train,
       y_train,
       epochs=200,
       shuffle=True, # shuffle data randomly.
       #NNs perform best on randomly shuffled data
       verbose=2 # this will tell keras to print more detailed info
       # during trainnig to know what is going on
       )

#run the test dataset
train_error_rate = model.evaluate(X_train, y_train, verbose=0)
print(
      "{} : {:.2f}%".format(model.metrics_names[1],
              train_error_rate[1]*100))
print(
      "{} : {:.2f}%".format(model.metrics_names[2],
              train_error_rate[2]*100))
print(
      "{} : {:.2f}%".format(model.metrics_names[3],
              train_error_rate[3]*100))
print(
      "{} : {:.2f}%".format(model.metrics_names[4],
              train_error_rate[4]*100))

test_error_rate = model.evaluate(X_test, y_test, verbose=0)
print(
      "{} : {:.2f}%".format(model.metrics_names[1],
              test_error_rate[1]*100))
print(
      "{} : {:.2f}%".format(model.metrics_names[2],
              test_error_rate[2]*100))
print(
      "{} : {:.2f}%".format(model.metrics_names[3],
              test_error_rate[3]*100))
print(
      "{} : {:.2f}%".format(model.metrics_names[4],
              test_error_rate[4]*100))


############Prediction##########################

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score,recall_score,confusion_matrix
#
#random_forest = RandomForestClassifier(n_estimators=100)
#random_forest.fit(X_train, y_train.Survived.values)
#y_pred = random_forest.predict(X_test)
#random_forest.score(X_train, y_train.Survived.values)
#
#print(accuracy_score(y_test,y_pred))
#print(recall_score(y_test,y_pred))
#print(confusion_matrix(y_test,y_pred))

#t = titanic_test_df.groupby(clean_feature_columns(titanic_test_df,['Sex', 'Pclass','Parch','SibSp','Fare']))['Cabin']
#t1 = titanic_test_df.groupby(['Sex', 'Pclass','Parch','SibSp'])['Cabin']
#counter = 0
#list_inx = []
#for x in t:
#    #list_inx.append(x[1].index.values)
#    list_inx = np.concatenate((list_inx,x[1].index.values),axis=None)
#    #print(x[1].index.values)
#    counter += len(x[1])
#print(counter)
#
#counter1 = 0
#list_inx1 = []
#for x in t1:
#    #list_inx.append(x[1].index.values)
#    list_inx1 = np.concatenate((list_inx1,x[1].index.values),axis=None)
#    #print(x[1].index.values)
#    counter1 += len(x[1])
#print(counter1)
#
#for x in list_inx1:
#    if x not in list_inx:
#        print("not in list_inx:",x)

#.transform(lambda x: x.fillna(np.nan if x.count()<=0 else x.mode()[0]))
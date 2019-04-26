# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 09:26:57 2019

@author: DevAccessa
"""

from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

def rhead(x, nrow = 6, ncol = 4):
    pd.set_option('display.expand_frame_repr', False)
    seq = np.arange(0, len(x.columns), ncol)
    for i in seq:
        print(x.loc[range(0, nrow), x.columns[
                range(i, min(i+ncol, len(x.columns)))]])
    pd.set_option('display.expand_frame_repr', True)


encoder = LabelBinarizer()
seed = 42

iris = datasets.load_iris()
iris_data_df = pd.DataFrame(data=iris.data, columns=iris.feature_names,
                       dtype=np.float32)
target = encoder.fit_transform(iris.target)
iris_target_df = pd.DataFrame(data=target, columns=iris.target_names)

X_train,X_test,y_train,y_test = train_test_split(iris_data_df,
                                                 iris_target_df,
                                                 test_size=0.30,
                                                 random_state=seed)

scaler = MinMaxScaler(feature_range=(0,1))

X_train = pd.DataFrame(scaler.fit_transform(X_train),
                               columns=X_train.columns,
                               index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test),
                           columns=X_test.columns,
                           index=X_test.index)

def model():
    """build the Keras model callback"""
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='tanh', name='layer_1'))
    model.add(Dense(10, activation='tanh', name='layer_2'))
    model.add(Dense(10, activation='tanh', name='layer_3'))
    model.add(Dense(3, activation='softmax', name='output_layer'))
    
    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
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
test_error_rate = model.evaluate(X_test, y_test, verbose=0)
print(
      "{} : {:.2f}%".format(model.metrics_names[1],
              test_error_rate[1]*100))
print(
      "{} : {:.2f}%".format(model.metrics_names[0],
              test_error_rate[0]*100))

## Lets do prediction
predicted_targets = model.predict_classes(X_test)
true_targets = encoder.inverse_transform(y_test.values)

def performance_tracker(actual, expected):
    flowers = {0:'setosa', 1:'versicolor', 2:'virginica'}
    print("Flowers in test set: Setosa={} Versicolor={} Virginica={}".format(
            y_test.setosa.sum(), y_test.versicolor.sum(),
            y_test.virginica.sum()))
    for act,exp in zip(actual, expected):
        if act != exp:
            print("ERROR: {} predicted as {}".format(flowers[exp],
                  flowers[act]))
            
performance_tracker(predicted_targets, true_targets)
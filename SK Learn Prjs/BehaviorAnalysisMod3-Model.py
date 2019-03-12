# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 13:21:18 2019

@author: DevAccessa
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

def rhead(x, nrow = 6, ncol = 4):
    pd.set_option('display.expand_frame_repr', False)
    seq = np.arange(0, len(x.columns), ncol)
    for i in seq:
        print(x.loc[range(0, nrow), x.columns[range(i, min(i+ncol, len(x.columns)))]])
    pd.set_option('display.expand_frame_repr', True)

dataset = pd.read_csv('new_appdata10.csv')

#### Data Pre-Processing ####

# Splitting Independent and Response Variables
response = dataset["enrolled"]
dataset = dataset.drop(columns=['enrolled'])

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset, response,
                                                    test_size = 0.2,
                                                    random_state = 0)

# Removing Identifiers
# We are not using user column right now but in the end
# we will need to associate our predictions with users.
identity_train = X_train['user']
identity_test = X_test['user']

X_train = X_train.drop(columns=['user'])
X_test = X_test.drop(columns=['user'])

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
# note that StandardScaler returns a numpy array. So we lose the indexes (which
# we will need to identify instances) and column names. To deal with this,
# we feed the result StandardScaler into a new dataframe and add the indexes 
# and column names to it

X_train2 = pd.DataFrame(scaler_X.fit_transform(X_train))
X_test2 = pd.DataFrame(scaler_X.transform(X_test))
X_train2.columns = X_test2.columns = X_train.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values

X_train = X_train2
X_test = X_test2

#### Model Building ####


# Fitting Model to the Training Set
from sklearn.linear_model import LogisticRegression
# if one screen is correlated to another (like if Loan
# and Credit buttons are next to eachother in the app,
# their screens will have high correlation among eachother
# since one visiting Loan might visit Credit simply because
# they are next to eachother. 
# Also, say a feature X may have strong correlation with 
# response variable as users who visit X mostly enroll.
# This can cause features W and Y also have high correlation
# to response variable by simply being screens next to X. 
# High correlation means higher weights. Large weights
# do not lead to smooth curves and cause overfitting. 
# Regularization, L1 or L2 helps here by ensuring that 
# all coefficients (weights) are small.
# see here fr an amazing explanation:
# https://www.youtube.com/watch?v=6B6C1gsGInU
classifier = LogisticRegression(random_state=0, penalty='l1')
classifier.fit(X_train,y_train)

# Predicting test set
y_pred = classifier.predict(X_test)

#Evaluating Results
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred) #(tp + tn) / m : m is the sample size
precision_score(y_test, y_pred) # tp / (tp + fp)
recall_score(y_test, y_pred) # tp / (tp + fn)
f1_score(y_test, y_pred) # 2*Precision*Recall/(Precision + Recall)
print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
print("Test Data Precision: %0.4f" % precision_score(y_test, y_pred))
print("Test Data Recall: %0.4f" % recall_score(y_test, y_pred))
print("Test Data F1: %0.4f" % f1_score(y_test, y_pred))

# since all the above evaluation scores are in 76% range, our model preforms good

# Visualize the results in a nice confusion matrix visualization
df_cm = pd.DataFrame(cm, index=(0,1),columns=(0,1))
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)
sns.heatmap(df_cm, annot=True, fmt='g')

# now lets double check if the model overfits the data by doing KFold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("SVM Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

# Analyzing Coefficients
# i.e. what coefficient each feature had
pd.concat([pd.DataFrame(dataset.drop(columns = 'user').columns, columns = ["features"]),
           pd.DataFrame(np.transpose(classifier.coef_), columns = ["coef"])
           ],axis = 1)

#### Model Tuning ####

## Grid Search (Round 1)
from sklearn.model_selection import GridSearchCV

# Select Regularization Method
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
# C is actually lambda for l1,l2
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

# Combine Parameters
parameters = dict(C=C, penalty=penalty)

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)
t0 = time.time()
grid_search = grid_search.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy = grid_search.best_score_
rf_best_parameters = grid_search.best_params_
rf_best_accuracy, rf_best_parameters


## Grid Search (Round 2)

# Select Regularization Method
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = [0.1, 0.5, 0.9, 1, 2, 5]

# Combine Parameters
parameters = dict(C=C, penalty=penalty)

grid_search_2 = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = "accuracy",
                           cv = 10,
                           n_jobs = -1)
t0 = time.time()
grid_search_2 = grid_search_2.fit(X_train, y_train)
t1 = time.time()
print("Took %0.2f seconds" % (t1 - t0))

rf_best_accuracy_2 = grid_search_2.best_score_
rf_best_parameters_2 = grid_search_2.best_params_
rf_best_accuracy_2, rf_best_parameters_2
grid_search_2.best_score_

#### End of Model ####


# Formatting Final Results
final_results = pd.concat([y_test, identity_test], axis = 1).dropna()
final_results['predicted_reach'] = y_pred
final_results = final_results[['user', 'enrolled', 'predicted_reach']].reset_index(drop=True)

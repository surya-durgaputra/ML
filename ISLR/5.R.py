# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:13:54 2019

@author: DevAccessa
"""

import pandas as pd
import numpy as np
from sklearn import linear_model, utils
import seaborn as sns

df = pd.read_csv('C:/Users/DevAccessa/Downloads/Xy.csv')

regr = linear_model.LinearRegression()
X = df.drop(columns=['y'])
y = df[['y']]

regr.fit(X,y)

acc = 0

itr = 1000
b1 = []
for _ in range(itr):
    boot = utils.resample(X, replace=True, n_samples=df.shape[0])
    regr = linear_model.LinearRegression()
    regr.fit(boot,y)
    b1.append(regr.coef_[0][0])
    
b1.sort()

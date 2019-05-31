# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:22:42 2019

@author: DevAccessa
"""

import pandas as pd
import numpy as np
from sklearn import linear_model, utils
import seaborn as sns

df = pd.read_csv('D:/VS/Pt/ML/Keras/hitters.csv')

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

from yellowbrick.regressor import AlphaSelection

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[:150]
#X = diabetes.data
#y = diabetes.target

alphas = np.logspace(-5, -0.5, 30)

# Instantiate the linear model and visualizer
model = LassoCV(alphas=alphas)
visualizer = AlphaSelection(model)

visualizer.fit(X, y)
g = visualizer.poof()


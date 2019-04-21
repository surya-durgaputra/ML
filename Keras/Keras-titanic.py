#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:23:41 2019

@author: helios
"""

import pandas as pd
import numpy as np

def rhead(x, nrow = 6, ncol = 4):
    pd.set_option('display.expand_frame_repr', False)
    seq = np.arange(0, len(x.columns), ncol)
    for i in seq:
        print(x.loc[range(0, nrow), x.columns[
                range(i, min(i+ncol, len(x.columns)))]])
    pd.set_option('display.expand_frame_repr', True)

titanic_df = pd.read_csv('titanic.csv')
useless_cols = ['PassengerId','Name','Ticket']
titanic_df = titanic_df.drop(useless_cols, axis=1)
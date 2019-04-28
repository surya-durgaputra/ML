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

titanic_df = pd.read_csv('titanic/train.csv')
titanic_test_df = pd.read_csv('titanic/test.csv')

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
titanic_df = fill_categoric_col(titanic_df,clean_feature_columns(titanic_df,['Sex', 'Pclass','Parch','SibSp']),'Embarked')
titanic_df = fill_categoric_col(titanic_df,clean_feature_columns(titanic_df,['Sex', 'Pclass','Parch','SibSp','Fare']),'Cabin')

titanic_test_df = fill_numeric_col(titanic_test_df,clean_feature_columns(titanic_test_df,['Sex', 'Pclass','Parch']),'Age')
titanic_test_df = fill_categoric_col(titanic_test_df,clean_feature_columns(titanic_test_df,['Sex', 'Pclass','Parch','SibSp']),'Embarked')
titanic_test_df = fill_categoric_col(titanic_test_df,clean_feature_columns(titanic_test_df,['Sex', 'Pclass','Parch','SibSp','Fare']),'Cabin')

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
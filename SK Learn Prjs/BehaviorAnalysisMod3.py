#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 08:37:18 2019

@author: helios
"""
import numpy as np
import pandas as pd
import seaborn as sns
from dateutil import parser
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

#pd.set_option('display.expand_frame_repr', False)
#pd.set_option('display.max_columns', None)  # or 1000
#pd.set_option('display.max_rows', None)  # or 1000
#pd.set_option('display.max_colwidth', -1)  # or 199

#a function to pretty print dataframe columns
# based on r's head() function
def rhead(x, nrow = 6, ncol = 4):
    pd.set_option('display.expand_frame_repr', False)
    seq = np.arange(0, len(x.columns), ncol)
    for i in seq:
        print(x.loc[range(0, nrow), x.columns[range(i, min(i+ncol, len(x.columns)))]])
    pd.set_option('display.expand_frame_repr', True)

dataset = pd.read_csv('appdata10.csv')

dataset["hour"] = dataset.hour.str.slice(1,3).astype(int)

### Plotting
# we remove all non numeric colmns as well as the response variable 'enrolled'
# Here we will plot a histogram of the correlation of numeric features with response variable 'enrolled'
dataset2 = dataset.copy().drop(columns=['user','first_open','screen_list','enrolled_date','enrolled'])

## Histograms
# plot these to take a look at the distribution of the numeric features
fig1 = plt.figure(figsize=(20,10))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(1, dataset2.shape[1] + 1):
    plt.subplot(3, 3, i)
    f = plt.gca()
#    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i - 1])

    vals = np.size(dataset2.iloc[:, i - 1].unique())
    
    plt.hist(dataset2.iloc[:, i - 1], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#plt.savefig('app_data_hist.jpg')

fig2 = plt.figure(figsize=(20,10))
## Correlation with Response Variable
dataset2.corrwith(dataset.enrolled).plot.bar(
                  title = 'Correlation with Reposnse variable',
                  fontsize = 15, rot = 45,
                  grid = True)

# Correlation matrix is a great way to see which features are independent
# ML algorithms generally work well with independent features
# A correlation matrix is generally a heatmap

## Correlation Matrix
sns.set(style="white", font_scale=2)

# Compute the correlation matrix
corr = dataset2.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(30, 15))
f.suptitle("Correlation Matrix", fontsize = 40)

# Generate a custom diverging colormap... Note.. not used as this color map not very clear
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True, fmt='.2f')

#### Feature Engineering ####

# Formatting the date column
dataset["first_open"] = [parser.parse(row_data) for row_data in dataset.first_open]
dataset["enrolled_date"] = [parser.parse(row_data) if isinstance(row_data,str) else row_data for row_data in dataset.enrolled_date]

dataset["difference"] = (dataset.enrolled_date-dataset.first_open).astype('timedelta64[h]')
plt.figure()
plt.hist(dataset.difference.dropna(), color='#3F5D7D')
plt.title('Distribution of Time-since-screen-reached')

dataset["difference"] = (dataset.enrolled_date-dataset.first_open).astype('timedelta64[h]')
plt.figure()
plt.hist(dataset.difference.dropna(), color='#3F5D7D', range=[1,500])
plt.title('Distribution of Time-since-screen-reached')

# to get a better understanding of within what time limit majority of the users lie
max_diff = np.max(dataset.difference)
fig = plt.figure(figsize=(20,10))
fig.add_subplot(221) 
dataset["proportion"] = [1 if diff <= max_diff else 0 for diff in dataset.difference]
sums = dataset.user.groupby(dataset.proportion).sum()
plt.axis('equal');plt.pie(sums, autopct='%1.0f%%', labels=[0,1]);
plt.title('Proportion of people enrolled in first '+ str(max_diff) +' hours')

#plt.figure(figsize=(20,10))
fig.add_subplot(222)
dataset["proportion"] = [1 if diff <= 100 else 0 for diff in dataset.difference]
sums = dataset.user.groupby(dataset.proportion).sum()
plt.axis('equal');plt.pie(sums, autopct='%1.0f%%', labels=[0,1]);
plt.title('Proportion of people enrolled in first 100 hours.')

#plt.figure(figsize=(20,10))
fig.add_subplot(223)
dataset["proportion"] = [1 if diff <= 48 else 0 for diff in dataset.difference]
sums = dataset.user.groupby(dataset.proportion).sum()
plt.axis('equal');plt.pie(sums, autopct='%1.0f%%', labels=[0,1]);
plt.title('Proportion of people enrolled in first 48 hours')


# so within 48 hours majority of people enroll
# so all the people who are currently not enrolled are already 0 ('enrolled' = 0)
# now we will also set everyone who enrolled after 48 hours as 0
dataset.loc[dataset.difference>48,'enrolled'] = 0

# now that we have utilized the dates and time to filter the list of people 
# we will drop the columns no longer needed
dataset = dataset.drop(columns=['enrolled_date', 'difference', 'first_open','proportion'])

# from our domain knowledge of the product, we know : the top screens used (we actually have
# it as a csv file) we will convert the top used screen into columns
top_screens = pd.read_csv('top_screens.csv')
top_screens = top_screens.top_screens.values

dataset["screen_list"] = dataset.screen_list.astype('str') + ','

for screen in top_screens:
    dataset[screen] = dataset.screen_list.str.contains(screen).astype('int')
    dataset.screen_list.str.replace(screen+',',"")

dataset["Other"] = dataset.screen_list.str.count(',')
dataset = dataset.drop(columns=['screen_list'])

#Funnels : these are groups of correlated screens
# we dont want correlated screens as correlated features make ML algos give bad results
# we makes these lists of correlated screens through domain knowledge of the product

#following screens are all related to savings
savings_screens = ["Saving1",
                    "Saving2",
                    "Saving2Amount",
                    "Saving4",
                    "Saving5",
                    "Saving6",
                    "Saving7",
                    "Saving8",
                    "Saving9",
                    "Saving10"]
# group them as one column
dataset['SavingCount'] = dataset[savings_screens].sum(axis=1)
#now remove these columns
dataset.drop(columns=savings_screens)

# do the same for all other related screens


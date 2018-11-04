# -*- coding: utf-8 -*-
'''
Created on Sun Jul 29 12:19:43 2018

@author: takalyan
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
from sklearn.cluster import MiniBatchKMeans
import pickle


import ml_functions as dp


#Import dataset
dataset_df = pd.read_csv('train.csv')


def plot_relplot(df,x,y):
    
    sns.relplot(x=x,y=y,
    data=df,aspect=1,height=5)
    
    plt.show()

##########################
#Exploratory Data Analysis
##########################

#Explore the data types
dataset_df.info()

#understand feature data types
dataset_df.get_dtype_counts()

#preview the data
preview_df = dataset_df.head()

#Summary stats of features
describe_df = dataset_df.describe()


#Show unique values
dataset_df.ExterQual.unique()
#Show counts of categorical variables
dataset_df.ExterQual.value_counts()

#List of all columns
col_list = list(dataset_df.columns)



#Display columns with missing data  
dp.display_missing_cols(dataset_df)


#display categorical features
print(dataset_df.dtypes[dataset_df.dtypes=='object'])

#Column list of all categorical features
obj_cols = list(dataset_df.select_dtypes(include=['object']).columns)


##### Correlation Maps #########

#Plot correlation matrix
dp.plot_heatmap(dataset_df)


#saleprice correlation matrix
#define top/bottom n features we want to look at vs target variable
n = 10
target = 'SalePrice'

#Plot features with highest correlation to target
highcorr_cols = dp.plot_tb_heatmap(dataset_df,target,n,'largest')
for cols in highcorr_cols:
    plot_relplot(dataset_df,cols,target)
    
#Plot features with lowest correlation to target    
lowcorr_cols = dp.plot_tb_heatmap(dataset_df,target,n,'smallest')
for cols in lowcorr_cols:
    plot_relplot(dataset_df,cols,target)



#### Visualizations ######


#Visualize target variable distribution
sns.distplot(dataset_df["SalePrice"])

#plot histo of all numerical values
dp.plot_all_hist(dataset_df)


    
#Pattern for anything that is an area feature 
pattern = re.compile('.*(area|sf).*',re.I)
#Store all area columns/features here
area_cols = []

#Iterate through df columns and plot vs target variable
for col in col_list:
   if (re.match(pattern,col)):
       area_cols.append(col)
       plot_relplot(dataset_df,col,'SalePrice')



#Pairplots
sns.pairplot(dataset_df,height=2,hue='OverallQual', vars=[
  'LotArea',
 'LotFrontage',
 'YearBuilt',
 'GrLivArea',
 'SalePrice'])


#Printing quantiles
var = dataset_df.LotArea
var.describe()
print(var.quantile([.5,0.75,0.90,0.95,0.99,1]))


#Rel plots
sns.relplot(x='LotArea',y='SalePrice',
            col='YrSold', col_wrap=2,
            data=dataset_df,aspect=1,height=8,)

#Cat plots
sns.catplot(x='MSZoning',y='SalePrice', data=dataset_df,
            kind = 'box',height=8
            )





################
# Data Cleansing
################



#Removing outliers for GrLivArea
plot_relplot(dataset_df,'GrLivArea','SalePrice')
new_df = dataset_df.drop(dataset_df[(dataset_df['GrLivArea']>4000) & (dataset_df['SalePrice'] < 300000)].index)
plot_relplot(new_df,'GrLivArea','SalePrice')

#Drop features missing > 90% of trainset
dp.display_missing_cols(dataset_df)
cols_to_drop = ['Alley','MiscFeature','PoolQC']
new_df = dataset_df.drop(cols_to_drop,axis=1)

#Drop low correlation vars
cols_to_drop = lowcorr_cols
new_df = new_df.drop(cols_to_drop,axis=1)

#Drop redundant, duplicate features
cols_to_drop = ['1stFlrSF','GarageCars']
new_df = new_df.drop(cols_to_drop,axis=1)


#Remove outliers using quantile values
col_list = ['GrLivArea','LotArea']
plot_relplot(dataset_df,'GrLivArea','SalePrice')
new_df = dp.remove_outliers_quant(new_df,col_list,0.99)
plot_relplot(new_df,'GrLivArea','SalePrice')

#Encode categorical variables
encoder, new_df  = dp.encode_df(new_df,'label')

#impute missing 
new_df = dp.impute_missing_mean(new_df)



####################
#Feature Engineering
####################


#Log transformations

#plot target var distribution
dp.plot_normplot(dataset_df,'SalePrice')


new_df["SalePrice"] = np.log(new_df["SalePrice"])
dp.plot_normplot(new_df,'SalePrice')

new_df['GrLivArea'] = np.log(new_df['GrLivArea'])


#Create a 2D array required for Kmeans clustering
lotArea_array = np.array(new_df["LotArea"]).reshape(-1,1)
#Use Kmeans with 5 groups for clustering data
kmeans = MiniBatchKMeans(n_clusters=5, batch_size=32).fit(lotArea_array)

#Assign new feature based on cluster
new_df['area_cluster'] = kmeans.predict(lotArea_array)

#Plot the clusters
sns.relplot(x='LotArea',y='LotFrontage',data=new_df,
            hue='area_cluster',style='area_cluster',height=8,
            palette='YlGnBu')



####################
# Save data
####################

#Save clean data to csv
#Random shuffle
new_df = new_df.sample(frac=1).reset_index(drop=True)

#Define train and test set
test_pct = 0.2
test_row_count = np.int(test_pct * new_df.shape[0])
train_row_count = new_df.shape[0] - test_row_count

#Create train and validation dataframe and dump to csv
train_df = new_df.loc[0:train_row_count-1,:]
test_df =  new_df.loc[train_row_count:,:]

train_df.to_csv('new_train.csv',index=False)
test_df.to_csv('new_test.csv',index=False)

pickle.dump(kmeans,open ('cluster_model.pickle','wb'))
pickle.dump(encoder,open('encoder.pickle','wb'))


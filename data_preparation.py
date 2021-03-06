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
from sklearn.decomposition import PCA
from scipy.stats import  skew 
from scipy.special import boxcox1p

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
print(dataset_df.info())


#understand feature data types
print(dataset_df.get_dtype_counts())

#preview the data
preview_df = dataset_df.head()

#Summary stats of features
describe_df = dataset_df.describe()



#List of all columns
col_list = list(dataset_df.columns)



#Display columns with missing data  
dp.display_missing_colrows(dataset_df,0)
#Display rows with missing data  
dp.display_missing_colrows(dataset_df,1)




#List of all categorical features
print(dataset_df.select_dtypes(include=['object']).columns)


##### Correlation Maps #########


#Plot correlation matrix
dp.plot_heatmap(dataset_df,vmax=0.8,vmin=-0.8)


#Add features with similarity to be removed later
cols_to_remove = ['GarageCars','GarageYrBlt']


#Get correlation values for feature pairs
corr_vals = dp.get_corrvals(dataset_df)

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
print(var.quantile([.5,0.75,0.90,0.95,0.99,1]))


#Rel plots
sns.relplot(x='LotArea',y='SalePrice',
            col='YrSold', col_wrap=2,
            data=dataset_df,aspect=1,height=8,)

#Cat plots
sns.catplot(x='OverallQual',y='SalePrice', data=dataset_df,
            kind = 'box',height=8
            )


#Check for skewness of features
numerical_feats = dataset_df.dtypes[dataset_df.dtypes != "object"].index
skewed_feats = dataset_df[numerical_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})

#Visualize skewness
sns.distplot(dataset_df['PoolArea'])

################
# Data Cleansing
################

#Copy dataframe
new_df=dataset_df.copy()




# Based on the description of the dataset, anything missing here is actually N/A 
# We should replace the values with "None" so that they are encoded  to a specific value
cols_to_mod = ['MSSubClass','MiscFeature','Alley','Fence','FireplaceQu','GarageType', 'GarageFinish',\
 'GarageQual','GarageCond','PoolQC','BsmtQual', 'BsmtCond', 'BsmtExposure',\
 'BsmtFinType1', 'BsmtFinType2','MasVnrType']

#Replace the values with "None"
for col in cols_to_mod:
    new_df[col] = dataset_df[col].fillna("None")


# Based on the description of the dataset, anything missing here is actually N/A 
# We should replace the values with 0 so that they are encoded  to a specific value
cols_to_mod = ['MasVnrArea','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',\
                'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

#Fill in missing values with a zero
for col in cols_to_mod:
    new_df[col] = new_df[col].fillna(0)


##Boxcox transformation for skewed features
#define minimum skewness for transformation
skewness = skewness[abs(skewness) > 0.75]
#Feature names are just the index values here
skewed_features = skewness.index




#Define lambda value for boxcox transform
lam = 0.15
#Transform skewed features
for col in skewed_features: 
    new_df[col] = boxcox1p(new_df[col], lam)


    
#Removing outliers for GrLivArea
plot_relplot(dataset_df,'GrLivArea','SalePrice')
new_df = new_df.drop(dataset_df[(dataset_df['GrLivArea']>4000) & (dataset_df['SalePrice'] < 300000)].index)
plot_relplot(new_df,'GrLivArea','SalePrice')

#Drop features missing > 90% of trainset
dp.display_missing_colrows(dataset_df,0)
cols_to_drop = ['Alley','MiscFeature','PoolQC']
new_df = dataset_df.drop(cols_to_drop,axis=1)

lowcorr_cols = ['YrSold',  'Id', 'MiscVal', 'BsmtHalfBath']
#Drop low correlation vars
new_df = new_df.drop(lowcorr_cols,axis=1)

#Drop redundant, duplicate features
new_df = new_df.drop(cols_to_remove,axis=1)

area_cols = ['LotArea',
 'MasVnrArea',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'GrLivArea',
 'GarageArea',
 'WoodDeckSF',
 'OpenPorchSF']

#Remove outliers using quantile values
col_list = area_cols

plot_relplot(new_df,'GrLivArea','SalePrice')
#Remove values > 99 percentile of the population for each column in the area list
new_df = dp.remove_outliers_quant(new_df,col_list,0.99)
plot_relplot(new_df,'GrLivArea','SalePrice')


#Encode categorical variables
encoder, new_df  = dp.encode_df(new_df,'label')


#impute missing 
new_df = dp.impute_missing_mean(new_df)

#reset index since we have dropped some rows 
new_df.reset_index(drop=True, inplace=True)

####################
#Feature Engineering
####################



#Create a 2D array required for Kmeans clustering
GrLivArea_array = np.array(new_df["GrLivArea"]).reshape(-1,1)
#Use Kmeans with 5 groups for clustering data
kmeans = MiniBatchKMeans(n_clusters=5, batch_size=32).fit(GrLivArea_array)

#Assign new feature based on cluster
new_df['area_cluster'] = kmeans.predict(GrLivArea_array)

#Plot the clusters
sns.relplot(x='GrLivArea',y='SalePrice',data=new_df,
            hue='area_cluster',style='area_cluster',height=8,
            palette='YlGnBu')


#Create new feature for overall living area
new_df['TotalSF'] = new_df['TotalBsmtSF'] + new_df['1stFlrSF'] +new_df['2ndFlrSF']

#Create new feature of area * (number or rooms + overall quality)
new_df['area_rooms']=new_df['TotalSF']*(new_df['TotRmsAbvGrd']+new_df['FullBath']+new_df['HalfBath']+new_df['OverallQual'])


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

#Check train and test distributions
sns.distplot(train_df["SalePrice"])
sns.distplot(test_df["SalePrice"])

train_df.to_csv('new_train.csv',index=False)
test_df.to_csv('new_test.csv',index=False)

pickle.dump(kmeans,open ('cluster_model.pickle','wb'))
pickle.dump(encoder,open('encoder.pickle','wb'))


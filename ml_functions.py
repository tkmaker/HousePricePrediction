
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import csv
import random
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy import stats
from scipy.stats import norm
from sys import exit

#Display columns in a dataframe by sorted list of missing %
def display_missing_cols (df):
    #Get missing values
    missing_df = df.isnull().sum(axis=0).reset_index()
    #rename columns
    missing_df.columns = ["feature","miss count"]
    #Preserve features with missing counts > 0
    missing_df = missing_df.loc[missing_df["miss count"] > 0]
    #Add miss ratio column
    missing_df["miss ratio"] = missing_df["miss count"]/df.shape[0]
    #Sort descending 
    missing_df = missing_df.sort_values(by='miss count',ascending=0)
    
    print ("Columns with missing data:\n",missing_df)

#Plot heatmap
def plot_heatmap (df):
    #correlation matrix
    corrmat = df.corr()
    f, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(corrmat, vmax=.8, square=True);

#Plots top/bottom n features in correlation matrix vs Target
def plot_tb_heatmap (df,target,n,target_type):
    
    corrmat = df.corr()

    if (target_type == 'largest'):
        cols = corrmat.nlargest(n, target)[target].index
        #Column list to return
        return_cols = cols
    elif (target_type == 'smallest'):
        cols = corrmat.nsmallest(n, target)[target].index
        #Column list to return
        return_cols = cols
        #Add Target variable to list
        cols_add = corrmat.nlargest(1, target)[target].index
        cols = cols.append(cols_add)
    else :
        print ("Specify 'largest' or 'smallest' as last argument. This controls highest or lowest corellation in heatmap.")
        exit()
    
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.25)
    f, ax = plt.subplots(figsize=(12, 9))

    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},\
                 yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    return return_cols.values

#Plot norm plots
def plot_normplot (df,col):

    sns.distplot(df[col], fit=norm);
    fig = plt.figure()
    res = stats.probplot(df[col], plot=plt)

#Encode categorical variables
def encode_df(df,encode_type) : 

        encoded_df = df.copy()

        if (encode_type == "label"):
            # Encoding categorical data - label encoding
            encoder = LabelEncoder()
        elif (encode_type == "1hot"):
            encoder = OneHotEncoder()
        else :
            print ("Encode type argument not recognized: Use label or 1hot\n")
            raise Exception('exit')    
        
        for col in df.columns.values:
                #Encoding only categorical variables
                if df[col].dtypes=='object':
                    encoded_df[col] = encoder.fit_transform(df[col].astype(str))
        
        
        
        return encoder, encoded_df




#Impute missing values in a column using mean of column
def impute_missing_mean(df):
    
    #Impute missing values with mean
    mean_values = df.mean(axis=0)

    df.fillna(mean_values, inplace=True)
    
    return df

#Plot histogram for all numberical columns in dataframe
def plot_all_hist(df):
    #Historgram of all numeric vars
    numeric_df = df.select_dtypes(exclude=['object'])

    #Iterate through numerical columns - skip any with null values and plot histograms for everything else
    for col in numeric_df.columns:
        #Skip if we have null values
        if numeric_df[col].isnull().any():
            continue
        print (col)
        plt.figure()
        sns.distplot(numeric_df[col])
        plt.title(col)
        plt.show()

#Remove outliers from a dataframe based on a column list and Quantile value eg. 0.9 = Remove outliers > 90% of distribution
def remove_outliers_quant(df,col_list,quant_val):
    
    new_df = df.copy()
        
    for col in col_list:
        #Removing outliers
        q = new_df[col].quantile(quant_val)
        print ("Removing outliers greater than {0} for {1}\n".format(q,col))
        new_df = new_df[new_df[col] < q]
    
    return new_df


def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

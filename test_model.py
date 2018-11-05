# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:16:06 2018

@author: Trushant Kalyanpur
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from regression_model import regression_model
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import pandas as pd

import xgboost as xgb

import ml_functions as dp

#Import test data
test_df  = pd.read_csv('new_test.csv')

#Add cluster column we created
kmeans = pickle.load(open('cluster_model.h5','rb'))
lotArea_array = np.array(test_df["LotArea"]).reshape(-1,1)
test_df["area_cluster"]  = kmeans.predict(lotArea_array)


#Initialize  and load pre-trained model
regressor = regression_model()
model_path = 'random_forest'
regressor.loadModel(model_path)

#Impute missing values with mean
X_imputed = dp.impute_missing_mean(test_df)
X = X_imputed

#Drop sale price since this is the target value
X_test = X.drop("SalePrice",axis=1)



#Target feature
y_true = test_df["SalePrice"]


#Predicted values 

#uncomment for non-xgboost models
y_pred = regressor.predict(X_test)

#Uncomment for xgboost
#y_pred = regressor.predict(X_test.values)

#Calculate Mean Absolute Error and R2 score based on predictions
mae = mean_absolute_error(y_true,y_pred)
r2 = r2_score(y_true,y_pred)

print ("Mean absolute error=",mae)
print ("R2 score=",r2)



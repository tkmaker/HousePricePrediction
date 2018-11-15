# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 22:16:06 2018

@author: Trushant Kalyanpur
"""

import numpy as np
import pandas as pd


from regression_model import regression_model

train_df = pd.read_csv('new_train.csv')
test_df  = pd.read_csv('new_test.csv')


#Drop Sale price column
X = train_df.drop("SalePrice",axis=1)
y = train_df["SalePrice"]


#Initialize regression model class
regressor  = regression_model()
#Choose number of Kfold val
regressor.num_folds = 5

#Train model now
regressor.trainModel(X,y)


#Save model 
model_path = 'random_forest_ww46'
regressor.saveModel(model_path)

                




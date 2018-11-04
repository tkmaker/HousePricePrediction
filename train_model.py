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

model  = regression_model()
model.num_folds = 5

model.trainModel(X,y)

model_path = 'random_forest'
model.saveModel(model_path)

                




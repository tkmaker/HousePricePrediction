# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 14:12:48 2018

@author: takalyan
"""

from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

train_df = pd.read_csv('new_train.csv')
test_df  = pd.read_csv('new_test.csv')


#Drop Sale price column
X = train_df.drop("SalePrice",axis=1)
y = train_df["SalePrice"]


# Create the parameter grid based on the results of random search 
param_grid = {
    #'max_depth': [80, 90, 100, 110],
    #'max_features': [2, 3],
    #'min_samples_leaf': [3, 4, 5],
    #'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 300,500,1000]
}


# Create a based model
model = RandomForestRegressor()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                          cv = 5, verbose = 2)


grid_search.fit(X, y)

#print (grid.grid_scores_)
grid_best_params = grid_search.best_params_
print (grid_best_params)

grid_best_score = grid_search.best_score_
print (grid_best_score)

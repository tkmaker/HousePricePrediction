# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 21:52:12 2018

@author: Trushant Kalyanpur
"""

from numpy import zeros
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import time
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.learning_curve import learning_curve
#from xgboost import XGBRegressor

class regression_model (object):
    
    def __init__ (self) :
        
        self.verbose = True
        #Number of K-Folds
        self.num_folds = 5
     
        #The encoder
        self.encoder = LabelEncoder()
        
        #Random forest regression model
        self.model = RandomForestRegressor(n_estimators=25) #,min_samples_leaf=3,min_samples_split=8)
        
      
        
        #XGBoost model - Gradient Boosted Descision tree
        #self.model = XGBRegressor(max_depth=10,\
         #           min_child_weight=4, subsample=0.7, colsample_bytree=0.6,\
          #          reg_alpha= 0.1, \
           #         objective= 'reg:linear', nthread=8, scale_pos_weight=1,seed=27)

    
    #Encode categorical variables
    def encode_train_df(self,df,encode_type) : 

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
        
        #assign encoder to class
        self.encoder = encoder
        
        return encoded_df
    
    #Encode categorical variables
    def encode_test_df(self,df,encode_type) : 

        encoded_df = df.copy()

        encoder = self.encoder
        
        if (encode_type != "label" and encode_type != "1hot" ):
            print ("Encode type argument not recognized: Use label or 1hot\n")
            raise Exception('exit')    
        
        for col in df.columns.values:
                #Encoding only categorical variables
                if df[col].dtypes=='object':
                    encoded_df[col] = encoder.transform(df[col].astype(str))
        
                
        return encoded_df
    
    def trainModel (self,X,y):
        
        print ("Training model using K-Fold cross val..\n")
        
        X = np.array(X)
        
        #K-Fold Cross validation
        kf = KFold(n_splits=self.num_folds,shuffle=True, random_state=1) 
        

        #Store cross val scores
        scores = []

        
        #track model training time
        start_time = time.time()

        #Do k-fold cross val across dataset and store scores
        i = 0
        for train_index, test_index in kf.split(X):
            
            
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Fitting the model to the training set
            self.model.fit(np.array(X_train), np.array(y_train))
            
          
            #Evaluate the model
            y_pred = self.model.predict(X_test)
            r2 = r2_score(y_test,y_pred)
            
            #R2 score for train set
            y_pred_train = self.model.predict(X_train)
            r2_train = r2_score(y_train,y_pred_train)
            scores.append(r2)
            
            #Mean absolute error
            mae_train = mean_absolute_error(y_train,y_pred_train)
            mae_val = mean_absolute_error(y_test,y_pred)
            
            if (self.verbose):
                print ("Kfold iteration = {0}:\n\t R2 train score= {1:.4f}, R2 val score={2:.4f}\n\t MAE (train)={3:.4f}, MAE(val) = {4:.4f}"\
                       .format(i+1,r2_train,r2,mae_train,mae_val))
            
           
            i+=1 
            
        # plot learning curve
        self.plot_lc(X,y)    
        
        end_time = time.time()
        train_time = (end_time - start_time)/60
  
        print("\nModel train time = %f minutes" % train_time)
        print("Overall R2 Score = %f (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))
    
    def predict(self,X):
        
        y_pred = self.model.predict(X)
        return y_pred
    
    #Plot learning curve
    def plot_lc(self,X,y):
        
        train_sizes, train_scores, test_scores = learning_curve(self.model, X, y, n_jobs=1, cv=self.num_folds, \
                                                                train_sizes=np.linspace(.1, 1.0, 5), verbose=0)

        # Create means and standard deviations of training set scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)

        # Create means and standard deviations of test set scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)

        # Draw lines
        plt.plot(train_sizes, train_mean,  color="r",  label="Training score")
        plt.plot(train_sizes, test_mean, color="g", label="Cross-validation score")

        # Draw bands
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

        # Create plot
        plt.title("Learning Curve")
        plt.xlabel("Training Set Size"), plt.ylabel("Accuracy Score"), plt.legend(loc="best")
        plt.tight_layout()
        plt.show()
        
  
        
    def saveModel (self,path):
        
        print ("Saving model..\n")
        
        encoder_path = path+"_encoder.pickle"
        model_path = path+"_model.h5"
        
        pickle.dump(self.encoder,open (encoder_path,'wb'))
        pickle.dump(self.model,open (model_path,'wb'))
    
        print ("Model saved as",model_path)
        print ("Encoder saved as",encoder_path)

    def loadModel (self,path):

        print ("Loading model..\n")
        
        encoder_path = path+"_encoder.pickle"
        model_path = path+"_model.h5"
        
        self.model = pickle.load(open(model_path,'rb'))
        self.encoder = pickle.load(open(encoder_path,'rb'))

        print ("Model loaded from",path)
        print ("Encoder loaded as",encoder_path)

        
        
    
    
        
        
    
    
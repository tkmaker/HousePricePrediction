# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 12:19:43 2018

@author: takalyan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import re
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

#Import dataset
dataset_df = pd.read_csv('train_data.csv')


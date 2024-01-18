# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:44:21 2023

@author: psccgoo
"""

# data manipulation
import pandas as pd
import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt
import statistics as stats
from collections import Counter
import seaborn as sns
import math as mt

# modules for computing SMOTE
import imblearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

## loading vehicle crash data
vehicle_data = pd.read_csv(r'C:\Users\psccgoo\OneDrive - University of Leeds\ITS\Work package 6.5 (HiDrive)\TME study\Data\Vehicle data\Processed data\data_for_analysis\takeover.responses.critical.csv')

## filter out participants
vehicle_data_filtered = vehicle_data \
    .loc[vehicle_data['ppid'] != 34] \
        .loc[vehicle_data['ppid'] != 37] \
            .loc[vehicle_data['ppid'] != 47]


# load in critical data
response_vehicle_eye = pd.read_csv(r'C:\Users\psccgoo\OneDrive - University of Leeds\ITS\Work package 6.5 (HiDrive)\TME study\Data\Vehicle data\Processed data\data_for_analysis\vehicle_response_and_entropy.csv')

# select GTE and SGE and convert to numpy array
X = response_vehicle_eye[['gte.norm', 'e.norm']]
X = X.to_numpy()

# select collision and make it into an array
y = response_vehicle_eye[['collision']]
y = y.to_numpy()

# Apply SMOTE to the training set
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# convert to a dataframe 
X_resampled = pd.DataFrame(X_resampled)
y_resampled = pd.DataFrame(y_resampled)

# concatenate together
resampled = pd.concat([X_resampled, y_resampled], axis = 1) 

# save resampled data
resampled.to_csv(r'C:\Users\psccgoo\OneDrive - University of Leeds\ITS\Work package 6.5 (HiDrive)\TME study\Data\Vehicle data\Processed data\data_for_analysis\vehicle_response_and_entropy_resampled.csv', index = False)




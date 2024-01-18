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
import re

# modules for computing SMOTE
import imblearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

## VEHICLE DATA
vehicle_data = pd.read_csv(r'C:\Users\psccgoo\OneDrive - University of Leeds\ITS\Work package 6.5 (HiDrive)\TME study\Data\Vehicle data\Processed data\data_for_analysis\takeover.responses.critical.csv')

## filter out participants and drop NAs
vehicle_data_filtered = vehicle_data \
    .loc[vehicle_data['ppid'] != 34] \
        .loc[vehicle_data['ppid'] != 37] \
            .loc[vehicle_data['ppid'] != 47] \
                .dropna()

# create new variable for collision
vehicle_data_filtered['collision_new'] = np.select([vehicle_data_filtered['collision'] == True, vehicle_data_filtered['collision'] == False], [1, 0])

# PARTICIPANT DATA 
participant_info = pd.read_csv(r'C:\Users\psccgoo\OneDrive - University of Leeds\gaze_entropy_heterogenous\data\participant_info.csv')
participant_info = participant_info.iloc[:, 0:5] \
    .rename(columns = {participant_info.columns[2]: 'annual_mileage'}) \
        .rename(columns = {participant_info.columns[3]: 'license_years'}) \
            .rename(columns = {participant_info.columns[4]: 'ppid'}) # select first 5 columns, then rename mileage, license years, and ppid
        
# GAZE DATA
gaze_entropy = pd.read_csv(r'C:\Users\psccgoo\OneDrive - University of Leeds\gaze_entropy_heterogenous\data\entropy.total.csv')

# filtering out participants and leavign on critical trials
gaze_entropy_filtered = gaze_entropy \
    .loc[gaze_entropy['ppid'] != 34] \
        .loc[gaze_entropy['ppid'] != 37] \
            .loc[gaze_entropy['ppid'] != 47] \
                .loc[gaze_entropy['critical'] == True]
       
# merging participant information and gaze data
gaze_entropy_filtered.merge(participant_info, how = "left", on = 'ppid')


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




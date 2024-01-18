# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 15:44:21 2023

@author: psccgoo
"""
"""

This file will focus on a machine learning technique called Synthetic Minority Oversampling Technique (SMOTE) to account for the fact that certain data types are imbalanced.
Modelling imbalanced is tricky, because the lack of one response can bias the model. However, the response that is lacking is usually the on of interest. 
In this example, the minority class is collisions in a driving experiment. However, this method of resmapling can be applied to many other disciplines.  

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

# merging gaze data with vehicle data
modelling_data = vehicle_data_filtered.merge(gaze_entropy_filtered, how = "left", on = "trialid") \
    .dropna()

## Initial plotting
"""
The data loaded above comes from a driving experiment. In the experiment, drivers we in an automated vehicle. 
In on condition, they have to complete a distracting task during the automation; in the other condition, they just monitored the road. 
After approximately 2 minutes, drivers had to takeover from automation because a vehicle in front of them began to decelerate. To takeover, drivers had to brake.
Most drivers managed this sucessfully, however some drivers crashed. 
For furture driver monitoring systems, it would be useful if we could predict whether people are going to crash, based on their behaviour during the automated drive. 
One way to do this is through eye movements. In the datafame, *gte.norm* is a measure of eye movements. Higher values = more erratic scanning patteners; lower values = less erratic scanning patterns. 

So, can we use this metric to predict whether drivers crashed when asked to takeover from an automated vehicle?
"""
# Here we plot the collision data as a function of GTE. 
# There are very few collisions overall. 
# However, people who do have collision tend to have higher GTE. 
# This could be indicative that more erratic gaze behaviours during automation are predictive of collisions.
plt.scatter(modelling_data['gte.norm'], modelling_data['collision_new'])
plt.xlabel("Gaze Transition Entropy (Higher = more erratic eye movements)")
plt.ylabel("Collision")

# how many people crash? 21 
modelling_data \
    .loc[modelling_data['collision'] == 1] \
        .groupby(['ppid_x']) \
            .head(1)[['ppid_x']] \
                .nunique()
                
# how many crashes per persons
modelling_data.loc[modelling_data['collision'] == 1].groupby(['ppid_x']).size().reset_index(name = "number_of_collisions")
    
                




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




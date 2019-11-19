#!/usr/bin/env python
# coding: utf-8

# ## Load packages and data


import pandas as pd
import numpy as np


# seed is used for reproducibility
SEED = 7
np.random.seed(SEED)


# Loading Data
raw_df=pd.read_csv('https://aisgaiap.blob.core.windows.net/aiap5-assessment-data/traffic_data.csv')
print(raw_df.head())
# Getting dataframe columns names
variables=raw_df.columns.values
variables


# separate var into df_cat and df_num  and df_dep

df_dep=raw_df[['traffic_volume']]
#drop 'traffic_volume' and 'snow_1h' as snow_1h is zero throughout all observations

df_var=raw_df.drop(['traffic_volume','snow_1h'],axis=1)

df_num=df_var[['temp', 'rain_1h','clouds_all']]

df_cat=df_var.drop(['temp', 'rain_1h','clouds_all'],axis=1)



# ## Preprocessing data

# standardize the df_num
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df_num)
# apply scaler to training data and testing data

ndArr_num_scaled=scaler.transform(df_num)
ndArr_num_scaled

#df_num_scaled,df_num
#preprocessing.scale(df_num),df_num_scaled


# create isHoliday, a binary variable to replace holiday
df_cat['isHoliday'] = np.where(df_cat['holiday']!='None', 1, 0).astype(int)


# date_time
df_cat['dTime']=pd.to_datetime(df_cat.date_time)
df_cat['Weekday']=df_cat['dTime'].dt.weekday_name
df_cat['hr']=df_cat['dTime'].dt.hour
df_cat['mth']=df_cat['dTime'].dt.month

# all observations collected in the same year 2013
print(df_cat['dTime'].dt.year.unique())


df_cat['hr_sin'] = np.sin(df_cat.hr*(2.*np.pi/24))
df_cat['hr_cos'] = np.cos(df_cat.hr*(2.*np.pi/24))
df_cat['mth_sin'] = np.sin((df_cat.mth-1)*(2.*np.pi/12))
df_cat['mth_cos'] = np.cos((df_cat.mth-1)*(2.*np.pi/12))




# drop ['holiday','date_time','dTime', 'weather_description', 'hr', 'mth']
df_cat_cleaned=df_cat.drop(['holiday','date_time','dTime', 'weather_description', 'hr', 'mth'],axis=1)


df_cat_cleaned


df_cat_with_dummies = pd.get_dummies(df_cat_cleaned, drop_first=True)
df_cat_with_dummies


ndArr_num_scaled



#convert to numpy.ndarray
ndArr_cat_with_dummies=df_cat_with_dummies.to_numpy()
ndArr_cat_with_dummies


#combine cat and num inputs 
ndArr_inputs=np.concatenate((ndArr_num_scaled,ndArr_cat_with_dummies),axis=1)



# columns names 
varNames=np.concatenate((df_num.columns.values, df_cat_with_dummies.columns.values), axis=None)


# reshape varNames
varNames=varNames.reshape(-1,1)
varNames.shape


# Shuffle the indices of the data, so the data is evenly distributed
shuffled_indices = np.arange(ndArr_inputs.shape[0])
np.random.shuffle(shuffled_indices)

# Use the shuffled indices to shuffle the inputs and targets.
shuffled_inputs = ndArr_inputs[shuffled_indices]

#convert target df_dep to ndArr_dep
ndArr_dep=df_dep.to_numpy()
shuffled_targets = ndArr_dep[shuffled_indices]



# ### Split the dataset into train, validation, and test

# Count the total number of samples
samples_count = shuffled_inputs.shape[0]

# Count the samples in each subset, assuming we want 80-10-10 distribution of training, validation, and test.
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)

# The 'test' dataset contains all remaining data.
test_samples_count = samples_count - train_samples_count - validation_samples_count

# Create variables that record the inputs and targets for training
# In our shuffled dataset, they are the first "train_samples_count" observations
train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

# Create variables that record the inputs and targets for validation.
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

# Create variables that record the inputs and targets for test.
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]


# Save the three datasets in *.npz.
np.savez('traffic_data_train', inputs=train_inputs, targets=train_targets)
np.savez('traffic_data_validation', inputs=validation_inputs, targets=validation_targets)
np.savez('traffic_data_test', inputs=test_inputs, targets=test_targets)


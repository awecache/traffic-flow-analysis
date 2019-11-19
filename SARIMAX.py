#ARIMA model

import sys
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np

inputs=sys.argv

periodicity=int(sys.argv[2]) #for m in SARIMAX model

raw_df=pd.read_csv('https://aisgaiap.blob.core.windows.net/aiap5-assessment-data/traffic_data.csv')

#size of data set
size=int(inputs[1])
data=raw_df.copy()
data.rename(columns={'date_time': 'date'}, inplace=True)

data.date = pd.to_datetime(data.date,infer_datetime_format=True)
data.set_index('date', inplace=True)

data_dummies=pd.get_dummies(data,drop_first=True)

data_dummies=data_dummies[:size]



#Key features are obtain from Exploratory Data Analysis
#through feature selection using f-regression
# ['temp', 'weather_main_Haze', 'weather_description_haze','weather_description_scattered clouds']
df_exog=data_dummies[['temp', 'weather_main_Haze', 'weather_description_haze','weather_description_scattered clouds']]
df_target=data_dummies[['traffic_volume']]


# split data into training and testing
# Create Train and Test
train =df_target[:round(size*0.85)]
test = df_target[round(size*0.85):]


train_exog=df_exog[:round(size*0.85)]
test_exog=df_exog[round(size*0.85):]


import pmdarima as pm

# SARIMAX Model
sxmodel = pm.auto_arima(train, exogenous=train_exog,
                           start_p=1, start_q=1,
                           test='adf',
                           max_p=10, max_q=10, m=periodicity,
                           start_P=0, seasonal=True,
                           d=None, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

sxmodel.summary()



#Forecast
#forcast period n
n= round(0.15*size)
fc, confint = sxmodel.predict(n_periods=n,
                             exogenous=test_exog,
                             return_conf_int=True)


# Accuracy metrics
from statsmodels.tsa.stattools import acf

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    mse = np.mean((forecast - actual)**2)             # MSE
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax

    

    #std
    std=np.std(actual)


    print("The Standard deviation of target value, Std is : %.1f" %std)
    print("The RMSE of our prediction is : %.1f" %rmse)
    print("The MSE of our prediction is : %.1f" %mse)
    r=rmse/std
    print("Ratio of RMSE to Std : %.2f" %r )
    print("If the RMSE of our predictions is close to the standard deviation of our target in magnitude, the model prediction is acceptable.")





forecast_accuracy(fc, test.traffic_volume.values)




#!/usr/bin/env python
# coding: utf-8


import sys

import numpy as np

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

inputs=sys.argv

iter=int(inputs[1])
past=int(sys.argv[2]) #recommended 48 or more
future=int(sys.argv[3]) # 24 


data = pd.read_csv('data_cleaned.csv', date_parser = True)



data_training = data[data['date_time']<'2013-09-20'].copy()
data_test = data[data['date_time']>='2013-09-20'].copy()


data_training = data_training.drop(['date_time'], axis = 1)
data_training.shape, data_test.shape


scaler = MinMaxScaler()
data_training = scaler.fit_transform(data_training)
data_training.shape


X_train = []
y_train = []


for i in range(past, data_training.shape[0]-future):
    X_train.append(data_training[i-past:i])
    y_train.append(data_training[i:i+future, 4])


X_train, y_train = np.array(X_train), np.array(y_train)


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


#create a Sequential model (a linear stack of layers)
regressor = Sequential()

regressor.add(LSTM(units = 60, activation = 'relu', return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 70, activation = 'relu', return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 90, activation = 'relu', return_sequences = True))
regressor.add(Dropout(0.4))

regressor.add(LSTM(units = 120, activation = 'relu'))
regressor.add(Dropout(0.5))

regressor.add(Dense(units = future))


regressor.summary()


regressor.compile(optimizer='adam', loss = 'mean_squared_error')



regressor.fit(X_train, y_train, epochs=iter , batch_size=120)




data_test=data_test.drop(['date_time'],axis=1)



data_test = scaler.fit_transform(data_test)



X_test, y_test = [], []

for i in range(past, data_test.shape[0]-future):
    X_test.append(data_test[i-past:i])
    y_test.append(data_test[i:i+future, 4])



X_test, y_test = np.array(X_test), np.array(y_test)


y_pred = regressor.predict(X_test)


scale=1/scaler.scale_[4]


y_pred = y_pred*scale
y_true = y_test*scale



from sklearn.metrics import mean_squared_error

def evaluate_model(y_true, y_predicted):
    scores = []
    
    #calculate rmse for each hours
    for i in range(y_true.shape[1]):
        mse = mean_squared_error(y_true[:, i], y_predicted[:, i])
        rmse = np.sqrt(mse)
        scores.append(rmse)
    
    #calculate rmse for all predictions
    total_score = 0
    for row in range(y_true.shape[0]):
        for col in range(y_predicted.shape[1]):
            total_score = total_score + (y_true[row, col] - y_predicted[row, col])**2
    mean_score = np.sqrt(total_score/(y_true.shape[0]*y_predicted.shape[1]))
    
    return mean_score, scores

#std
std=np.std(y_true)

score=evaluate_model(y_true, y_pred)

print("The Standard deviation of target value, Std is : %.1f" %std)
print("The RMSE of our prediction is : %.1f" %score[0])
print("The MSE of our prediction is : %.1f" %score[0]**2)
r=score[0]/std
print("Ratio of RMSE to Std : %.2f" %r)
print("If the RMSE of our predictions is close to the standard deviation of our target in magnitude, the model prediction is acceptable.")





















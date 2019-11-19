#!/usr/bin/env python
# coding: utf-8


import sys

import numpy as np
#import tensorflow
import tensorflow as tf


# User inputs
# $maxEpochs $layerSize $tolerance $lossFun
inputs=sys.argv

#max_epochs = 500
maxEpochs=int(inputs[1]) 
#hidden_layer_size = 50
layerSize=int(inputs[2]) 
#loss='mean_squared_error'
lossFun=inputs[4] 
#patience=50
tolerance=int(inputs[3]) 

# ### Load Data

# create npz for tensorflow model
npz = np.load('traffic_data_train.npz')


train_inputs = npz['inputs'].astype(np.float)
train_targets = npz['targets'].astype(np.int)

npz = np.load('traffic_data_validation.npz')
validation_inputs, validation_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)

npz = np.load('traffic_data_test.npz')
test_inputs, test_targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)



# Set the input and output sizes
input_size = train_inputs.shape[1]
output_size = train_targets.shape[1]

# hidden_layer_size = (input_size+output_size)/2
hidden_layer_size = layerSize
    
# model
model = tf.keras.Sequential([
    # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
    # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    # the final layer is no different, we just make sure to activate it with softmax
    tf.keras.layers.Dense(output_size, activation='linear') # output layer
])


#optimizer and the loss function
model.compile(optimizer='adam', loss=lossFun, metrics=['mse'])

batch_size = 100

# maximum number of training epochs
max_epochs = maxEpochs

# an early stopping mechanism
# monitor the validation loss and stop the training process the first time the validation loss starts incresing
early_stopping = tf.keras.callbacks.EarlyStopping(patience=tolerance)

# fit the model
history=model.fit(train_inputs, 
          train_targets, 
          batch_size=batch_size, 
          epochs=max_epochs, 
          callbacks=[early_stopping], # early stopping
          validation_data=(validation_inputs, validation_targets), # validation data
          verbose = 2 
          )  



# model evaluation
train_mse = model.evaluate(train_inputs, train_targets, verbose=0)
validation_mse = model.evaluate(validation_inputs, validation_targets, verbose=0)
test_mse = model.evaluate(test_inputs, test_targets, verbose=0)



#std
std=np.std(test_targets)


print("The Standard deviation of target value, Std is : %.1f" %std)
print("The RMSE of our prediction is : %.1f" %test_mse[1]**0.5)
print("The MSE of our prediction is : %.1f" %test_mse[1])
r=(test_mse[1]**0.5)/std
print("Ratio of RMSE to Std : %.2f" %r)
print("If the RMSE of our predictions is close to the standard deviation of our target in magnitude, the model prediction is acceptable.")

#good for estimating heavy traffic
#in most cases, we are more interested in siutations where the traffic is heavy







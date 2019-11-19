#!/usr/bin/env bash

echo " For ARIMA with exogenous terms, please enter 1"
echo " For Deep Neural Network, please enter 2"
echo " For Recurrent Neural Network, please enter 3"
read -p " 1 , 2 or 3 ?  " option


if [ $option -eq 1 ];
then
    echo "ARIMA with seaonality and exogenous terms(SARIMAX)"
    echo "Please set the parameters for the neural network..."
    read -p "Number of samples used for training and validating model (recommended >200, <8573): " size
    size=${size:-500}
    if [ $size -gt 8573 ]
    then
        size=8573
    fi 

    read -p "number of observations per seasonal cycle (i.e. 24, 168, 8766): " m
    m=${m:-24}

    echo "Training..."
    python3 ./SARIMAX.py $size $m
    
    read -n 1 -r -s -p "press any key to continue "
    echo " "

elif [ $option -eq 2 ]
then
    echo "Deep Neural Netwrok"
    echo "Please set the parameters for the neural network..."

    # echo "Max number of iterations for training : "
    # read maxEpochs

    read -p "Max number of iterations for training : " maxEpochs
    maxEpochs=${maxEpochs:-800}

    read -p "Size of hidden layer : " layerSize
    layerSize=${layerSize:-50}

    read -p "Tolerance for early stopping :" tolerance
    tolerance=${tolerance:-50}

    read -p "Choice of loss function : " lossFun
    lossFun=${lossFun:-mean_squared_error}

    echo "$maxEpochs $layerSize $tolerance $lossFun  "

    echo "loading data and creating npz files...."
    python3 ./data_prep.py

    echo "training model... "
    # python model_eval.py $maxEpochs $layerSize $tolerance $lossFun
    python3 ./model_eval.py $maxEpochs $layerSize $tolerance $lossFun



    #max_epochs = 1000
    #hidden_layer_size = 50
    #loss='mean_squared_error'
    #patience=50
    
    read -n 1 -r -s -p "press any key to continue "
    echo " "
else

    echo "Recurrent Neural Netwrok, LSTM"
    echo "Please enter the parameters for recurrent neural network..."

    python3 ./data_prep2.py

    read -p "Number of iterations for training : " epochs
    epochs=${epochs:-50}

    echo "data are recorded hourly"
    read -p "Number of past data points for prediction ( recommended > 24) : " past
    past=${past:-48}

    read -p "Number of hours of traffic condition for prediction : " future
    future=${future:-24}

    echo "runnng Recurrent Neural Netwrok LSTM"
    python3 ./lstm.py $epochs $past $future

    read -n 1 -r -s -p "press any key to continue " 
    echo " "
fi








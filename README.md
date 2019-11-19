# Time Series for Traffic flow Analysis

## For virtualenv to install all files in the requirements.txt file.

cd to the directory where requirements.txt is located
activate your virtualenv
run: pip install -r requirements.txt in your shell

## run the programme
run ./run.sh  or .run.sh in bash 

you may replace python3 with python in run.sh script if run.sh fails to execute i.e. python3 permission denies

choose among three models (SARIMAX, deep neural network and recurrent neural network) for model prediction


## Input hyperparameters

Enter without passing in a value will result in model using default values for hyperparameters

Here are some other possible loss functions for deep neural network model :
mean_absolute_error
hinge
squared_hinge
mean_absolute_percentage_error
mean_squared_logarithmic_error
logcosh
kullback_leibler_divergence


## SARIMAX
Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors model has been implemented but the parameters are hard to determined especially for problems with large number of data set. Modelling with SARIMAX may be computationally intensive.
There are several hyperparameters in the model need to be determined

ARIMA models are made up of three different terms:

p: The order of the auto-regressive (AR) model (i.e., the number of lag observations)
d: The degree of differencing.
q: The order of the moving average (MA) model. This is essentially the size of the “window” function over your time series

To model seasonality three additional parameters P, D, Q which heavily resembles that of ARIMA are introduced.

In addition m, the number of observations per seasonal cycle must be determined before modelling
Typically m corresponds to
7 - daily
12 - monthly
52 - weekly

SARIMAX may be used as as a baseline for prediction accuracy.

In the model, data is preprocessed and one-hot encoded. The key feastures based on our exploratory data analysis are selected for exogenous regressors.

There are more than 8 hyperparameters need to be tuned for optimizating our model. Thankfully most of them are taken care of by the autoregressor model from pmdarima.api

We can focus on tuning m and the sample size for optimizing our model



## Outline of model development:

For deep neural network (DNN):
1. Data is divied into categorical and numerical data and treated separately. 
2. Numberical data is normalized.
3. One hot encoding is applied on categorical data.
4. Since all data is collected in 2013, we can ignore the year. Dates are  then represented using a combination of sin and cos function in order to charaterize the continuity of time. This enables us to parameterize date instead of treating date as a categorical variable
4. Numerical and categorical data are combined
5. Combined Data is shuffled and splitted into 3 for training, validation and testing and save as 3 npz files which will be used for model building and validation subsequently.
7. model is built using training data and validation data.
8. early stopping criteria is applied when performance metrix of validation worsens and exceeds the treshold set by user. This prevents model from overfitting.
9. Tensorflow is used for build deep neural netwrok.
10. model is built with 3 layers: 2 hidden layers with relu as their activation function. The activation function for outer layer is linear as the model is used for regression. Adam, an adaptive moment estimation is used as the optimization algorithm.

### default hyperparameters for DNN model 
max_epochs = 1000
hidden_layer_size = 50
loss='mean_squared_error'
patience=50


For Recursive Neural Network (RNN):
1. Data is imported from url and date variable is used to indexed data
2. Data is selected from January till October 26 as missing data are suspected in later dates
3. The more significant variables are selected based on feature selection in exploratory data analysis earlier.
4. LSTM is picked as it is a more suitable neural network model for time-series analysis as the node not only receive signal from inputs but also from past data. This models data having 'memory' of its past inputs.
5. Data is divided into training and test for measuring model performance.
6. Users can choose the number of iteration for model training
7. User may also specify the number of past data used for prediction and the number of data to be predicted. Data is measured every hour.
8. Metric for performance used is square mean error and root mean square error
9. This allows us to compare both deep neural network and recurrent neural network models.
10. In addition, RMSE can be compared to the standard deviation of the target variable to give us a sense of the effectiveness of our model. The closer RMSE to the statndard deveiation of our target variable, the more accurate is our model in prediction.

### Default hyperparameter:
past =48    (2 days of past traffic data required for prediction)
future =24  (1 day of traffic data to be predicted)
Iter= 50   



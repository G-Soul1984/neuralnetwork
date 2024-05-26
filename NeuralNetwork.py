import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pandas as pd
import numpy as np
from yahoo_fin import stock_info as si
from datetime import datetime

# Define the symbol, start date, and end date

ticker = 'AAPL'  # yahoo ticker
StartDate = datetime(2022, 1,1)
FutureTimeSteps = 5

# Fetch historical stock data
StockData = si.get_data(ticker, start_date=StartDate)

# Check the shape of the data
print(StockData.shape)

StockData['TradeDate']=StockData.index

FullData = StockData[['close']].values

from sklearn.preprocessing import StandardScaler, MinMaxScaler

sc = MinMaxScaler()
DataScaler = sc.fit(FullData)
X = DataScaler.transform(FullData)

X_samples = list()
y_samples = list()

NumerOfRows = len(X)

TimeSteps = 10

for i in range(TimeSteps , NumerOfRows - FutureTimeSteps):
    x_sample = X[i-TimeSteps:i]
    y_sample = X[i:i+FutureTimeSteps]
    X_samples.append(x_sample)
    y_samples.append(y_sample)

X_data=np.array(X_samples)
X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)

y_data=np.array(y_samples)

TestingRecords = 5

X_train=X_data[:-TestingRecords]
X_test=X_data[-TestingRecords:]
y_train=y_data[:-TestingRecords]
y_test=y_data[-TestingRecords:]

TimeSteps=X_train.shape[1]
TotalFeatures=X_train.shape[2]


# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, InputLayer
 
# Initialising the RNN
regressor = Sequential()
 
# Adding the First input hidden layer and the LSTM layer
# return_sequences = True, means the output of every time step to be shared with hidden next layer
regressor.add(InputLayer(shape = (TimeSteps, TotalFeatures)))
regressor.add(LSTM(units = 10, activation = 'relu', return_sequences=True))

# Adding the Second Second hidden layer and the LSTM layer
regressor.add(LSTM(units = 5, activation = 'relu', return_sequences=True))

# Adding the Second Third hidden layer and the LSTM layer
regressor.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))
 
 
# Adding the output layer
regressor.add(Dense(units = FutureTimeSteps))
 
# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
 
##################################################
 
import time
# Measuring the time taken by the model to train
StartTime=time.time()
 
# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)

EndTime=time.time()
print("## Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ##')

# Making predictions on test data
predicted_Price = regressor.predict(X_test)
predicted_Price = DataScaler.inverse_transform(predicted_Price)
print('#### Predicted Prices ####')
print(predicted_Price)
 
# Getting the original price values for testing data
orig=y_test.reshape(y_test.shape[0], y_test.shape[1])
orig=DataScaler.inverse_transform(orig)
print('\n#### Original Prices ####')
print(orig)

Last10DaysPrices = np.array(FullData[-10:])
Last10DaysPrices=Last10DaysPrices.reshape(-1, 1)

# Scaling the data on the same level on which model was trained
X_test=DataScaler.transform(Last10DaysPrices)

NumberofSamples=1
TimeSteps=X_test.shape[0]
NumberofFeatures=X_test.shape[1]
# Reshaping the data as 3D input
X_test=X_test.reshape(NumberofSamples,TimeSteps,NumberofFeatures)

# Generating the predictions for next 5 days
NextDaysPrice = regressor.predict(X_test)

# Generating the prices in original scale
NextDaysPrice = DataScaler.inverse_transform(NextDaysPrice)
print(*NextDaysPrice)
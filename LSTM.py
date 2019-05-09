from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
test_size = 0.33
Bdata = pd.read_csv('C:/Users/yanru/Desktop/Stock_LSTM/Apple.csv')
train, test = train_test_split(Bdata, test_size=test_size, shuffle=False)

# What metric to model and predict
metric = 'Close'
# Maximum number of timestamps to learn from
timestamp = 64
# Number of layers in LSTM
layers = 3
# Number of neurons in each layer
neurons = 264
# Number of times the entire data needs to be looped upon
epochs = 64
# Weights updated after n rows
batch_size = 16
# Percentage of data to validate the model when training
validation_split = 0.1
# Regularisation parameter
dropout = 0.2
optimizer = 'adam'
loss = 'mean_squared_error'

loc = train.columns.get_loc(metric)
train_data = train.iloc[:, loc:loc + 1].values

sc = MinMaxScaler(feature_range=(0, 1))
train_data = sc.fit_transform(train_data)

train_x = []
train_y = []
for i in range(0, len(train_data) - timestamp - 1):
    train_x.append(train_data[i:i + timestamp, 0])
    train_y.append(train_data[i + timestamp, 0])
train_x, train_y = np.array(train_x), np.array(train_y)

train_x = np.reshape(train_x, (train_x.shape[0],train_x.shape[1],1))

regressor = Sequential()
regressor.add(LSTM(neurons, return_sequences=True, input_shape=(train_x.shape[1],1)))
regressor.add(Dropout(dropout))

if layers > 2:
    for i in range(2, layers):
        regressor.add(LSTM(neurons, return_sequences=True))
        regressor.add(Dropout(dropout))

regressor.add(LSTM(neurons))
regressor.add(Dropout(dropout))

regressor.add(Dense(units=1))

regressor.compile(optimizer=optimizer, loss=loss)

regressor.summary()

regressor.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

test_data = pd.concat((train[len(train) - timestamp:][metric], test[metric]), axis=0)
test_data = test_data.values
test_data = test_data.reshape(-1, 1)

test_data = sc.transform(test_data)

test_x = []
test_y = []
for i in range(0, len(test_data) - timestamp - 1):
    test_x.append(test_data[i: i + timestamp, 0])
    test_y.append(test_data[i + timestamp, 0])
test_x, test_y = np.array(test_x), np.array(test_y)

test_x = np.reshape(test_x, (test_x.shape[0],test_x.shape[1],1))

predicted_price = sc.inverse_transform(regressor.predict(test_x))
real_stock_price = test[metric]


def plot_pred(metric, real_stock_price, predicted_price):
    days = np.array(test['Date'], dtype="datetime64[ms]")
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(days, real_stock_price, 'bo-', label='actual')
    axes.plot(days, predicted_price, 'r+-', label='predicted')
    axes.set_title('Apple')
    fig.autofmt_xdate()
    plt.legend()
    plt.show()


plot_pred(metric, real_stock_price, predicted_price)






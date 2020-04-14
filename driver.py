import pandas as pd
import numpy as np
import keras
import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go


df = pd.read_csv('data/2019-plu-total-hab-data.csv')

# set date as index
df['Date'] = pd.to_datetime(df['Current Year Week Ending'])
df = df.sort_values('Date')
df.set_axis(df['Date'], inplace=True)

# use only rows for Total U.S and Conventional Type
df = df[df['Type'].str.match('Conventional')]
df = df[df['Geography'].str.match('Total U.S.')]

df.drop(
    columns=['Geography', 'Current Year Week Ending', 'Timeframe', 'Type', '4046 Units',
             '4225 Units', '4770 Units', 'TotalBagged Units', 'SmlBagged Units', 'LrgBagged Units',
             'X-LrgBagged Units'], inplace=True)

print(df)
asp_data = df['ASP Current Year'].values
asp_data = asp_data.reshape((-1, 1))

# use 80% for training, 20 for testing
split_percent = 0.80
split = int(split_percent * len(asp_data))

asp_train = asp_data[:split]
asp_test = asp_data[split:]

date_train = df['Date'][:split]
date_test = df['Date'][split:]


look_back = 2


train_generator = TimeseriesGenerator(asp_train, asp_train, length=look_back, batch_size=1)
test_generator = TimeseriesGenerator(asp_test, asp_test, length=look_back, batch_size=1)

model = Sequential()
activation = 'linear'
loss = 'mse'
optimizer = 'SGD'

model.add(
    LSTM(10,
         activation=activation,
         input_shape=(look_back, 1))
)
model.add(Dense(1))
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

num_epochs = 1000
model.fit_generator(train_generator, epochs=num_epochs, verbose=1)

prediction = model.predict_generator(test_generator, 10)

close_train = asp_train.reshape((-1))
close_test = asp_test.reshape((-1))
prediction = prediction.reshape((-1))



trace1 = go.Scatter(
    x = date_train,
    y = close_train,
    mode = 'lines',
    name = 'Data'
)
trace2 = go.Scatter(
    x = date_test,
    y = prediction,
    mode = 'lines',
    name = 'Prediction'
)
trace3 = go.Scatter(
    x = date_test,
    y = close_test,
    mode='lines',
    name = 'Ground Truth'
)
layout = go.Layout(
    title = "Avocado ASP Prediction epochs: %d, lookback: %d" % (num_epochs, look_back),
    xaxis = {'title' : "Date"},
    yaxis = {'title' : "Average Selling Price ($)"}
)
fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
fig.show()

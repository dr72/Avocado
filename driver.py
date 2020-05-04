import numpy as np
import keras
import tensorflow as tf
import tkinter as tk
from tkinter import ttk
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go

window = tk.Tk()

window.title("Avocado Pricing Prediction")
window.geometry('400x400')

root = tk.Frame(window)
root.grid(row=0, column=0)
root.grid(padx=20, pady=20)

labelEpochs = tk.Label(root, text="Epochs")
labelEpochs.grid(column=0, row=0, sticky='W', pady=10)
scaleEpochs = tk.Scale(root, from_=1, to=5000, length=200, orient="horizontal")
scaleEpochs.grid(column=1, row=0, sticky='W', pady=10, padx=10)

labelLookBack = tk.Label(root, text="Look Back")
labelLookBack.grid(column=0, row=2, sticky='W', pady=10)
scaleLookBack = tk.Scale(root, from_=1, to=10, length=200, orient="horizontal")
scaleLookBack.grid(column=1, row=2, sticky='W', pady=10, padx=10)

labelLoss = tk.Label(root, text="Loss Function")
labelLoss.grid(column=0, row=4, sticky='W', pady=10)
lossOptions = ["mean_squared_error", "mean_absolute_error", "mean_squared_logarithmic_error", "hinge"]
lossCombo = ttk.Combobox(root, values=lossOptions, width=30)
lossCombo.current(0)
lossCombo.grid(column=1, row=4, sticky='W', pady=10, padx=10)

labelActivation = tk.Label(root, text="Activation")
labelActivation.grid(column=0, row=6, sticky='W', pady=10)
activationOptions = ["linear", "relu", "softmax"]
activationCombo = ttk.Combobox(root, values=activationOptions, width=30)
activationCombo.current(0)
activationCombo.grid(column=1, row=6, sticky='W', pady=10, padx=10)

labelOptimization = tk.Label(root, text="Optimization")
labelOptimization.grid(column=0, row=8, sticky='W', pady=10)
optimizationOptions = ["SGD", "Adam", "Adagrad"]
optimizationCombo = ttk.Combobox(root, values=optimizationOptions, width=30)
optimizationCombo.current(0)
optimizationCombo.grid(column=1, row=8, sticky='W', pady=10, padx=10)


def run():
    import pandas as pd
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
    dates = df
    df.drop(columns=['Date'])
    # print(df)
    asp_data = df['ASP Current Year'].values
    total_data = df['Total Bulk and Bags Units'].values

    full_data = np.stack((asp_data, total_data))
    test = []
    for x in range(len(asp_data)):
        test.append([asp_data[x], total_data[x]])

    pd = pd.DataFrame(test)
    # print(pd)
    scaler = MinMaxScaler(feature_range=(0, 1))
    pd = scaler.fit_transform(pd.values)

    # print(full_data)
    # use 80% for training, 20 for testing
    split_percent = 0.80
    split = int(split_percent * len(test))

    asp_train = pd[:split, 0]

    asp_test = pd[split:, 0]

    date_train = df['Date'][:split]
    date_test = df['Date'][split:]

    test_train = pd[:split]
    test_test = pd[split:]

    look_back = scaleLookBack.get()

    train_generator = TimeseriesGenerator(test_train, asp_train, length=look_back, batch_size=1)
    test_generator = TimeseriesGenerator(test_test, asp_test, length=look_back, batch_size=1)

    model = Sequential()
    activation = activationCombo.get()
    loss = lossCombo.get()
    optimizer = optimizationCombo.get()

    model.add(
        LSTM(20,
             activation=activation,
             input_shape=(look_back, 2))
    )
    model.add(Dense(1, activation=activation))
    model.add(Dense(1, activation=activation))
    model.add(Dense(1, activation=activation))
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    num_epochs = scaleEpochs.get()
    model.fit_generator(train_generator, epochs=num_epochs, verbose=1)

    prediction = model.predict_generator(test_generator, 10)

    close_train = asp_train.reshape((-1))
    close_test = asp_test.reshape((-1))
    prediction = prediction.reshape((-1))

    trace1 = go.Scatter(
        x=date_train,
        y=close_train,
        mode='lines',
        name='Data'
    )
    trace2 = go.Scatter(
        x=date_test,
        y=prediction,
        mode='lines',
        name='Prediction'
    )
    trace3 = go.Scatter(
        x=date_test,
        y=close_test,
        mode='lines',
        name='Ground Truth'
    )
    layout = go.Layout(
        title="Avocado ASP Prediction -- epochs: %d | lookback: %d | activation: %s" % (num_epochs, look_back, activation),
        xaxis={'title': "Date"},
        yaxis={'title': "Average Selling Price ($)"}
    )
    fig = go.Figure(data=[trace1, trace2, trace3], layout=layout)
    fig.show()


btStart = tk.Button(root, text="Start", command=run)
btStart.grid(column=1, row=10, pady=10, columnspan=2, sticky="we")

window.mainloop()

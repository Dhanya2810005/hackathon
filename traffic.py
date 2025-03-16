import pandas as pd
import numpy as np

timestamps = pd.date_range(start="2023-01-01", periods=100, freq="H")
speeds = np.random.randint(30, 100, size=100) 

df = pd.DataFrame({
    "timestamp": timestamps,
    "speed": speeds
})


df.to_csv("traffic_data.csv", index=False)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df.dropna(inplace=True)
df.sort_values(by="timestamp", inplace=True)

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(df["timestamp"], df["speed"])
plt.title("Traffic Speed Over Time")
plt.xlabel("Time")
plt.ylabel("Speed (KMPH)")
plt.show()

from statsmodels.tsa.seasonal import seasonal_decompose

period = 24  
if len(df["speed"]) >= 2 * period:
    result = seasonal_decompose(df["speed"], model="additive", period=period)
    result.plot()
    plt.show()
else:
    print(f"Not enough data for seasonal decomposition. Requires at least {2 * period} observations.")

from statsmodels.tsa.stattools import adfuller

result = adfuller(df["speed"])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")


train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train["speed"], order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))

from sklearn.metrics import mean_absolute_error, mean_squared_error


mae = mean_absolute_error(test["speed"], forecast)

mse = mean_squared_error(test["speed"], forecast)  
rmse = mse ** 0.5  
print(f"MAE: {mae}, RMSE: {rmse}")


import streamlit as st

st.title("Traffic Speed Forecasting")
st.write("Forecasted Values:")
st.line_chart(forecast)
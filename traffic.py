import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

timestamps = pd.date_range(start="2023-01-01", periods=100, freq="H")
speeds = np.random.randint(30, 100, size=100) 

df = pd.DataFrame({
    "timestamp": timestamps,
    "speed": speeds
})

df["timestamp"] = pd.to_datetime(df["timestamp"])
df.dropna(inplace=True)
df.sort_values(by="timestamp", inplace=True)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df["timestamp"], df["speed"])
ax.set_title("Traffic Speed Over Time")
ax.set_xlabel("Time")
ax.set_ylabel("Speed (KMPH)")
st.pyplot(fig)

period = 24  
if len(df["speed"]) >= 2 * period:
    result = seasonal_decompose(df["speed"], model="additive", period=period)
    fig = result.plot()
    st.pyplot(fig)
else:
    st.write(f"Not enough data for seasonal decomposition. Requires at least {2 * period} observations.")

result_adf = adfuller(df["speed"])
st.write(f"ADF Statistic: {result_adf[0]}")
st.write(f"p-value: {result_adf[1]}")

train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

model = ARIMA(train["speed"], order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))

mae = mean_absolute_error(test["speed"], forecast)
mse = mean_squared_error(test["speed"], forecast)
rmse = mse ** 0.5  
st.write(f"MAE: {mae}, RMSE: {rmse}")

st.title("Traffic Speed Forecasting")
st.line_chart(forecast)

correlation_matrix = df.corr()
st.write("Correlation Matrix:")
st.write(correlation_matrix)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

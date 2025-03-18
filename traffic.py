import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st


API_KEY = "aKpt6KemLwa0TMjuOF4FDelFEXlSK8L6"
API_URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
CSV_FILE = "traffic_data.csv"

params = {
    "point": "52.41072,4.84239",  
    "key": API_KEY
}


response = requests.get(API_URL, params=params)
data = response.json()


if "flowSegmentData" in data:
    speeds = data["flowSegmentData"]["currentSpeed"]  # Get current speed
    timestamp = pd.Timestamp.now()
    new_data = pd.DataFrame({"timestamp": [timestamp], "speed": [speeds]})
    

    try:
        existing_data = pd.read_csv(CSV_FILE)
        existing_data = pd.concat([existing_data, new_data], ignore_index=True)
    except FileNotFoundError:
        existing_data = new_data
    
    existing_data.to_csv(CSV_FILE, index=False)
else:
    raise ValueError("Invalid API response. Check API key and request parameters.")

# Load CSV data
df = pd.read_csv(CSV_FILE)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.dropna(inplace=True)
df.sort_values(by="timestamp", inplace=True)

# Step 3: Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 6))
plt.plot(df["timestamp"], df["speed"], label="Traffic Speed")
plt.title("Traffic Speed Over Time")
plt.xlabel("Time")
plt.ylabel("Speed (KMPH)")
plt.legend()
plt.show()

# Seasonal decomposition (if enough data points)
period = 24
if len(df) >= 2 * period:
    result = seasonal_decompose(df["speed"], model="additive", period=period)
    result.plot()
    plt.show()
else:
    print(f"Not enough data for seasonal decomposition. Requires at least {2 * period} observations.")


result = adfuller(df["speed"])
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")


df["hour"] = df["timestamp"].dt.hour
df["day"] = df["timestamp"].dt.date
heatmap_data = df.pivot_table(values="speed", index="day", columns="hour", aggfunc=np.mean)
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".1f")
plt.title("Traffic Speed Heatmap")
plt.xlabel("Hour of the Day")
plt.ylabel("Date")
plt.show()


train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

model = ARIMA(train["speed"], order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))


mae = mean_absolute_error(test["speed"], forecast)
rmse = mean_squared_error(test["speed"], forecast) ** 0.5
print(f"MAE: {mae}, RMSE: {rmse}")


st.title("Traffic Speed Forecasting Web App")
st.write("Forecasted Traffic Speeds:")
st.line_chart(forecast)

st.write("### Traffic Speed Heatmap")
st.pyplot(sns.heatmap(heatmap_data, cmap="coolwarm", annot=True, fmt=".1f").figure)
